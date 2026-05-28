from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, cos, exp, pi, sin, sqrt
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

from .room import Room
from .simulator import SensorPacket


@dataclass
class Estimate:
    time_ms: int
    x_m: float
    y_m: float
    heading_rad: float
    confidence: float
    motion_state: str
    tof_used: bool
    best_tof_error_m: Optional[float]


class RuleBasedEstimator:
    """First-pass coordinate estimator.

    This is intentionally simple and transparent:
      1. Integrate gyro z for heading.
      2. Use IMU motion energy to decide stationary vs moving.
      3. Predict position using a nominal walking speed when moving.
      4. If ToF is valid, search nearby grid cells whose expected raycast
         distance matches the measurement and blend toward the best cell.
      5. Keep confidence as a bounded scalar, not a truth claim.
    """

    def __init__(
        self,
        room: Room,
        start_x_m: float = 0.5,
        start_y_m: float = 0.5,
        start_heading_rad: float = 0.0,
        grid_spacing_m: float = 0.25,
        nominal_speed_mps: float = 0.5,
        max_tof_range_m: float = 1.2,
        gyro_bias_calibration_samples: int = 20,
    ):
        self.room = room
        self.x_m = start_x_m
        self.y_m = start_y_m
        self.heading_rad = start_heading_rad
        self.grid_spacing_m = grid_spacing_m
        self.nominal_speed_mps = nominal_speed_mps
        self.max_tof_range_m = max_tof_range_m
        self.confidence = 1.0
        self.last_time_ms: Optional[int] = None
        self.gyro_bias_rad_s = 0.0
        self._calibration_samples = gyro_bias_calibration_samples
        self._gyro_calibration_buffer: List[float] = []
        self._energy_window: Deque[float] = deque(maxlen=8)
        self._grid = self._make_grid()

    def update(self, packet: SensorPacket) -> Estimate:
        if self.last_time_ms is None:
            self.last_time_ms = packet.time_ms
            self._collect_gyro_bias(packet)
            return Estimate(
                time_ms=packet.time_ms,
                x_m=self.x_m,
                y_m=self.y_m,
                heading_rad=self.heading_rad,
                confidence=self.confidence,
                motion_state="initializing",
                tof_used=False,
                best_tof_error_m=None,
            )

        dt = max(0.001, (packet.time_ms - self.last_time_ms) / 1000.0)
        self.last_time_ms = packet.time_ms

        if len(self._gyro_calibration_buffer) < self._calibration_samples:
            self._collect_gyro_bias(packet)
            motion_state = "calibrating"
        else:
            motion_state = self._classify_motion(packet)

        # Heading prediction from gyro z.
        gz_corrected = packet.gz - self.gyro_bias_rad_s
        self.heading_rad = _wrap_angle(self.heading_rad + gz_corrected * dt)

        # Position prediction from motion state. We intentionally avoid raw
        # double-integration of acceleration.
        if motion_state in {"moving", "turning_moving"}:
            self.x_m += self.nominal_speed_mps * dt * cos(self.heading_rad)
            self.y_m += self.nominal_speed_mps * dt * sin(self.heading_rad)
            self.confidence *= 0.995
        elif motion_state == "turning":
            self.confidence *= 0.992
        else:
            self.confidence *= 0.999

        self.x_m, self.y_m = self._clamp_to_room(self.x_m, self.y_m)

        tof_used = False
        best_error: Optional[float] = None
        if packet.distance_valid:
            tof_used, best_error = self._apply_tof_correction(packet.distance_mm / 1000.0)

        if not packet.distance_valid:
            # If no distance correction is available, coordinate uncertainty grows.
            self.confidence *= 0.992

        self.confidence = max(0.05, min(1.0, self.confidence))

        return Estimate(
            time_ms=packet.time_ms,
            x_m=self.x_m,
            y_m=self.y_m,
            heading_rad=self.heading_rad,
            confidence=self.confidence,
            motion_state=motion_state,
            tof_used=tof_used,
            best_tof_error_m=best_error,
        )

    def _collect_gyro_bias(self, packet: SensorPacket) -> None:
        self._gyro_calibration_buffer.append(packet.gz)
        if len(self._gyro_calibration_buffer) >= self._calibration_samples:
            self.gyro_bias_rad_s = sum(self._gyro_calibration_buffer) / len(self._gyro_calibration_buffer)

    def _classify_motion(self, packet: SensorPacket) -> str:
        accel_norm = sqrt(packet.ax * packet.ax + packet.ay * packet.ay + packet.az * packet.az)
        gyro_norm = sqrt(packet.gx * packet.gx + packet.gy * packet.gy + packet.gz * packet.gz)
        # Motion energy is deliberately simple for first version.
        energy = abs(accel_norm - 9.81) + 0.20 * gyro_norm
        self._energy_window.append(energy)
        smoothed = sum(self._energy_window) / len(self._energy_window)

        turning = abs(packet.gz - self.gyro_bias_rad_s) > 0.12
        moving = smoothed > 0.12
        if moving and turning:
            return "turning_moving"
        if turning:
            return "turning"
        if moving:
            return "moving"
        return "stationary"

    def _apply_tof_correction(self, measured_distance_m: float) -> Tuple[bool, Optional[float]]:
        """Search local grid cells and blend toward best ToF-compatible cell."""
        search_radius_m = 0.90 + (1.0 - self.confidence) * 1.25
        best_score = float("inf")
        best_cell: Optional[Tuple[float, float]] = None
        best_error: Optional[float] = None

        for gx, gy in self._grid:
            dx = gx - self.x_m
            dy = gy - self.y_m
            dist_from_pred = sqrt(dx * dx + dy * dy)
            if dist_from_pred > search_radius_m:
                continue

            expected = self.room.raycast(gx, gy, self.heading_rad, max_range_m=self.max_tof_range_m)
            if expected is None:
                continue

            error = abs(expected.distance_m - measured_distance_m)
            # Prefer cells close to the prediction, but ToF match dominates.
            score = error + 0.20 * dist_from_pred
            if score < best_score:
                best_score = score
                best_cell = (gx, gy)
                best_error = error

        if best_cell is None or best_error is None:
            self.confidence *= 0.98
            return False, None

        # Reject weak matches. This prevents a random valid ToF reading from
        # snapping the estimate to a bad grid cell.
        if best_error > 0.18:
            self.confidence *= 0.98
            return False, best_error

        # Blend toward the candidate. Strong matches get stronger correction.
        correction_strength = max(0.15, min(0.75, 0.75 - best_error / 0.24))
        self.x_m = (1.0 - correction_strength) * self.x_m + correction_strength * best_cell[0]
        self.y_m = (1.0 - correction_strength) * self.y_m + correction_strength * best_cell[1]
        self.x_m, self.y_m = self._clamp_to_room(self.x_m, self.y_m)

        # Useful ToF correction increases confidence.
        self.confidence = min(1.0, self.confidence + 0.08 * (1.0 - best_error / 0.18))
        return True, best_error

    def _make_grid(self) -> List[Tuple[float, float]]:
        cells: List[Tuple[float, float]] = []
        n_x = int(round(self.room.width_m / self.grid_spacing_m))
        n_y = int(round(self.room.height_m / self.grid_spacing_m))
        for ix in range(n_x + 1):
            x = ix * self.grid_spacing_m
            for iy in range(n_y + 1):
                y = iy * self.grid_spacing_m
                if self.room.is_valid_position(x, y, margin=0.05):
                    cells.append((x, y))
        return cells

    def _clamp_to_room(self, x: float, y: float) -> Tuple[float, float]:
        margin = 0.05
        x = max(margin, min(self.room.width_m - margin, x))
        y = max(margin, min(self.room.height_m - margin, y))
        return x, y


def _wrap_angle(a: float) -> float:
    while a > pi:
        a -= 2.0 * pi
    while a <= -pi:
        a += 2.0 * pi
    return a


def heading_name(rad: float) -> str:
    """Human-readable coarse heading name."""
    # 0=east, pi/2=north.
    deg = (_wrap_angle(rad) * 180.0 / pi) % 360.0
    if 45.0 <= deg < 135.0:
        return "north"
    if 135.0 <= deg < 225.0:
        return "west"
    if 225.0 <= deg < 315.0:
        return "south"
    return "east"
