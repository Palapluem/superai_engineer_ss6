from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, pi
from random import Random
from typing import Iterable, List, Optional, Sequence

from .room import Room


@dataclass
class TrueState:
    t_s: float
    x_m: float
    y_m: float
    heading_rad: float
    speed_mps: float
    angular_velocity_rad_s: float


@dataclass
class SensorPacket:
    """Arduino Nano-like packet.

    This is intentionally close to what the real Nano can later stream over
    Serial. Units:
        ax,ay,az: m/s^2
        gx,gy,gz: rad/s
        distance_mm: millimeters, max range if invalid
    """

    time_ms: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    distance_mm: int
    distance_valid: bool

    def to_csv_row(self) -> str:
        return (
            f"{self.time_ms},{self.ax:.5f},{self.ay:.5f},{self.az:.5f},"
            f"{self.gx:.5f},{self.gy:.5f},{self.gz:.5f},"
            f"{self.distance_mm},{int(self.distance_valid)}"
        )


@dataclass
class SimulationRecord:
    truth: TrueState
    packet: SensorPacket


class PathBuilder:
    """Build a piecewise path with moves, turns, and stops."""

    def __init__(self, x0: float, y0: float, heading0_rad: float, dt_s: float = 0.1):
        self.x = x0
        self.y = y0
        self.heading = heading0_rad
        self.t = 0.0
        self.dt_s = dt_s
        self.states: List[TrueState] = []

    def stop(self, duration_s: float) -> "PathBuilder":
        steps = max(1, int(round(duration_s / self.dt_s)))
        for _ in range(steps):
            self._append(speed=0.0, omega=0.0)
        return self

    def move(self, distance_m: float, speed_mps: float) -> "PathBuilder":
        duration_s = abs(distance_m) / speed_mps
        steps = max(1, int(round(duration_s / self.dt_s)))
        signed_speed = speed_mps if distance_m >= 0.0 else -speed_mps
        for _ in range(steps):
            self.x += signed_speed * self.dt_s * cos(self.heading)
            self.y += signed_speed * self.dt_s * sin(self.heading)
            self._append(speed=abs(signed_speed), omega=0.0)
        return self

    def turn(self, delta_heading_rad: float, duration_s: float) -> "PathBuilder":
        steps = max(1, int(round(duration_s / self.dt_s)))
        omega = delta_heading_rad / (steps * self.dt_s)
        for _ in range(steps):
            self.heading += omega * self.dt_s
            self._append(speed=0.0, omega=omega)
        return self

    def _append(self, speed: float, omega: float) -> None:
        self.states.append(
            TrueState(
                t_s=self.t,
                x_m=self.x,
                y_m=self.y,
                heading_rad=self.heading,
                speed_mps=speed,
                angular_velocity_rad_s=omega,
            )
        )
        self.t += self.dt_s

    def build(self) -> List[TrueState]:
        return self.states


def make_demo_path(dt_s: float = 0.1) -> List[TrueState]:
    """Ground-truth path for the first simulation.

    Starts at (0.5,0.5), facing east. It intentionally approaches multiple
    walls/objects so the distance sensor gets useful correction events.
    """
    east = 0.0
    north = pi / 2.0
    west = pi
    south = -pi / 2.0

    b = PathBuilder(x0=0.5, y0=0.5, heading0_rad=east, dt_s=dt_s)
    return (
        b.stop(2.0)
        .move(distance_m=2.8, speed_mps=0.5)      # near right wall
        .stop(0.5)
        .turn(delta_heading_rad=north - east, duration_s=2.0)
        .move(distance_m=3.0, speed_mps=0.5)      # passes near desk landmark
        .stop(0.5)
        .turn(delta_heading_rad=west - north, duration_s=2.0)
        .move(distance_m=2.25, speed_mps=0.5)     # toward left side
        .stop(0.5)
        .turn(delta_heading_rad=south - west, duration_s=2.0)
        .move(distance_m=2.4, speed_mps=0.5)      # toward lower-left area
        .stop(1.0)
        .build()
    )


class SensorSimulator:
    def __init__(
        self,
        room: Room,
        max_tof_range_m: float = 1.2,
        rng_seed: int = 7,
        gyro_bias_rad_s: float = 0.006,
    ):
        self.room = room
        self.max_tof_range_m = max_tof_range_m
        self.rng = Random(rng_seed)
        self.gyro_bias_rad_s = gyro_bias_rad_s

    def generate_packet(self, state: TrueState) -> SensorPacket:
        t = state.t_s
        moving = state.speed_mps > 0.05
        turning = abs(state.angular_velocity_rad_s) > 0.02

        # Fake IMU. This is not a physics-perfect accelerometer model. It is a
        # realistic-enough signal for testing moving/still/turning logic.
        accel_noise = lambda scale: self.rng.gauss(0.0, scale)
        gyro_noise = lambda scale: self.rng.gauss(0.0, scale)

        if moving:
            # Human-like step oscillation. Constant velocity still creates body
            # acceleration variation when walking.
            step_wave = sin(2.0 * pi * 1.8 * t)
            ax = 0.20 * step_wave * cos(state.heading_rad) + accel_noise(0.04)
            ay = 0.20 * step_wave * sin(state.heading_rad) + accel_noise(0.04)
            az = 9.81 + 0.55 * abs(step_wave) + accel_noise(0.05)
        else:
            ax = accel_noise(0.03)
            ay = accel_noise(0.03)
            az = 9.81 + accel_noise(0.03)

        gx = gyro_noise(0.004)
        gy = gyro_noise(0.004)
        gz = state.angular_velocity_rad_s + self.gyro_bias_rad_s + gyro_noise(0.004)

        hit = self.room.raycast(
            state.x_m,
            state.y_m,
            state.heading_rad,
            max_range_m=None,
        )
        if hit is not None and hit.distance_m <= self.max_tof_range_m:
            noisy_distance_m = max(0.02, hit.distance_m + self.rng.gauss(0.0, 0.015))
            distance_mm = int(round(noisy_distance_m * 1000.0))
            distance_valid = True
            # Occasional invalid readings simulate missed surfaces.
            if self.rng.random() < 0.025:
                distance_mm = int(round(self.max_tof_range_m * 1000.0))
                distance_valid = False
        else:
            distance_mm = int(round(self.max_tof_range_m * 1000.0))
            distance_valid = False

        return SensorPacket(
            time_ms=int(round(state.t_s * 1000.0)),
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz,
            distance_mm=distance_mm,
            distance_valid=distance_valid,
        )

    def run(self, path: Sequence[TrueState]) -> List[SimulationRecord]:
        return [SimulationRecord(truth=s, packet=self.generate_packet(s)) for s in path]


def csv_header() -> str:
    return "time_ms,ax,ay,az,gx,gy,gz,distance_mm,distance_valid"
