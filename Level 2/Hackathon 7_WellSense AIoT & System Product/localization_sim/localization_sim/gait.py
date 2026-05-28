from __future__ import annotations

import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import List, Sequence

from .estimator import Estimate
from .simulator import SensorPacket


@dataclass
class GaitWindow:
    start_time_ms: int
    end_time_ms: int
    center_x_m: float
    center_y_m: float
    step_count: int
    cadence_spm: float
    dynamic_accel_rms_mps2: float
    lateral_sway_rms_mps2: float
    turn_rate_rms_rad_s: float
    mean_speed_mps: float
    stop_ratio: float
    mean_confidence: float
    risk_score: float
    risk_level: str


def analyze_gait_windows(
    packets: Sequence[SensorPacket],
    estimates: Sequence[Estimate],
    window_s: float = 5.0,
    stride_s: float = 2.5,
) -> List[GaitWindow]:
    """Extract demo-grade gait risk features from IMU and coordinate estimates."""
    if not packets or not estimates:
        return []

    paired = list(zip(packets, estimates))
    start_ms = paired[0][0].time_ms
    end_ms = paired[-1][0].time_ms
    window_ms = int(round(window_s * 1000.0))
    stride_ms = max(1, int(round(stride_s * 1000.0)))

    windows: List[GaitWindow] = []
    cursor = start_ms
    while cursor + window_ms <= end_ms + 1:
        subset = [(p, e) for p, e in paired if cursor <= p.time_ms < cursor + window_ms]
        if len(subset) >= 3:
            windows.append(_analyze_one_window(subset, cursor, cursor + window_ms))
        cursor += stride_ms
    return windows


def write_gait_windows(windows: Sequence[GaitWindow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_time_ms",
                "end_time_ms",
                "center_x_m",
                "center_y_m",
                "step_count",
                "cadence_spm",
                "dynamic_accel_rms_mps2",
                "lateral_sway_rms_mps2",
                "turn_rate_rms_rad_s",
                "mean_speed_mps",
                "stop_ratio",
                "mean_confidence",
                "risk_score",
                "risk_level",
            ]
        )
        for w in windows:
            writer.writerow(
                [
                    w.start_time_ms,
                    w.end_time_ms,
                    f"{w.center_x_m:.4f}",
                    f"{w.center_y_m:.4f}",
                    w.step_count,
                    f"{w.cadence_spm:.2f}",
                    f"{w.dynamic_accel_rms_mps2:.4f}",
                    f"{w.lateral_sway_rms_mps2:.4f}",
                    f"{w.turn_rate_rms_rad_s:.4f}",
                    f"{w.mean_speed_mps:.4f}",
                    f"{w.stop_ratio:.4f}",
                    f"{w.mean_confidence:.4f}",
                    f"{w.risk_score:.4f}",
                    w.risk_level,
                ]
            )


def _analyze_one_window(
    subset: Sequence[tuple[SensorPacket, Estimate]],
    start_time_ms: int,
    end_time_ms: int,
) -> GaitWindow:
    packets = [p for p, _ in subset]
    estimates = [e for _, e in subset]
    duration_s = max(0.001, (end_time_ms - start_time_ms) / 1000.0)

    dynamic_accel = [_accel_dynamic_mps2(p) for p in packets]
    lateral_accel = [sqrt(p.ax * p.ax + p.ay * p.ay) for p in packets]
    turn_rates = [abs(p.gz) for p in packets]

    step_count = _count_steps(packets, dynamic_accel)
    cadence_spm = step_count / duration_s * 60.0
    dynamic_rms = _rms(dynamic_accel)
    lateral_rms = _rms(lateral_accel)
    turn_rate_rms = _rms(turn_rates)
    center_x = mean(e.x_m for e in estimates)
    center_y = mean(e.y_m for e in estimates)
    stop_ratio = sum(1 for e in estimates if e.motion_state in {"stationary", "calibrating", "initializing"}) / len(estimates)
    mean_confidence = mean(e.confidence for e in estimates)
    mean_speed = _path_distance(estimates) / duration_s
    risk_score = _risk_score(
        step_count=step_count,
        cadence_spm=cadence_spm,
        dynamic_rms=dynamic_rms,
        lateral_rms=lateral_rms,
        turn_rate_rms=turn_rate_rms,
        mean_speed_mps=mean_speed,
        stop_ratio=stop_ratio,
        mean_confidence=mean_confidence,
    )

    return GaitWindow(
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        center_x_m=center_x,
        center_y_m=center_y,
        step_count=step_count,
        cadence_spm=cadence_spm,
        dynamic_accel_rms_mps2=dynamic_rms,
        lateral_sway_rms_mps2=lateral_rms,
        turn_rate_rms_rad_s=turn_rate_rms,
        mean_speed_mps=mean_speed,
        stop_ratio=stop_ratio,
        mean_confidence=mean_confidence,
        risk_score=risk_score,
        risk_level=_risk_level(risk_score),
    )


def _count_steps(packets: Sequence[SensorPacket], dynamic_accel: Sequence[float]) -> int:
    if len(packets) < 3:
        return 0
    avg = mean(dynamic_accel)
    variance = mean((v - avg) ** 2 for v in dynamic_accel)
    threshold = max(0.16, avg + 0.45 * sqrt(variance))
    min_interval_ms = 380
    last_peak_ms = -min_interval_ms
    count = 0
    for i in range(1, len(dynamic_accel) - 1):
        value = dynamic_accel[i]
        if value < threshold:
            continue
        if value < dynamic_accel[i - 1] or value < dynamic_accel[i + 1]:
            continue
        now_ms = packets[i].time_ms
        if now_ms - last_peak_ms >= min_interval_ms:
            count += 1
            last_peak_ms = now_ms
    return count


def _risk_score(
    step_count: int,
    cadence_spm: float,
    dynamic_rms: float,
    lateral_rms: float,
    turn_rate_rms: float,
    mean_speed_mps: float,
    stop_ratio: float,
    mean_confidence: float,
) -> float:
    active = step_count > 0 or mean_speed_mps > 0.05
    if not active:
        return _clamp(0.08 + 0.18 * (1.0 - mean_confidence))

    low_cadence = _clamp((80.0 - cadence_spm) / 60.0)
    high_cadence = _clamp((cadence_spm - 140.0) / 70.0)
    cadence_factor = max(low_cadence, high_cadence)
    dynamic_factor = _clamp((dynamic_rms - 0.18) / 0.65)
    sway_factor = _clamp((lateral_rms - 0.10) / 0.40)
    turn_factor = _clamp((turn_rate_rms - 0.18) / 0.75)
    stop_factor = _clamp((stop_ratio - 0.25) / 0.55)
    confidence_factor = _clamp(1.0 - mean_confidence)

    return _clamp(
        0.22 * cadence_factor
        + 0.24 * dynamic_factor
        + 0.24 * sway_factor
        + 0.14 * turn_factor
        + 0.10 * stop_factor
        + 0.06 * confidence_factor
    )


def _risk_level(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def _accel_dynamic_mps2(packet: SensorPacket) -> float:
    accel_norm = sqrt(packet.ax * packet.ax + packet.ay * packet.ay + packet.az * packet.az)
    return abs(accel_norm - 9.81)


def _path_distance(estimates: Sequence[Estimate]) -> float:
    total = 0.0
    for prev, cur in zip(estimates, estimates[1:]):
        total += sqrt((cur.x_m - prev.x_m) ** 2 + (cur.y_m - prev.y_m) ** 2)
    return total


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sqrt(mean(v * v for v in values))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
