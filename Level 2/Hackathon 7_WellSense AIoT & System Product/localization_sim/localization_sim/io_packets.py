from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .estimator import Estimate, heading_name
from .simulator import SensorPacket


REQUIRED_PACKET_COLUMNS = (
    "time_ms",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "distance_mm",
    "distance_valid",
)

_COLUMN_ALIASES = {
    "timestamp_ms": "time_ms",
    "t_ms": "time_ms",
    "time": "time_ms",
    "accel_x": "ax",
    "accel_y": "ay",
    "accel_z": "az",
    "gyro_x": "gx",
    "gyro_y": "gy",
    "gyro_z": "gz",
    "distance": "distance_mm",
    "tof_mm": "distance_mm",
    "range_mm": "distance_mm",
    "tof_valid": "distance_valid",
    "range_valid": "distance_valid",
}


@dataclass
class PacketLoadResult:
    packets: List[SensorPacket]
    warnings: List[str]
    source_columns: List[str]


def load_sensor_packets(path: Path) -> PacketLoadResult:
    """Load Arduino Nano-style sensor packets from a CSV/serial log.

    Lines starting with "#" are ignored, so the Arduino sketch can print status
    comments before the CSV header.
    """
    warnings: List[str] = []
    with path.open(newline="") as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]

    if not lines:
        raise ValueError(f"No CSV data found in {path}")

    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"Missing CSV header in {path}")

    normalized_fields = [_normalize_column_name(name) for name in reader.fieldnames]
    source_columns = list(reader.fieldnames)
    missing = [name for name in REQUIRED_PACKET_COLUMNS if name not in normalized_fields]
    gyro_missing = [name for name in ("gx", "gy", "gz") if name in missing]
    fatal_missing = [name for name in missing if name not in gyro_missing]

    if gyro_missing:
        warnings.append(
            "Gyro columns are missing; heading will not turn correctly. "
            "The loader fills missing gx/gy/gz with 0.0."
        )
    if fatal_missing:
        raise ValueError(f"Missing required packet columns in {path}: {', '.join(fatal_missing)}")

    packets: List[SensorPacket] = []
    previous_time_ms = -1
    for line_number, row in enumerate(reader, start=2):
        normalized = {_normalize_column_name(k): v for k, v in row.items() if k is not None}
        try:
            packet = SensorPacket(
                time_ms=int(float(_value(normalized, "time_ms"))),
                ax=float(_value(normalized, "ax")),
                ay=float(_value(normalized, "ay")),
                az=float(_value(normalized, "az")),
                gx=float(normalized.get("gx") or 0.0),
                gy=float(normalized.get("gy") or 0.0),
                gz=float(normalized.get("gz") or 0.0),
                distance_mm=int(float(_value(normalized, "distance_mm"))),
                distance_valid=_parse_bool(_value(normalized, "distance_valid")),
            )
        except ValueError as exc:
            raise ValueError(f"Bad packet value at {path}:{line_number}: {exc}") from exc

        if packet.time_ms <= previous_time_ms:
            warnings.append(
                f"Non-increasing time_ms at input line {line_number}; "
                "estimator will clamp dt to a small positive value."
            )
        previous_time_ms = packet.time_ms
        packets.append(packet)

    if len(packets) < 10:
        warnings.append("Very few packets were loaded; localization confidence may be low.")

    return PacketLoadResult(packets=packets, warnings=warnings, source_columns=source_columns)


def write_coordinate_estimates(
    packets: Sequence[SensorPacket],
    estimates: Sequence[Estimate],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_ms",
                "x_m",
                "y_m",
                "heading_rad",
                "heading_name",
                "confidence",
                "motion_state",
                "distance_mm",
                "distance_valid",
                "tof_used",
                "best_tof_error_m",
            ]
        )
        for packet, est in zip(packets, estimates):
            writer.writerow(
                [
                    est.time_ms,
                    f"{est.x_m:.4f}",
                    f"{est.y_m:.4f}",
                    f"{est.heading_rad:.5f}",
                    heading_name(est.heading_rad),
                    f"{est.confidence:.4f}",
                    est.motion_state,
                    packet.distance_mm,
                    int(packet.distance_valid),
                    int(est.tof_used),
                    "" if est.best_tof_error_m is None else f"{est.best_tof_error_m:.4f}",
                ]
            )


def _normalize_column_name(name: str) -> str:
    key = name.strip().lower()
    return _COLUMN_ALIASES.get(key, key)


def _value(row: Dict[str, str], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"missing {key}")
    return value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "valid"}
