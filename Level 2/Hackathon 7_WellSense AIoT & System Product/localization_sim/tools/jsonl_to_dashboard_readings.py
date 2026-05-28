from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


DASHBOARD_PATH = [
    ("Bedroom", 176, 154),
    ("Bedroom", 228, 188),
    ("Hallway", 280, 238),
    ("Hallway", 286, 292),
    ("Bathroom", 166, 344),
    ("Bathroom", 206, 316),
    ("Hallway", 322, 338),
    ("Kitchen", 430, 340),
    ("Living Room", 536, 232),
    ("Balcony", 620, 304),
    ("Living Room", 524, 232),
    ("Hallway", 286, 292),
]


def convert_jsonl_to_dashboard_readings(
    input_jsonl: Path,
    output_json: Path,
    room_mode: str = "path",
    window_size: int = 6,
) -> None:
    raw_rows = _read_jsonl(input_jsonl)
    if not raw_rows:
        raise ValueError(f"No JSONL rows found in {input_jsonl}")

    readings = []
    for index, row in enumerate(raw_rows):
        window = raw_rows[max(0, index - window_size + 1) : index + 1]
        readings.append(_make_dashboard_reading(index, row, window, room_mode))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(readings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(readings)} readings to {output_json}")


def _read_jsonl(input_jsonl: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with input_jsonl.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {input_jsonl}:{line_number}: {exc}") from exc
    return rows


def _make_dashboard_reading(
    index: int,
    row: Dict[str, Any],
    window: List[Dict[str, Any]],
    room_mode: str,
) -> Dict[str, Any]:
    ax = _num(row, "ax")
    ay = _num(row, "ay")
    az = _num(row, "az")
    gx = _num(row, "gx")
    gy = _num(row, "gy")
    gz = _num(row, "gz")

    accel_norms = [_acc_norm(r) for r in window]
    gyro_norms = [_gyro_norm(r) for r in window]
    dynamic_accel = abs(_acc_norm(row) - 1.0)
    gyro_norm = _gyro_norm(row)
    sway = mean(abs(_num(r, "ay")) + abs(_num(r, "az")) * 0.35 for r in window)
    instability = _clamp(dynamic_accel * 28.0 + gyro_norm * 0.11 + _std(accel_norms) * 34.0, 0.0, 100.0)
    fall_risk = _clamp(instability + (18.0 if gyro_norm > 260.0 else 0.0) + (16.0 if dynamic_accel > 1.4 else 0.0), 0.0, 100.0)
    near_fall = fall_risk >= 82.0 or (dynamic_accel > 2.0 and gyro_norm > 220.0)
    room, base_x, base_y = DASHBOARD_PATH[index % len(DASHBOARD_PATH)]
    if room_mode == "risk" and near_fall:
        room, base_x, base_y = ("Bathroom", 166, 344)

    return {
        "timestamp": _timestamp(row),
        "room": room,
        "x": round(base_x + _clamp(ax, -2.0, 2.0) * 8.0, 2),
        "y": round(base_y + _clamp(ay, -2.0, 2.0) * 6.0, 2),
        "gait_speed": round(_clamp(1.15 - instability / 120.0, 0.25, 1.25), 3),
        "sway": round(_clamp(sway * 2.5, 1.0, 9.0), 3),
        "cadence": round(_clamp(110.0 - instability * 0.42, 55.0, 120.0), 2),
        "turning_velocity": round(_clamp(100.0 - gyro_norm * 0.10, 10.0, 110.0), 2),
        "instability_score": round(instability, 2),
        "fall_risk": round(fall_risk, 2),
        "near_fall": near_fall,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
    }


def _timestamp(row: Dict[str, Any]) -> str:
    value = row.get("pc_ts") or row.get("timestamp") or row.get("time")
    if isinstance(value, str) and value:
        return value
    return datetime.now(timezone.utc).isoformat()


def _num(row: Dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _acc_norm(row: Dict[str, Any]) -> float:
    return sqrt(_num(row, "ax") ** 2 + _num(row, "ay") ** 2 + _num(row, "az") ** 2)


def _gyro_norm(row: Dict[str, Any]) -> float:
    return sqrt(_num(row, "gx") ** 2 + _num(row, "gy") ** 2 + _num(row, "gz") ** 2)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sqrt(mean((value - avg) ** 2 for value in values))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UNO Q BLE JSONL into dashboard SensorReading JSON.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--room-mode", choices=["path", "risk"], default="path")
    parser.add_argument("--window-size", type=int, default=6)
    args = parser.parse_args()

    convert_jsonl_to_dashboard_readings(
        input_jsonl=args.input_jsonl,
        output_json=args.output_json,
        room_mode=args.room_mode,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
