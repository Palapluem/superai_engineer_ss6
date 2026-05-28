from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence


ACTIVITY_LABELS = {
    "D01": "walk_pick_object",
    "D02": "slow_walk",
    "D03": "limping_walk",
    "D04": "normal_walk",
    "D05": "lying",
    "D06": "standing",
    "D07": "sit_stand_transition",
    "F01": "gradual_fall",
    "F02": "slow_collapse",
}

ACTIVITY_LABELS_TH = {
    "D01": "เดินจับของระหว่างทาง",
    "D02": "เดินช้า",
    "D03": "เดินกระเพก",
    "D04": "เดินปกติ",
    "D05": "นอน",
    "D06": "ยืน",
    "D07": "ลุกยืนสลับนั่ง",
    "F01": "ค่อยๆล้ม",
    "F02": "ล้มแบบค่อยๆทรุด",
}

SUBJECT_GROUPS = {
    "SA": "young_adult",
    "SE": "elderly",
}

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")

_COLUMN_ALIASES = {
    "accel_x": "ax",
    "accel_y": "ay",
    "accel_z": "az",
    "accelerometer_x": "ax",
    "accelerometer_y": "ay",
    "accelerometer_z": "az",
    "gyro_x": "gx",
    "gyro_y": "gy",
    "gyro_z": "gz",
    "gyroscope_x": "gx",
    "gyroscope_y": "gy",
    "gyroscope_z": "gz",
    "timestamp_ms": "ms",
    "time_ms": "ms",
    "t_ms": "ms",
}


@dataclass
class SixAxisSample:
    index: int
    time_ms: Optional[int]
    ax_mps2: float
    ay_mps2: float
    az_mps2: float
    gx_rad_s: float
    gy_rad_s: float
    gz_rad_s: float

    def paper_int_row(self) -> List[int]:
        """Paper-like row: accel in milli-g, gyro in deg/s, all integers."""
        return [
            round(self.ax_mps2 / 9.80665 * 1000.0),
            round(self.ay_mps2 / 9.80665 * 1000.0),
            round(self.az_mps2 / 9.80665 * 1000.0),
            round(self.gx_rad_s * 180.0 / pi),
            round(self.gy_rad_s * 180.0 / pi),
            round(self.gz_rad_s * 180.0 / pi),
        ]


@dataclass
class FormatResult:
    samples: List[SixAxisSample]
    accel_unit: str
    gyro_unit: str
    source_columns: List[str]


def convert_sensor_csv_to_dataset_format(
    input_csv: Path,
    output_dir: Path,
    subject_group: str,
    activity_code: str,
    trial_id: str = "01",
    accel_unit: str = "auto",
    gyro_unit: str = "auto",
) -> Dict[str, Path]:
    subject_group = subject_group.upper()
    activity_code = activity_code.upper()
    _validate_metadata(subject_group, activity_code)

    result = load_six_axis_csv(input_csv, accel_unit=accel_unit, gyro_unit=gyro_unit)
    prefix = f"{subject_group}_{activity_code}_{trial_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_path = output_dir / f"{prefix}_dataset_6ch.txt"
    preprocessed_path = output_dir / f"{prefix}_preprocessed.csv"
    manifest_path = output_dir / f"{prefix}_manifest.json"

    write_paper_6ch_txt(result.samples, paper_path)
    write_preprocessed_csv(
        result.samples,
        preprocessed_path,
        subject_group=subject_group,
        activity_code=activity_code,
        trial_id=trial_id,
    )
    write_manifest_json(
        manifest_path,
        input_csv=input_csv,
        paper_path=paper_path,
        preprocessed_path=preprocessed_path,
        result=result,
        subject_group=subject_group,
        activity_code=activity_code,
        trial_id=trial_id,
    )
    return {
        "paper_6ch": paper_path,
        "preprocessed_csv": preprocessed_path,
        "manifest_json": manifest_path,
    }


def load_six_axis_csv(
    input_csv: Path,
    accel_unit: str = "auto",
    gyro_unit: str = "auto",
) -> FormatResult:
    rows = _read_csv_rows(input_csv)
    if not rows:
        raise ValueError(f"No data rows found in {input_csv}")

    source_columns = list(rows[0].keys())
    normalized_rows = [{_normalize_column(k): v for k, v in row.items() if k is not None} for row in rows]
    missing = [name for name in CHANNELS if name not in normalized_rows[0]]
    if missing:
        raise ValueError(f"Missing required IMU columns in {input_csv}: {', '.join(missing)}")

    raw_values = []
    for index, row in enumerate(normalized_rows):
        raw_values.append(
            {
                "index": index,
                "time_ms": _parse_optional_ms(row),
                "ax": float(row["ax"]),
                "ay": float(row["ay"]),
                "az": float(row["az"]),
                "gx": float(row["gx"]),
                "gy": float(row["gy"]),
                "gz": float(row["gz"]),
            }
        )

    resolved_accel_unit = _resolve_accel_unit(raw_values, accel_unit)
    resolved_gyro_unit = _resolve_gyro_unit(raw_values, gyro_unit)

    samples: List[SixAxisSample] = []
    for row in raw_values:
        if resolved_accel_unit == "g":
            ax_mps2 = row["ax"] * 9.80665
            ay_mps2 = row["ay"] * 9.80665
            az_mps2 = row["az"] * 9.80665
        elif resolved_accel_unit == "mps2":
            ax_mps2 = row["ax"]
            ay_mps2 = row["ay"]
            az_mps2 = row["az"]
        else:
            raise ValueError(f"Unsupported accel unit: {resolved_accel_unit}")

        if resolved_gyro_unit == "deg_s":
            gx_rad_s = row["gx"] * pi / 180.0
            gy_rad_s = row["gy"] * pi / 180.0
            gz_rad_s = row["gz"] * pi / 180.0
        elif resolved_gyro_unit == "rad_s":
            gx_rad_s = row["gx"]
            gy_rad_s = row["gy"]
            gz_rad_s = row["gz"]
        else:
            raise ValueError(f"Unsupported gyro unit: {resolved_gyro_unit}")

        samples.append(
            SixAxisSample(
                index=row["index"],
                time_ms=row["time_ms"],
                ax_mps2=ax_mps2,
                ay_mps2=ay_mps2,
                az_mps2=az_mps2,
                gx_rad_s=gx_rad_s,
                gy_rad_s=gy_rad_s,
                gz_rad_s=gz_rad_s,
            )
        )

    return FormatResult(
        samples=samples,
        accel_unit=resolved_accel_unit,
        gyro_unit=resolved_gyro_unit,
        source_columns=source_columns,
    )


def write_paper_6ch_txt(samples: Sequence[SixAxisSample], output_path: Path) -> None:
    """Write headerless paper-like rows: ax,ay,az,gx,gy,gz;"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        for sample in samples:
            f.write(",".join(str(value) for value in sample.paper_int_row()) + ";\n")


def write_preprocessed_csv(
    samples: Sequence[SixAxisSample],
    output_path: Path,
    subject_group: str,
    activity_code: str,
    trial_id: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_index",
                "time_ms",
                "subject_group",
                "subject_group_name",
                "activity_group",
                "activity_code",
                "activity_name",
                "activity_name_th",
                "trial_id",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "gx_rad_s",
                "gy_rad_s",
                "gz_rad_s",
                "paper_ax_mg",
                "paper_ay_mg",
                "paper_az_mg",
                "paper_gx_deg_s",
                "paper_gy_deg_s",
                "paper_gz_deg_s",
            ]
        )
        for sample in samples:
            paper = sample.paper_int_row()
            writer.writerow(
                [
                    sample.index,
                    "" if sample.time_ms is None else sample.time_ms,
                    subject_group,
                    SUBJECT_GROUPS[subject_group],
                    "fall" if activity_code.startswith("F") else "adl",
                    activity_code,
                    ACTIVITY_LABELS[activity_code],
                    ACTIVITY_LABELS_TH[activity_code],
                    trial_id,
                    f"{sample.ax_mps2:.6f}",
                    f"{sample.ay_mps2:.6f}",
                    f"{sample.az_mps2:.6f}",
                    f"{sample.gx_rad_s:.6f}",
                    f"{sample.gy_rad_s:.6f}",
                    f"{sample.gz_rad_s:.6f}",
                    *paper,
                ]
            )


def write_manifest_json(
    output_path: Path,
    input_csv: Path,
    paper_path: Path,
    preprocessed_path: Path,
    result: FormatResult,
    subject_group: str,
    activity_code: str,
    trial_id: str,
) -> None:
    manifest = {
        "source_file": str(input_csv),
        "sample_count": len(result.samples),
        "source_columns": result.source_columns,
        "detected_units": {
            "accel": result.accel_unit,
            "gyro": result.gyro_unit,
        },
        "subject_group": subject_group,
        "subject_group_name": SUBJECT_GROUPS[subject_group],
        "activity_group": "fall" if activity_code.startswith("F") else "adl",
        "activity_code": activity_code,
        "activity_name": ACTIVITY_LABELS[activity_code],
        "activity_name_th": ACTIVITY_LABELS_TH[activity_code],
        "trial_id": trial_id,
        "outputs": {
            "paper_6ch": str(paper_path),
            "preprocessed_csv": str(preprocessed_path),
        },
        "paper_6ch_format": "ax_mg,ay_mg,az_mg,gx_deg_s,gy_deg_s,gz_deg_s;",
    }
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv_rows(input_csv: Path) -> List[Dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"Missing CSV header in {input_csv}")
    return list(reader)


def _normalize_column(name: str) -> str:
    key = name.strip().lower()
    return _COLUMN_ALIASES.get(key, key)


def _parse_optional_ms(row: Dict[str, str]) -> Optional[int]:
    value = row.get("ms")
    if value is None or value == "":
        return None
    return int(float(value))


def _resolve_accel_unit(raw_values: Sequence[Dict[str, float]], accel_unit: str) -> str:
    if accel_unit != "auto":
        return accel_unit
    norms = [sqrt(row["ax"] ** 2 + row["ay"] ** 2 + row["az"] ** 2) for row in raw_values]
    mid = median(norms)
    return "g" if 0.5 <= mid <= 2.0 else "mps2"


def _resolve_gyro_unit(raw_values: Sequence[Dict[str, float]], gyro_unit: str) -> str:
    if gyro_unit != "auto":
        return gyro_unit
    max_abs = max(max(abs(row["gx"]), abs(row["gy"]), abs(row["gz"])) for row in raw_values)
    return "deg_s" if max_abs > 5.0 else "rad_s"


def _validate_metadata(subject_group: str, activity_code: str) -> None:
    if subject_group not in SUBJECT_GROUPS:
        raise ValueError(f"Unsupported subject group {subject_group}. Use one of: {', '.join(SUBJECT_GROUPS)}")
    if activity_code not in ACTIVITY_LABELS:
        raise ValueError(f"Unsupported activity code {activity_code}. Use one of: {', '.join(ACTIVITY_LABELS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Format WellSense sensor CSV to paper dataset-like 6-axis rows.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Raw sensor CSV with ax,ay,az,gx,gy,gz columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_formatted"))
    parser.add_argument("--subject-group", choices=sorted(SUBJECT_GROUPS), required=True, help="SA=young adult, SE=elderly.")
    parser.add_argument("--activity-code", choices=sorted(ACTIVITY_LABELS), required=True, help="DXX for ADL, FXX for fall.")
    parser.add_argument("--trial-id", default="01")
    parser.add_argument("--accel-unit", choices=["auto", "g", "mps2"], default="auto")
    parser.add_argument("--gyro-unit", choices=["auto", "deg_s", "rad_s"], default="auto")
    args = parser.parse_args()

    outputs = convert_sensor_csv_to_dataset_format(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        subject_group=args.subject_group,
        activity_code=args.activity_code,
        trial_id=args.trial_id,
        accel_unit=args.accel_unit,
        gyro_unit=args.gyro_unit,
    )
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
