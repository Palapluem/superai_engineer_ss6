from __future__ import annotations

import argparse
import csv
from math import sqrt
from pathlib import Path
from statistics import mean, median
from typing import List, Optional, Sequence

from .estimator import Estimate, RuleBasedEstimator, heading_name
from .gait import analyze_gait_windows, write_gait_windows
from .io_packets import load_sensor_packets, write_coordinate_estimates
from .plot_results import (
    plot_confidence,
    plot_estimated_track,
    plot_home_test_map,
    plot_risk_heatmap,
    plot_trajectory,
)
from .report import write_summary_report
from .risk import build_risk_heatmap, write_risk_heatmap
from .room import make_example_room
from .simulator import SensorPacket, SensorSimulator, SimulationRecord, csv_header, make_demo_path


def run(
    output_dir: Path,
    input_csv: Optional[Path] = None,
    start_x_m: float = 0.5,
    start_y_m: float = 0.5,
    start_heading_rad: float = 0.0,
    grid_spacing_m: float = 0.25,
    nominal_speed_mps: float = 0.5,
    max_tof_range_m: float = 1.2,
    heatmap_cell_size_m: float = 0.5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    room = make_example_room()
    input_warnings: List[str] = []
    records: Optional[List[SimulationRecord]] = None

    if input_csv is None:
        path = make_demo_path(dt_s=0.1)
        simulator = SensorSimulator(room=room, max_tof_range_m=max_tof_range_m, rng_seed=7)
        records = simulator.run(path)
        packets = [record.packet for record in records]
        sensor_csv = output_dir / "simulated_nano_packets.csv"
        _write_simulated_packets(records, sensor_csv)
    else:
        loaded = load_sensor_packets(input_csv)
        packets = loaded.packets
        input_warnings = loaded.warnings
        sensor_csv = input_csv

    estimator = RuleBasedEstimator(
        room=room,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        start_heading_rad=start_heading_rad,
        grid_spacing_m=grid_spacing_m,
        nominal_speed_mps=nominal_speed_mps,
        max_tof_range_m=max_tof_range_m,
        gyro_bias_calibration_samples=20,
    )

    estimates: List[Estimate] = []
    for packet in packets:
        estimates.append(estimator.update(packet))

    coordinates_csv = output_dir / "localization_coordinates.csv"
    write_coordinate_estimates(packets, estimates, coordinates_csv)

    results_csv: Optional[Path] = None
    trajectory_png = output_dir / "trajectory.png"
    if records is not None:
        results_csv = output_dir / "estimated_path.csv"
        _write_truth_debug_csv(records, estimates, results_csv)
        plot_trajectory(room, records, estimates, trajectory_png)
    else:
        plot_estimated_track(room, estimates, trajectory_png)

    confidence_png = output_dir / "confidence.png"
    plot_confidence(estimates, confidence_png)

    gait_windows = analyze_gait_windows(packets, estimates, window_s=5.0, stride_s=2.5)
    gait_csv = output_dir / "gait_windows.csv"
    write_gait_windows(gait_windows, gait_csv)

    heatmap_cells = build_risk_heatmap(
        room=room,
        estimates=estimates,
        gait_windows=gait_windows,
        cell_size_m=heatmap_cell_size_m,
    )
    heatmap_csv = output_dir / "risk_heatmap.csv"
    heatmap_png = output_dir / "risk_heatmap.png"
    home_test_png = output_dir / "home_test_map.png"
    write_risk_heatmap(heatmap_cells, heatmap_csv)
    plot_risk_heatmap(room, heatmap_cells, heatmap_png)
    plot_home_test_map(
        room=room,
        packets=packets,
        estimates=estimates,
        output_path=home_test_png,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        grid_spacing_m=heatmap_cell_size_m,
    )

    report_txt = output_dir / "summary_report.txt"
    write_summary_report(
        output_path=report_txt,
        packets=packets,
        estimates=estimates,
        gait_windows=gait_windows,
        heatmap_cells=heatmap_cells,
        input_warnings=input_warnings,
        records=records,
    )

    _print_summary(
        packets=packets,
        estimates=estimates,
        records=records,
        sensor_csv=sensor_csv,
        coordinates_csv=coordinates_csv,
        results_csv=results_csv,
        gait_csv=gait_csv,
        heatmap_csv=heatmap_csv,
        trajectory_png=trajectory_png,
        confidence_png=confidence_png,
        heatmap_png=heatmap_png,
        home_test_png=home_test_png,
        report_txt=report_txt,
        input_warnings=input_warnings,
    )


def _write_simulated_packets(records: Sequence[SimulationRecord], sensor_csv: Path) -> None:
    with sensor_csv.open("w", newline="") as f:
        f.write(csv_header() + "\n")
        for record in records:
            f.write(record.packet.to_csv_row() + "\n")


def _write_truth_debug_csv(
    records: Sequence[SimulationRecord],
    estimates: Sequence[Estimate],
    results_csv: Path,
) -> None:
    with results_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_ms",
                "true_x_m",
                "true_y_m",
                "true_heading_rad",
                "est_x_m",
                "est_y_m",
                "est_heading_rad",
                "est_heading_name",
                "confidence",
                "motion_state",
                "distance_mm",
                "distance_valid",
                "tof_used",
                "best_tof_error_m",
            ]
        )
        for record, est in zip(records, estimates):
            writer.writerow(
                [
                    record.packet.time_ms,
                    f"{record.truth.x_m:.4f}",
                    f"{record.truth.y_m:.4f}",
                    f"{record.truth.heading_rad:.5f}",
                    f"{est.x_m:.4f}",
                    f"{est.y_m:.4f}",
                    f"{est.heading_rad:.5f}",
                    heading_name(est.heading_rad),
                    f"{est.confidence:.4f}",
                    est.motion_state,
                    record.packet.distance_mm,
                    int(record.packet.distance_valid),
                    int(est.tof_used),
                    "" if est.best_tof_error_m is None else f"{est.best_tof_error_m:.4f}",
                ]
            )


def _print_summary(
    packets: Sequence[SensorPacket],
    estimates: Sequence[Estimate],
    records: Optional[Sequence[SimulationRecord]],
    sensor_csv: Path,
    coordinates_csv: Path,
    results_csv: Optional[Path],
    gait_csv: Path,
    heatmap_csv: Path,
    trajectory_png: Path,
    confidence_png: Path,
    heatmap_png: Path,
    home_test_png: Path,
    report_txt: Path,
    input_warnings: Sequence[str],
) -> None:
    errors = [
        sqrt((record.truth.x_m - est.x_m) ** 2 + (record.truth.y_m - est.y_m) ** 2)
        for record, est in zip(records, estimates)
    ] if records is not None else []
    final_record = records[-1] if records is not None else None
    final_est = estimates[-1] if estimates else None
    used_tof_count = sum(1 for e in estimates if e.tof_used)
    valid_tof_count = sum(1 for p in packets if p.distance_valid)

    print("Localization pipeline complete")
    print(f"Records: {len(packets)}")
    print(f"Valid ToF readings: {valid_tof_count}")
    print(f"ToF corrections used: {used_tof_count}")
    if errors and final_record is not None:
        print(f"Mean position error: {mean(errors):.3f} m")
        print(f"Median position error: {median(errors):.3f} m")
        print(f"Final true position: ({final_record.truth.x_m:.2f}, {final_record.truth.y_m:.2f})")
    if final_est is not None:
        print(f"Final estimated position: ({final_est.x_m:.2f}, {final_est.y_m:.2f})")
        print(f"Final confidence: {final_est.confidence:.2f}")
    for warning in input_warnings:
        print(f"Warning: {warning}")
    print(f"Wrote: {sensor_csv}")
    print(f"Wrote: {coordinates_csv}")
    if results_csv is not None:
        print(f"Wrote: {results_csv}")
    print(f"Wrote: {gait_csv}")
    print(f"Wrote: {heatmap_csv}")
    print(f"Wrote: {trajectory_png}")
    print(f"Wrote: {confidence_png}")
    print(f"Wrote: {heatmap_png}")
    print(f"Wrote: {home_test_png}")
    print(f"Wrote: {report_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run indoor localization simulation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional Arduino Nano packet CSV. If omitted, the demo simulator is used.",
    )
    parser.add_argument("--start-x", type=float, default=0.5, help="Known start x coordinate in meters.")
    parser.add_argument("--start-y", type=float, default=0.5, help="Known start y coordinate in meters.")
    parser.add_argument(
        "--start-heading-deg",
        type=float,
        default=0.0,
        help="Known start heading in degrees. 0=east, 90=north.",
    )
    parser.add_argument("--grid-spacing", type=float, default=0.25, help="Estimator grid spacing in meters.")
    parser.add_argument("--nominal-speed", type=float, default=0.5, help="Nominal walking speed in m/s.")
    parser.add_argument("--max-tof-range", type=float, default=1.2, help="Maximum usable ToF range in meters.")
    parser.add_argument("--heatmap-cell-size", type=float, default=0.5, help="Risk heatmap cell size in meters.")
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        input_csv=args.input_csv,
        start_x_m=args.start_x,
        start_y_m=args.start_y,
        start_heading_rad=args.start_heading_deg * 3.141592653589793 / 180.0,
        grid_spacing_m=args.grid_spacing,
        nominal_speed_mps=args.nominal_speed,
        max_tof_range_m=args.max_tof_range,
        heatmap_cell_size_m=args.heatmap_cell_size,
    )


if __name__ == "__main__":
    main()
