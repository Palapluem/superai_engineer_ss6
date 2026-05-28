from __future__ import annotations

from math import sqrt
from pathlib import Path
from statistics import mean, median
from typing import List, Optional, Sequence

from .estimator import Estimate
from .gait import GaitWindow
from .risk import RiskHeatmapCell, top_risk_cells
from .simulator import SensorPacket, SimulationRecord


def write_summary_report(
    output_path: Path,
    packets: Sequence[SensorPacket],
    estimates: Sequence[Estimate],
    gait_windows: Sequence[GaitWindow],
    heatmap_cells: Sequence[RiskHeatmapCell],
    input_warnings: Sequence[str],
    records: Optional[Sequence[SimulationRecord]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_summary_report(
            packets=packets,
            estimates=estimates,
            gait_windows=gait_windows,
            heatmap_cells=heatmap_cells,
            input_warnings=input_warnings,
            records=records,
        ),
        encoding="utf-8",
    )


def build_summary_report(
    packets: Sequence[SensorPacket],
    estimates: Sequence[Estimate],
    gait_windows: Sequence[GaitWindow],
    heatmap_cells: Sequence[RiskHeatmapCell],
    input_warnings: Sequence[str],
    records: Optional[Sequence[SimulationRecord]] = None,
) -> str:
    packet_count = len(packets)
    duration_s = _duration_s(packets)
    valid_tof_count = sum(1 for p in packets if p.distance_valid)
    tof_used_count = sum(1 for e in estimates if e.tof_used)
    valid_tof_rate = valid_tof_count / packet_count if packet_count else 0.0
    tof_used_rate = tof_used_count / packet_count if packet_count else 0.0
    final_estimate = estimates[-1] if estimates else None
    final_confidence = final_estimate.confidence if final_estimate else 0.0
    moving_count = sum(1 for e in estimates if e.motion_state in {"moving", "turning_moving"})
    moving_rate = moving_count / len(estimates) if estimates else 0.0

    status_notes: List[str] = []
    if packet_count < 50:
        status_notes.append("Need more packets for a stable demo run.")
    if valid_tof_rate < 0.05:
        status_notes.append("ToF correction is sparse; position will drift between visible walls/objects.")
    if final_confidence < 0.20:
        status_notes.append("Final confidence is low; check start pose, gyro bias, and sensor mounting.")
    status_notes.extend(input_warnings)
    usable = packet_count >= 50 and final_confidence >= 0.20

    lines = [
        "WellSense localization demo report",
        "===================================",
        "",
        f"Can it work tonight?: {'YES - prototype/demo ready' if usable else 'PARTIAL - needs sensor/data check'}",
        "",
        "What works",
        "- Known room map + known start coordinate produces an x,y estimate over time.",
        "- Modulino Movement IMU provides motion/turning cues without double-integrating acceleration.",
        "- Modulino Distance provides absolute correction events when a wall/object is in front of the user.",
        "- x,y estimates are combined with gait windows to create a risk heatmap for dashboard use.",
        "",
        "Current run",
        f"- Packets: {packet_count}",
        f"- Duration: {duration_s:.1f} s",
        f"- Valid ToF readings: {valid_tof_count} ({valid_tof_rate:.1%})",
        f"- ToF corrections used: {tof_used_count} ({tof_used_rate:.1%})",
        f"- Moving packet ratio: {moving_rate:.1%}",
        f"- Final confidence: {final_confidence:.2f}",
    ]

    if final_estimate is not None:
        lines.append(f"- Latest estimated position: ({final_estimate.x_m:.2f}, {final_estimate.y_m:.2f}) m")

    if records:
        errors = [
            sqrt((record.truth.x_m - est.x_m) ** 2 + (record.truth.y_m - est.y_m) ** 2)
            for record, est in zip(records, estimates)
        ]
        lines.extend(
            [
                f"- Mean position error: {mean(errors):.3f} m",
                f"- Median position error: {median(errors):.3f} m",
            ]
        )

    if gait_windows:
        mean_gait_risk = mean(w.risk_score for w in gait_windows)
        high_windows = sum(1 for w in gait_windows if w.risk_level == "high")
        medium_windows = sum(1 for w in gait_windows if w.risk_level == "medium")
        lines.extend(
            [
                "",
                "Gait risk windows",
                f"- Windows: {len(gait_windows)}",
                f"- Mean gait risk score: {mean_gait_risk:.2f}",
                f"- Medium/high windows: {medium_windows}/{high_windows}",
            ]
        )

    risk_cells = top_risk_cells(heatmap_cells, limit=5)
    if risk_cells:
        lines.extend(["", "Top risk heatmap cells"])
        for cell in risk_cells:
            lines.append(
                "- "
                f"({cell.x_center_m:.2f}, {cell.y_center_m:.2f}) m: "
                f"score={cell.heatmap_score:.2f}, level={cell.danger_level}, "
                f"dwell={cell.dwell_s:.1f}s, gait={cell.mean_gait_risk:.2f}, env={cell.environment_risk:.2f}"
            )

    lines.extend(
        [
            "",
            "Known limitations / blockers",
            "- It needs a known start coordinate and heading, or a dashboard/manual reset step.",
            "- Heading drifts when gyro bias changes; a future version should use a particle filter and map constraints.",
            "- Distance correction only happens when the forward-facing ToF sees a mapped wall/object within range.",
            "- Sensor placement should be near waist/L4-L5 for gait features; wrist/hand mounting changes thresholds.",
            "- Real-home accuracy depends on an up-to-date room layout and obstacle map.",
        ]
    )

    if status_notes:
        lines.extend(["", "Run warnings"])
        lines.extend(f"- {note}" for note in status_notes)

    lines.extend(
        [
            "",
            "Recommended demo sentence",
            "This is a prevention pipeline: the device estimates indoor x,y, extracts gait risk, and turns repeated risky movement near mapped obstacles into a caregiver heatmap.",
            "",
        ]
    )
    return "\n".join(lines)


def _duration_s(packets: Sequence[SensorPacket]) -> float:
    if len(packets) < 2:
        return 0.0
    return max(0.0, (packets[-1].time_ms - packets[0].time_ms) / 1000.0)
