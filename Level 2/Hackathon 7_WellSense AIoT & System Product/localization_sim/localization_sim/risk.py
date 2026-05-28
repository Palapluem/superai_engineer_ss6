from __future__ import annotations

import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .estimator import Estimate
from .gait import GaitWindow
from .room import Point, Room, Segment


@dataclass
class RiskHeatmapCell:
    ix: int
    iy: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    x_center_m: float
    y_center_m: float
    visit_count: int
    dwell_s: float
    gait_window_count: int
    mean_gait_risk: float
    max_gait_risk: float
    mean_confidence: float
    environment_risk: float
    heatmap_score: float
    danger_level: str


@dataclass
class _CellAccumulator:
    ix: int
    iy: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    visit_count: int = 0
    dwell_s: float = 0.0
    confidence_sum: float = 0.0
    confidence_count: int = 0
    gait_risk_sum: float = 0.0
    gait_risk_max: float = 0.0
    gait_window_count: int = 0

    @property
    def x_center_m(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def y_center_m(self) -> float:
        return (self.y_min + self.y_max) / 2.0


def build_risk_heatmap(
    room: Room,
    estimates: Sequence[Estimate],
    gait_windows: Sequence[GaitWindow],
    cell_size_m: float = 0.5,
) -> List[RiskHeatmapCell]:
    """Combine x,y occupancy, gait instability, and mapped obstacles."""
    cells = _make_cells(room, cell_size_m)

    for idx, est in enumerate(estimates):
        cell = _cell_for_position(cells, cell_size_m, est.x_m, est.y_m)
        if cell is None:
            continue
        if idx < len(estimates) - 1:
            dt_s = max(0.0, (estimates[idx + 1].time_ms - est.time_ms) / 1000.0)
        else:
            dt_s = 0.0
        cell.visit_count += 1
        cell.dwell_s += dt_s
        cell.confidence_sum += est.confidence
        cell.confidence_count += 1

    for window in gait_windows:
        cell = _cell_for_position(cells, cell_size_m, window.center_x_m, window.center_y_m)
        if cell is None:
            continue
        cell.gait_risk_sum += window.risk_score
        cell.gait_risk_max = max(cell.gait_risk_max, window.risk_score)
        cell.gait_window_count += 1

    max_dwell_s = max((cell.dwell_s for cell in cells.values()), default=0.0)
    max_visit_count = max((cell.visit_count for cell in cells.values()), default=0)

    output: List[RiskHeatmapCell] = []
    for cell in cells.values():
        mean_gait = cell.gait_risk_sum / cell.gait_window_count if cell.gait_window_count else 0.0
        mean_conf = cell.confidence_sum / cell.confidence_count if cell.confidence_count else 1.0
        dwell_norm = cell.dwell_s / max_dwell_s if max_dwell_s > 0.0 else 0.0
        visit_norm = cell.visit_count / max_visit_count if max_visit_count > 0 else 0.0
        traffic = 0.65 * dwell_norm + 0.35 * visit_norm
        env_risk = _environment_risk(room, cell.x_center_m, cell.y_center_m)
        confidence_penalty = 1.0 - mean_conf
        visited = cell.visit_count > 0 or cell.gait_window_count > 0

        score = _clamp(
            0.45 * mean_gait
            + 0.30 * traffic
            + 0.20 * env_risk
            + 0.05 * confidence_penalty
        )
        if not visited:
            score *= 0.25

        output.append(
            RiskHeatmapCell(
                ix=cell.ix,
                iy=cell.iy,
                x_min=cell.x_min,
                y_min=cell.y_min,
                x_max=cell.x_max,
                y_max=cell.y_max,
                x_center_m=cell.x_center_m,
                y_center_m=cell.y_center_m,
                visit_count=cell.visit_count,
                dwell_s=cell.dwell_s,
                gait_window_count=cell.gait_window_count,
                mean_gait_risk=mean_gait,
                max_gait_risk=cell.gait_risk_max,
                mean_confidence=mean_conf,
                environment_risk=env_risk,
                heatmap_score=score,
                danger_level=_danger_level(score),
            )
        )

    return sorted(output, key=lambda c: (c.iy, c.ix))


def write_risk_heatmap(cells: Sequence[RiskHeatmapCell], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ix",
                "iy",
                "x_center_m",
                "y_center_m",
                "visit_count",
                "dwell_s",
                "gait_window_count",
                "mean_gait_risk",
                "max_gait_risk",
                "mean_confidence",
                "environment_risk",
                "heatmap_score",
                "danger_level",
            ]
        )
        for cell in cells:
            writer.writerow(
                [
                    cell.ix,
                    cell.iy,
                    f"{cell.x_center_m:.3f}",
                    f"{cell.y_center_m:.3f}",
                    cell.visit_count,
                    f"{cell.dwell_s:.3f}",
                    cell.gait_window_count,
                    f"{cell.mean_gait_risk:.4f}",
                    f"{cell.max_gait_risk:.4f}",
                    f"{cell.mean_confidence:.4f}",
                    f"{cell.environment_risk:.4f}",
                    f"{cell.heatmap_score:.4f}",
                    cell.danger_level,
                ]
            )


def top_risk_cells(cells: Sequence[RiskHeatmapCell], limit: int = 5) -> List[RiskHeatmapCell]:
    visited = [cell for cell in cells if cell.visit_count > 0 or cell.gait_window_count > 0]
    return sorted(visited, key=lambda c: c.heatmap_score, reverse=True)[:limit]


def _make_cells(room: Room, cell_size_m: float) -> Dict[Tuple[int, int], _CellAccumulator]:
    cells: Dict[Tuple[int, int], _CellAccumulator] = {}
    nx = int(room.width_m / cell_size_m)
    ny = int(room.height_m / cell_size_m)
    if nx * cell_size_m < room.width_m:
        nx += 1
    if ny * cell_size_m < room.height_m:
        ny += 1

    for ix in range(nx):
        x_min = ix * cell_size_m
        x_max = min(room.width_m, x_min + cell_size_m)
        for iy in range(ny):
            y_min = iy * cell_size_m
            y_max = min(room.height_m, y_min + cell_size_m)
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            if not room.is_valid_position(x_center, y_center, margin=0.02):
                continue
            cells[(ix, iy)] = _CellAccumulator(ix=ix, iy=iy, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    return cells


def _cell_for_position(
    cells: Dict[Tuple[int, int], _CellAccumulator],
    cell_size_m: float,
    x_m: float,
    y_m: float,
) -> Optional[_CellAccumulator]:
    ix = int(x_m / cell_size_m)
    iy = int(y_m / cell_size_m)
    return cells.get((ix, iy))


def _environment_risk(room: Room, x_m: float, y_m: float) -> float:
    distances = [_point_segment_distance((x_m, y_m), seg) for _, seg in room.all_segments()]
    nearest = min(distances) if distances else 1.0
    obstacle_or_wall = _clamp((0.60 - nearest) / 0.60)
    doorway = 0.0
    if y_m <= 0.55 and abs(x_m - room.doorway_center_x) <= room.doorway_width_m:
        doorway = 0.35
    return max(obstacle_or_wall, doorway)


def _point_segment_distance(point: Point, segment: Segment) -> float:
    (px, py) = point
    (ax, ay), (bx, by) = segment
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = _clamp(t)
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def _danger_level(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
