from __future__ import annotations

from math import cos, sin
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .estimator import Estimate
from .risk import RiskHeatmapCell
from .room import Room
from .simulator import SensorPacket, SimulationRecord


def plot_trajectory(
    room: Room,
    records: Sequence[SimulationRecord],
    estimates: Sequence[Estimate],
    output_path: Path,
) -> None:
    true_x = [r.truth.x_m for r in records]
    true_y = [r.truth.y_m for r in records]
    est_x = [e.x_m for e in estimates]
    est_y = [e.y_m for e in estimates]

    fig, ax = plt.subplots(figsize=(7, 9))
    _draw_room(ax, room, "Simulated indoor coordinate estimate")

    ax.plot(true_x, true_y, label="true path", linewidth=2)
    ax.plot(est_x, est_y, label="estimated path", linestyle="--", linewidth=2)
    ax.scatter([0.5], [0.5], marker="o", label="known start")
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_estimated_track(room: Room, estimates: Sequence[Estimate], output_path: Path) -> None:
    est_x = [e.x_m for e in estimates]
    est_y = [e.y_m for e in estimates]

    fig, ax = plt.subplots(figsize=(7, 9))
    _draw_room(ax, room, "Indoor coordinate estimate")
    ax.plot(est_x, est_y, label="estimated path", linestyle="--", linewidth=2)
    if estimates:
        ax.scatter([est_x[0]], [est_y[0]], marker="o", label="known start")
        ax.scatter([est_x[-1]], [est_y[-1]], marker="x", label="latest estimate")
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confidence(estimates: Sequence[Estimate], output_path: Path) -> None:
    t_s = [e.time_ms / 1000.0 for e in estimates]
    conf = [e.confidence for e in estimates]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title("Estimator confidence")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("confidence [0..1]")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.plot(t_s, conf, linewidth=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_risk_heatmap(room: Room, cells: Sequence[RiskHeatmapCell], output_path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("wellsense_risk", ["#e9f7ef", "#ffe08a", "#e74c3c"])

    fig, ax = plt.subplots(figsize=(7, 9))
    _draw_room(ax, room, "Gait + location risk heatmap")

    for cell in cells:
        color = cmap(cell.heatmap_score)
        rect = plt.Rectangle(
            (cell.x_min, cell.y_min),
            cell.x_max - cell.x_min,
            cell.y_max - cell.y_min,
            facecolor=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.78 if cell.visit_count or cell.gait_window_count else 0.22,
        )
        ax.add_patch(rect)
        if cell.heatmap_score >= 0.33 and (cell.visit_count or cell.gait_window_count):
            ax.text(
                cell.x_center_m,
                cell.y_center_m,
                f"{cell.heatmap_score:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0.0, 1.0)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("risk score")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_home_test_map(
    room: Room,
    packets: Sequence[SensorPacket],
    estimates: Sequence[Estimate],
    output_path: Path,
    start_x_m: float = 0.5,
    start_y_m: float = 0.5,
    grid_spacing_m: float = 0.5,
) -> None:
    """Presentation-style floor plan from one real/simulated device test."""
    if not estimates:
        return

    latest = estimates[-1]
    latest_packet = packets[-1] if packets else None
    x_path = [e.x_m for e in estimates]
    y_path = [e.y_m for e in estimates]
    ray_length_m = (
        latest_packet.distance_mm / 1000.0
        if latest_packet is not None and latest_packet.distance_valid
        else 1.2
    )
    ray_end_x = latest.x_m + ray_length_m * cos(latest.heading_rad)
    ray_end_y = latest.y_m + ray_length_m * sin(latest.heading_rad)
    heading_end_x = latest.x_m + 0.42 * cos(latest.heading_rad)
    heading_end_y = latest.y_m + 0.42 * sin(latest.heading_rad)
    fig, (ax, panel_ax) = plt.subplots(
        1,
        2,
        figsize=(14, 9),
        gridspec_kw={"width_ratios": [4.8, 1.55], "wspace": 0.05},
    )
    ax.set_xlim(-1.25, room.width_m + 0.45)
    ax.set_ylim(-0.96, room.height_m + 0.92)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    panel_ax.axis("off")

    _draw_presentation_grid(ax, room, grid_spacing_m)
    _draw_presentation_room(ax, room)

    ax.plot(x_path, y_path, color="#2f80ed", linewidth=2.5, alpha=0.45, label="estimated coordinate path")
    ax.scatter(x_path, y_path, s=10, color="#2f80ed", alpha=0.32)

    ax.add_patch(plt.Circle((start_x_m, start_y_m), 0.14, facecolor="#b48bea", edgecolor="#3d1268", linewidth=2.0))
    ax.add_patch(plt.Rectangle((start_x_m - 0.035, start_y_m + 0.14), 0.07, 0.12, facecolor="#6aa0d8", edgecolor="black"))
    ax.text(start_x_m + 0.18, start_y_m + 0.04, "Start", color="#4b1680", fontsize=9, weight="bold")

    ax.add_patch(plt.Circle((latest.x_m, latest.y_m), 0.16, facecolor="#5da9e9", edgecolor="#16324f", linewidth=2.0))
    ax.add_patch(plt.Circle((latest.x_m, latest.y_m + 0.06), 0.07, facecolor="#333333", edgecolor="black", linewidth=1.0))
    ax.annotate(
        "",
        xy=(heading_end_x, heading_end_y),
        xytext=(latest.x_m, latest.y_m),
        arrowprops=dict(arrowstyle="-|>", color="#0b57d0", linewidth=2.4, mutation_scale=18),
    )
    ax.plot([latest.x_m, ray_end_x], [latest.y_m, ray_end_y], color="#0b57d0", linestyle="--", linewidth=1.9)
    ax.annotate(
        "",
        xy=(ray_end_x, ray_end_y),
        xytext=(latest.x_m, latest.y_m),
        arrowprops=dict(arrowstyle="-|>", color="#0b57d0", linestyle="--", linewidth=1.8, mutation_scale=16),
    )
    ax.text(latest.x_m + 0.18, latest.y_m + 0.08, "Latest", color="#16324f", fontsize=9, weight="bold")
    ax.text(
        max(0.22, min(room.width_m - 0.55, (latest.x_m + ray_end_x) / 2.0 + 0.10)),
        max(0.22, min(room.height_m - 0.22, (latest.y_m + ray_end_y) / 2.0)),
        "ToF",
        color="#0b57d0",
        fontsize=8,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15,rounding_size=0.03", facecolor="white", edgecolor="none", alpha=0.85),
    )

    ax.text(
        room.width_m / 2.0,
        room.height_m + 0.72,
        "Example floor-plan view from device coordinate test",
        ha="center",
        fontsize=15,
        weight="bold",
    )
    _draw_home_test_side_panel(
        panel_ax=panel_ax,
        room=room,
        packet_count=len(packets),
        latest=latest,
        latest_packet=latest_packet,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        grid_spacing_m=grid_spacing_m,
        ray_length_m=ray_length_m,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.06, wspace=0.06)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_room(ax, room: Room, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(-0.2, room.width_m + 0.2)
    ax.set_ylim(-0.2, room.height_m + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5)

    ax.plot([0, room.width_m, room.width_m, 0, 0], [0, 0, room.height_m, room.height_m, 0], linewidth=2)

    gap_left = room.doorway_center_x - room.doorway_width_m / 2.0
    gap_right = room.doorway_center_x + room.doorway_width_m / 2.0
    ax.plot([gap_left, gap_right], [0, 0], linewidth=5)
    ax.text(room.doorway_center_x, -0.15, "doorway", ha="center", va="top")

    for obj in room.objects:
        rect = plt.Rectangle(
            (obj.x_min, obj.y_min),
            obj.x_max - obj.x_min,
            obj.y_max - obj.y_min,
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        cx, cy = obj.center
        ax.text(cx, cy, obj.name, ha="center", va="center", fontsize=8)


def _draw_presentation_grid(ax, room: Room, grid_spacing_m: float) -> None:
    x = 0.0
    while x <= room.width_m + 1e-9:
        ax.plot([x, x], [0.0, room.height_m], color="#b8b8b8", linestyle="--", linewidth=0.7, zorder=0)
        if x > 0.0 and x < room.width_m:
            ax.text(x, -0.13, f"{x:.1f}", ha="center", va="top", fontsize=9)
        x += grid_spacing_m

    y = 0.0
    while y <= room.height_m + 1e-9:
        ax.plot([0.0, room.width_m], [y, y], color="#b8b8b8", linestyle="--", linewidth=0.7, zorder=0)
        if y > 0.0:
            ax.text(-0.12, y, f"{y:.1f}", ha="right", va="center", fontsize=10, weight="bold")
        y += grid_spacing_m


def _draw_home_test_side_panel(
    panel_ax,
    room: Room,
    packet_count: int,
    latest: Estimate,
    latest_packet: SensorPacket | None,
    start_x_m: float,
    start_y_m: float,
    grid_spacing_m: float,
    ray_length_m: float,
) -> None:
    panel_ax.set_xlim(0.0, 1.0)
    panel_ax.set_ylim(0.0, 1.0)

    _panel_box(
        panel_ax,
        y_top=0.94,
        title="Room",
        body=(
            f"Size: {room.width_m:.1f} m x {room.height_m:.1f} m\n"
            f"Grid: {grid_spacing_m:.1f} m\n"
            "Origin: bottom-left"
        ),
        edge_color="black",
    )
    _panel_box(
        panel_ax,
        y_top=0.70,
        title="Device Test",
        body=(
            f"Packets: {packet_count}\n"
            f"Latest x,y: ({latest.x_m:.2f}, {latest.y_m:.2f}) m\n"
            f"Heading: {latest.heading_rad:.2f} rad\n"
            f"Confidence: {latest.confidence:.2f}\n"
            f"State: {latest.motion_state}"
        ),
        edge_color="#6d9eeb",
        face_color="#f8fbff",
    )
    tof_text = (
        f"Distance: {ray_length_m:.2f} m\n"
        f"Valid: {int(latest_packet.distance_valid)}"
        if latest_packet is not None
        else "Distance: n/a\nValid: 0"
    )
    _panel_box(
        panel_ax,
        y_top=0.39,
        title="ToF",
        body=tof_text,
        edge_color="#0b57d0",
        face_color="#f7fbff",
    )
    _panel_box(
        panel_ax,
        y_top=0.20,
        title="Known Start",
        body=f"Reset dock:\n({start_x_m:.1f}, {start_y_m:.1f}) m",
        edge_color="#b48bea",
        face_color="#fbf8ff",
    )


def _panel_box(
    panel_ax,
    y_top: float,
    title: str,
    body: str,
    edge_color: str,
    face_color: str = "white",
) -> None:
    panel_ax.text(
        0.05,
        y_top,
        f"{title}\n{body}",
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.55,rounding_size=0.06",
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.25,
        ),
    )


def _draw_presentation_room(ax, room: Room) -> None:
    wall_width = 2.6
    ax.plot([0.0, room.width_m], [room.height_m, room.height_m], color="black", linewidth=wall_width)
    ax.plot([0.0, 0.0], [0.0, room.height_m], color="black", linewidth=wall_width)
    ax.plot([room.width_m, room.width_m], [0.0, room.height_m], color="black", linewidth=wall_width)

    gap_left = room.doorway_center_x - room.doorway_width_m / 2.0
    gap_right = room.doorway_center_x + room.doorway_width_m / 2.0
    ax.plot([0.0, gap_left], [0.0, 0.0], color="black", linewidth=wall_width)
    ax.plot([gap_right, room.width_m], [0.0, 0.0], color="black", linewidth=wall_width)
    ax.text(room.doorway_center_x, -0.32, "Doorway", ha="center", va="top", fontsize=9)
    ax.text(room.doorway_center_x, -0.47, "x ~= 2.0, y = 0", ha="center", va="top", fontsize=8)

    ax.scatter([0.0, room.width_m, room.width_m, 0.0], [0.0, 0.0, room.height_m, room.height_m], s=75, color="black", zorder=5)
    ax.text(0.08, room.height_m + 0.08, "(0, 6.0)", fontsize=10, weight="bold")
    ax.text(room.width_m - 0.42, room.height_m + 0.08, "(4.0, 6.0)", fontsize=10, weight="bold")
    ax.text(-0.04, -0.25, "(0, 0)", fontsize=10, weight="bold", ha="left")
    ax.text(room.width_m - 0.26, -0.25, "(4.0, 0)", fontsize=10, weight="bold", ha="left")

    ax.annotate("", xy=(room.width_m + 0.48, 0.0), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.8))
    ax.annotate("", xy=(0.0, room.height_m + 0.48), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.8))
    ax.text(room.width_m + 0.43, -0.18, "x", fontsize=15, weight="bold")
    ax.text(-0.22, room.height_m + 0.54, "y", fontsize=15, weight="bold")
    ax.text(room.width_m / 2.0, -0.78, "4.0 m (x direction)", ha="center", fontsize=12, weight="bold")
    ax.text(
        -1.02,
        room.height_m / 2.0,
        "6.0 m\n(y direction)",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.04", facecolor="white", edgecolor="none", alpha=0.9),
    )

    object_styles = {
        "panel_A": ("#dff2df", "A", "Panel A"),
        "panel_B": ("#dce9ff", "B", "Panel B"),
        "desk": ("#ead9c9", "", "Desk center"),
    }
    for obj in room.objects:
        facecolor, symbol, label = object_styles.get(obj.name, ("#eeeeee", "", obj.name))
        ax.add_patch(
            plt.Rectangle(
                (obj.x_min, obj.y_min),
                obj.x_max - obj.x_min,
                obj.y_max - obj.y_min,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.2,
            )
        )
        cx, cy = obj.center
        if symbol:
            ax.text(cx, cy, symbol, ha="center", va="center", fontsize=15, weight="bold")
            ax.text(cx, obj.y_min - 0.12, f"{label}\n({cx:.1f}, {cy:.1f})", ha="center", va="top", fontsize=9, weight="bold")
        else:
            ax.text(cx, obj.y_max + 0.12, f"{label}\n({cx:.1f}, {cy:.1f})", ha="center", va="bottom", fontsize=9, weight="bold")
            ax.plot([cx - 0.07, cx + 0.07], [cy, cy], color="black", linewidth=1.0)
            ax.plot([cx, cx], [cy - 0.07, cy + 0.07], color="black", linewidth=1.0)
