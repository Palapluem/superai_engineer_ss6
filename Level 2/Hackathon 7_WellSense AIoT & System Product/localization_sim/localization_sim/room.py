from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, isfinite
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


@dataclass(frozen=True)
class RectObject:
    """Axis-aligned rectangular object in the room map.

    Coordinates are meters in the room coordinate frame.
    """

    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> Point:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x_min - margin <= x <= self.x_max + margin
            and self.y_min - margin <= y <= self.y_max + margin
        )

    def segments(self) -> List[Segment]:
        p1 = (self.x_min, self.y_min)
        p2 = (self.x_max, self.y_min)
        p3 = (self.x_max, self.y_max)
        p4 = (self.x_min, self.y_max)
        return [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]


@dataclass
class RaycastHit:
    distance_m: float
    target_name: str
    hit_point: Point


@dataclass
class Room:
    """Known room geometry.

    The room is rectangular, with optional doorway gap and known rectangular
    landmarks/objects that can be seen by the forward-facing distance sensor.
    """

    width_m: float
    height_m: float
    doorway_center_x: float = 2.0
    doorway_width_m: float = 0.4
    objects: Sequence[RectObject] = ()

    def wall_segments(self) -> List[Tuple[str, Segment]]:
        w, h = self.width_m, self.height_m
        gap_left = max(0.0, self.doorway_center_x - self.doorway_width_m / 2.0)
        gap_right = min(w, self.doorway_center_x + self.doorway_width_m / 2.0)

        segments: List[Tuple[str, Segment]] = [
            ("left_wall", ((0.0, 0.0), (0.0, h))),
            ("right_wall", ((w, 0.0), (w, h))),
            ("top_wall", ((0.0, h), (w, h))),
        ]

        # Bottom wall has a doorway gap.
        if gap_left > 0.0:
            segments.append(("bottom_wall", ((0.0, 0.0), (gap_left, 0.0))))
        if gap_right < w:
            segments.append(("bottom_wall", ((gap_right, 0.0), (w, 0.0))))
        return segments

    def all_segments(self) -> List[Tuple[str, Segment]]:
        out = list(self.wall_segments())
        for obj in self.objects:
            for seg in obj.segments():
                out.append((obj.name, seg))
        return out

    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        return margin <= x <= self.width_m - margin and margin <= y <= self.height_m - margin

    def is_blocked(self, x: float, y: float, margin: float = 0.05) -> bool:
        """Return True if a candidate point is inside a mapped obstacle."""
        return any(obj.contains(x, y, margin=margin) for obj in self.objects)

    def is_valid_position(self, x: float, y: float, margin: float = 0.05) -> bool:
        return self.in_bounds(x, y, margin=margin) and not self.is_blocked(x, y, margin=margin)

    def raycast(
        self,
        x: float,
        y: float,
        heading_rad: float,
        max_range_m: Optional[float] = None,
    ) -> Optional[RaycastHit]:
        """Cast a ray from (x,y) along heading and return nearest hit.

        If max_range_m is provided and the nearest hit is farther than that,
        return None. Headings are radians: 0=east, pi/2=north.
        """
        direction = (cos(heading_rad), sin(heading_rad))
        best_t = float("inf")
        best_name = ""
        best_point: Optional[Point] = None

        for name, segment in self.all_segments():
            t = _ray_segment_intersection((x, y), direction, segment)
            if t is not None and 0.0 <= t < best_t:
                best_t = t
                best_name = name
                best_point = (x + t * direction[0], y + t * direction[1])

        if not isfinite(best_t) or best_point is None:
            return None
        if max_range_m is not None and best_t > max_range_m:
            return None
        return RaycastHit(distance_m=best_t, target_name=best_name, hit_point=best_point)


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _ray_segment_intersection(origin: Point, direction: Point, segment: Segment) -> Optional[float]:
    """Return distance parameter t where ray hits segment, or None.

    Ray:     origin + t * direction, t >= 0
    Segment: p + u * (q - p), 0 <= u <= 1
    direction should be unit-length for t to be metric distance.
    """
    p, q = segment
    s = _sub(q, p)
    denom = _cross(direction, s)
    if abs(denom) < 1e-9:
        return None

    p_minus_o = _sub(p, origin)
    t = _cross(p_minus_o, s) / denom
    u = _cross(p_minus_o, direction) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


def make_example_room() -> Room:
    """Example 4.0 m × 6.0 m room from the diagram."""
    objects = [
        RectObject("panel_A", 0.85, 4.65, 1.15, 4.95),
        RectObject("panel_B", 0.65, 1.65, 0.95, 1.95),
        RectObject("desk", 2.65, 4.20, 3.35, 4.80),
    ]
    return Room(width_m=4.0, height_m=6.0, objects=objects)
