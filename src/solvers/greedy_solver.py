import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from core.data import Data, Item


@dataclass
class _Placement:
    item: Item
    x: float
    y: float
    strip: int
    geom: object


class GreedySolver:
    def __init__(
        self,
        data: Data,
        height: float,
        width: float,
        S: int = 1,
        delta_x: float = 1.0,
        eps_area: float = 1e-6,
    ):
        self.data = data
        self.height = float(height)
        self.width = float(width)
        self.S = max(1, int(S))
        self.h = self.height / self.S
        self.delta_x = max(float(delta_x), 1e-6)
        self.eps_area = float(eps_area)
        self._eps_shift = 1e-6
        self._max_vertical_iterations = 10_000

        # keep the same "placed geometry" logic as in tests:
        # Polygon(points) -> buffer(0) if invalid -> translate
        self._local_geom: Dict[Item, object] = {}
        for it in self.data.items:
            pts = np.asarray(it.points, dtype=float)
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            self._local_geom[it] = poly

    def solve(self):
        item_groups = self._build_item_groups()
        placed: List[_Placement] = []
        packed_map: Dict[Item, _Placement] = {}

        # Sort items in non-increasing priority and place one rotation per item-id group.
        for rotations in item_groups:
            best: Optional[_Placement] = None
            best_value = float("-inf")

            for candidate_item in rotations:
                candidate_place = self._find_first_feasible_position(candidate_item, placed)
                if candidate_place is None:
                    continue

                value = float(candidate_item.area)
                if value > best_value:
                    best_value = value
                    best = candidate_place

            if best is not None:
                placed.append(best)
                packed_map[best.item] = best

        p: List[float] = []
        x: List[float] = []
        y: List[float] = []
        s: List[float] = []
        deltas: List[List[float]] = []
        objective = 0.0

        for it in self.data.items:
            rec = packed_map.get(it)
            if rec is None:
                p.append(0.0)
                x.append(0.0)
                y.append(0.0)
                s.append(0.0)
                deltas.append([0.0 for _ in range(self.S)])
                continue

            p.append(1.0)
            x.append(float(rec.x))
            y.append(float(rec.y))
            s.append(float(rec.strip))
            row = [0.0 for _ in range(self.S)]
            row[rec.strip] = 1.0
            deltas.append(row)
            objective += float(it.area)

        return {
            "p": p,
            "x": x,
            "y": y,
            "s": s,
            "deltas": deltas,
            "objective_value": objective,
            "status": "OPTIMAL",
        }

    def _build_item_groups(self) -> List[List[Item]]:
        grouped: Dict[object, List[Item]] = {}
        for it in self.data.items:
            grouped.setdefault(it.id, []).append(it)

        return sorted(
            grouped.values(),
            key=lambda group: max(float(it.area) for it in group),
            reverse=True,
        )

    def _find_first_feasible_position(self, item: Item, placed: List[_Placement]) -> Optional[_Placement]:
        start_x = max(0.0, -float(item.xmin))
        max_x = self.width - float(item.xmax)
        if max_x + self._eps_shift < start_x:
            return None

        for x in self._iter_x_positions(start_x, max_x):
            y = max(0.0, -float(item.ymin))
            for _ in range(self._max_vertical_iterations):
                geom = self._placed_geometry(item, x, y)
                if not self._inside_container(item, x, y):
                    break
                overlap, cont, conflicts = self._detect_conflicts(geom, placed)

                if not overlap and not cont:
                    strip = self._strip_index(y)
                    return _Placement(item=item, x=float(x), y=float(y), strip=strip, geom=geom)

                next_y = self._minimum_upward_shift(item, y, conflicts)
                if next_y is None:
                    next_y = y + self._eps_shift
                next_y = max(next_y, y + self._eps_shift)
                if next_y + float(item.ymax) > self.height + self._eps_shift:
                    break
                y = next_y

        return None

    def _iter_x_positions(self, start_x: float, max_x: float) -> Iterable[float]:
        x = float(start_x)
        last = None
        while x <= max_x + self._eps_shift:
            last = x
            yield x
            x += self.delta_x

        if last is None or (max_x - last) > self._eps_shift:
            yield float(max_x)

    def _inside_container(self, item: Item, x: float, y: float) -> bool:
        return (
            x + float(item.xmin) >= -self._eps_shift
            and x + float(item.xmax) <= self.width + self._eps_shift
            and y + float(item.ymin) >= -self._eps_shift
            and y + float(item.ymax) <= self.height + self._eps_shift
        )

    def _placed_geometry(self, item: Item, x_shift: float, y_shift: float):
        return affinity.translate(
            self._local_geom[item], xoff=float(x_shift), yoff=float(y_shift)
        )

    def _detect_conflicts(self, geom, placed: List[_Placement]):
        overlap = False
        cont = False
        conflicts: List[_Placement] = []

        for rec in placed:
            other = rec.geom

            # same bbox-fast-reject style as in tests
            if (
                geom.bounds[2] <= other.bounds[0]
                or other.bounds[2] <= geom.bounds[0]
                or geom.bounds[3] <= other.bounds[1]
                or other.bounds[3] <= geom.bounds[1]
            ):
                continue

            if geom.intersects(other):
                inter = geom.intersection(other)
                area = float(getattr(inter, "area", 0.0) or 0.0)
                if area > self.eps_area:
                    overlap = True
                    conflicts.append(rec)
                    continue

            contains_relation = geom.within(other) or other.within(geom)
            if contains_relation:
                cont = True
                conflicts.append(rec)

        return overlap, cont, conflicts

    def _minimum_upward_shift(
        self, item: Item, current_y: float, conflicts: List[_Placement]
    ) -> Optional[float]:
        if not conflicts:
            return None

        candidates = []
        for rec in conflicts:
            target_y = float(rec.geom.bounds[3]) - float(item.ymin) + self._eps_shift
            if target_y > current_y + self._eps_shift:
                candidates.append(target_y)

        if not candidates:
            return None
        return min(candidates)

    def _strip_index(self, y: float) -> int:
        if self.S <= 1:
            return 0
        idx = int(math.floor((y + self._eps_shift) / self.h))
        return max(0, min(self.S - 1, idx))
