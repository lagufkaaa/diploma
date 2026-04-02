import copy
import math
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import unary_union

from core.data import Data, Item
from solvers.greedy_result_cache import (
    GLOBAL_GREEDY_RESULT_MEMORY_CACHE,
    GreedyResultDiskCache,
    build_greedy_result_cache_key,
    resolve_greedy_result_cache_path,
)


@dataclass
class _Placement:
    item: Item
    x: float
    y: float
    strip: int
    geom: object


class GreedySolver:
    """
    Continuous Bottom-Left Fill greedy packing.

    For every item (one chosen rotation per id group), it builds the continuous
    feasible anchor region as:
        domain(item) \ union(translated_nfp(item, placed_item))
    and then selects the point with minimal y, and minimal x as a tie-breaker.
    """

    def __init__(
        self,
        data: Data,
        height: float,
        width: float,
        S: int = 1,
        delta_x: float = 1.0,
        eps_area: float = 1e-6,
        enable_progress_log: bool = False,
        log_interval_sec: float = 2.0,
        log_prefix: str = "[greedy]",
        use_result_cache: bool = True,
        result_cache_path: Optional[str] = None,
        result_cache_ttl_days: Optional[float] = None,
        shared_result_cache: Optional[MutableMapping[str, bytes]] = None,
    ):
        self.data = data
        self.height = float(height)
        self.width = float(width)
        self.S = max(1, int(S))
        self.h = self.height / self.S
        # Kept for backward compatibility with callers.
        # In continuous BLF mode this value is intentionally unused.
        self.delta_x = max(float(delta_x), 1e-12)
        self.eps_area = float(eps_area)
        self._eps_shift = 1e-9
        self._coord_eps = 1e-8
        self.enable_progress_log = bool(enable_progress_log)
        self.log_interval_sec = max(0.2, float(log_interval_sec))
        self.log_prefix = str(log_prefix)
        self.use_result_cache = bool(use_result_cache)
        self.result_cache_path = (
            resolve_greedy_result_cache_path(result_cache_path) if self.use_result_cache else None
        )
        self.result_cache_ttl_seconds = (
            None
            if result_cache_ttl_days is None
            else max(0.0, float(result_cache_ttl_days) * 24.0 * 3600.0)
        )
        if self.use_result_cache:
            self.result_memory_cache: Optional[MutableMapping[str, bytes]] = (
                shared_result_cache
                if shared_result_cache is not None
                else GLOBAL_GREEDY_RESULT_MEMORY_CACHE
            )
        else:
            self.result_memory_cache = None
        self.last_result_cache_hit = False
        self.last_result_cache_key: Optional[str] = None
        self._solve_start_ts = 0.0
        self._last_progress_log_ts = 0.0

        self._local_geom: Dict[Item, object] = {}
        self._anchor_local: Dict[Item, Tuple[float, float]] = {}
        for it in self.data.items:
            pts = np.asarray(it.points, dtype=float)
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            self._local_geom[it] = poly
            self._anchor_local[it] = (float(pts[0][0]), float(pts[0][1]))

    def solve(self):
        self._solve_start_ts = time.perf_counter()
        self._last_progress_log_ts = self._solve_start_ts - self.log_interval_sec
        self.last_result_cache_hit = False
        self.last_result_cache_key = None

        cache_key: Optional[str] = None
        if self.use_result_cache:
            cache_key = build_greedy_result_cache_key(
                self.data,
                height=self.height,
                width=self.width,
                S=self.S,
                eps_area=self.eps_area,
            )
            self.last_result_cache_key = cache_key

            cached_payload = None
            if self.result_memory_cache is not None:
                cached_payload = self.result_memory_cache.get(cache_key)

            if cached_payload is None and self.result_cache_path is not None:
                disk_cache = GreedyResultDiskCache(self.result_cache_path)
                try:
                    cached_payload = disk_cache.get(cache_key, ttl_seconds=self.result_cache_ttl_seconds)
                finally:
                    disk_cache.close()
                if cached_payload is not None and self.result_memory_cache is not None:
                    self.result_memory_cache[cache_key] = cached_payload

            if cached_payload is not None:
                try:
                    cached_result = pickle.loads(cached_payload)
                except Exception:
                    cached_result = None
                if isinstance(cached_result, dict):
                    self.last_result_cache_hit = True
                    self._progress_log("greedy result cache hit", force=True)
                    return copy.deepcopy(cached_result)

        item_groups = self._build_item_groups()
        placed: List[_Placement] = []
        packed_map: Dict[Item, _Placement] = {}
        item_index = {it: idx for idx, it in enumerate(self.data.items)}
        placement_order: List[int] = []
        total_groups = len(item_groups)

        self._progress_log(
            f"start BLF: groups={total_groups}, items={len(self.data.items)}, strips={self.S}",
            force=True,
        )

        for group_idx, rotations in enumerate(item_groups, start=1):
            best: Optional[_Placement] = None
            self._progress_log(
                f"group {group_idx}/{total_groups}: trying {len(rotations)} rotation(s), already_placed={len(placed)}",
                force=True,
            )

            for rotation_idx, candidate_item in enumerate(rotations, start=1):
                candidate_place = self._find_bottom_left_position(
                    candidate_item,
                    placed,
                    group_idx=group_idx,
                    total_groups=total_groups,
                    rotation_idx=rotation_idx,
                    rotations_total=len(rotations),
                )
                if candidate_place is None:
                    continue
                if best is None or self._is_better_bottom_left(candidate_place, best):
                    best = candidate_place

            if best is not None:
                placed.append(best)
                packed_map[best.item] = best
                best_idx = item_index.get(best.item)
                if best_idx is not None:
                    placement_order.append(int(best_idx))
                self._progress_log(
                    (
                        f"group {group_idx}/{total_groups}: placed id={best.item.id} at "
                        f"(x={best.x:.3f}, y={best.y:.3f}), packed={len(placed)}"
                    ),
                    force=True,
                )
            else:
                self._progress_log(
                    f"group {group_idx}/{total_groups}: not placed, packed={len(placed)}",
                    force=True,
                )

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

        result = {
            "p": p,
            "x": x,
            "y": y,
            "s": s,
            "deltas": deltas,
            "placement_order": placement_order,
            "objective_value": objective,
            "status": "OPTIMAL",
        }
        self._progress_log(
            (
                f"done BLF: packed={len(placed)}/{total_groups}, "
                f"objective={objective:.3f}"
            ),
            force=True,
        )

        if self.use_result_cache and cache_key is not None:
            payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            if self.result_memory_cache is not None:
                self.result_memory_cache[cache_key] = payload
            if self.result_cache_path is not None:
                disk_cache = GreedyResultDiskCache(self.result_cache_path)
                try:
                    disk_cache.put(cache_key, payload)
                finally:
                    disk_cache.close()

        return result

    def _progress_log(self, message: str, *, force: bool = False) -> None:
        if not self.enable_progress_log:
            return
        now = time.perf_counter()
        if not force and (now - self._last_progress_log_ts) < self.log_interval_sec:
            return
        elapsed = now - self._solve_start_ts if self._solve_start_ts > 0.0 else 0.0
        print(f"{self.log_prefix} t={elapsed:7.2f}s {message}", flush=True)
        self._last_progress_log_ts = now

    def _format_progress_context(
        self,
        item: Item,
        *,
        group_idx: Optional[int],
        total_groups: Optional[int],
        rotation_idx: Optional[int],
        rotations_total: Optional[int],
    ) -> str:
        parts: List[str] = []
        if group_idx is not None and total_groups is not None:
            parts.append(f"group {group_idx}/{total_groups}")
        if rotation_idx is not None and rotations_total is not None:
            parts.append(f"rot {rotation_idx}/{rotations_total}")
        parts.append(f"id={item.id}")
        return ", ".join(parts)

    def _build_item_groups(self) -> List[List[Item]]:
        grouped: Dict[object, List[Item]] = {}
        for it in self.data.items:
            grouped.setdefault(it.id, []).append(it)

        return sorted(
            grouped.values(),
            key=lambda group: max(float(it.area) for it in group),
            reverse=True,
        )

    def _is_better_bottom_left(self, a: _Placement, b: _Placement) -> bool:
        if a.y < b.y - self._coord_eps:
            return True
        if abs(a.y - b.y) <= self._coord_eps and a.x < b.x - self._coord_eps:
            return True
        return False

    # Backward-compatible alias used by older callers/tests.
    def _find_first_feasible_position(
        self, item: Item, placed: List[_Placement]
    ) -> Optional[_Placement]:
        return self._find_bottom_left_position(item, placed)

    def _find_bottom_left_position(
        self,
        item: Item,
        placed: List[_Placement],
        *,
        group_idx: Optional[int] = None,
        total_groups: Optional[int] = None,
        rotation_idx: Optional[int] = None,
        rotations_total: Optional[int] = None,
    ) -> Optional[_Placement]:
        ctx = self._format_progress_context(
            item,
            group_idx=group_idx,
            total_groups=total_groups,
            rotation_idx=rotation_idx,
            rotations_total=rotations_total,
        )
        domain = self._build_anchor_domain(item)
        if domain is None or domain.is_empty:
            self._progress_log(f"{ctx}: empty anchor domain")
            return None

        self._progress_log(f"{ctx}: build forbidden union, placed={len(placed)}")
        forbidden_union = self._build_translated_forbidden_union(item, placed)
        self._progress_log(f"{ctx}: build feasible anchor geometry")
        feasible = self._build_feasible_anchor_geometry(domain, forbidden_union)
        if feasible.is_empty:
            self._progress_log(f"{ctx}: no feasible anchors after NFP")
            return None

        x_checked = 0
        y_level_count = 0
        for y in self._iter_candidate_y_levels(feasible):
            y_level_count += 1
            self._progress_log(
                f"{ctx}: scan y-level {y_level_count}, y={float(y):.3f}, x_checked={x_checked}",
            )
            for x in self._iter_x_candidates_on_level(feasible, y):
                x_checked += 1
                if x_checked % 200 == 0:
                    self._progress_log(
                        f"{ctx}: checked x-candidates={x_checked} on main pass",
                    )
                if not self._inside_container(item, x, y):
                    continue
                if not self._non_overlapping_by_nfp(item, x, y, placed):
                    continue

                geom = self._placed_geometry(item, x, y)
                strip = self._strip_index(y)
                self._progress_log(
                    (
                        f"{ctx}: placed at (x={float(x):.3f}, y={float(y):.3f}) "
                        f"after y_levels={y_level_count}, x_checked={x_checked}"
                    )
                )
                return _Placement(item=item, x=float(x), y=float(y), strip=strip, geom=geom)

        # Fallback to point-based candidates for unusual degenerate geometries.
        fallback_checked = 0
        for x, y in self._iter_bottom_left_candidates(feasible):
            fallback_checked += 1
            if fallback_checked % 100 == 0:
                self._progress_log(
                    f"{ctx}: fallback checked={fallback_checked}, main_x_checked={x_checked}",
                )
            if not self._inside_container(item, x, y):
                continue
            if not self._non_overlapping_by_nfp(item, x, y, placed):
                continue

            geom = self._placed_geometry(item, x, y)
            strip = self._strip_index(y)
            self._progress_log(
                (
                    f"{ctx}: placed in fallback at (x={float(x):.3f}, y={float(y):.3f}), "
                    f"fallback_checked={fallback_checked}, main_x_checked={x_checked}"
                )
            )
            return _Placement(item=item, x=float(x), y=float(y), strip=strip, geom=geom)

        self._progress_log(
            (
                f"{ctx}: no placement found "
                f"(y_levels={y_level_count}, main_x_checked={x_checked}, fallback_checked={fallback_checked})"
            )
        )
        return None

    def _build_feasible_anchor_geometry(self, domain, forbidden_union):
        if forbidden_union is None or forbidden_union.is_empty:
            return domain

        # Allow touching: remove only the forbidden closed set and then add back
        # the boundary points/segments where anchor touching is valid.
        strict_free = domain.difference(forbidden_union)
        boundary_touch = domain.intersection(forbidden_union.boundary)
        return unary_union([strict_free, boundary_touch])

    def _build_anchor_domain(self, item: Item):
        min_x = max(0.0, -float(item.xmin))
        max_x = self.width - float(item.xmax)
        min_y = max(0.0, -float(item.ymin))
        max_y = self.height - float(item.ymax)

        if max_x + self._eps_shift < min_x or max_y + self._eps_shift < min_y:
            return None

        if abs(max_x - min_x) <= self._eps_shift and abs(max_y - min_y) <= self._eps_shift:
            return Point(min_x, min_y)
        if abs(max_x - min_x) <= self._eps_shift:
            return LineString([(min_x, min_y), (min_x, max_y)])
        if abs(max_y - min_y) <= self._eps_shift:
            return LineString([(min_x, min_y), (max_x, min_y)])
        return box(min_x, min_y, max_x, max_y)

    def _build_translated_forbidden_union(self, item: Item, placed: List[_Placement]):
        if not placed:
            return None

        translated = []
        nfp_map = item.nfp or {}
        for rec in placed:
            nfp = nfp_map.get(rec.item)
            if nfp is None:
                continue
            geom = affinity.translate(nfp, xoff=float(rec.x), yoff=float(rec.y))
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            translated.append(geom)

        if not translated:
            return None
        return unary_union(translated)

    def _iter_bottom_left_candidates(self, feasible) -> Iterable[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        seen = set()

        for x, y in self._iter_geometry_points(feasible):
            rx = round(float(x), 8)
            ry = round(float(y), 8)
            key = (rx, ry)
            if key in seen:
                continue
            seen.add(key)

            pt = Point(float(x), float(y))
            if not feasible.covers(pt):
                continue
            points.append((float(x), float(y)))

        if not points and hasattr(feasible, "representative_point"):
            rp = feasible.representative_point()
            if not rp.is_empty and feasible.covers(rp):
                points.append((float(rp.x), float(rp.y)))

        points.sort(key=lambda p: (p[1], p[0]))
        for p in points:
            yield p

    def _iter_candidate_y_levels(self, feasible) -> Iterable[float]:
        ys: List[float] = []
        seen = set()
        for _x, y in self._iter_geometry_points(feasible):
            ry = round(float(y), 8)
            if ry in seen:
                continue
            seen.add(ry)
            ys.append(float(y))
        ys.sort()
        for y in ys:
            yield y

    def _iter_x_candidates_on_level(self, feasible, y: float) -> Iterable[float]:
        minx, _, maxx, _ = feasible.bounds
        line = LineString(
            [
                (float(minx) - 1.0, float(y)),
                (float(maxx) + 1.0, float(y)),
            ]
        )
        inter = feasible.intersection(line)

        xs = self._extract_xs_from_horizontal_intersection(inter)
        if not xs:
            return

        xs.sort()
        seen = set()
        for x in xs:
            rx = round(float(x), 8)
            if rx in seen:
                continue
            seen.add(rx)
            pt = Point(float(x), float(y))
            if feasible.covers(pt):
                yield float(x)

    def _extract_xs_from_horizontal_intersection(self, geom) -> List[float]:
        if geom is None or geom.is_empty:
            return []

        if isinstance(geom, Point):
            return [float(geom.x)]

        if isinstance(geom, MultiPoint):
            return [float(pt.x) for pt in geom.geoms]

        if isinstance(geom, LineString):
            xs = [float(x) for x, _y in geom.coords]
            return [min(xs)] if xs else []

        if isinstance(geom, MultiLineString):
            out: List[float] = []
            for line in geom.geoms:
                xs = [float(x) for x, _y in line.coords]
                if xs:
                    out.append(min(xs))
            return out

        if isinstance(geom, GeometryCollection):
            out: List[float] = []
            for g in geom.geoms:
                out.extend(self._extract_xs_from_horizontal_intersection(g))
            return out

        if isinstance(geom, Polygon):
            # Degenerate fallback, normally not expected for line intersection.
            return [float(geom.bounds[0])]

        if isinstance(geom, MultiPolygon):
            return [float(poly.bounds[0]) for poly in geom.geoms]

        if hasattr(geom, "coords"):
            xs = [float(x) for x, _y in geom.coords]
            return [min(xs)] if xs else []

        return []

    def _iter_geometry_points(self, geom) -> Iterable[Tuple[float, float]]:
        if geom is None or geom.is_empty:
            return

        if isinstance(geom, Point):
            yield (float(geom.x), float(geom.y))
            return

        if isinstance(geom, MultiPoint):
            for pt in geom.geoms:
                yield (float(pt.x), float(pt.y))
            return

        if isinstance(geom, LineString):
            for x, y in geom.coords:
                yield (float(x), float(y))
            return

        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                for x, y in line.coords:
                    yield (float(x), float(y))
            return

        if isinstance(geom, Polygon):
            for x, y in geom.exterior.coords:
                yield (float(x), float(y))
            for ring in geom.interiors:
                for x, y in ring.coords:
                    yield (float(x), float(y))
            return

        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                yield from self._iter_geometry_points(poly)
            return

        if isinstance(geom, GeometryCollection):
            for g in geom.geoms:
                yield from self._iter_geometry_points(g)
            return

        if hasattr(geom, "coords"):
            for x, y in geom.coords:
                yield (float(x), float(y))

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

    def _non_overlapping_by_nfp(
        self, item: Item, x: float, y: float, placed: List[_Placement]
    ) -> bool:
        if not placed:
            return True

        ax_local, ay_local = self._anchor_local[item]
        anchor = Point(float(x) + ax_local, float(y) + ay_local)
        nfp_map = item.nfp or {}
        candidate_geom = None
        for rec in placed:
            nfp = nfp_map.get(rec.item)
            if nfp is not None:
                translated_nfp = affinity.translate(
                    nfp, xoff=float(rec.x), yoff=float(rec.y)
                )
                # "contains" excludes boundary, so touching is allowed.
                if translated_nfp.contains(anchor):
                    return False
                # NFP is a coarse geometric guard; due numeric effects we still
                # validate area-overlap exactly below.

            if candidate_geom is None:
                candidate_geom = self._placed_geometry(item, x, y)

            other = rec.geom
            if (
                candidate_geom.bounds[2] <= other.bounds[0]
                or other.bounds[2] <= candidate_geom.bounds[0]
                or candidate_geom.bounds[3] <= other.bounds[1]
                or other.bounds[3] <= candidate_geom.bounds[1]
            ):
                continue
            if candidate_geom.intersects(other):
                inter = candidate_geom.intersection(other)
                if float(getattr(inter, "area", 0.0) or 0.0) > self.eps_area:
                    return False

        return True

    def _strip_index(self, y: float) -> int:
        if self.S <= 1:
            return 0
        idx = int(math.floor((y + self._eps_shift) / self.h))
        return max(0, min(self.S - 1, idx))
