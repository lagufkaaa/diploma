import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from core.data import Data, Item
from solvers.greedy_solver import GreedySolver
from solvers.model_hybrid import Problem as HybridProblem


@dataclass
class _PackedRecord:
    idx: int
    item: Item
    x: float
    y: float
    strip: int
    geom: object


class HybridSolver:
    """
    Hybrid strategy:
    1. Build an initial solution with greedy.
    2. Unpack the last N greedy-packed items.
    3. Build a new top-cropped model container by cut height.
    4. Reoptimize the partial solution with the MIP model inside that container.
    """

    def __init__(
        self,
        data: Data,
        height: float,
        width: float,
        S: int = 1,
        *,
        solver_name: str = "SCIP",
        greedy_delta_x: float = 1.0,
        greedy_eps_area: float = 1e-6,
    ):
        self.data = data
        self.height = float(height)
        self.width = float(width)
        self.S = int(S)
        self.solver_name = solver_name
        self.greedy_delta_x = float(greedy_delta_x)
        self.greedy_eps_area = float(greedy_eps_area)

        self._local_geom: Dict[Item, object] = {}
        for it in self.data.items:
            pts = np.asarray(it.points, dtype=float)
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            self._local_geom[it] = poly

    def solve(
        self,
        *,
        unpack_last_n: int,
        crop_height: float,
        use_top_crop: bool = True,
        free_space_improvement: float = 1.0,
        solver_gap: float = 1.0,
        model_time_limit_sec: Optional[float] = None,
        stop_after_first_solution: bool = True,
        lock_greedy_unpacked: bool = True,
        max_model_unfixed_items: Optional[int] = None,
        model_enable_output: bool = False,
    ):
        t0 = time.perf_counter()
        crop_h_clamped = max(0.0, min(self.height, float(crop_height)))
        model_height = crop_h_clamped if use_top_crop else self.height
        y_offset = self.height - crop_h_clamped if use_top_crop else 0.0
        if use_top_crop:
            packing_y_min = self.height - crop_h_clamped
            packing_y_max = self.height
        else:
            packing_y_min = 0.0
            packing_y_max = self.height

        greedy = GreedySolver(
            self.data,
            height=self.height,
            width=self.width,
            S=self.S,
            delta_x=self.greedy_delta_x,
            eps_area=self.greedy_eps_area,
        )
        greedy_start = time.perf_counter()
        greedy_results = greedy.solve()
        greedy_time = time.perf_counter() - greedy_start

        if greedy_results.get("status") != "OPTIMAL":
            return {
                "status": "GREEDY_FAILED",
                "final": greedy_results,
                "visualization": {
                    "greedy_solution": greedy_results,
                    "packed_indices": [],
                    "candidate_indices": [],
                    "fixed_indices": [],
                    "use_top_crop": bool(use_top_crop),
                    "used_crop_height": crop_h_clamped,
                    "crop_y_min": packing_y_min,
                    "packing_y_min": packing_y_min,
                    "packing_y_max": packing_y_max,
                },
                "hybrid_stats": {
                    "greedy_time_sec": greedy_time,
                    "model_time_sec": 0.0,
                    "total_time_sec": time.perf_counter() - t0,
                },
            }

        packed_records = self._collect_packed_records(greedy_results)
        packed_records.sort(key=lambda rec: (rec.y, rec.x))
        packed_ids = {rec.item.id for rec in packed_records}
        all_item_ids = {it.id for it in self.data.items}
        greedy_unpacked_ids = all_item_ids - packed_ids

        candidate_ids = self._select_candidate_ids(
            packed_records=packed_records,
            unpack_last_n=unpack_last_n,
            max_model_unfixed_items=max_model_unfixed_items,
        )
        model_item_ids: Set[object] = set(candidate_ids)
        if not lock_greedy_unpacked:
            model_item_ids |= greedy_unpacked_ids

        fixed_records = [rec for rec in packed_records if rec.item.id not in model_item_ids]
        forced_unpacked_ids = greedy_unpacked_ids - model_item_ids

        packed_indices = [rec.idx for rec in packed_records]
        candidate_indices = [rec.idx for rec in packed_records if rec.item.id in candidate_ids]
        fixed_indices = [rec.idx for rec in fixed_records]

        visualization_payload = {
            "greedy_solution": greedy_results,
            "packed_indices": packed_indices,
            "candidate_indices": candidate_indices,
            "fixed_indices": fixed_indices,
            "use_top_crop": bool(use_top_crop),
            "used_crop_height": crop_h_clamped,
            "crop_y_min": packing_y_min,
            "packing_y_min": packing_y_min,
            "packing_y_max": packing_y_max,
        }

        greedy_obj = float(greedy_results.get("objective_value") or 0.0)
        greedy_free = self._free_space_percent(greedy_obj)

        target_free_space = None
        if free_space_improvement is not None:
            free_space_improvement = max(0.0, float(free_space_improvement))
            target_free_space = max(0.0, greedy_free - free_space_improvement)

        fixed_area = sum(float(rec.item.area) for rec in fixed_records)
        min_local_objective = None
        if target_free_space is not None:
            container_area = self.width * self.height
            min_total_area = container_area * (1.0 - float(target_free_space) / 100.0)
            min_local_objective = max(0.0, min_total_area - fixed_area)

        model_start = time.perf_counter()
        model_data = self._build_local_model_data(model_item_ids)
        if model_data is None:
            local_results = {"status": "OPTIMAL", "objective_value": 0.0, "p": [], "x": [], "s": [], "deltas": []}
        else:
            problem = HybridProblem(
                model_data,
                S=self.S,
                R=model_data.R,
                height=model_height,
                width=self.width,
                solver_name=self.solver_name,
                enable_output=model_enable_output,
                min_objective_value=min_local_objective,
                relative_gap=solver_gap,
                time_limit_sec=model_time_limit_sec,
                stop_after_first_solution=stop_after_first_solution,
            )
            local_results = problem.solve()

        model_results = self._assemble_full_solution(
            fixed_records=fixed_records,
            model_data=model_data,
            local_results=local_results,
            y_offset=y_offset,
            local_height=model_height,
        )
        if model_results.get("objective_value") is not None and self._has_solution_overlaps(model_results):
            model_results["status"] = "INFEASIBLE"
            model_results["objective_value"] = None
        model_time = time.perf_counter() - model_start

        model_obj = model_results.get("objective_value")
        model_status = str(model_results.get("status", ""))
        model_ok = model_obj is not None and model_status in {"OPTIMAL", "FEASIBLE"}

        improved = model_ok and float(model_obj) > greedy_obj + 1e-9
        model_free = self._free_space_percent(float(model_obj)) if model_obj is not None else None

        full_search_mode = (
            model_time_limit_sec is None
            and not stop_after_first_solution
            and (solver_gap is None or float(solver_gap) <= 0.0)
        )
        proven_global_optimal = model_status == "OPTIMAL"
        can_claim_no_improvement = proven_global_optimal

        if not improved:
            fail_status = "NOT_IMPROVED" if can_claim_no_improvement else "NOT_PROVEN"
            use_model_as_final = fail_status == "NOT_PROVEN" and model_ok

            final_solution = model_results if use_model_as_final else greedy_results
            final_objective = float(model_obj) if use_model_as_final and model_obj is not None else greedy_obj
            final_free_space = (
                float(model_free) if use_model_as_final and model_free is not None else greedy_free
            )
            final_selected = "model" if use_model_as_final else "greedy"
            final_improvement = greedy_free - final_free_space

            return {
                "status": fail_status,
                "selected_solution": final_selected,
                "final": final_solution,
                "model_result": model_results,
                "visualization": visualization_payload,
                "hybrid_stats": {
                    "packed_by_greedy": len(packed_records),
                    "unpack_last_n": int(max(0, unpack_last_n)),
                    "use_top_crop": bool(use_top_crop),
                    "used_crop_height": crop_h_clamped,
                    "candidate_items_for_model": len(candidate_ids),
                    "model_item_ids_count": len(model_item_ids),
                    "restricted_items_in_window": len(model_item_ids),
                    "fixed_items_in_model": len(fixed_records),
                    "forced_unpacked_ids": len(forced_unpacked_ids),
                    "greedy_objective_value": greedy_obj,
                    "model_objective_value": float(model_obj) if model_obj is not None else None,
                    "final_objective_value": final_objective,
                    "greedy_free_space_percent": greedy_free,
                    "model_free_space_percent": model_free,
                    "final_free_space_percent": final_free_space,
                    "free_space_improvement_percent": final_improvement,
                    "target_free_space_percent": target_free_space,
                    "greedy_time_sec": greedy_time,
                    "model_time_sec": model_time,
                    "total_time_sec": time.perf_counter() - t0,
                    "model_status": model_status,
                    "full_search_mode": full_search_mode,
                    "proven_global_optimal": proven_global_optimal,
                    "can_claim_no_improvement": can_claim_no_improvement,
                },
            }

        success_status = "OK" if model_status == "OPTIMAL" else "NOT_PROVEN"
        final_obj = float(model_obj)
        final_free = float(model_free)
        free_space_improvement_abs = greedy_free - final_free

        return {
            "status": success_status,
            "selected_solution": "model",
            "final": model_results,
            "model_result": model_results,
            "visualization": visualization_payload,
            "hybrid_stats": {
                "packed_by_greedy": len(packed_records),
                "unpack_last_n": int(max(0, unpack_last_n)),
                "use_top_crop": bool(use_top_crop),
                "used_crop_height": crop_h_clamped,
                "candidate_items_for_model": len(candidate_ids),
                "model_item_ids_count": len(model_item_ids),
                "restricted_items_in_window": len(model_item_ids),
                "fixed_items_in_model": len(fixed_records),
                "forced_unpacked_ids": len(forced_unpacked_ids),
                "greedy_objective_value": greedy_obj,
                "final_objective_value": final_obj,
                "greedy_free_space_percent": greedy_free,
                "final_free_space_percent": final_free,
                "free_space_improvement_percent": free_space_improvement_abs,
                "target_free_space_percent": target_free_space,
                "greedy_time_sec": greedy_time,
                "model_time_sec": model_time,
                "total_time_sec": time.perf_counter() - t0,
                "model_status": model_status,
                "full_search_mode": full_search_mode,
                "proven_global_optimal": proven_global_optimal,
            },
        }

    def _collect_packed_records(self, greedy_results: dict) -> List[_PackedRecord]:
        p_vals = greedy_results.get("p", [])
        x_vals = greedy_results.get("x", [])
        y_vals = greedy_results.get("y", [])
        s_vals = greedy_results.get("s", [])

        packed_records: List[_PackedRecord] = []
        for idx, item in enumerate(self.data.items):
            if idx >= len(p_vals) or float(p_vals[idx]) <= 0.5:
                continue

            x_val = float(x_vals[idx]) if idx < len(x_vals) else 0.0
            y_val = float(y_vals[idx]) if idx < len(y_vals) else 0.0
            strip_val = int(round(float(s_vals[idx]))) if idx < len(s_vals) else self._strip_from_y(y_val)

            packed_records.append(
                _PackedRecord(
                    idx=idx,
                    item=item,
                    x=x_val,
                    y=y_val,
                    strip=max(0, min(self.S - 1, strip_val)),
                    geom=self._placed_geometry(item, x_val, y_val),
                )
            )

        return packed_records

    def _select_candidate_ids(
        self,
        *,
        packed_records: List[_PackedRecord],
        unpack_last_n: int,
        max_model_unfixed_items: Optional[int],
    ) -> Set[object]:
        candidate_ids: List[object] = []
        seen: Set[object] = set()

        def add_candidate(item_id):
            if item_id in seen:
                return
            seen.add(item_id)
            candidate_ids.append(item_id)

        unpack_last_n = max(0, int(unpack_last_n))
        if unpack_last_n > 0:
            for rec in packed_records[-unpack_last_n:]:
                add_candidate(rec.item.id)

        if max_model_unfixed_items is not None:
            limit = max(0, int(max_model_unfixed_items))
            candidate_ids = candidate_ids[:limit]

        return set(candidate_ids)

    def _build_partial_assignment(
        self,
        *,
        packed_records: List[_PackedRecord],
        candidate_ids: Set[object],
        lock_greedy_unpacked: bool,
    ) -> Tuple[Dict[Item, Tuple[float, int]], Set[object]]:
        fixed_item_assignments: Dict[Item, Tuple[float, int]] = {}
        packed_ids: Set[object] = set()

        for rec in packed_records:
            packed_ids.add(rec.item.id)
            if rec.item.id in candidate_ids:
                continue
            fixed_item_assignments[rec.item] = (rec.x, rec.strip)

        forced_unpacked_ids: Set[object] = set()
        if lock_greedy_unpacked:
            all_item_ids = {it.id for it in self.data.items}
            greedy_unpacked_ids = all_item_ids - packed_ids
            for item_id in greedy_unpacked_ids:
                if item_id not in candidate_ids:
                    forced_unpacked_ids.add(item_id)

        return fixed_item_assignments, forced_unpacked_ids

    def _placed_geometry(self, item: Item, x_shift: float, y_shift: float):
        return affinity.translate(self._local_geom[item], xoff=float(x_shift), yoff=float(y_shift))

    def _strip_from_y(self, y_val: float) -> int:
        if self.S <= 1:
            return 0
        h = self.height / self.S
        idx = int(y_val // h)
        return max(0, min(self.S - 1, idx))

    def _free_space_percent(self, packed_area: float) -> float:
        container_area = self.width * self.height
        if container_area <= 0.0:
            return 0.0
        free = max(0.0, container_area - float(packed_area))
        return 100.0 * free / container_area

    def _build_local_model_data(self, model_item_ids: Set[object]) -> Optional[Data]:
        if not model_item_ids:
            return None

        source_by_id: Dict[object, Item] = {}
        for it in self.data.items:
            if it.id not in model_item_ids:
                continue
            if it.id not in source_by_id:
                source_by_id[it.id] = it
            if abs(float(it.rotation) % 360.0) <= 1e-6:
                source_by_id[it.id] = it

        seeds: List[Item] = []
        for item_id in model_item_ids:
            src = source_by_id.get(item_id)
            if src is None:
                continue
            clone = Item(np.asarray(src.points, dtype=float).copy())
            clone.id = src.id
            clone.rotation = 0
            seeds.append(clone)

        if not seeds:
            return None

        cache_ttl_days = None
        ttl_seconds = getattr(self.data, "cache_ttl_seconds", None)
        if ttl_seconds is not None:
            cache_ttl_days = float(ttl_seconds) / (24.0 * 3600.0)

        return Data(
            seeds,
            R=self.data.R,
            parallel_nfp=bool(getattr(self.data, "parallel_nfp", True)),
            nfp_workers=getattr(self.data, "nfp_workers", None),
            use_cache=bool(getattr(self.data, "use_cache", True)),
            cache_path=str(getattr(self.data, "cache_path", "")) if getattr(self.data, "cache_path", None) else None,
            cache_ttl_days=cache_ttl_days,
            use_memory_cache=bool(getattr(self.data, "use_memory_cache", True)),
            shared_memory_cache=getattr(self.data, "memory_cache", None),
        )

    def _assemble_full_solution(
        self,
        *,
        fixed_records: List[_PackedRecord],
        model_data: Optional[Data],
        local_results: dict,
        y_offset: float,
        local_height: float,
    ) -> dict:
        status = str(local_results.get("status", "NOT_SOLVED"))
        if status not in {"OPTIMAL", "FEASIBLE"}:
            return {"status": status, "objective_value": None}

        n_full = len(self.data.items)
        p = [0.0 for _ in range(n_full)]
        x = [0.0 for _ in range(n_full)]
        y = [0.0 for _ in range(n_full)]
        s = [0.0 for _ in range(n_full)]
        deltas = [[0.0 for _ in range(self.S)] for _ in range(n_full)]

        used_indices: Set[int] = set()
        total_area = 0.0

        for rec in fixed_records:
            idx = rec.idx
            used_indices.add(idx)
            p[idx] = 1.0
            x[idx] = float(rec.x)
            y[idx] = float(rec.y)
            strip_idx = max(0, min(self.S - 1, int(rec.strip)))
            s[idx] = float(strip_idx)
            deltas[idx][strip_idx] = 1.0
            total_area += float(rec.item.area)

        if model_data is not None and model_data.items:
            local_p = local_results.get("p", [])
            local_x = local_results.get("x", [])
            local_s = local_results.get("s", [])
            local_d = local_results.get("deltas", [])
            h_local = float(local_height) / float(max(1, self.S))

            by_id_rot: Dict[Tuple[object, int], List[int]] = defaultdict(list)
            by_id: Dict[object, List[int]] = defaultdict(list)
            for idx, it in enumerate(self.data.items):
                rot_key = int(round(float(it.rotation))) % 360
                by_id_rot[(it.id, rot_key)].append(idx)
                by_id[it.id].append(idx)

            for li, lit in enumerate(model_data.items):
                if li >= len(local_p) or float(local_p[li]) <= 0.5:
                    continue

                rot_key = int(round(float(lit.rotation))) % 360
                candidates = by_id_rot.get((lit.id, rot_key), [])
                full_idx = next((idx for idx in candidates if idx not in used_indices), None)
                if full_idx is None:
                    fallback = by_id.get(lit.id, [])
                    full_idx = next((idx for idx in fallback if idx not in used_indices), None)
                if full_idx is None:
                    continue

                used_indices.add(full_idx)
                local_x_val = float(local_x[li]) if li < len(local_x) else 0.0
                if li < len(local_s):
                    local_strip = int(round(float(local_s[li])))
                elif li < len(local_d):
                    row = local_d[li] if isinstance(local_d[li], list) else []
                    local_strip = next((j for j, val in enumerate(row) if float(val) > 0.5), 0)
                else:
                    local_strip = 0

                local_strip = max(0, min(self.S - 1, int(local_strip)))
                global_y = float(y_offset) + float(local_strip) * h_local
                global_strip = self._strip_from_y(global_y)

                p[full_idx] = 1.0
                x[full_idx] = local_x_val
                y[full_idx] = global_y
                s[full_idx] = float(global_strip)
                deltas[full_idx][global_strip] = 1.0
                total_area += float(self.data.items[full_idx].area)

        return {
            "status": status,
            "p": p,
            "x": x,
            "y": y,
            "s": s,
            "deltas": deltas,
            "objective_value": total_area,
        }

    def _has_solution_overlaps(self, solution: dict) -> bool:
        p_vals = solution.get("p", [])
        x_vals = solution.get("x", [])
        y_vals = solution.get("y", [])

        placed = []
        for idx, it in enumerate(self.data.items):
            if idx >= len(p_vals) or float(p_vals[idx]) <= 0.5:
                continue
            x_val = float(x_vals[idx]) if idx < len(x_vals) else 0.0
            y_val = float(y_vals[idx]) if idx < len(y_vals) else 0.0
            placed.append(self._placed_geometry(it, x_val, y_val))

        eps_area = max(1e-9, float(self.greedy_eps_area))
        for i in range(len(placed)):
            ga = placed[i]
            for j in range(i + 1, len(placed)):
                gb = placed[j]
                if (
                    ga.bounds[2] <= gb.bounds[0]
                    or gb.bounds[2] <= ga.bounds[0]
                    or ga.bounds[3] <= gb.bounds[1]
                    or gb.bounds[3] <= ga.bounds[1]
                ):
                    continue
                if ga.intersects(gb):
                    inter = ga.intersection(gb)
                    area = float(getattr(inter, "area", 0.0) or 0.0)
                    if area > eps_area:
                        return True

        return False
