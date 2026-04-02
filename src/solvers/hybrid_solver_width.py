import time
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Set, Tuple

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


class HybridSolverWidth:
    """
    Hybrid strategy:
    1. Build an initial solution with greedy.
    2. Unpack the last N greedy-packed items.
    3. Build a new right-cropped model container by cut width.
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
        greedy_enable_output: bool = True,
        greedy_log_interval_sec: float = 2.0,
        greedy_use_result_cache: bool = True,
        greedy_result_cache_path: Optional[str] = None,
        greedy_result_cache_ttl_days: Optional[float] = None,
        greedy_shared_result_cache: Optional[MutableMapping[str, bytes]] = None,
        hybrid_enable_output: bool = True,
        hybrid_log_interval_sec: float = 2.0,
    ):
        self.data = data
        self.height = float(height)
        self.width = float(width)
        self.S = int(S)
        self.solver_name = solver_name
        self.greedy_delta_x = float(greedy_delta_x)
        self.greedy_eps_area = float(greedy_eps_area)
        self.greedy_enable_output = bool(greedy_enable_output)
        self.greedy_log_interval_sec = max(0.2, float(greedy_log_interval_sec))
        self.greedy_use_result_cache = bool(greedy_use_result_cache)
        self.greedy_result_cache_path = greedy_result_cache_path
        self.greedy_result_cache_ttl_days = greedy_result_cache_ttl_days
        self.greedy_shared_result_cache = greedy_shared_result_cache
        self.hybrid_enable_output = bool(hybrid_enable_output)
        self.hybrid_log_interval_sec = max(0.2, float(hybrid_log_interval_sec))

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
        crop_width: float,
        greedy_delta_x: Optional[float] = None,
        use_right_crop: bool = True,
        free_space_improvement: float = 1.0,
        solver_gap: float = 1.0,
        model_time_limit_sec: Optional[float] = None,
        model_num_threads: Optional[int] = None,
        stop_after_first_solution: bool = True,
        lock_greedy_unpacked: bool = True,
        max_model_unfixed_items: Optional[int] = None,
        model_enable_output: bool = False,
        random_iterations: int = 1,
        random_seed: Optional[int] = 0,
        random_sample_size: Optional[int] = None,
        greedy_enable_output: Optional[bool] = None,
        greedy_log_interval_sec: Optional[float] = None,
        hybrid_enable_output: Optional[bool] = None,
        hybrid_log_interval_sec: Optional[float] = None,
    ):
        t0 = time.perf_counter()
        hybrid_progress_on = (
            self.hybrid_enable_output
            if hybrid_enable_output is None
            else bool(hybrid_enable_output)
        )
        hybrid_log_interval = (
            self.hybrid_log_interval_sec
            if hybrid_log_interval_sec is None
            else max(0.2, float(hybrid_log_interval_sec))
        )
        last_hybrid_log_ts = t0 - hybrid_log_interval

        def _hybrid_log(message: str, *, force: bool = False) -> None:
            nonlocal last_hybrid_log_ts
            if not hybrid_progress_on:
                return
            now = time.perf_counter()
            if not force and (now - last_hybrid_log_ts) < hybrid_log_interval:
                return
            elapsed = now - t0
            print(f"[hybrid-width] t={elapsed:7.2f}s {message}", flush=True)
            last_hybrid_log_ts = now

        _hybrid_log(
            (
                f"start solve: use_right_crop={bool(use_right_crop)}, unpack_last_n={int(max(0, unpack_last_n))}, "
                f"random_iterations={int(max(1, random_iterations))}"
            ),
            force=True,
        )
        crop_w_clamped = max(0.0, min(self.width, float(crop_width)))
        if use_right_crop:
            packing_x_min = self.width - crop_w_clamped
            packing_x_max = self.width
        else:
            packing_x_min = 0.0
            packing_x_max = self.width

        greedy_dx = self.greedy_delta_x if greedy_delta_x is None else float(greedy_delta_x)
        greedy_progress_on = (
            self.greedy_enable_output
            if greedy_enable_output is None
            else bool(greedy_enable_output)
        )
        greedy_log_interval = (
            self.greedy_log_interval_sec
            if greedy_log_interval_sec is None
            else max(0.2, float(greedy_log_interval_sec))
        )
        greedy = GreedySolver(
            self.data,
            height=self.height,
            width=self.width,
            S=self.S,
            delta_x=greedy_dx,
            eps_area=self.greedy_eps_area,
            enable_progress_log=greedy_progress_on,
            log_interval_sec=greedy_log_interval,
            log_prefix="[hybrid-width-greedy]",
            use_result_cache=self.greedy_use_result_cache,
            result_cache_path=self.greedy_result_cache_path,
            result_cache_ttl_days=self.greedy_result_cache_ttl_days,
            shared_result_cache=self.greedy_shared_result_cache,
        )
        greedy_start = time.perf_counter()
        greedy_results = greedy.solve()
        greedy_time = time.perf_counter() - greedy_start
        greedy_cache_hit = bool(getattr(greedy, "last_result_cache_hit", False))
        _hybrid_log(f"greedy finished in {greedy_time:.2f}s", force=True)

        if greedy_results.get("status") != "OPTIMAL":
            return {
                "status": "GREEDY_FAILED",
                "final": greedy_results,
                "visualization": {
                    "greedy_solution": greedy_results,
                    "packed_indices": [],
                    "candidate_indices": [],
                    "fixed_indices": [],
                    "use_right_crop": bool(use_right_crop),
                    "used_crop_width": crop_w_clamped,
                    "crop_x_min": packing_x_min,
                    "packing_x_min": packing_x_min,
                    "packing_x_max": packing_x_max,
                },
                "hybrid_stats": {
                    "greedy_cache_hit": greedy_cache_hit,
                    "greedy_time_sec": greedy_time,
                    "model_time_sec": 0.0,
                    "total_time_sec": time.perf_counter() - t0,
                },
            }

        packed_records = self._collect_packed_records(greedy_results)
        self._sort_packed_records_by_greedy_time(packed_records, greedy_results)
        packed_ids = {rec.item.id for rec in packed_records}
        all_item_ids = {it.id for it in self.data.items}
        greedy_unpacked_ids = all_item_ids - packed_ids
        _hybrid_log(
            f"post-greedy: packed_records={len(packed_records)}, greedy_unpacked_ids={len(greedy_unpacked_ids)}",
            force=True,
        )

        candidate_ids = self._select_candidate_ids(
            packed_records=packed_records,
            unpack_last_n=unpack_last_n,
            max_model_unfixed_items=None,
        )
        unpack_ids: Set[object] = set(candidate_ids)
        _hybrid_log(f"selected candidates for model: {len(unpack_ids)}", force=True)

        model_pool_ids: Set[object] = set(unpack_ids)
        if not lock_greedy_unpacked:
            extra_unpacked_ids: List[object] = []
            seen_extra_ids: Set[object] = set()
            for it in self.data.items:
                item_id = it.id
                if item_id in unpack_ids:
                    continue
                if item_id not in greedy_unpacked_ids:
                    continue
                if item_id in seen_extra_ids:
                    continue
                seen_extra_ids.add(item_id)
                extra_unpacked_ids.append(item_id)

            if max_model_unfixed_items is not None:
                extra_limit = max(0, int(max_model_unfixed_items))
                extra_unpacked_ids = extra_unpacked_ids[:extra_limit]

            model_pool_ids |= set(extra_unpacked_ids)

        greedy_obj = float(greedy_results.get("objective_value") or 0.0)
        greedy_free = self._free_space_percent(greedy_obj)

        target_free_space = None
        if free_space_improvement is not None:
            free_space_improvement = max(0.0, float(free_space_improvement))
            target_free_space = max(0.0, greedy_free - free_space_improvement)

        ordered_pool_ids: List[object] = []
        seen_pool_ids: Set[object] = set()
        for it in self.data.items:
            if it.id in model_pool_ids and it.id not in seen_pool_ids:
                seen_pool_ids.add(it.id)
                ordered_pool_ids.append(it.id)

        if random_sample_size is None:
            sample_size = max(0, int(unpack_last_n)) + 1
        else:
            sample_size = max(0, int(random_sample_size))
        random_iterations = max(1, int(random_iterations))
        rng = random.Random(random_seed) if random_seed is not None else random.Random()
        _hybrid_log(
            (
                f"sampling config: pool_ids will be built, sample_size={sample_size}, "
                f"iterations={random_iterations}, seed={random_seed}"
            ),
            force=True,
        )

        fixed_records_base = [rec for rec in packed_records if rec.item.id not in unpack_ids]
        fixed_item_assignments = {rec.item: (float(rec.x), int(rec.strip)) for rec in fixed_records_base}
        fixed_ids_base = {rec.item.id for rec in fixed_records_base}
        unpacked_greedy_count = len(packed_ids - fixed_ids_base)

        min_total_objective = None
        if target_free_space is not None:
            container_area = self.width * self.height
            min_total_objective = container_area * (1.0 - float(target_free_space) / 100.0)

        best_model_results: dict = {"status": "NOT_SOLVED", "objective_value": None}
        best_model_status = "NOT_SOLVED"
        best_model_obj: Optional[float] = None
        best_model_ok = False
        best_model_status_rank = 0
        best_model_iteration = 1
        best_model_item_ids: Set[object] = set()
        best_forced_unpacked_ids: Set[object] = set(greedy_unpacked_ids | unpack_ids)
        model_time = 0.0

        def _status_rank(status_val: str) -> int:
            if status_val == "OPTIMAL":
                return 2
            if status_val == "FEASIBLE":
                return 1
            return 0

        for iter_idx in range(random_iterations):
            if not ordered_pool_ids:
                model_item_ids: Set[object] = set()
            else:
                k = min(len(ordered_pool_ids), sample_size)
                if k >= len(ordered_pool_ids):
                    sampled_ids = list(ordered_pool_ids)
                else:
                    sampled_ids = rng.sample(ordered_pool_ids, k=k)
                model_item_ids = set(sampled_ids)

            fixed_records = fixed_records_base
            forced_unpacked_ids = (greedy_unpacked_ids | unpack_ids) - model_item_ids
            _hybrid_log(
                (
                    f"iter {iter_idx + 1}/{random_iterations}: model_items={len(model_item_ids)}, "
                    f"fixed_records={len(fixed_records)}, forced_unpacked={len(forced_unpacked_ids)}"
                ),
                force=True,
            )

            iter_start = time.perf_counter()
            _hybrid_log(f"iter {iter_idx + 1}/{random_iterations}: build model", force=True)
            problem = HybridProblem(
                self.data,
                S=self.S,
                R=self.data.R,
                height=self.height,
                width=self.width,
                solver_name=self.solver_name,
                enable_output=model_enable_output,
                fixed_item_assignments=fixed_item_assignments,
                forced_unpacked_ids=forced_unpacked_ids,
                restricted_item_ids=set(model_item_ids) if use_right_crop else set(),
                packing_x_min=packing_x_min if use_right_crop else None,
                packing_x_max=packing_x_max if use_right_crop else None,
                min_objective_value=min_total_objective,
                relative_gap=solver_gap,
                time_limit_sec=model_time_limit_sec,
                num_threads=model_num_threads,
                stop_after_first_solution=stop_after_first_solution,
            )
            _hybrid_log(
                (
                    f"iter {iter_idx + 1}/{random_iterations}: start model.solve() "
                    f"(enable_output={bool(model_enable_output)})"
                ),
                force=True,
            )
            raw_model_results = problem.solve()
            _hybrid_log(f"iter {iter_idx + 1}/{random_iterations}: model.solve() finished", force=True)
            model_results = self._assemble_solution_with_fixed_records(
                fixed_records=fixed_records,
                model_results=raw_model_results,
                model_item_ids=model_item_ids,
            )
            if model_results.get("objective_value") is not None:
                _hybrid_log(
                    f"iter {iter_idx + 1}/{random_iterations}: check overlaps in merged solution",
                    force=True,
                )
            if model_results.get("objective_value") is not None and self._has_solution_overlaps(model_results):
                model_results = {"status": "INFEASIBLE", "objective_value": None}
                _hybrid_log(
                    f"iter {iter_idx + 1}/{random_iterations}: overlap detected, mark as INFEASIBLE",
                    force=True,
                )
            iter_time = time.perf_counter() - iter_start
            model_time += iter_time

            iter_status = str(model_results.get("status", "NOT_SOLVED"))
            iter_obj = model_results.get("objective_value")
            iter_ok = iter_obj is not None and iter_status in {"OPTIMAL", "FEASIBLE"}
            iter_rank = _status_rank(iter_status)
            _hybrid_log(
                (
                    f"iter {iter_idx + 1}/{random_iterations}: "
                    f"status={iter_status}, objective={iter_obj}, iter_time={iter_time:.2f}s"
                ),
                force=True,
            )

            choose_iteration = iter_idx == 0
            if not choose_iteration:
                if iter_ok and not best_model_ok:
                    choose_iteration = True
                elif iter_ok and best_model_ok:
                    best_obj_cmp = float(best_model_obj) if best_model_obj is not None else float("-inf")
                    iter_obj_cmp = float(iter_obj)
                    if iter_obj_cmp > best_obj_cmp + 1e-9:
                        choose_iteration = True
                    elif abs(iter_obj_cmp - best_obj_cmp) <= 1e-9 and iter_rank > best_model_status_rank:
                        choose_iteration = True
                elif not best_model_ok:
                    if best_model_status == "NOT_SOLVED" and iter_status != "NOT_SOLVED":
                        choose_iteration = True
                    elif iter_rank > best_model_status_rank:
                        choose_iteration = True

            if choose_iteration:
                best_model_results = model_results
                best_model_status = iter_status
                best_model_obj = float(iter_obj) if iter_obj is not None else None
                best_model_ok = bool(iter_ok)
                best_model_status_rank = int(iter_rank)
                best_model_iteration = int(iter_idx + 1)
                best_model_item_ids = set(model_item_ids)
                best_forced_unpacked_ids = set(forced_unpacked_ids)
                _hybrid_log(
                    (
                        f"iter {iter_idx + 1}/{random_iterations}: new best "
                        f"(status={best_model_status}, objective={best_model_obj})"
                    ),
                    force=True,
                )

        model_results = best_model_results
        model_item_ids = best_model_item_ids
        fixed_records = fixed_records_base
        forced_unpacked_ids = best_forced_unpacked_ids
        _hybrid_log(
            (
                f"model phase finished: best_iteration={best_model_iteration}, "
                f"best_status={best_model_status}, best_objective={best_model_obj}, model_time={model_time:.2f}s"
            ),
            force=True,
        )
        selected_packed_in_model = sum(1 for rec in packed_records if rec.item.id in model_item_ids)
        selected_unpacked_in_model = len(model_item_ids - packed_ids)

        packed_indices = [rec.idx for rec in packed_records]
        candidate_indices = [rec.idx for rec in packed_records if rec.item.id in unpack_ids]
        fixed_indices = [rec.idx for rec in fixed_records]

        visualization_payload = {
            "greedy_solution": greedy_results,
            "packed_indices": packed_indices,
            "candidate_indices": candidate_indices,
            "fixed_indices": fixed_indices,
            "use_right_crop": bool(use_right_crop),
            "used_crop_width": crop_w_clamped,
            "crop_x_min": packing_x_min,
            "packing_x_min": packing_x_min,
            "packing_x_max": packing_x_max,
        }

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
                    "greedy_cache_hit": greedy_cache_hit,
                    "packed_by_greedy": len(packed_records),
                    "unpack_last_n": int(max(0, unpack_last_n)),
                    "use_right_crop": bool(use_right_crop),
                    "used_crop_width": crop_w_clamped,
                    "candidate_items_for_model": len(candidate_ids),
                    "unpack_ids_count": len(unpack_ids),
                    "actual_unpacked_from_greedy": int(unpacked_greedy_count),
                    "model_pool_ids_count": len(model_pool_ids),
                    "model_item_ids_count": len(model_item_ids),
                    "selected_packed_items_for_model": int(selected_packed_in_model),
                    "selected_unpacked_items_for_model": int(selected_unpacked_in_model),
                    "restricted_items_in_window": len(model_item_ids),
                    "fixed_items_in_model": len(fixed_records),
                    "forced_unpacked_ids": len(forced_unpacked_ids),
                    "random_iterations_requested": int(random_iterations),
                    "random_iterations_executed": int(random_iterations),
                    "random_sample_size_requested": int(sample_size),
                    "random_sample_size": int(min(len(ordered_pool_ids), sample_size)),
                    "best_model_iteration": int(best_model_iteration),
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
                    "greedy_delta_x_used": float(greedy_dx),
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
                "greedy_cache_hit": greedy_cache_hit,
                "packed_by_greedy": len(packed_records),
                "unpack_last_n": int(max(0, unpack_last_n)),
                "use_right_crop": bool(use_right_crop),
                "used_crop_width": crop_w_clamped,
                "candidate_items_for_model": len(candidate_ids),
                "unpack_ids_count": len(unpack_ids),
                "actual_unpacked_from_greedy": int(unpacked_greedy_count),
                "model_pool_ids_count": len(model_pool_ids),
                "model_item_ids_count": len(model_item_ids),
                "selected_packed_items_for_model": int(selected_packed_in_model),
                "selected_unpacked_items_for_model": int(selected_unpacked_in_model),
                "restricted_items_in_window": len(model_item_ids),
                "fixed_items_in_model": len(fixed_records),
                "forced_unpacked_ids": len(forced_unpacked_ids),
                "random_iterations_requested": int(random_iterations),
                "random_iterations_executed": int(random_iterations),
                "random_sample_size_requested": int(sample_size),
                "random_sample_size": int(min(len(ordered_pool_ids), sample_size)),
                "best_model_iteration": int(best_model_iteration),
                "greedy_objective_value": greedy_obj,
                "final_objective_value": final_obj,
                "greedy_free_space_percent": greedy_free,
                "final_free_space_percent": final_free,
                "free_space_improvement_percent": free_space_improvement_abs,
                "target_free_space_percent": target_free_space,
                "greedy_time_sec": greedy_time,
                "model_time_sec": model_time,
                "total_time_sec": time.perf_counter() - t0,
                "greedy_delta_x_used": float(greedy_dx),
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
            if idx < len(s_vals):
                strip_hint = int(round(float(s_vals[idx])))
            else:
                strip_hint = self._strip_from_y(y_val)
            strip_val = self._project_strip_for_model(item, y_val, strip_hint)

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

    def _sort_packed_records_by_greedy_time(self, packed_records: List[_PackedRecord], greedy_results: dict) -> None:
        placement_order = greedy_results.get("placement_order", [])
        if not isinstance(placement_order, list) or not placement_order:
            packed_records.sort(key=lambda rec: (rec.y, rec.x))
            return

        rank_by_idx: Dict[int, int] = {}
        rank = 0
        for raw_idx in placement_order:
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if idx in rank_by_idx:
                continue
            rank_by_idx[idx] = rank
            rank += 1

        if not rank_by_idx:
            packed_records.sort(key=lambda rec: (rec.y, rec.x))
            return

        fallback_rank = len(rank_by_idx) + 1
        packed_records.sort(key=lambda rec: (rank_by_idx.get(int(rec.idx), fallback_rank), int(rec.idx)))

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

    def _project_strip_for_model(self, item: Item, y_val: float, strip_hint: Optional[int] = None) -> int:
        if self.S <= 1:
            return 0

        h = self.height / self.S
        if h <= 0.0:
            return 0

        if strip_hint is None:
            strip_nominal = int(round(float(y_val) / h))
        else:
            strip_nominal = int(strip_hint)

        s_min = int(np.ceil((-float(item.ymin)) / h - 1e-9))
        s_max = int(np.floor((self.height - float(item.ymax)) / h + 1e-9))

        s_min = max(0, min(self.S - 1, s_min))
        s_max = max(0, min(self.S - 1, s_max))
        strip_nominal = max(0, min(self.S - 1, strip_nominal))

        if s_min > s_max:
            return strip_nominal
        return max(s_min, min(s_max, strip_nominal))

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

    def _assemble_solution_with_fixed_records(
        self,
        *,
        fixed_records: List[_PackedRecord],
        model_results: dict,
        model_item_ids: Set[object],
    ) -> dict:
        status = str(model_results.get("status", "NOT_SOLVED"))
        if status not in {"OPTIMAL", "FEASIBLE"}:
            return {"status": status, "objective_value": None}

        n_full = len(self.data.items)
        p = [0.0 for _ in range(n_full)]
        x = [0.0 for _ in range(n_full)]
        y = [0.0 for _ in range(n_full)]
        s = [0.0 for _ in range(n_full)]
        deltas = [[0.0 for _ in range(self.S)] for _ in range(n_full)]

        used_ids: Set[object] = set()
        total_area = 0.0
        h = self.height / self.S if self.S > 0 else self.height

        for rec in fixed_records:
            idx = int(rec.idx)
            if idx < 0 or idx >= n_full:
                continue
            strip_idx = self._strip_from_y(float(rec.y))
            p[idx] = 1.0
            x[idx] = float(rec.x)
            y[idx] = float(rec.y)
            s[idx] = float(strip_idx)
            if self.S > 0:
                deltas[idx][strip_idx] = 1.0
            used_ids.add(rec.item.id)
            total_area += float(rec.item.area)

        raw_p = model_results.get("p", [])
        raw_x = model_results.get("x", [])
        raw_s = model_results.get("s", [])
        raw_deltas = model_results.get("deltas", [])

        for idx, it in enumerate(self.data.items):
            if it.id in used_ids:
                continue
            if model_item_ids and it.id not in model_item_ids:
                continue
            if idx >= len(raw_p) or float(raw_p[idx]) <= 0.5:
                continue

            if idx < len(raw_s):
                strip_idx = int(round(float(raw_s[idx])))
            elif idx < len(raw_deltas):
                row = raw_deltas[idx] if isinstance(raw_deltas[idx], list) else []
                strip_idx = next((j for j, val in enumerate(row) if float(val) > 0.5), 0)
            else:
                strip_idx = 0

            strip_idx = max(0, min(self.S - 1, strip_idx))
            x_val = float(raw_x[idx]) if idx < len(raw_x) else 0.0
            y_val = float(strip_idx) * float(h)

            p[idx] = 1.0
            x[idx] = x_val
            y[idx] = y_val
            s[idx] = float(strip_idx)
            if self.S > 0:
                deltas[idx][strip_idx] = 1.0
            used_ids.add(it.id)
            total_area += float(it.area)

        return {
            "status": status,
            "p": p,
            "x": x,
            "y": y,
            "s": s,
            "deltas": deltas,
            "objective_value": total_area,
        }

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
        s_vals = solution.get("s", [])
        h = self.height / self.S if self.S > 0 else self.height

        placed = []
        for idx, it in enumerate(self.data.items):
            if idx >= len(p_vals) or float(p_vals[idx]) <= 0.5:
                continue
            x_val = float(x_vals[idx]) if idx < len(x_vals) else 0.0
            if idx < len(y_vals):
                y_val = float(y_vals[idx])
            elif idx < len(s_vals):
                y_val = float(s_vals[idx]) * float(h)
            else:
                y_val = 0.0
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
