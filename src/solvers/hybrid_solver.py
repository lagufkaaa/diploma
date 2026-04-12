import time
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Set, Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point, Polygon

from core.data import Data, Item
from solvers.free_space_improvement import resolve_free_space_improvement_requirement
from solvers.greedy_solver import GreedySolver
from solvers.greedy_solver_random import GreedySolverRandom
from solvers.hybrid_hard_timeout import solve_hybrid_problem_with_hard_timeout
from solvers.model_hybrid import Problem as HybridProblem


@dataclass
class _PackedRecord:
    idx: int
    item: Item
    x: float
    y: float
    strip: int
    geom: object


@dataclass
class _GridAnchorSpec:
    strip: int
    global_x: float
    local_shift_x: float
    local_shift_y: float


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
        greedy_eps_area: float = 1e-6,
        greedy_enable_output: bool = True,
        greedy_log_interval_sec: float = 2.0,
        greedy_use_result_cache: bool = True,
        greedy_result_cache_path: Optional[str] = None,
        greedy_result_cache_ttl_days: Optional[float] = None,
        greedy_shared_result_cache: Optional[MutableMapping[str, bytes]] = None,
        greedy_order_strategy: str = "deterministic",
        greedy_random_seed: Optional[int] = None,
        hybrid_enable_output: bool = True,
        hybrid_log_interval_sec: float = 2.0,
    ):
        self.data = data
        self.height = float(height)
        self.width = float(width)
        self.S = int(S)
        self.solver_name = solver_name
        self.greedy_eps_area = float(greedy_eps_area)
        self.greedy_enable_output = bool(greedy_enable_output)
        self.greedy_log_interval_sec = max(0.2, float(greedy_log_interval_sec))
        self.greedy_use_result_cache = bool(greedy_use_result_cache)
        self.greedy_result_cache_path = greedy_result_cache_path
        self.greedy_result_cache_ttl_days = greedy_result_cache_ttl_days
        self.greedy_shared_result_cache = greedy_shared_result_cache
        self.greedy_order_strategy = self._normalize_greedy_order_strategy(
            greedy_order_strategy
        )
        self.greedy_random_seed = (
            None if greedy_random_seed is None else int(greedy_random_seed)
        )
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
        crop_height: float,
        crop_selection_mode: str = "fixed_height",
        crop_zero_tolerance: float = 1e-6,
        crop_lowest_multiplier: float = 1.0,
        use_top_crop: bool = True,
        free_space_improvement: object = False,
        early_stop_free_space_improvement: object = None,
        solver_gap: float = 1.0,
        model_time_limit_sec: Optional[float] = None,
        model_num_threads: Optional[int] = None,
        stop_after_first_solution: bool = True,
        model_enable_output: bool = False,
        min_unpacked_in_sample: int = 0,
        random_iterations: int = 1,
        random_seed: Optional[int] = 0,
        random_sample_size: Optional[int] = None,
        greedy_enable_output: Optional[bool] = None,
        greedy_log_interval_sec: Optional[float] = None,
        greedy_order_strategy: Optional[str] = None,
        greedy_random_seed: Optional[int] = None,
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
            print(f"[hybrid] t={elapsed:7.2f}s {message}", flush=True)
            last_hybrid_log_ts = now

        _hybrid_log(
            (
                f"start solve: use_top_crop={bool(use_top_crop)}, unpack_last_n={int(max(0, unpack_last_n))}, "
                f"random_iterations={int(max(1, random_iterations))}, "
                f"crop_selection_mode={self._normalize_crop_selection_mode(crop_selection_mode)}"
            ),
            force=True,
        )
        base_crop_h_clamped = max(0.0, min(self.height, float(crop_height)))
        crop_selection_mode_effective = self._normalize_crop_selection_mode(
            crop_selection_mode
        )
        crop_zero_tolerance_effective = max(0.0, float(crop_zero_tolerance))
        crop_lowest_multiplier_effective = max(0.0, float(crop_lowest_multiplier))
        used_crop_height = base_crop_h_clamped if use_top_crop else self.height
        crop_decision = "top_crop_disabled" if not use_top_crop else "fixed_height_mode"
        lowest_unpacked_y_for_crop: Optional[float] = None
        lowest_distance_to_top_for_crop: Optional[float] = None
        scaled_lowest_distance_for_crop: Optional[float] = None
        raw_unpacked_candidates_for_crop_count = 0
        if use_top_crop:
            packing_y_min = self.height - used_crop_height
            packing_y_max = self.height
        else:
            packing_y_min = 0.0
            packing_y_max = self.height

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
        greedy_order_strategy_effective = (
            self.greedy_order_strategy
            if greedy_order_strategy is None
            else self._normalize_greedy_order_strategy(greedy_order_strategy)
        )
        greedy_random_seed_requested = (
            self.greedy_random_seed
            if greedy_random_seed is None
            else int(greedy_random_seed)
        )
        if (
            greedy_order_strategy_effective == "random"
            and greedy_random_seed_requested is None
            and random_seed is not None
        ):
            greedy_random_seed_requested = int(random_seed)

        greedy_common_kwargs = dict(
            data=self.data,
            height=self.height,
            width=self.width,
            S=self.S,
            eps_area=self.greedy_eps_area,
            enable_progress_log=greedy_progress_on,
            log_interval_sec=greedy_log_interval,
            use_result_cache=self.greedy_use_result_cache,
            result_cache_path=self.greedy_result_cache_path,
            result_cache_ttl_days=self.greedy_result_cache_ttl_days,
            shared_result_cache=self.greedy_shared_result_cache,
        )
        if greedy_order_strategy_effective == "random":
            greedy = GreedySolverRandom(
                **greedy_common_kwargs,
                log_prefix="[hybrid-greedy-random]",
                random_seed=greedy_random_seed_requested,
            )
        else:
            greedy = GreedySolver(
                **greedy_common_kwargs,
                log_prefix="[hybrid-greedy]",
            )
        greedy_start = time.perf_counter()
        greedy_results = greedy.solve()
        greedy_time = time.perf_counter() - greedy_start
        greedy_cache_hit = bool(getattr(greedy, "last_result_cache_hit", False))
        greedy_random_seed_used = getattr(greedy, "last_random_seed_used", None)
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
                    "use_top_crop": bool(use_top_crop),
                    "used_crop_height": used_crop_height,
                    "crop_y_min": packing_y_min,
                    "packing_y_min": packing_y_min,
                    "packing_y_max": packing_y_max,
                },
                "hybrid_stats": {
                    "greedy_cache_hit": greedy_cache_hit,
                    "greedy_order_strategy": greedy_order_strategy_effective,
                    "greedy_random_seed_requested": greedy_random_seed_requested,
                    "greedy_random_seed_used": greedy_random_seed_used,
                    "sampling_strategy": "random",
                    "crop_selection_mode_requested": str(crop_selection_mode),
                    "crop_selection_mode_effective": crop_selection_mode_effective,
                    "crop_zero_tolerance": float(crop_zero_tolerance_effective),
                    "crop_lowest_multiplier": float(crop_lowest_multiplier_effective),
                    "crop_decision": crop_decision,
                    "lowest_unpacked_y_for_crop": lowest_unpacked_y_for_crop,
                    "lowest_distance_to_top_for_crop": lowest_distance_to_top_for_crop,
                    "scaled_lowest_distance_for_crop": scaled_lowest_distance_for_crop,
                    "raw_unpacked_candidates_for_crop_count": int(
                        raw_unpacked_candidates_for_crop_count
                    ),
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

        if use_top_crop and crop_selection_mode_effective == "unpacked_lowest_y":
            raw_unpacked_candidate_ids_for_crop = self._select_candidate_ids(
                packed_records=packed_records,
                unpack_last_n=unpack_last_n,
            )
            raw_unpacked_candidates_for_crop_count = len(raw_unpacked_candidate_ids_for_crop)
            lowest_unpacked_y_for_crop = self._lowest_packed_point_y_by_item_ids(
                packed_records=packed_records,
                item_ids=raw_unpacked_candidate_ids_for_crop,
            )
            if lowest_unpacked_y_for_crop is None:
                crop_decision = "adaptive_no_candidates_fallback_crop_height"
            elif float(lowest_unpacked_y_for_crop) <= crop_zero_tolerance_effective + 1e-9:
                crop_decision = "adaptive_lowest_near_zero_fallback_crop_height"
            else:
                lowest_distance_to_top_for_crop = max(
                    0.0,
                    self.height - float(lowest_unpacked_y_for_crop),
                )
                scaled_lowest_distance_for_crop = (
                    float(lowest_distance_to_top_for_crop)
                    * float(crop_lowest_multiplier_effective)
                )
                used_crop_height = max(
                    float(base_crop_h_clamped),
                    float(scaled_lowest_distance_for_crop),
                )
                used_crop_height = max(0.0, min(self.height, float(used_crop_height)))
                crop_decision = "adaptive_scaled_distance_vs_crop_height"
            packing_y_min = self.height - used_crop_height
            packing_y_max = self.height
        elif use_top_crop:
            crop_decision = "fixed_height_mode"

        _hybrid_log(
            (
                f"crop selection: mode={crop_selection_mode_effective}, decision={crop_decision}, "
                f"base_crop_height={base_crop_h_clamped:.3f}, used_crop_height={used_crop_height:.3f}, "
                f"lowest_unpacked_y={lowest_unpacked_y_for_crop}, "
                f"distance_to_top={lowest_distance_to_top_for_crop}, "
                f"scaled_distance={scaled_lowest_distance_for_crop}, "
                f"multiplier={crop_lowest_multiplier_effective:.6f}, "
                f"zero_tolerance={crop_zero_tolerance_effective:.6f}, "
                f"raw_candidates={raw_unpacked_candidates_for_crop_count}"
            ),
            force=True,
        )

        packed_crossing_cut_ids: Set[object] = set()
        packed_below_cut_ids: Set[object] = set()
        if use_top_crop:
            packed_crossing_cut_ids = self._collect_packed_ids_crossing_y_line(
                packed_records=packed_records,
                y_line=packing_y_min,
            )
            packed_below_cut_ids = self._collect_packed_ids_below_y_line(
                packed_records=packed_records,
                y_line=packing_y_min,
            )
            _hybrid_log(
                f"packed items crossing cut line (kept fixed): {len(packed_crossing_cut_ids)}",
                force=True,
            )
            _hybrid_log(
                f"packed items below cut line (cannot unpack): {len(packed_below_cut_ids)}",
                force=True,
            )

        candidate_source_records = (
            [
                rec
                for rec in packed_records
                if rec.item.id not in packed_crossing_cut_ids
                and rec.item.id not in packed_below_cut_ids
            ]
            if use_top_crop
            else packed_records
        )
        candidate_ids = self._select_candidate_ids(
            packed_records=candidate_source_records,
            unpack_last_n=unpack_last_n,
        )
        unpack_ids: Set[object] = set(candidate_ids)
        _hybrid_log(f"selected unpack_last_n candidates: {len(unpack_ids)}", force=True)

        model_pool_ids: Set[object] = set(unpack_ids) | set(greedy_unpacked_ids)
        ordered_pool_ids: List[object] = []
        seen_pool_ids: Set[object] = set()
        for it in self.data.items:
            if it.id not in model_pool_ids or it.id in seen_pool_ids:
                continue
            seen_pool_ids.add(it.id)
            ordered_pool_ids.append(it.id)
        _hybrid_log(
            (
                "selection pool (unpack_last_n + greedy_unpacked, no differentiation): "
                f"{len(ordered_pool_ids)}"
            ),
            force=True,
        )

        greedy_obj = float(greedy_results.get("objective_value") or 0.0)
        greedy_free = self._free_space_percent(greedy_obj)

        container_area = self.width * self.height
        improvement_requirement = resolve_free_space_improvement_requirement(
            free_space_improvement,
            greedy_objective=greedy_obj,
            container_area=container_area,
        )
        early_stop_requirement = resolve_free_space_improvement_requirement(
            early_stop_free_space_improvement,
            greedy_objective=greedy_obj,
            container_area=container_area,
        )
        require_improvement = bool(improvement_requirement.require_improvement)
        min_total_objective = improvement_requirement.min_total_objective
        min_improvement_area = improvement_requirement.min_improvement_area
        target_free_space = improvement_requirement.target_free_space_percent
        early_stop_total_objective = early_stop_requirement.min_total_objective
        early_stop_target_free_space = early_stop_requirement.target_free_space_percent
        early_stop_enabled = bool(
            early_stop_requirement.require_improvement
            and early_stop_total_objective is not None
        )

        requested_sample_size = (
            10 if random_sample_size is None else max(0, int(random_sample_size))
        )
        min_unpacked_requested = max(0, int(min_unpacked_in_sample))
        sample_size = int(requested_sample_size)
        random_iterations = max(1, int(random_iterations))
        if random_seed is None:
            random_seed_used = random.SystemRandom().randrange(0, 2**63)
            random_seed_requested = None
        else:
            random_seed_used = int(random_seed)
            random_seed_requested = int(random_seed)
        rng = random.Random(random_seed_used)
        _hybrid_log(
            (
                f"sampling config: pool_ids will be built, sample_size={sample_size}, "
                f"iterations={random_iterations}, seed={random_seed_used}, "
                f"require_improvement={require_improvement}, "
                f"improvement_mode={improvement_requirement.mode}, "
                f"required_improvement_percent={improvement_requirement.required_improvement_percent}, "
                f"early_stop_mode={early_stop_requirement.mode}, "
                f"early_stop_required_improvement_percent={early_stop_requirement.required_improvement_percent}, "
                f"min_unpacked_in_sample={min_unpacked_requested}"
            ),
            force=True,
        )

        fixed_records_base = [rec for rec in packed_records if rec.item.id not in unpack_ids]
        fixed_ids_base = {rec.item.id for rec in fixed_records_base}
        unpacked_greedy_count = len(packed_ids - fixed_ids_base)
        ordered_unpack_ids = [item_id for item_id in ordered_pool_ids if item_id in unpack_ids]

        best_model_results: dict = {"status": "NOT_SOLVED", "objective_value": None}
        best_model_status = "NOT_SOLVED"
        best_model_obj: Optional[float] = None
        best_model_ok = False
        best_model_status_rank = 0
        best_model_iteration = 1
        best_model_item_ids: Set[object] = set()
        best_sampled_item_ids: Set[object] = set()
        best_min_objective_fallback_used = False
        model_time = 0.0
        iteration_stats: List[dict] = []

        def _status_rank(status_val: str) -> int:
            if status_val == "OPTIMAL":
                return 2
            if status_val == "FEASIBLE":
                return 1
            return 0

        for iter_idx in range(random_iterations):
            if not ordered_pool_ids or sample_size <= 0:
                sampled_item_ids: Set[object] = set()
            else:
                k = min(len(ordered_pool_ids), sample_size)
                mandatory_unpacked = min(k, min_unpacked_requested, len(ordered_unpack_ids))
                if mandatory_unpacked > 0:
                    if mandatory_unpacked >= len(ordered_unpack_ids):
                        sampled_unpack_ids = list(ordered_unpack_ids)
                    else:
                        sampled_unpack_ids = rng.sample(ordered_unpack_ids, k=mandatory_unpacked)
                else:
                    sampled_unpack_ids = []
                sampled_unpack_set = set(sampled_unpack_ids)
                remaining_slots = k - len(sampled_unpack_ids)
                remaining_pool = [item_id for item_id in ordered_pool_ids if item_id not in sampled_unpack_set]
                if remaining_slots >= len(remaining_pool):
                    sampled_remaining_ids = list(remaining_pool)
                else:
                    sampled_remaining_ids = rng.sample(remaining_pool, k=remaining_slots)
                sampled_ids = list(sampled_unpack_ids) + list(sampled_remaining_ids)
                sampled_item_ids = set(sampled_ids)

            model_item_ids = set(sampled_item_ids)

            fixed_records = [rec for rec in fixed_records_base if rec.item.id not in model_item_ids]
            forced_by_window = 0
            fixed_blocker_records = (
                [
                    rec
                    for rec in fixed_records
                    if self._record_intersects_y_window(
                        rec,
                        y_low=packing_y_min,
                        y_high=packing_y_max,
                    )
                ]
                if use_top_crop
                else list(fixed_records)
            )
            fixed_grid_anchor_specs = self._build_fixed_grid_anchor_specs(
                fixed_blocker_records=fixed_blocker_records,
            )
            fixed_seed_overrides = self._build_seed_overrides_for_fixed_grid_anchors(
                fixed_blocker_records=fixed_blocker_records,
                fixed_grid_anchor_specs=fixed_grid_anchor_specs,
            )
            fixed_blocker_ids = {rec.item.id for rec in fixed_blocker_records}
            local_data_item_ids = set(model_item_ids) | set(fixed_blocker_ids)

            fixed_area = sum(float(rec.item.area) for rec in fixed_records)
            fixed_blocker_area = sum(float(rec.item.area) for rec in fixed_blocker_records)
            fixed_area_outside_local = max(0.0, float(fixed_area) - float(fixed_blocker_area))
            local_max_objective = self._max_packable_area_by_item_ids(local_data_item_ids)
            local_min_objective = None
            if min_total_objective is not None:
                local_min_candidate = max(
                    0.0,
                    float(min_total_objective) - float(fixed_area_outside_local),
                )
                if local_min_candidate > 1e-9:
                    local_min_objective = float(local_min_candidate)
            local_objective_stop = None
            if early_stop_total_objective is not None:
                local_stop_candidate = max(
                    0.0,
                    float(early_stop_total_objective) - float(fixed_area_outside_local),
                )
                if local_stop_candidate > 1e-9:
                    local_objective_stop = float(local_stop_candidate)
            _hybrid_log(
                (
                    f"iter {iter_idx + 1}/{random_iterations}: model_items={len(model_item_ids)} "
                    f"(sampled={len(sampled_item_ids)}, forced_window={forced_by_window}), "
                    f"fixed_records={len(fixed_records)}, "
                    f"fixed_blockers={len(fixed_blocker_records)}, "
                    f"grid_anchored_fixed={len(fixed_grid_anchor_specs)}, "
                    f"fixed_area={fixed_area:.3f}, local_min_objective={local_min_objective}, "
                    f"local_objective_stop={local_objective_stop}, "
                    f"local_max_objective={local_max_objective:.3f}"
                ),
                force=True,
            )

            iter_label = f"iter {iter_idx + 1}/{random_iterations}"
            iter_start = time.perf_counter()
            local_model_data = self._build_local_model_data(
                local_data_item_ids,
                seed_overrides_by_id=fixed_seed_overrides,
            )
            if local_model_data is None or not local_model_data.items:
                model_results = {"status": "NOT_SOLVED", "objective_value": None}
                _hybrid_log(
                    f"{iter_label}: local model data is empty, skip model iteration",
                    force=True,
                )
                iter_time = time.perf_counter() - iter_start
                model_time += iter_time
                iter_status = str(model_results.get("status", "NOT_SOLVED"))
                iter_obj = model_results.get("objective_value")
                iter_ok = False
                iter_rank = _status_rank(iter_status)
                _hybrid_log(
                    (
                        f"iter {iter_idx + 1}/{random_iterations}: "
                        f"status={iter_status}, objective={iter_obj}, iter_time={iter_time:.2f}s"
                    ),
                    force=True,
                )

                choose_iteration = iter_idx == 0
                if not choose_iteration and not best_model_ok:
                    if best_model_status == "NOT_SOLVED" and iter_status != "NOT_SOLVED":
                        choose_iteration = True
                    elif iter_rank > best_model_status_rank:
                        choose_iteration = True
                if choose_iteration:
                    best_model_results = model_results
                    best_model_status = iter_status
                    best_model_obj = None
                    best_model_ok = bool(iter_ok)
                    best_model_status_rank = int(iter_rank)
                    best_model_iteration = int(iter_idx + 1)
                    best_model_item_ids = set(model_item_ids)
                    best_sampled_item_ids = set(sampled_item_ids)
                    best_min_objective_fallback_used = False
                iteration_stats.append(
                    {
                        "iteration": int(iter_idx + 1),
                        "sampled_model_item_ids_count": int(len(sampled_item_ids)),
                        "model_item_ids_count": int(len(model_item_ids)),
                        "fixed_records_count": int(len(fixed_records)),
                        "fixed_blockers_count": int(len(fixed_blocker_records)),
                        "local_min_objective": (
                            float(local_min_objective) if local_min_objective is not None else None
                        ),
                        "local_objective_stop": (
                            float(local_objective_stop) if local_objective_stop is not None else None
                        ),
                        "local_max_objective": float(local_max_objective),
                        "status": str(iter_status),
                        "objective_value": float(iter_obj) if iter_obj is not None else None,
                        "iter_time_sec": float(iter_time),
                        "selected_as_best": bool(choose_iteration),
                        "model_data_empty": True,
                    }
                )
                continue
            local_fixed_assignments = self._build_local_fixed_assignments(
                model_data=local_model_data,
                fixed_records=fixed_blocker_records,
                fixed_grid_anchor_specs=fixed_grid_anchor_specs,
            )
            # Keep all fixed greedy placements in final merge (including blockers
            # that were also added to local model only for overlap constraints).
            # This preserves their original continuous Y from greedy instead of
            # snapping them to strip*h through local model reconstruction.
            fixed_records_for_merge = list(fixed_records)
            _hybrid_log(
                (
                    f"{iter_label}: local model data prepared "
                    f"(physical_ids={len(model_item_ids)}, fixed_blockers={len(fixed_blocker_ids)}, "
                    f"expanded_items={len(local_model_data.items)})"
                ),
                force=True,
            )

            def _solve_local_problem_once(
                min_objective_value: Optional[float],
                objective_stop_value: Optional[float],
                *,
                run_label: str,
            ) -> dict:
                run_build_t0 = time.perf_counter()
                min_obj_text = (
                    "None"
                    if min_objective_value is None
                    else f"{float(min_objective_value):.6f}"
                )
                objective_stop_text = (
                    "None"
                    if objective_stop_value is None
                    else f"{float(objective_stop_value):.6f}"
                )
                _hybrid_log(
                    (
                        f"{iter_label}: build model (Problem init) start "
                        f"[run={run_label}, min_objective={min_obj_text}, "
                        f"objective_stop={objective_stop_text}]"
                    ),
                    force=True,
                )
                base_problem_kwargs = dict(
                    data=local_model_data,
                    S=self.S,
                    R=local_model_data.R,
                    height=self.height,
                    width=self.width,
                    solver_name=self.solver_name,
                    enable_output=model_enable_output,
                    fixed_item_assignments=local_fixed_assignments,
                    no_lower_y_bound_item_ids=set(fixed_grid_anchor_specs.keys()),
                    restricted_item_ids=set(model_item_ids) if use_top_crop else set(),
                    packing_y_min=packing_y_min if use_top_crop else None,
                    packing_y_max=packing_y_max if use_top_crop else None,
                    min_objective_value=min_objective_value,
                    objective_stop_value=objective_stop_value,
                    relative_gap=solver_gap,
                    time_limit_sec=model_time_limit_sec,
                    num_threads=model_num_threads,
                    stop_after_first_solution=stop_after_first_solution,
                    progress_label="[model]",
                )
                if model_time_limit_sec is not None:
                    run_results, timed_out_hard, hard_error = solve_hybrid_problem_with_hard_timeout(
                        problem_kwargs=base_problem_kwargs,
                        timeout_sec=model_time_limit_sec,
                    )
                    if timed_out_hard:
                        _hybrid_log(
                            f"{iter_label}: hard timeout reached ({float(model_time_limit_sec):.2f}s), process terminated",
                            force=True,
                        )
                    if hard_error:
                        _hybrid_log(
                            f"{iter_label}: hard-timeout helper note: {hard_error}",
                            force=True,
                        )
                else:
                    problem = HybridProblem(
                        **base_problem_kwargs,
                        progress_callback=lambda msg, label=iter_label: _hybrid_log(
                            f"{label}: {msg}",
                            force=True,
                        ),
                    )
                    _hybrid_log(
                        (
                            f"{iter_label}: build model (Problem init) finished in "
                            f"{time.perf_counter() - run_build_t0:.2f}s [run={run_label}]"
                        ),
                        force=True,
                    )
                    _hybrid_log(
                        (
                            f"{iter_label}: start model.solve() "
                            f"(enable_output={bool(model_enable_output)}, run={run_label})"
                        ),
                        force=True,
                    )
                    run_results = problem.solve()
                run_status = str(run_results.get("status", "NOT_SOLVED"))
                _hybrid_log(
                    f"{iter_label}: model.solve() finished [run={run_label}, status={run_status}]",
                    force=True,
                )
                return run_results

            raw_model_results = _solve_local_problem_once(
                local_min_objective,
                local_objective_stop,
                run_label="primary",
            )
            model_results = self._assemble_full_solution(
                fixed_records=fixed_records_for_merge,
                model_data=local_model_data,
                local_results=raw_model_results,
                y_offset=0.0,
                local_height=self.height,
            )
            if model_results.get("objective_value") is not None:
                _hybrid_log(
                    f"iter {iter_idx + 1}/{random_iterations}: check overlaps in merged solution",
                    force=True,
                )
            if model_results.get("objective_value") is not None and self._has_solution_overlaps(
                model_results,
                active_item_ids=set(model_item_ids),
            ):
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
                best_sampled_item_ids = set(sampled_item_ids)
                best_min_objective_fallback_used = False
                _hybrid_log(
                    (
                        f"iter {iter_idx + 1}/{random_iterations}: new best "
                        f"(status={best_model_status}, objective={best_model_obj})"
                    ),
                    force=True,
                )
            iteration_stats.append(
                {
                    "iteration": int(iter_idx + 1),
                    "sampled_model_item_ids_count": int(len(sampled_item_ids)),
                    "model_item_ids_count": int(len(model_item_ids)),
                    "fixed_records_count": int(len(fixed_records)),
                    "fixed_blockers_count": int(len(fixed_blocker_records)),
                    "local_min_objective": (
                        float(local_min_objective) if local_min_objective is not None else None
                    ),
                    "local_objective_stop": (
                        float(local_objective_stop) if local_objective_stop is not None else None
                    ),
                    "local_max_objective": float(local_max_objective),
                    "status": str(iter_status),
                    "objective_value": float(iter_obj) if iter_obj is not None else None,
                    "iter_time_sec": float(iter_time),
                    "selected_as_best": bool(choose_iteration),
                    "model_data_empty": False,
                }
            )

        model_results = best_model_results
        model_item_ids = best_model_item_ids
        sampled_item_ids = best_sampled_item_ids
        fixed_records = [rec for rec in fixed_records_base if rec.item.id not in model_item_ids]
        forced_by_window = 0
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
        # For panel-2 visualization keep the deterministic "after unpack_last_n" layout:
        # greedy packed items minus the last-N unpacked candidates.
        fixed_indices = [rec.idx for rec in fixed_records_base]

        visualization_payload = {
            "greedy_solution": greedy_results,
            "packed_indices": packed_indices,
            "candidate_indices": candidate_indices,
            "fixed_indices": fixed_indices,
            "use_top_crop": bool(use_top_crop),
            "used_crop_height": used_crop_height,
            "crop_y_min": packing_y_min,
            "packing_y_min": packing_y_min,
            "packing_y_max": packing_y_max,
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
        can_claim_no_improvement = bool(full_search_mode and proven_global_optimal)

        if not improved:
            fail_status = "NOT_IMPROVED" if can_claim_no_improvement else "NOT_PROVEN"
            use_model_as_final = fail_status == "NOT_PROVEN" and model_ok and (float(model_obj) > greedy_obj + 1e-9)

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
                    "greedy_order_strategy": greedy_order_strategy_effective,
                    "greedy_random_seed_requested": greedy_random_seed_requested,
                    "greedy_random_seed_used": greedy_random_seed_used,
                    "sampling_strategy": "random",
                    "packed_by_greedy": len(packed_records),
                    "unpack_last_n": int(max(0, unpack_last_n)),
                    "use_top_crop": bool(use_top_crop),
                    "used_crop_height": used_crop_height,
                    "crop_selection_mode_requested": str(crop_selection_mode),
                    "crop_selection_mode_effective": crop_selection_mode_effective,
                    "crop_zero_tolerance": float(crop_zero_tolerance_effective),
                    "crop_lowest_multiplier": float(crop_lowest_multiplier_effective),
                    "crop_decision": crop_decision,
                    "lowest_unpacked_y_for_crop": lowest_unpacked_y_for_crop,
                    "lowest_distance_to_top_for_crop": lowest_distance_to_top_for_crop,
                    "scaled_lowest_distance_for_crop": scaled_lowest_distance_for_crop,
                    "raw_unpacked_candidates_for_crop_count": int(
                        raw_unpacked_candidates_for_crop_count
                    ),
                    "candidate_items_for_model": len(model_pool_ids),
                    "unpack_ids_count": len(unpack_ids),
                    "actual_unpacked_from_greedy": int(unpacked_greedy_count),
                    "model_pool_ids_count": len(model_pool_ids),
                    "model_item_ids_count": len(model_item_ids),
                    "sampled_model_item_ids_count": len(sampled_item_ids),
                    "window_forced_model_items_count": int(forced_by_window),
                    "selected_packed_items_for_model": int(selected_packed_in_model),
                    "selected_unpacked_items_for_model": int(selected_unpacked_in_model),
                    "restricted_items_in_window": len(model_item_ids),
                    "fixed_items_in_model": len(fixed_records),
                    "random_iterations_requested": int(random_iterations),
                    "random_iterations_executed": int(random_iterations),
                    "min_unpacked_in_sample": int(min_unpacked_requested),
                    "random_sample_size_requested": int(requested_sample_size),
                    "random_sample_size": int(min(len(ordered_pool_ids), sample_size)),
                    "random_seed_requested": random_seed_requested,
                    "random_seed_used": int(random_seed_used),
                    "best_model_iteration": int(best_model_iteration),
                    "greedy_objective_value": greedy_obj,
                    "model_objective_value": float(model_obj) if model_obj is not None else None,
                    "final_objective_value": final_objective,
                    "greedy_free_space_percent": greedy_free,
                    "model_free_space_percent": model_free,
                    "final_free_space_percent": final_free_space,
                    "free_space_improvement_percent": final_improvement,
                    "target_free_space_percent": target_free_space,
                    "require_improvement": bool(require_improvement),
                    "free_space_improvement_mode": improvement_requirement.mode,
                    "required_free_space_improvement_percent": (
                        improvement_requirement.required_improvement_percent
                    ),
                    "early_stop_enabled": bool(early_stop_enabled),
                    "early_stop_improvement_mode": early_stop_requirement.mode,
                    "early_stop_required_improvement_percent": (
                        early_stop_requirement.required_improvement_percent
                    ),
                    "early_stop_target_objective_value": (
                        float(early_stop_total_objective)
                        if early_stop_total_objective is not None
                        else None
                    ),
                    "early_stop_target_free_space_percent": early_stop_target_free_space,
                    "min_improvement_area": float(min_improvement_area),
                    "greedy_time_sec": greedy_time,
                    "model_time_sec": model_time,
                    "total_time_sec": time.perf_counter() - t0,
                    "model_status": model_status,
                    "min_objective_fallback_used": bool(best_min_objective_fallback_used),
                    "full_search_mode": full_search_mode,
                    "proven_global_optimal": proven_global_optimal,
                    "can_claim_no_improvement": can_claim_no_improvement,
                    "iteration_stats": list(iteration_stats),
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
                "greedy_order_strategy": greedy_order_strategy_effective,
                "greedy_random_seed_requested": greedy_random_seed_requested,
                "greedy_random_seed_used": greedy_random_seed_used,
                "sampling_strategy": "random",
                "packed_by_greedy": len(packed_records),
                "unpack_last_n": int(max(0, unpack_last_n)),
                "use_top_crop": bool(use_top_crop),
                "used_crop_height": used_crop_height,
                "crop_selection_mode_requested": str(crop_selection_mode),
                "crop_selection_mode_effective": crop_selection_mode_effective,
                "crop_zero_tolerance": float(crop_zero_tolerance_effective),
                "crop_lowest_multiplier": float(crop_lowest_multiplier_effective),
                "crop_decision": crop_decision,
                "lowest_unpacked_y_for_crop": lowest_unpacked_y_for_crop,
                "lowest_distance_to_top_for_crop": lowest_distance_to_top_for_crop,
                "scaled_lowest_distance_for_crop": scaled_lowest_distance_for_crop,
                "raw_unpacked_candidates_for_crop_count": int(
                    raw_unpacked_candidates_for_crop_count
                ),
                "candidate_items_for_model": len(model_pool_ids),
                "unpack_ids_count": len(unpack_ids),
                "actual_unpacked_from_greedy": int(unpacked_greedy_count),
                "model_pool_ids_count": len(model_pool_ids),
                "model_item_ids_count": len(model_item_ids),
                "sampled_model_item_ids_count": len(sampled_item_ids),
                "window_forced_model_items_count": int(forced_by_window),
                "selected_packed_items_for_model": int(selected_packed_in_model),
                "selected_unpacked_items_for_model": int(selected_unpacked_in_model),
                "restricted_items_in_window": len(model_item_ids),
                "fixed_items_in_model": len(fixed_records),
                "random_iterations_requested": int(random_iterations),
                "random_iterations_executed": int(random_iterations),
                "min_unpacked_in_sample": int(min_unpacked_requested),
                "random_sample_size_requested": int(requested_sample_size),
                "random_sample_size": int(min(len(ordered_pool_ids), sample_size)),
                "random_seed_requested": random_seed_requested,
                "random_seed_used": int(random_seed_used),
                "best_model_iteration": int(best_model_iteration),
                "greedy_objective_value": greedy_obj,
                "final_objective_value": final_obj,
                "greedy_free_space_percent": greedy_free,
                "final_free_space_percent": final_free,
                "free_space_improvement_percent": free_space_improvement_abs,
                "target_free_space_percent": target_free_space,
                "require_improvement": bool(require_improvement),
                "free_space_improvement_mode": improvement_requirement.mode,
                "required_free_space_improvement_percent": (
                    improvement_requirement.required_improvement_percent
                ),
                "early_stop_enabled": bool(early_stop_enabled),
                "early_stop_improvement_mode": early_stop_requirement.mode,
                "early_stop_required_improvement_percent": (
                    early_stop_requirement.required_improvement_percent
                ),
                "early_stop_target_objective_value": (
                    float(early_stop_total_objective)
                    if early_stop_total_objective is not None
                    else None
                ),
                "early_stop_target_free_space_percent": early_stop_target_free_space,
                "min_improvement_area": float(min_improvement_area),
                "greedy_time_sec": greedy_time,
                "model_time_sec": model_time,
                "total_time_sec": time.perf_counter() - t0,
                "model_status": model_status,
                "min_objective_fallback_used": bool(best_min_objective_fallback_used),
                "full_search_mode": full_search_mode,
                "proven_global_optimal": proven_global_optimal,
                "iteration_stats": list(iteration_stats),
            },
        }

    @staticmethod
    def _normalize_greedy_order_strategy(strategy: Optional[str]) -> str:
        value = "deterministic" if strategy is None else str(strategy).strip().lower()
        if value in {"deterministic", "default", "area_desc", "sorted"}:
            return "deterministic"
        if value in {"random", "shuffle", "random_order"}:
            return "random"
        raise ValueError(
            f"Unsupported greedy_order_strategy='{strategy}'. "
            "Allowed: deterministic, random."
        )

    @staticmethod
    def _normalize_crop_selection_mode(mode: Optional[str]) -> str:
        value = "fixed_height" if mode is None else str(mode).strip().lower()
        if value in {"fixed_height", "fixed", "default", "crop_height"}:
            return "fixed_height"
        if value in {
            "unpacked_lowest_y",
            "unpacked_lowest",
            "lowest_unpacked_y",
            "adaptive_lowest",
        }:
            return "unpacked_lowest_y"
        raise ValueError(
            f"Unsupported crop_selection_mode='{mode}'. "
            "Allowed: fixed_height, unpacked_lowest_y."
        )

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

        return set(candidate_ids)

    def _lowest_packed_point_y_by_item_ids(
        self,
        *,
        packed_records: List[_PackedRecord],
        item_ids: Set[object],
    ) -> Optional[float]:
        if not item_ids:
            return None
        min_y: Optional[float] = None
        for rec in packed_records:
            if rec.item.id not in item_ids:
                continue
            item_low = float(rec.y) + float(rec.item.ymin)
            if min_y is None or item_low < min_y:
                min_y = item_low
        return None if min_y is None else float(min_y)

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

    def _max_packable_area_by_item_ids(self, item_ids: Set[object]) -> float:
        if not item_ids:
            return 0.0
        best_area_by_id: Dict[object, float] = {}
        for it in self.data.items:
            item_id = it.id
            if item_id not in item_ids:
                continue
            area = float(it.area)
            prev = best_area_by_id.get(item_id)
            if prev is None or area > prev:
                best_area_by_id[item_id] = area
        return float(sum(best_area_by_id.values()))

    def _collect_packed_ids_in_y_window(
        self,
        *,
        packed_records: List[_PackedRecord],
        y_min: float,
        y_max: float,
    ) -> Set[object]:
        ids: Set[object] = set()
        y_low = float(min(y_min, y_max))
        y_high = float(max(y_min, y_max))
        for rec in packed_records:
            if self._record_intersects_y_window(rec, y_low=y_low, y_high=y_high):
                ids.add(rec.item.id)
        return ids

    @staticmethod
    def _record_intersects_y_window(rec: _PackedRecord, *, y_low: float, y_high: float) -> bool:
        item_low = float(rec.y) + float(rec.item.ymin)
        item_high = float(rec.y) + float(rec.item.ymax)
        eps = 1e-9
        return (item_high > y_low + eps) and (item_low < y_high - eps)

    def _collect_packed_ids_crossing_y_line(
        self,
        *,
        packed_records: List[_PackedRecord],
        y_line: float,
    ) -> Set[object]:
        ids: Set[object] = set()
        y_cut = float(y_line)
        for rec in packed_records:
            if self._record_crosses_y_line(rec, y_line=y_cut):
                ids.add(rec.item.id)
        return ids

    def _collect_packed_ids_below_y_line(
        self,
        *,
        packed_records: List[_PackedRecord],
        y_line: float,
    ) -> Set[object]:
        ids: Set[object] = set()
        y_cut = float(y_line)
        for rec in packed_records:
            if self._record_below_y_line(rec, y_line=y_cut):
                ids.add(rec.item.id)
        return ids

    @staticmethod
    def _record_crosses_y_line(rec: _PackedRecord, *, y_line: float) -> bool:
        item_low = float(rec.y) + float(rec.item.ymin)
        item_high = float(rec.y) + float(rec.item.ymax)
        eps = 1e-9
        return (item_low < y_line - eps) and (item_high > y_line + eps)

    @staticmethod
    def _record_below_y_line(rec: _PackedRecord, *, y_line: float) -> bool:
        item_high = float(rec.y) + float(rec.item.ymax)
        eps = 1e-9
        return item_high <= y_line + eps

    def _build_fixed_grid_anchor_specs(
        self,
        *,
        fixed_blocker_records: List[_PackedRecord],
    ) -> Dict[object, _GridAnchorSpec]:
        specs: Dict[object, _GridAnchorSpec] = {}
        if self.S <= 1:
            return specs
        for rec in fixed_blocker_records:
            item_id = rec.item.id
            if item_id in specs:
                continue
            spec = self._pick_grid_anchor_spec_for_record(rec)
            if spec is None:
                continue
            specs[item_id] = spec
        return specs

    def _build_seed_overrides_for_fixed_grid_anchors(
        self,
        *,
        fixed_blocker_records: List[_PackedRecord],
        fixed_grid_anchor_specs: Dict[object, _GridAnchorSpec],
    ) -> Dict[object, Item]:
        overrides: Dict[object, Item] = {}
        if not fixed_grid_anchor_specs:
            return overrides

        for rec in fixed_blocker_records:
            item_id = rec.item.id
            spec = fixed_grid_anchor_specs.get(item_id)
            if spec is None or item_id in overrides:
                continue

            src_pts = np.asarray(rec.item.points, dtype=float)
            shift_vec = np.array([float(spec.local_shift_x), float(spec.local_shift_y)], dtype=float)
            shifted_pts = src_pts - shift_vec

            # Preserve exact shifted coordinates: Item() normalizes by first point,
            # so we explicitly restore the intended point cloud afterward.
            seed = self._make_item_with_exact_points(shifted_pts)
            seed.id = item_id
            seed.rotation = float(rec.item.rotation) % 360.0
            overrides[item_id] = seed

        return overrides

    def _pick_grid_anchor_spec_for_record(self, rec: _PackedRecord) -> Optional[_GridAnchorSpec]:
        if self.S <= 1:
            return None
        h = float(self.height) / float(self.S)
        if h <= 0.0:
            return None

        for strip_idx in range(1, self.S):
            y_line = float(strip_idx) * h
            point = self._pick_intersection_point_with_y_line(rec.geom, y_line)
            if point is None:
                continue
            px, py = point
            return _GridAnchorSpec(
                strip=int(strip_idx),
                global_x=float(px),
                local_shift_x=float(px) - float(rec.x),
                local_shift_y=float(py) - float(rec.y),
            )
        return None

    def _pick_intersection_point_with_y_line(
        self,
        geom,
        y_line: float,
    ) -> Optional[Tuple[float, float]]:
        line = LineString([(0.0, float(y_line)), (float(self.width), float(y_line))])
        inter = geom.intersection(line)
        if inter.is_empty:
            return None

        candidates: List[Tuple[float, float]] = []

        def _collect_points(g) -> None:
            if g is None or g.is_empty:
                return
            if isinstance(g, Point):
                candidates.append((float(g.x), float(g.y)))
                return
            if isinstance(g, MultiPoint):
                for sub in g.geoms:
                    _collect_points(sub)
                return
            if isinstance(g, LineString):
                xs = [float(pt[0]) for pt in g.coords]
                if not xs:
                    return
                candidates.append((min(xs), float(y_line)))
                return
            if isinstance(g, MultiLineString):
                for sub in g.geoms:
                    _collect_points(sub)
                return
            if isinstance(g, GeometryCollection):
                for sub in g.geoms:
                    _collect_points(sub)
                return

        _collect_points(inter)
        if not candidates:
            return None

        candidates.sort(key=lambda p: (p[0], p[1]))
        px, py = candidates[0]
        return float(px), float(py)

    def _build_local_model_data(
        self,
        model_item_ids: Set[object],
        *,
        seed_overrides_by_id: Optional[Dict[object, Item]] = None,
    ) -> Optional[Data]:
        if not model_item_ids:
            return None

        seed_overrides = seed_overrides_by_id or {}
        source_by_id: Dict[object, Item] = {}
        for it in self.data.items:
            if it.id not in model_item_ids:
                continue
            if it.id not in source_by_id:
                source_by_id[it.id] = it
            if abs(float(it.rotation) % 360.0) <= 1e-6:
                source_by_id[it.id] = it

        ordered_ids: List[object] = []
        seen_ids: Set[object] = set()
        for it in self.data.items:
            if it.id not in model_item_ids or it.id in seen_ids:
                continue
            seen_ids.add(it.id)
            ordered_ids.append(it.id)

        seeds: List[Item] = []
        for item_id in ordered_ids:
            src = seed_overrides.get(item_id)
            src_from_override = src is not None
            if src is None:
                src = source_by_id.get(item_id)
            if src is None:
                continue

            src_points = np.asarray(src.points, dtype=float).copy()
            if src_from_override:
                clone = self._make_item_with_exact_points(src_points)
            else:
                clone = Item(src_points)
            clone.id = src.id
            clone.rotation = float(getattr(src, "rotation", 0.0)) % 360.0
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

    @staticmethod
    def _make_item_with_exact_points(points: np.ndarray) -> Item:
        pts = np.asarray(points, dtype=float).copy()
        item = Item(pts.copy())

        item.points = pts
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        item.polygon = poly
        item.area = float(poly.area)

        if pts.size == 0:
            item.xmin = 0.0
            item.xmax = 0.0
            item.ymin = 0.0
            item.ymax = 0.0
            return item

        item.xmin = float(np.min(pts[:, 0]))
        item.xmax = float(np.max(pts[:, 0]))
        item.ymin = float(np.min(pts[:, 1]))
        item.ymax = float(np.max(pts[:, 1]))
        return item

    def _build_local_fixed_assignments(
        self,
        *,
        model_data: Optional[Data],
        fixed_records: List[_PackedRecord],
        fixed_grid_anchor_specs: Optional[Dict[object, _GridAnchorSpec]] = None,
    ) -> Dict[Item, Tuple[float, int]]:
        assignments: Dict[Item, Tuple[float, int]] = {}
        if model_data is None or not model_data.items or not fixed_records:
            return assignments

        grid_specs = fixed_grid_anchor_specs or {}
        by_id_rot: Dict[Tuple[object, int], List[Item]] = defaultdict(list)
        by_id: Dict[object, List[Item]] = defaultdict(list)
        for it in model_data.items:
            rot_key = int(round(float(it.rotation))) % 360
            by_id_rot[(it.id, rot_key)].append(it)
            by_id[it.id].append(it)

        assigned_ids: Set[object] = set()
        for rec in fixed_records:
            item_id = rec.item.id
            if item_id in assigned_ids:
                continue

            rot_key = int(round(float(rec.item.rotation))) % 360
            candidates = by_id_rot.get((item_id, rot_key), [])
            local_item = candidates[0] if candidates else None
            if local_item is None:
                fallback = by_id.get(item_id, [])
                local_item = fallback[0] if fallback else None
            if local_item is None:
                continue

            spec = grid_specs.get(item_id)
            if spec is not None:
                x_value = float(spec.global_x)
                strip_idx = int(spec.strip)
            else:
                x_value = float(rec.x)
                strip_idx = int(rec.strip)

            strip_idx = max(0, min(self.S - 1, strip_idx))
            assignments[local_item] = (x_value, int(strip_idx))
            assigned_ids.add(item_id)

        return assignments

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
        used_item_ids: Set[object] = set()
        total_area = 0.0

        for rec in fixed_records:
            idx = rec.idx
            used_indices.add(idx)
            used_item_ids.add(rec.item.id)
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
                if lit.id in used_item_ids:
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
                used_item_ids.add(lit.id)
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

    def _has_solution_overlaps(
        self,
        solution: dict,
        *,
        active_item_ids: Optional[Set[object]] = None,
    ) -> bool:
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
            placed.append((it.id, self._placed_geometry(it, x_val, y_val)))

        eps_area = max(1e-9, float(self.greedy_eps_area))
        for i in range(len(placed)):
            id_a, ga = placed[i]
            for j in range(i + 1, len(placed)):
                id_b, gb = placed[j]
                if active_item_ids and (id_a not in active_item_ids) and (id_b not in active_item_ids):
                    continue
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
