from collections import defaultdict
import math
import time
from typing import Callable, Dict, Optional, Set, Tuple

from ortools.linear_solver import pywraplp

from core.data import Data, Item
from core.encoding import Encoding


class Problem:
    def __init__(
        self,
        data: Data,
        S: int,
        R: int,
        height: float,
        width: float,
        solver_name: str = "SCIP",
        *,
        enable_output: bool = True,
        fixed_item_assignments: Optional[Dict[Item, Tuple[float, int]]] = None,
        no_lower_y_bound_item_ids: Optional[Set[object]] = None,
        forced_unpacked_ids: Optional[Set[object]] = None,
        restricted_item_ids: Optional[Set[object]] = None,
        packing_x_min: Optional[float] = None,
        packing_x_max: Optional[float] = None,
        packing_y_min: Optional[float] = None,
        packing_y_max: Optional[float] = None,
        packing_height_limit: Optional[float] = None,
        min_objective_value: Optional[float] = None,
        max_free_space_percent: Optional[float] = None,
        relative_gap: Optional[float] = None,
        time_limit_sec: Optional[float] = None,
        stop_after_first_solution: bool = False,
        num_threads: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        progress_label: str = "[model]",
    ):
        self._progress_callback = progress_callback
        self._progress_label = str(progress_label)
        init_t0 = time.perf_counter()
        self._progress("init: start")

        self.data = data
        self.solver_name = solver_name
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        if self.solver is None:
            raise RuntimeError(f"Failed to create solver '{solver_name}'")

        self.enable_output = bool(enable_output)
        if self.enable_output:
            self.solver.EnableOutput()

        self.S = int(S)
        self.R = int(R)
        self.height = float(height)
        self.width = float(width)
        self.h = self.height / self.S

        self.fixed_item_assignments = fixed_item_assignments or {}
        self.no_lower_y_bound_item_ids = set(no_lower_y_bound_item_ids or set())
        self.forced_unpacked_ids = set(forced_unpacked_ids or set())
        self.restricted_item_ids = set(restricted_item_ids or set())

        x_min = None if packing_x_min is None else float(packing_x_min)
        x_max = None if packing_x_max is None else float(packing_x_max)
        if x_min is not None:
            x_min = max(0.0, min(x_min, self.width))
        if x_max is not None:
            x_max = max(0.0, min(x_max, self.width))
        if x_min is not None and x_max is not None and x_min > x_max:
            x_min = x_max

        y_min = None if packing_y_min is None else float(packing_y_min)
        y_max = None if packing_y_max is None else float(packing_y_max)
        if y_min is not None:
            y_min = max(0.0, min(y_min, self.height))
        if y_max is not None:
            y_max = max(0.0, min(y_max, self.height))

        # Backward compatibility for the previous "pack only below limit" mode.
        if y_min is None and y_max is None and packing_height_limit is not None:
            y_min = 0.0
            y_max = max(0.0, min(float(packing_height_limit), self.height))

        if y_min is not None and y_max is not None and y_min > y_max:
            y_min = y_max

        self.packing_x_min = x_min
        self.packing_x_max = x_max
        self.packing_y_min = y_min
        self.packing_y_max = y_max
        self.min_objective_value = min_objective_value
        self.max_free_space_percent = max_free_space_percent
        self.relative_gap = relative_gap
        self.time_limit_sec = time_limit_sec
        self.stop_after_first_solution = bool(stop_after_first_solution)
        self.num_threads = num_threads

        enc_t0 = time.perf_counter()
        self._progress("init: build encoding")
        self.encoding = Encoding(data, self.S, self.height)
        self._progress(
            f"init: encoding ready in {time.perf_counter() - enc_t0:.2f}s "
            f"(enc_keys={len(self.encoding.enc)})"
        )

        self.a: Dict[Tuple[Item, Item, int], list[float]] = {}
        self.b: Dict[Tuple[Item, Item, int], list[float]] = {}
        self.C: Dict[Tuple[Item, Item, int], int] = {}
        abc_t0 = time.perf_counter()
        self._progress("init: build a/b/C coefficients")
        self.build_abc()
        self._progress(
            f"init: a/b/C ready in {time.perf_counter() - abc_t0:.2f}s "
            f"(keys={len(self.C)})"
        )

        self.big_M, big_m_meta = self._compute_auto_big_m()
        min_a_text = "n/a" if big_m_meta["min_a"] is None else f"{float(big_m_meta['min_a']):.3f}"
        max_b_text = "n/a" if big_m_meta["max_b"] is None else f"{float(big_m_meta['max_b']):.3f}"
        self._progress(
            (
                f"init: big_M auto={self.big_M:.3f} "
                f"(boundary_req={float(big_m_meta['boundary_req']):.3f}, "
                f"non_overlap_req={float(big_m_meta['non_overlap_req']):.3f}, "
                f"min_a={min_a_text}, max_b={max_b_text})"
            )
        )

        self.x: Dict[Item, pywraplp.Variable] = {}
        self.p: Dict[Item, pywraplp.Variable] = {}
        self.str_var: Dict[Item, pywraplp.Variable] = {}
        self.deltas: Dict[Tuple[Item, int], pywraplp.Variable] = {}

        self.gammas: Dict[Tuple[Item, Item, int, int], pywraplp.Variable] = {}
        self._configure_solver()
        self._progress(f"init: done in {time.perf_counter() - init_t0:.2f}s")

    # ---------- preprocessing ----------
    def build_abc(self):
        """
        Build a/b/C from Encoding.enc.
        Encoding.enc expected: enc[(i, j, k)] = list of segments [[[x1,y],[x2,y]], ...]
        """
        for key, segs in self.encoding.enc.items():
            if not (isinstance(key, tuple) and len(key) == 3):
                raise ValueError(f"Encoding.enc key must be (i,j,k), got: {key}")

            temp_a = []
            temp_b = []
            for seg in segs:
                x1 = float(min(seg[0][0], seg[1][0]))
                x2 = float(max(seg[0][0], seg[1][0]))
                temp_a.append(x1)
                temp_b.append(x2)

            self.a[key] = temp_a
            self.b[key] = temp_b
            self.C[key] = len(temp_a)

    # ---------- solve ----------
    def solve(self):
        solve_t0 = time.perf_counter()

        t0 = time.perf_counter()
        self._progress("solve: build variables")
        self._build_variables()
        self._progress(
            (
                f"solve: build variables done in {time.perf_counter() - t0:.2f}s "
                f"(p={len(self.p)}, x={len(self.x)}, s={len(self.str_var)}, "
                f"deltas={len(self.deltas)}, gammas={len(self.gammas)})"
            )
        )

        t0 = time.perf_counter()
        self._progress("solve: set objective")
        self._set_objective()
        self._progress(f"solve: set objective done in {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self._progress("solve: add strip linking constraints")
        self._add_strip_linking_constraints()
        self._progress(f"solve: strip linking done in {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self._progress("solve: add boundary constraints")
        self._add_boundary_constraints()
        self._progress(f"solve: boundary constraints done in {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self._progress("solve: add single-use constraints")
        self._add_single_use_constraints()
        self._progress(f"solve: single-use constraints done in {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self._progress("solve: add fixed-item constraints")
        self._add_fixed_item_constraints()
        self._progress(f"solve: fixed-item constraints done in {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self._progress("solve: add minimum-quality constraint")
        self._add_minimum_quality_constraint()
        self._progress(
            f"solve: minimum-quality constraint done in {time.perf_counter() - t0:.2f}s"
        )

        t0 = time.perf_counter()
        self._progress("solve: add non-overlap constraints")
        self._add_non_overlap_constraints()
        self._progress(f"solve: non-overlap constraints done in {time.perf_counter() - t0:.2f}s")

        solve_call_t0 = time.perf_counter()
        self._progress("solve: call solver")
        status = self.solver.Solve()
        solve_call_dt = time.perf_counter() - solve_call_t0
        status_text = self._status_name(status)

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            results = {
                "p": [self.p[it].solution_value() for it in self.data.items],
                "x": [self.x[it].solution_value() for it in self.data.items],
                "s": [self.str_var[it].solution_value() for it in self.data.items],
                "deltas": [
                    [self.deltas[(it, s)].solution_value() for s in range(self.S)]
                    for it in self.data.items
                ],
                "objective_value": self.solver.Objective().Value(),
                "status": status_text,
            }
        else:
            results = {"status": status_text, "objective_value": None}

        self._progress(
            (
                f"solve: finished status={status_text}, objective={results.get('objective_value')}, "
                f"solve_call={solve_call_dt:.2f}s, total={time.perf_counter() - solve_t0:.2f}s"
            )
        )

        return results

    # ---------- variables ----------
    def _build_variables(self) -> None:
        for it in self.data.items:
            self.p[it] = self.solver.BoolVar(f"p_{it}")
            self.x[it] = self.solver.NumVar(0.0, self.width, f"x_{it}")
            self.str_var[it] = self.solver.IntVar(0, self.S - 1, f"s_{it}")

            for s in range(self.S):
                self.deltas[(it, s)] = self.solver.BoolVar(f"delta_{it}_{s}")

        for (i, j, k), _segs in self.encoding.enc.items():
            c_count = self.C.get((i, j, k), 0)
            for c in range(c_count):
                self.gammas[(i, j, k, c)] = self.solver.BoolVar(f"gamma_{i}_{j}_{k}_{c}")

    # ---------- constraints ----------
    def _add_strip_linking_constraints(self) -> None:
        for it in self.data.items:
            self.solver.Add(sum(self.deltas[(it, s)] for s in range(self.S)) == self.p[it])
            self.solver.Add(self.str_var[it] == sum(s * self.deltas[(it, s)] for s in range(self.S)))

    def _add_boundary_constraints(self) -> None:
        for it in self.data.items:
            restrict_this_item = (
                bool(self.restricted_item_ids) and (it.id in self.restricted_item_ids)
            )
            x_lower = 0.0
            x_upper = float(self.width)
            y_lower = 0.0
            y_upper = float(self.height)
            if restrict_this_item:
                if self.packing_x_min is not None:
                    x_lower = float(self.packing_x_min)
                if self.packing_x_max is not None:
                    x_upper = float(self.packing_x_max)
                if self.packing_y_min is not None and it.id not in self.no_lower_y_bound_item_ids:
                    y_lower = float(self.packing_y_min)
                if self.packing_y_max is not None:
                    y_upper = float(self.packing_y_max)

            self.solver.Add(self.x[it] + it.xmax <= x_upper + self.big_M * (1 - self.p[it]))
            self.solver.Add(self.x[it] + it.xmin >= x_lower - self.big_M * (1 - self.p[it]))
            self.solver.Add(self.str_var[it] * self.h + it.ymin >= y_lower - self.big_M * (1 - self.p[it]))
            self.solver.Add(self.str_var[it] * self.h + it.ymax <= y_upper + self.big_M * (1 - self.p[it]))

    def _add_single_use_constraints(self) -> None:
        items_by_id = defaultdict(list)
        for it in self.data.items:
            items_by_id[it.id].append(it)

        for group in items_by_id.values():
            all_deltas = []
            for it in group:
                for s in range(self.S):
                    all_deltas.append(self.deltas[(it, s)])
            self.solver.Add(sum(all_deltas) <= 1)

    def _add_fixed_item_constraints(self) -> None:
        if not self.fixed_item_assignments and not self.forced_unpacked_ids:
            return

        fixed_by_id = {}
        for item, (x_value, strip_idx) in self.fixed_item_assignments.items():
            if item not in self.p:
                raise ValueError("fixed_item_assignments contains an item from another Data instance")
            if item.id in fixed_by_id:
                raise ValueError("fixed_item_assignments contains multiple orientations for one item.id")
            fixed_by_id[item.id] = (item, float(x_value), int(strip_idx))

        items_by_id = defaultdict(list)
        for it in self.data.items:
            items_by_id[it.id].append(it)

        for item_id, group in items_by_id.items():
            if item_id in self.forced_unpacked_ids:
                for it in group:
                    self._force_item_not_packed(it)
                continue

            fixed_entry = fixed_by_id.get(item_id)
            if fixed_entry is None:
                continue

            fixed_item, x_value, strip_idx = fixed_entry
            strip_idx = max(0, min(self.S - 1, int(strip_idx)))
            for it in group:
                if it is fixed_item:
                    self._force_item_packed(it, x_value, strip_idx)
                else:
                    self._force_item_not_packed(it)

    def _force_item_packed(self, it: Item, x_value: float, strip_idx: int) -> None:
        self.solver.Add(self.p[it] == 1)
        self.solver.Add(self.x[it] == float(x_value))
        self.solver.Add(self.str_var[it] == int(strip_idx))
        for s in range(self.S):
            self.solver.Add(self.deltas[(it, s)] == (1 if s == strip_idx else 0))

    def _force_item_not_packed(self, it: Item) -> None:
        self.solver.Add(self.p[it] == 0)
        self.solver.Add(self.x[it] == 0.0)
        self.solver.Add(self.str_var[it] == 0)
        for s in range(self.S):
            self.solver.Add(self.deltas[(it, s)] == 0)

    def _add_minimum_quality_constraint(self) -> None:
        min_objective = None

        if self.min_objective_value is not None:
            min_objective = float(self.min_objective_value)

        if self.max_free_space_percent is not None:
            free_pct = max(0.0, min(100.0, float(self.max_free_space_percent)))
            container_area = self.width * self.height
            area_from_free_pct = container_area * (1.0 - free_pct / 100.0)
            min_objective = (
                area_from_free_pct if min_objective is None else max(min_objective, area_from_free_pct)
            )

        if min_objective is None:
            return

        objective_expr = sum(float(it.area) * self.p[it] for it in self.data.items)
        self.solver.Add(objective_expr >= float(min_objective))

    def _add_non_overlap_constraints(self) -> None:
        # Greedy-fixed placements are continuous by Y, while this model is strip-indexed.
        # Fixed-vs-fixed checks in strip space can be inconsistent with the original greedy
        # geometry, so we keep overlap checks only for pairs involving at least one variable item.
        fixed_items = set(self.fixed_item_assignments.keys())

        for i in self.data.items:
            for j in self.data.items:
                if i.id == j.id:
                    continue
                if i in fixed_items and j in fixed_items:
                    continue

                if hasattr(self.encoding, "k_bounds"):
                    n_min, n_max = self.encoding.k_bounds.get((i, j), (-(self.S - 1), self.S - 1))
                else:
                    n_min, n_max = (-(self.S - 1), self.S - 1)

                for si in range(self.S):
                    di = self.deltas[(i, si)]
                    for sj in range(self.S):
                        dj = self.deltas[(j, sj)]
                        k = si - sj

                        if k < n_min or k > n_max:
                            continue

                        key = (i, j, k)
                        Ck = self.C.get(key, 0)
                        if Ck <= 0:
                            continue

                        for c in range(Ck):
                            gamma = self.gammas[(i, j, k, c)]
                            a_val = self.a[key][c]
                            b_val = self.b[key][c]

                            self.solver.Add(
                                self.x[i] <= self.x[j] + a_val
                                + self.big_M * gamma
                                + self.big_M * (1 - di)
                                + self.big_M * (1 - dj)
                                + self.big_M * (1 - self.p[i])
                                + self.big_M * (1 - self.p[j])
                            )

                            self.solver.Add(
                                self.x[i] >= self.x[j] + b_val
                                - self.big_M * (1 - gamma)
                                - self.big_M * (1 - di)
                                - self.big_M * (1 - dj)
                                - self.big_M * (1 - self.p[i])
                                - self.big_M * (1 - self.p[j])
                            )

    # ---------- objective ----------
    def _set_objective(self) -> None:
        obj = self.solver.Objective()
        for it in self.data.items:
            obj.SetCoefficient(self.p[it], float(it.area))
        obj.SetMaximization()

    # ---------- solver config ----------
    def _configure_solver(self) -> None:
        threads = self._resolve_num_threads(self.num_threads)
        self.solver.SetNumThreads(threads)

        if self.time_limit_sec is not None:
            self.solver.SetTimeLimit(int(max(0.0, float(self.time_limit_sec)) * 1000.0))

        if self.solver_name.upper() != "SCIP":
            return

        params = []
        params.append(f"parallel/maxnthreads = {threads}")
        if self.relative_gap is not None:
            params.append(f"limits/gap = {max(0.0, float(self.relative_gap))}")
        if self.stop_after_first_solution:
            params.append("limits/solutions = 1")
        if not self.enable_output:
            params.append("display/verblevel = 0")

        if params:
            self.solver.SetSolverSpecificParametersAsString("\n".join(params))

    @staticmethod
    def _resolve_num_threads(num_threads: Optional[int]) -> int:
        if num_threads is not None:
            return max(1, int(num_threads))
        return 1

    def _compute_auto_big_m(self) -> Tuple[float, Dict[str, Optional[float]]]:
        items = list(self.data.items)
        if not items:
            return 1.0, {
                "boundary_req": 1.0,
                "non_overlap_req": 0.0,
                "min_a": None,
                "max_b": None,
            }

        width = float(self.width)
        height = float(self.height)

        min_xmin = min(float(it.xmin) for it in items)
        max_xmax = max(float(it.xmax) for it in items)
        min_ymin = min(float(it.ymin) for it in items)
        max_ymax = max(float(it.ymax) for it in items)

        x_lower_candidates = [0.0]
        x_upper_candidates = [width]
        y_lower_candidates = [0.0]
        y_upper_candidates = [height]

        if self.packing_x_min is not None:
            x_lower_candidates.append(float(self.packing_x_min))
        if self.packing_x_max is not None:
            x_upper_candidates.append(float(self.packing_x_max))
        if self.packing_y_min is not None:
            y_lower_candidates.append(float(self.packing_y_min))
        if self.packing_y_max is not None:
            y_upper_candidates.append(float(self.packing_y_max))

        boundary_req = 0.0
        for x_upper in x_upper_candidates:
            boundary_req = max(boundary_req, width + max_xmax - float(x_upper))
        for x_lower in x_lower_candidates:
            boundary_req = max(boundary_req, float(x_lower) - min_xmin)
        for y_upper in y_upper_candidates:
            boundary_req = max(boundary_req, max_ymax - float(y_upper))
        for y_lower in y_lower_candidates:
            boundary_req = max(boundary_req, float(y_lower) - min_ymin)

        min_a: Optional[float] = None
        max_b: Optional[float] = None
        for vals in self.a.values():
            if not vals:
                continue
            local_min = min(float(v) for v in vals)
            min_a = local_min if min_a is None else min(min_a, local_min)
        for vals in self.b.values():
            if not vals:
                continue
            local_max = max(float(v) for v in vals)
            max_b = local_max if max_b is None else max(max_b, local_max)

        non_overlap_req = 0.0
        if min_a is not None:
            non_overlap_req = max(non_overlap_req, width - float(min_a))
        if max_b is not None:
            non_overlap_req = max(non_overlap_req, width + float(max_b))

        required = max(1.0, boundary_req, non_overlap_req)
        if not math.isfinite(required):
            required = 1e6

        # Small safety margin for numerical tolerance.
        big_m = required * 1.05 + 1.0
        return float(big_m), {
            "boundary_req": float(boundary_req),
            "non_overlap_req": float(non_overlap_req),
            "min_a": min_a,
            "max_b": max_b,
        }

    # ---------- utils ----------
    def _status_name(self, status: int) -> str:
        names = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        model_invalid = getattr(pywraplp.Solver, "MODEL_INVALID", None)
        if model_invalid is not None:
            names[model_invalid] = "MODEL_INVALID"
        return names.get(status, f"STATUS_{status}")

    def _progress(self, message: str) -> None:
        callback = self._progress_callback
        if callback is not None:
            callback(f"{self._progress_label} {message}")
