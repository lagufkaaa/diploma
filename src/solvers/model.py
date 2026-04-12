from ortools.linear_solver import pywraplp
import math
from typing import Dict, Optional, Tuple

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
        enable_output: bool = False,
        objective_stop_value: Optional[float] = None,
        relative_gap: Optional[float] = None,
        time_limit_sec: Optional[float] = None,
        stop_after_first_solution: bool = False,
        num_threads: Optional[int] = None,
    ):
        self.data = data
        self.solver_name = str(solver_name)
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        if self.solver is None:
            raise RuntimeError(f"Failed to create solver '{solver_name}'")

        self.enable_output = bool(enable_output)
        self.objective_stop_value = objective_stop_value
        self.relative_gap = relative_gap
        self.time_limit_sec = time_limit_sec
        self.stop_after_first_solution = bool(stop_after_first_solution)
        self.num_threads = num_threads
        if self.enable_output:
            self.solver.EnableOutput()

        self.S = S
        self.R = R
        self.height = float(height)
        self.width = float(width)
        self.h = self.height / self.S

        self.encoding = Encoding(data, S, height)

        self.a: Dict[Tuple[Item, Item, int], list[float]] = {}
        self.b: Dict[Tuple[Item, Item, int], list[float]] = {}
        self.C: Dict[Tuple[Item, Item, int], int] = {}
        self.build_abc()

        self.big_M = 1e8

        self.x: Dict[Item, pywraplp.Variable] = {}
        self.p: Dict[Item, pywraplp.Variable] = {}
        self.str_var: Dict[Item, pywraplp.Variable] = {}
        self.deltas: Dict[Tuple[Item, int], pywraplp.Variable] = {}

        self.gammas: Dict[Tuple[Item, Item, int, int], pywraplp.Variable] = {}
        self._configure_solver()

    # ---------- preprocessing ----------
    def build_abc(self):
        """
        Build a/b/C from Encoding.enc.
        Encoding.enc expected: enc[(i, j, k)] = list of segments [[[x1,y],[x2,y]], ...]
        """
        for key, segs in self.encoding.enc.items():
            # key must be (i, j, k)
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
        self._build_variables()
        self._set_objective()

        self._add_strip_linking_constraints()
        self._add_boundary_constraints()
        self._add_single_use_constraints()
        self._add_non_overlap_constraints()

        status = self.solver.Solve()

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
                "status": self._status_name(status),
            }
        else:
            results = {"status": self._status_name(status), "objective_value": None}

        return results

    # ---------- variables ----------
    def _build_variables(self) -> None:
        # item variables
        for it in self.data.items:
            self.p[it] = self.solver.BoolVar(f"p_{it}")
            self.x[it] = self.solver.NumVar(0.0, self.width, f"x_{it}")
            self.str_var[it] = self.solver.IntVar(0, self.S - 1, f"s_{it}")

            for s in range(self.S):
                self.deltas[(it, s)] = self.solver.BoolVar(f"delta_{it}_{s}")

        # gamma variables: one binary per NFP segment c
        for (i, j, k), _segs in self.encoding.enc.items():
            c_count = self.C.get((i, j, k), 0)
            for c in range(c_count):
                self.gammas[(i, j, k, c)] = self.solver.BoolVar(f"gamma_{i}_{j}_{k}_{c}")

    # ---------- constraints ----------
    def _add_strip_linking_constraints(self) -> None:
        for it in self.data.items:
            # sum_s delta(it,s) = p(it)
            self.solver.Add(sum(self.deltas[(it, s)] for s in range(self.S)) == self.p[it])

            # s(it) = Σ s * delta(it,s)
            self.solver.Add(self.str_var[it] == sum(s * self.deltas[(it, s)] for s in range(self.S)))

    def _add_boundary_constraints(self) -> None:
        for it in self.data.items:
            # X bounds (relaxed if not packed)
            self.solver.Add(self.x[it] + it.xmax <= self.width + self.big_M * (1 - self.p[it]))
            self.solver.Add(self.x[it] + it.xmin >= -self.big_M * (1 - self.p[it]))

            # ---- keep your Y global constraints too (optional but OK) ----
            self.solver.Add(self.str_var[it] * self.h + it.ymin >= -self.big_M * (1 - self.p[it]))
            self.solver.Add(self.str_var[it] * self.h + it.ymax <= self.height + self.big_M * (1 - self.p[it]))
    
    def _add_single_use_constraints(self) -> None:
        """
        Если item.id повторяется (одна форма в разных rotation / копиях),
        разрешаем выбрать максимум одну из группы.
        """
        from collections import defaultdict

        items_by_id = defaultdict(list)
        for it in self.data.items:
            items_by_id[it.id].append(it)

        for group in items_by_id.values():
            # суммарно по группе и всем полосам не более 1
            all_deltas = []
            for it in group:
                for s in range(self.S):
                    all_deltas.append(self.deltas[(it, s)])
                # у каждой копии sum delta = p
                # self.solver.Add(sum(self.deltas[(it, s)] for s in range(self.S)) == self.p[it])

            self.solver.Add(sum(all_deltas) <= 1)

    def _add_zero_checking_deltas(self) -> None:
        for i in self.data.items:
            self.solver.Add((1 - self.p[i])*self.M <= self.str_var[i])
            
    def _add_non_overlap_constraints(self) -> None:
        for i in self.data.items:
            for j in self.data.items:
                if i.id == j.id:
                    continue
                n_min, n_max = None, None
                if hasattr(self.encoding, "k_bounds"):
                    n_min, n_max = self.encoding.k_bounds.get((i, j), (-(self.S - 1), self.S - 1))
                else:
                    n_min, n_max = (-(self.S - 1), self.S - 1)

                for si in range(self.S):
                    di = self.deltas[(i, si)]
                    for sj in range(self.S):
                        dj = self.deltas[(j, sj)]
                        # Encoding enc[(i,j,k)] is built for k = s_i - s_j.
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

        params = [f"parallel/maxnthreads = {threads}"]
        if self.relative_gap is not None:
            params.append(f"limits/gap = {max(0.0, float(self.relative_gap))}")
        if self.objective_stop_value is not None:
            objective_stop = float(self.objective_stop_value)
            if math.isfinite(objective_stop):
                params.append(f"limits/objectivestop = {objective_stop}")
        if self.stop_after_first_solution:
            params.append("limits/solutions = 1")
        if not self.enable_output:
            params.append("display/verblevel = 0")

        self.solver.SetSolverSpecificParametersAsString("\n".join(params))

    @staticmethod
    def _resolve_num_threads(num_threads: Optional[int]) -> int:
        if num_threads is not None:
            return max(1, int(num_threads))
        return 1

    @staticmethod
    def _status_name(status: int) -> str:
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
