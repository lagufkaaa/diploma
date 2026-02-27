from ortools.linear_solver import pywraplp
from typing import Dict, Tuple

from core.data import Data, Item
from core.encoding import Encoding

import math

class Problem:
    def __init__(
        self,
        data: Data,
        S: int,
        R: int,
        height: float,
        width: float,
        solver_name: str = "SCIP",
    ):
        self.data = data
        self.solver = pywraplp.Solver.CreateSolver(solver_name)

        self.S = S
        self.R = R
        self.height = float(height)
        self.width = float(width)
        self.h = self.height / self.S

        self.encoding = Encoding(data, S, height)

        # a/b/C indexed by (i, j, k) where k = s_j - s_i
        self.a: Dict[Tuple[Item, Item, int], list[float]] = {}
        self.b: Dict[Tuple[Item, Item, int], list[float]] = {}
        self.C: Dict[Tuple[Item, Item, int], int] = {}
        self.build_abc()

        self.big_M = 1e8

        # decision vars
        self.x: Dict[Item, pywraplp.Variable] = {}
        self.p: Dict[Item, pywraplp.Variable] = {}
        self.str_var: Dict[Item, pywraplp.Variable] = {}
        self.deltas: Dict[Tuple[Item, int], pywraplp.Variable] = {}

        # gamma indexed by (i, j, k, c)
        self.gammas: Dict[Tuple[Item, Item, int, int], pywraplp.Variable] = {}

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

        if status == pywraplp.Solver.OPTIMAL:
            results = {
                "p": [self.p[it].solution_value() for it in self.data.items],
                "x": [self.x[it].solution_value() for it in self.data.items],
                "s": [self.str_var[it].solution_value() for it in self.data.items],
                "deltas": [
                    [self.deltas[(it, s)].solution_value() for s in range(self.S)]
                    for it in self.data.items
                ],
                "objective_value": self.solver.Objective().Value(),
                "status": "OPTIMAL",
            }
        else:
            results = {"status": "NOT_OPTIMAL", "objective_value": None}

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

        # gamma variables: only where segments exist
        for (i, j, k), _segs in self.encoding.enc.items():
            Ck = self.C[(i, j, k)]
            for c in range(Ck):
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
            self.solver.Add(self.x[it] + it.xmin >= 0.0 - self.big_M * (1 - self.p[it]))

            # ---- NEW: bounds on str_var (integer rows) ----
            # lower bound for s: s >= ceil((-ymin)/h)
            s_low = math.ceil((-float(it.ymin)) / self.h)

            # upper bound for s: s <= floor((height - ymax)/h)
            s_high = math.floor((self.height - float(it.ymax)) / self.h)

            # clamp into [0, S-1] just in case
            s_low = max(0, s_low)
            s_high = min(self.S - 1, s_high)

            # If item cannot fit vertically at all, forbid packing it:
            if s_low > s_high:
                self.solver.Add(self.p[it] == 0)
                continue

            # enforce bounds when packed
            self.solver.Add(self.str_var[it] >= s_low - self.big_M * (1 - self.p[it]))
            self.solver.Add(self.str_var[it] <= s_high + self.big_M * (1 - self.p[it]))

            # ---- keep your Y global constraints too (optional but OK) ----
            self.solver.Add(self.str_var[it] * self.h + it.ymin >= 0.0 - self.big_M * (1 - self.p[it]))
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
                self.solver.Add(sum(self.deltas[(it, s)] for s in range(self.S)) == self.p[it])

            self.solver.Add(sum(all_deltas) <= 1)

    def _add_non_overlap_constraints(self) -> None:
        """
        Direct translation of your formula using:
        - k = s_j - s_i
        - coefficients a[(i,j,k)][c], b[(i,j,k)][c]
        - activation via deltas(i,si), deltas(j,sj)
        """
        printed = False
        for i in self.data.items:
            for j in self.data.items:
                if i.id == j.id:
                    continue

                # bounds for k (difference of strips)
                # expecting dict: encoding.k_bounds[(i,j)] = (n_min, n_max)
                if hasattr(self.encoding, "k_bounds"):
                    n_min, n_max = self.encoding.k_bounds.get((i, j), (-(self.S - 1), self.S - 1))
                else:
                    n_min, n_max = (-(self.S - 1), self.S - 1)

                for si in range(self.S):
                    di = self.deltas[(i, si)]
                    for sj in range(self.S):
                        dj = self.deltas[(j, sj)]
                        k = sj - si
                        if (not printed) and k == 0:
                            key = (i, j, k)
                            Ck = self.C.get(key, 0)
                            print("\n--- DEBUG Encoding for k=0 ---")
                            print("pair:", i, j, "si,sj:", si, sj, "k:", k)
                            print("Ck:", Ck)
                            if Ck > 0:
                                pairs = list(zip(self.a[key][:10], self.b[key][:10]))
                                print("a,b first 10:", pairs)
                                print("min a:", min(self.a[key]), "max b:", max(self.b[key]))
                            else:
                                print("NO SEGMENTS for key", key)
                            printed = True

                        if k < n_min or k > n_max:
                            continue

                        key = (i, j, k)
                        Ck = self.C.get(key, 0)
                        if Ck == 0:
                            continue

                        for c in range(Ck):
                            gamma = self.gammas[(i, j, k, c)]
                            a_val = self.a[key][c]
                            b_val = self.b[key][c]

                            # x_i <= x_j - b + gamma*M + (1-di)M + (1-dj)M
                            self.solver.Add(
                                self.x[i] <= self.x[j] - b_val
                                + self.big_M * gamma
                                + self.big_M * (1 - di)
                                + self.big_M * (1 - dj)
                            )

                            # x_i >= x_j - a - (1-gamma)M - (1-di)M - (1-dj)M
                            self.solver.Add(
                                self.x[i] >= self.x[j] - a_val
                                - self.big_M * (1 - gamma)
                                - self.big_M * (1 - di)
                                - self.big_M * (1 - dj)
                            )

    def _set_objective(self) -> None:
        obj = self.solver.Objective()
        for it in self.data.items:
            obj.SetCoefficient(self.p[it], float(it.area))
        obj.SetMaximization()