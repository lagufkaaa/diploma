from ortools.linear_solver import pywraplp
from core.data import Data
from core.encoding import Encoding
from typing import Dict, Tuple

class Problem:
    def __init__(self, data: Data, S : int, R :int, height: float, width: float, solver_name: str = "SCIP"):
        self.data = data 
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        self.S = S
        self.R = R
        self.height = height
        self.width = width
        self.encoding = Encoding(data, S, height)
        self.a = {}
        self.b = {}
        self.C = {}
        self.build_abc()

        self.big_M = 10e8
        
        self.p: Dict[str, pywraplp.Variable] = {}
        self.x: Dict[str, pywraplp.Variable] = {}
        self.str_var: Dict[str, pywraplp.Variable] = {}
        self.deltas: Dict[Tuple[str, int], pywraplp.Variable] = {}
        self.gammas: Dict[str, pywraplp.Variable] = {}
    
    def build_abc(self):
        for nfp in self.data.nfp:
            temp1 = []
            temp2 = []
            for seg in self.encoding.enc[nfp]:
                temp1.append(seg[0][0])
                temp2.append(seg[1][0])
                
            self.a[nfp] = temp1
            self.b[nfp] = temp2
            self.C[nfp] = len(temp1)

    def solve(self):
        self._build_variables()
        self._set_objective()

        self._add_strip_linking_constraints()
        self._add_boundary_constraints()
        self._add_single_use_constraints()
        
        solver = self.solver

        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            results = {
                'p': [self.p[p].solution_value() for p in self.data.items],
                'x': [self.x[p].solution_value() for p in self.data.items],
                's': [self.str_var[p].solution_value() for p in self.data.items],
                'objective_value': solver.Objective().Value(),
                'status': 'OPTIMAL'
            }
        else:
            results = {
                'status': 'NOT_OPTIMAL',
                'objective_value': None
            }
        
        return results
    
    def _build_variables(self) -> None:
        for p in self.data.items: 
            self.p[p] = self.solver.BoolVar(f"p_{p}")
            self.x[p] = self.solver.NumVar(0.0, self.width, f"x_{p}")
            self.str_var[p] = self.solver.IntVar(0, self.S, f"s_{p}")

            for s in range(self.S):
                self.deltas[(p, s)] = self.solver.BoolVar(f"deltas_{p}_{s}")

    def _add_strip_linking_constraints(self) -> None: #TODO !!! перепроверить нулевые случаи
        """
        Связь дискретных delta_{p,s} с переменной s_p (str_var[p]).
        Если предмет выбран (p[p]=1), то:
        - выбирается ровно одна полоса: sum_s delta_{p,s} = 1
        - s_p = sum_s s * delta_{p,s}
        Если предмет НЕ выбран (p[p]=0), то:
        - ни одна полоса не выбирается: sum_s delta_{p,s} = 0
        - (и тогда s_p автоматически будет 0, если добавить второе равенство)
        """
        for p in self.data.items:
            # 1) one-hot, но учитываем "включенность" предмета
            self.solver.Add(
                sum(self.deltas[(p, s)] for s in range(self.S)) == self.p[p]
            )

            # 2) связь индекса полосы с выбранной дельтой
            self.solver.Add(
                self.str_var[p] == sum(s * self.deltas[(p, s)] for s in range(self.S))
            )

    def _add_boundary_constraints(self) -> None:
        for i in self.data.items: 
            self.solver.Add(self.x[i] + i.xmax <= self.width + (1 - self.p[i])*self.big_M)
            self.solver.Add(self.x[i] >= -i.xmin)

    def _add_single_use_constraints(self) -> None: #TODO
        for i in self.data.items:
            for s in range(self.S):
                pass

    def _set_objective(self) -> None:
        obj = self.solver.Objective()
        for p in self.data.items: 
            obj.SetCoefficient(self.p[p], float(p.area))
        obj.SetMaximization()