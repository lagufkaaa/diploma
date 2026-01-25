from ortools.linear_solver import pywraplp
from core.data import Data
from core.encoding import Encoding
from typing import Dict, List

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
        
        self.p: Dict[str, pywraplp.Variable] = {}
        self.x: Dict[str, pywraplp.Variable] = {}
        self.str_var: Dict[str, pywraplp.Variable] = {}
        self.gammas: Dict[str, pywraplp.Variable] = {}
        
    def solve(self):
        self._build_variables()
        self._set_objective()
        
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
            
    def _set_objective(self) -> None:
        obj = self.solver.Objective()
        for p in self.data.items: 
            obj.SetCoefficient(self.p[p], float(p.area))
        obj.SetMaximization()
        
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
