from ortools.linear_solver import pywraplp
from core.data import Data
from core.encoding import Encoding

class Problem:
    def __init__(self, data: Data, S : int, R :int, height: float, width: float, solver_name: str = "SCIP"):
        self.data = data 
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        self.S = S
        self.R = R
        self.height = height
        self.width = width
        self.encoding = Encoding(data, S, height)
        
        self.x: Dict[str, pywraplp.Variable] = {}
        
    def solve(self, data : Data, S :int, R :int, height: float, width: float):
        self._build_variables()
        self._set_objective()
        pass
    
    def _build_variables(self) -> None:
        for p in self.data.items: # TODO items или items_with_rotation?
            self.x[p] = self.solver.BoolVar(f"x_{p}")
            
    def _set_objective(self) -> None:
        obj = self.solver.Objective()
        for p in self.data.items: # TODO items или items_with_rotation?
            obj.SetCoefficient(self.x[p], float(p.area))
        obj.SetMaximization()

