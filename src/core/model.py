from ortools.linear_solver import pywraplp
from core.data import Data, Item
from core.encoding import Encoding
from typing import Dict, Tuple

class Problem:
    def __init__(self, data: Data, S : int, R :int, height: float, width: float, solver_name: str = "SCIP"):
        self.data = data 
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        self.S = S
        self.R = R
        self.height = height
        self.h = height/S
        self.width = width
        self.encoding = Encoding(data, S, height)
        self.a = {}
        self.b = {}
        self.C = {}
        self.build_abc()

        self.big_M = 10e8
        
        self.x: Dict[Item, pywraplp.Variable] = {}
        self.p: Dict[Item, pywraplp.Variable] = {}
        self.str_var: Dict[Item, pywraplp.Variable] = {}
        self.deltas: Dict[Tuple[Item, int], pywraplp.Variable] = {}
        self.gammas: Dict[Tuple[Item, Item, int], pywraplp.Variable] = {}
    
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
        self._add_zero_checking_delta()
        
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
            self.x[p] = self.solver.NumVar(0.0, self.width, f"x_{p}")
            self.str_var[p] = self.solver.IntVar(0, self.S, f"s_{p}") #TODO возможно тут не должно быть ограничения сверху из-за нового ограничения, что если все дельты - нули, то оно улетает в никуда

            for s in range(self.S):
                self.deltas[(p, s)] = self.solver.BoolVar(f"deltas_{p}_{s}")
            
            for other in self.data.items:
                if p.id != other.id:
                    for s in range(self.S):
                        self.gammas[(p, other, s)] = self.solver.BoolVar(f"gammas_{p}_{other}_{s}")

    def _add_strip_linking_constraints(self) -> None: #TODO !!! перепроверить нулевые случаи
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
            self.solver.Add(self.x[i] + i.xmax <= self.width)
            self.solver.Add(self.x[i] >= -i.xmin)
            
            self.solver.Add(self.str_var[i]*self.h + i.ymin >= 0)
            self.solver.Add(self.str_var[i]*self.h + i.ymax <= self.height)

    def _add_single_use_constraints(self) -> None:
        from collections import defaultdict

        items_by_id = defaultdict(list)
        for item in self.data.items:
            items_by_id[item.id].append(item)

        for item_list in items_by_id.values():
            delta_vars = []
            for item in item_list:            
                for s in range(self.S):        
                    delta_vars.append(self.deltas[(item, s)])
                
                self.solver.Add(sum(delta_vars) == self.p[item])

            self.solver.Add(sum(delta_vars) <= 1)
            
    def _add_zero_checking_delta(self) -> None: # если все дельты == 0, то предмет отправляется в никуда
        for item in self.data.item:
            self.solver.Add((1 - self.p[item]) * self.M <= self.str_var[item])

    def _set_objective(self) -> None:
        obj = self.solver.Objective()
        for item in self.data.items: 
            obj.SetCoefficient(self.p[item], float(item.area)) #TODO понять сработает ли такой вариант
        obj.SetMaximization()