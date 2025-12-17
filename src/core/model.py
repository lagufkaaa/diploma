import numpy as np
import ortools
from ortools.linear_solver import pywraplp
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

from .encoding import Encoding
from .nfp import NFP
from src.tests.test_geometry import Test_NFP
from src.tests.test_encoding import Test_Encoding
from ..utils.helpers import util_model 

class Model:
    def __init__(self):
        pass

    def model_func(items, W, H, R, N, S):
        h = H/S
        
        min_y, max_y = 0, H
        H = max_y - min_y
        
        Y = [min_y + h * i for i in range(S + 1)]

        M = 1e17
        solver = pywraplp.Solver.CreateSolver('SCIP')
        Area = [Polygon(item).area for item in items]
        
        p = [[solver.BoolVar(f'p_{n}_{r}') for r in range(R)] for n in range(N)]
        
        for n in range(N):
            solver.Add(sum(p[n][r] for r in range(R)) <= 1)
            
        deltas = [[[solver.BoolVar(f'deltas_{s}_{n}_{r}') for r in range(R)] for n in range(N)] for s in range(S)]
        for n in range(N):
            solver.Add(sum(sum(deltas[s][n][r] for s in range(S)) for r in range(R)) <= 1) # TODO какой тут знак????
            
        str_vars = [[solver.IntVar(0, S, f's_{r}_{n}') for n in range(N)] for r in range(R)]
        for n in range(N):
            for r in range(R):
                # str_vars[r][n] должно равняться s, для которого deltas[s][n][r] = 1
                # Создаем выражение: str_vars[r][n] = sum(s * deltas[s][n][r] for s in range(S))
                solver.Add(str_vars[r][n] == sum(s * deltas[s][n][r] for s in range(S))) # TODO если все дельты = 0, то равно 0!!!! неправильно
                
            x = [[solver.NumVar(0, W, f'x_{r}_{n}') for r in range(R)] for n in range(N)]

        x_min = []
        x_max = []
        y_min = []
        y_max = []

        for i, item in enumerate(items):
            # Создаем копию для нормализации, не изменяя оригинал
            normalized_item = []
            for j, point in enumerate(item):
                if j == 0:
                    normalized_item.append([0, 0])
                else:
                    normalized_item.append([point[0] - item[0][0], point[1] - item[0][1]])
            
            x_min.append([])
            x_max.append([])
            y_min.append([])
            y_max.append([])
            
            for r in range(R):
                # Получаем bounding box как словарь
                bbox = util_model.find_bounding_box_numpy(normalized_item)
                
                # Извлекаем значения из словаря
                xmin = bbox['min_x']
                xmax = bbox['max_x'] 
                ymin = bbox['min_y']
                ymax = bbox['max_y']
                
                x_min[i].append(xmin)
                x_max[i].append(xmax)
                y_min[i].append(ymin)
                y_max[i].append(ymax)

                solver.Add(str_vars[r][i]*h + ymin >= 0)
                solver.Add(str_vars[r][i]*h + ymax <= H)
                
        for i in range(N):
            for r in range(R):
                solver.Add(x[i][r] >= -x_min[i][r])
                solver.Add((x[i][r] + x_max[i][r]) <= (W + (1 - p[i][r])*M))
                
        NFPs = [[[[None for _ in range(R)] for _ in range(N)] for _ in range(R)] for _ in range(N)]
        Enc = [[[[None for _ in range(R)] for _ in range(N)] for _ in range(R)] for _ in range(N)]
        C = [[[[None for _ in range(R)] for _ in range(N)] for _ in range(R)] for _ in range(N)]
        a = [[[[[] for _ in range(R)] for _ in range(N)] for _ in range(R)] for _ in range(N)]
        b = [[[[[] for _ in range(R)] for _ in range(N)] for _ in range(R)] for _ in range(N)]

        for i in range(N):
            for r1 in range(R):
                for j in range(N): 
                    for r2 in range(R):
                        if i == j: 
                            NFPs[i][r1][j][r2] = None
                            Enc[i][r1][j][r2] = None
                            C[i][r1][j][r2] = None
                        else:
                            onfp = NFP.outer_no_fit_polygon(Polygon(items[i]), Polygon(items[j]), items[i][0])
                            
                            onfp = util_model.normalize_polygon(onfp)
                            # Test_NFP.vis_nfp(Polygon(items[i]), Polygon(items[j]), items[i][0])
                            NFPs[i][r1][j][r2] = onfp
                            
                            Enc[i][r1][j][r2] = Encoding.encode_polygon(onfp, Y)
                            # Test_Encoding.vis_encoding(onfp, Y)
                            
                            # Создаем промежуточные переменные
                            diff_var = solver.IntVar(-S, S, f'diff_{i}_{r1}_{j}_{r2}')
                            abs_diff_var = solver.IntVar(0, S, f'abs_diff_{i}_{r1}_{j}_{r2}')
                            is_positive = solver.BoolVar(f'pos_{i}_{r1}_{j}_{r2}')
                            
                            # Ограничения
                            solver.Add(diff_var == str_vars[r1][i] - str_vars[r2][j])
                            solver.Add(abs_diff_var >= diff_var)
                            solver.Add(abs_diff_var >= -diff_var)
                            solver.Add(abs_diff_var <= diff_var + M * (1 - is_positive))
                            solver.Add(abs_diff_var <= -diff_var + M * is_positive)
                            solver.Add(diff_var >= -M * (1 - is_positive))
                            solver.Add(diff_var <= M * is_positive - 1e-9)
                            
                            target_value = abs_diff_var * h
                                                        
                            if Enc[i][r1][j][r2] != []:
                                # C[i][r1][j][r2] = round(sum(1 for seg in Enc[i][r1][j][r2] if (seg[0][1] == target_value))/ 2)
                                
                                for seg in Enc[i][r1][j][r2]:
                                    if seg[1][1] == target_value: 
                                        a[i][r1][j][r2].append(seg[0][0])
                                        b[i][r1][j][r2].append(seg[1][0])
                                # print(len(a[i][r1][j][r2]))
                                # print(len(b[i][r1][j][r2]))
                                C[i][r1][j][r2] = len(a[i][r1][j][r2])
                            else:
                                C[i][r1][j][r2] = 0
                            

        gammas = [
            [
                [
                    [
                        [
                            solver.BoolVar(f'gammas_{i}_{r1}_{j}_{r2}_{c}')
                            for c in range(int(C[i][r1][j][r2])) 
                        ] if C[i][r1][j][r2] is not None else []
                        for r2 in range(R)
                    ]
                    for j in range(N)
                ]
                for r1 in range(R)
            ]
            for i in range(N)
        ]

        # TODO точно ли все циклы указаны??? 

        for i in range(N):
            for r1 in range(R):
                for j in range(i + 1, N): # TODO точно ли i + 1
                    for r2 in range(R):
                        for c in range(C[i][r1][j][r2]):
                            diff_var_x = solver.NumVar(-W, W, f'diff_{i}_{r1}_{j}_{r2}')
                            abs_diff_var_x = solver.NumVar(0, W, f'abs_diff_{i}_{r1}_{j}_{r2}')

                            solver.Add(diff_var_x == - x[i][r1] + x[j][r2])
                            solver.Add(abs_diff_var_x >= diff_var_x)
                            solver.Add(abs_diff_var_x >= -diff_var_x)
                            solver.Add(abs_diff_var_x <= diff_var_x + M * (1 - gammas[i][r1][j][r2][c]))
                            solver.Add(abs_diff_var_x <= -diff_var_x + M * gammas[i][r1][j][r2][c])
                            solver.Add(diff_var_x >= -M * (1 - gammas[i][r1][j][r2][c]))
                            solver.Add(diff_var_x <= M * gammas[i][r1][j][r2][c] - 1e-9)


                            solver.Add(x[i][r1] - x[j][r2] >= -M * (1 - gammas[i][r1][j][r2][c]))

                            epsilon = 1e-5
                            solver.Add(x[j][r2] - x[i][r1] + epsilon <= M * gammas[i][r1][j][r2][c])

                            for s in range(S):
                                solver.Add(x[i][r1] - (x[j][r2] - b[i][r1][j][r2][c] + 
                                        gammas[i][r1][j][r2][c] * M + 
                                        (1 - p[i][r1]) * M + 
                                        (1 - p[j][r2]) * M) <= (2 - deltas[s][i][r1] - deltas[s][j][r2]) * M)
                                
                                solver.Add((2 - deltas[s][i][r1] - deltas[s][j][r2]) * M >= x[j][r2] - a[i][r1][j][r2][c] - 
                                        (1 - gammas[i][r1][j][r2][c]) * M - 
                                        (1 - p[i][r1]) * M - 
                                        (1 - p[j][r2]) * M - x[i][r1])


                        # delta_i_active = sum(deltas[s][i][r1] for s in range(S)) # TODO ошибка скорее всего тут!!!
                        # delta_j_active = sum(deltas[s][j][r2] for s in range(S)) # TODO ошибка скорее всего тут!!!
                        
                        # solver.Add(x[i][r1] <= x[j][r2] - b[i][r1][j][r2][c] + 
                        #         gammas[i][r1][j][r2][c] * M + 
                        #         (1 - delta_i_active) * M + 
                        #         (1 - delta_j_active) * M + 
                        #         (1 - p[i][r1]) * M + 
                        #         (1 - p[j][r2]) * M)
                        
                        # solver.Add(x[i][r1] >= x[j][r2] - a[i][r1][j][r2][c] - 
                        #         (1 - gammas[i][r1][j][r2][c]) * M - 
                        #         (1 - delta_i_active) * M - 
                        #         (1 - delta_j_active) * M - 
                        #         (1 - p[i][r1]) * M - 
                        #         (1 - p[j][r2]) * M)
                        
        

        # Правильное вычисление целевой функции
        total_value = 0
        for i in range(N):
            for r in range(R):
                # Добавляем площадь предмета i, если он размещен в повороте r
                total_value += p[i][r] * Area[i]

        solver.Maximize(total_value)

        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            results = {
                'p': [[p[i][r].solution_value() for r in range(R)] for i in range(N)],
                'x': [[x[i][r].solution_value() for r in range(R)] for i in range(N)],
                's': [[str_vars[r][i].solution_value() for r in range(R)] for i in range(N)],
                'objective_value': solver.Objective().Value(),
                'status': 'OPTIMAL'
            }
        else:
            results = {
                'status': 'NOT_OPTIMAL',
                'objective_value': None
            }
        
        return results
