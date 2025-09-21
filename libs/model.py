import numpy as np
import ortools
from ortools.linear_solver import pywraplp
from shapely.geometry import Polygon
from libs.nfp import NFP
from libs.auxiliary import Auxiliary
from libs.encoding import Encoding

def model_func(items, W, H, R, N, S):
    h = H/S
    M = 1e9
    solver = pywraplp.Solver.CreateSolver('SCIP')

    p = [[solver.BoolVar(f'p_{n}_{r}') for n in range(N)] for r in range(R)]

    for n in range(N):
        solver.Add(sum(p[n][r] for r in range(R)) == 1)

    deltas = [[[solver.BoolVar(f'deltas_{s}_{n}_{r}') for s in range(S)] for n in range(N)] for r in range(R)]

    for n in range(N):
        solver.Add(sum(sum(deltas[s][n][r] for s in range(S)) for r in range(R)) == 1)

    str = [[(solver.IntVar(0, S, f's_{n}_{r}')) for n in range(N)] for r in range(R)]
    for n in range(N):
        for r in range(R):
            for s in range(S):
                solver.Add(str[r][n] == s).OnlyEnforceIf(deltas[s][n][r] == 1)


    x = [[solver.NumVar(0, W, f'x_{n}_{r}') for n in range(N)] for r in range(R)]

    x_min = []
    x_max = []
    y_min = []
    y_max = []

    for item in items: 
        for i in range(len(item)): # TODO понять влияет ли это на алгоритм работы
            if i != 0: 
                item[i] -= item[0]
        item[0] = np.array([0, 0])

        x_min.append([])
        x_max.append([])
        y_min.append([])
        y_max.append([]) 
        for r in range(R):
            # TODO ТУТ ДОЛЖЕН БЫТЬ УМНЫЙ ПОВОРОТ
            xmin, xmax, ymin, ymax = Auxiliary.find_bounding_box_numpy(item)
            x_min[i].append(xmin)
            x_max[i].append(xmax)
            y_min[i].append(ymin)
            y_max[i].append(ymax) 
        
        # TODO если item[i] = item[j], то тогда просто дублировать из таблицы а не просчитывать по новой

    for i in range(N):
        for r in range(R):
            solver.Add(x[i][r] >= -x_min[i][r])
            solver.Add(x[i][r] + x_max[i][r] <= W + (1 - p[i][r])*M)

    NFPs = [[[[None for _ in range(R)] for _ in range(N)] for _ in range(R)] for _ in range(N)]
    Enc = [[[[None for _ in range(R)] for _ in range(N)] for _ in range(R)] for _ in range(N)]
    C = [[[[None for _ in range(R)] for _ in range(N)] for _ in range(R)] for _ in range(N)]
    a = []
    b = []

    for i in range(N):
        a.append([])
        b.append([])
        for r1 in range(R):
            a[i].append([])
            b[i].append([])
            for j in range(N): 
                a[i][r1].append([])
                b[i][r1].append([])
                for r2 in range(R):
                    a[i][r1][j].append([])
                    b[i][r1][j].append([])
                    if i == j: 
                        NFPs[i][r1][j][r2] = None
                        Enc[i][r1][j][r2] = None
                    else: # TODO если item[i] = item[j], то тогда просто дублировать из таблицы а не просчитывать по новой
                        onfp = NFP.outer_no_fit_polygon(Polygon(items[i]), Polygon(items[j]), items[i][0]) # TODO понять какую использовать ведущую точку
                        NFPs[i][r1][j][r2] = onfp
                        Enc[i][r1][j][r2] = Encoding.cod_model(H, S, onfp)
                        C[i][r1][j][r2] = sum(1 for point in Enc[i][r1][j][r2][0] if point[1] == abs(s[i][r1] - s[j][r2])*h)/2 + sum(1 for point in Enc[i][r1][j][r2][1] if point[1] == abs(s[i][r1] - s[j][r2])*h) # TODO памагите
                        for seg in Enc[i][r1][j][r2][0]:
                            if seg[1] == abs(s[i][r1] - s[j][r2]) * h: 
                                a[i][r1][j][r2].append(seg[0])
                                b[i][r1][j][r2].append(seg[1])
                        for point in Enc[i][r1][j][r2][1]:
                            if point[1] == abs(s[i][r1] - s[j][r2]) * h:
                                a[i][r1][j][r2].append(point)
                                b[i][r1][j][r2].append(point)

    gammas = [
            [
                [
                    [
                        [
                            solver.BoolVar(f'gammas_{c}_{n1}_{r1}_{n2}_{r2}')
                            for c in range(C[n1][r1][n2][r2])  # исправлено r1 на r2
                        ]
                        for n2 in range(N)
                    ]
                    for r2 in range(R)
                ]
                for n1 in range(N)
            ]
            for r1 in range(R)
        ]


    # TODO точно ли все циклы указаны??? 

    for i in range(N):
        for r1 in range(R):
            for j in range(i + 1, N): # TODO точно ли i + 1
                for r2 in range(R):
                    for c in range(C[i][r1][j][r2]):
                        solver.Add(x[i][r1] <= x[j][r2] -  b[i][r1][j][r2][c] + gammas[i][r1][j][r2][c]*M + (1 - deltas[s[i]][i][r1])*M + (1 - deltas[s[j]][j][r2])*M + (1 - p[i][r1])*M + (1 - p[j][r2])*M)
                        solver.Add(x[i][r1] >= x[j][r2] -  a[i][r1][j][r2][c] - (1 - gammas[i][r1][j][r2][c])*M - (1 - deltas[s[i]][i][r1])*M - (1 - deltas[s[j]][j][r2])*M - (1 - p[i][r1])*M - (1 - p[j][r2])*M)


    Area = [Polygon(item).area for item in items]

    total_value = sum(p[i] * Area[i] for i in range(N))
    solver.Maximize(total_value)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        results = {
            'p': [[p[i][r].solution_value() for r in range(R)] for i in range(N)],
            'x': [[x[i][r].solution_value() for r in range(R)] for i in range(N)],
            's': [[str[i][r].solution_value() for r in range(R)] for i in range(N)],
            'objective_value': solver.Objective().Value(),
            'status': 'OPTIMAL'
        }
    else:
        results = {
            'status': 'NOT_OPTIMAL',
            'objective_value': None
        }
    
    return results
    
def parse(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    items = []

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('PIECE'):
            i += 1

            quantity_line = lines[i].strip()
            quantity = int(quantity_line.split()[1])
            i += 1

            vertex_line = lines[i].strip()
            vertex_count = int(vertex_line.split()[3])
            i += 1

            if not lines[i].strip().startswith('VERTICES'):
                raise ValueError(f"Ожидалась строка 'VERTICES', получено: {lines[i]}")
            i += 1

            vertices = []
            for _ in range(vertex_count):
                x_str, y_str = lines[i].strip().split()
                x, y = float(x_str), float(y_str)
                vertices.append([x, y])
                i += 1

            shape = np.array(vertices)
            for _ in range(quantity):
                items.append(shape)

        else:
            i += 1

    return items
