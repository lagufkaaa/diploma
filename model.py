import ortools
from ortools.linear_solver import pywraplp
import numpy as np
from nfp import NFP
from shapely.geometry import Polygon
from encoding import Encoding

def model_func(items, H, W, amount_rot, n):
    solver = pywraplp.Solver.CreateSolver('GLOP')  # Или 'SCIP' если нужны булевы/целые переменные

    if not solver:
        raise RuntimeError("Решатель не создан — проверь, установлен ли OR-Tools и выбрано ли правильное имя решателя.")

    h = W / n
    X_vals = [(h * i) for i in range(n + 1)]

    amount_items = len(items)
    M = 1e9 

    X = [[solver.NumVar(0.0, W, f'X_{i}_{j}') for j in range(amount_rot)] for i in range(amount_items)]

    used = [[solver.BoolVar(f'used_{i}_{j}') for j in range(amount_rot)] for i in range(amount_items)]

    for i in range(amount_items):
        solver.Add(solver.Sum(used[i]) == 1)

    areas = [item.area for item in items]
    total_value = solver.Sum(used[i][j] * areas[i] for i in range(amount_items) for j in range(amount_rot))
    solver.Maximize(total_value)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Найдено оптимальное решение")
        for i in range(amount_items):
            for j in range(amount_rot):
                if used[i][j].solution_value() > 0.5:
                    print(f"Item {i} uses rotation {j}, X = {X[i][j].solution_value():.2f}")
    else:
        print("Решение не найдено")

    return solver

def parse(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    items = []

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('PIECE'):
            i += 1  # Пропускаем 'PIECE'

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
