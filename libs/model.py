import numpy as np
import ortools
from ortools.linear_solver import pywraplp
from shapely.geometry import Polygon
from libs.nfp import NFP
from libs.auxiliary import Auxiliary
from libs.encoding import Encoding
from libs.test import Test

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

def model_func(items, W, H, R, N, S):
    h = H/S
    M = 1e9
    solver = pywraplp.Solver.CreateSolver('SCIP')

    p = [[solver.BoolVar(f'p_{n}_{r}') for r in range(R)] for n in range(N)]

    for n in range(N):
        solver.Add(sum(p[n][r] for r in range(R)) <= 1)

    deltas = [[[solver.BoolVar(f'deltas_{s}_{n}_{r}') for r in range(R)] for n in range(N)] for s in range(S)]

    for n in range(N):
        solver.Add(sum(sum(deltas[s][n][r] for s in range(S)) for r in range(R)) <= 1)

    str_vars = [[solver.IntVar(0, S, f's_{r}_{n}') for n in range(N)] for r in range(R)]
    for n in range(N):
        for r in range(R):
            # str_vars[r][n] должно равняться s, для которого deltas[s][n][r] = 1
            # Создаем выражение: str_vars[r][n] = sum(s * deltas[s][n][r] for s in range(S))
            solver.Add(str_vars[r][n] == sum(s * deltas[s][n][r] for s in range(S)))


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
            bbox = Auxiliary.find_bounding_box_numpy(normalized_item)
            
            # Извлекаем значения из словаря
            xmin = bbox['min_x']
            xmax = bbox['max_x'] 
            ymin = bbox['min_y']
            ymax = bbox['max_y']
            
            x_min[i].append(xmin)
            x_max[i].append(xmax)
            y_min[i].append(ymin)
            y_max[i].append(ymax)
        
        # TODO если item[i] = item[j], то тогда просто дублировать из таблицы а не просчитывать по новой

    for i in range(N):
        for r in range(R):
            solver.Add(x[i][r] >= -x_min[i][r])
            solver.Add(x[i][r] + x_max[i][r] <= (W + (1 - p[i][r])*M))

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
                        NFPs[i][r1][j][r2] = onfp
                        Enc[i][r1][j][r2] = Encoding.cod_model(H, S, onfp)

                        print(Enc[i][r1][j][r2])
                        
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
                        
                        if Enc[i][r1][j][r2][0] != []:
                            C[i][r1][j][r2] = round(sum(1 for point in Enc[i][r1][j][r2][0][0] if point[1] == target_value) / 2 
                                                    + sum(1 for point in Enc[i][r1][j][r2][1] if point[1] == target_value)
                                                    )
                            
                            for seg in Enc[i][r1][j][r2][0][0]:
                                if seg[1] == target_value: 
                                    a[i][r1][j][r2].append(seg[0])
                                    b[i][r1][j][r2].append(seg[1])
                            
                            for point in Enc[i][r1][j][r2][1]:
                                if point[1] == target_value:
                                    a[i][r1][j][r2].append(point[0])
                                    b[i][r1][j][r2].append(point[1]) 

                        else:
                            C[i][r1][j][r2] = 0
                        
                        # print(C[i][r1][j][r2])
                        

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
                        solver.Add(x[i][r1] - x[j][r2] >= -M * (1 - gammas[i][r1][j][r2][c]))

                        epsilon = 1e-5
                        solver.Add(x[j][r2] - x[i][r1] + epsilon <= M * gammas[i][r1][j][r2][c])

                    delta_i_active = sum(deltas[s][i][r1] for s in range(S))
                    delta_j_active = sum(deltas[s][j][r2] for s in range(S))
                    
                    solver.Add(x[i][r1] <= x[j][r2] - b[i][r1][j][r2][c] + 
                              gammas[i][r1][j][r2][c] * M + 
                              (1 - delta_i_active) * M + 
                              (1 - delta_j_active) * M + 
                              (1 - p[i][r1]) * M + 
                              (1 - p[j][r2]) * M)
                    
                    solver.Add(x[i][r1] >= x[j][r2] - a[i][r1][j][r2][c] - 
                              (1 - gammas[i][r1][j][r2][c]) * M - 
                              (1 - delta_i_active) * M - 
                              (1 - delta_j_active) * M - 
                              (1 - p[i][r1]) * M - 
                              (1 - p[j][r2]) * M)


    Area = [Polygon(item).area for item in items]

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

            # Нормализация по первой точке
            if len(vertices) > 0:
                first_point = vertices[0].copy()  # Сохраняем первую точку
                normalized_vertices = []
                for vertex in vertices:
                    # Вычитаем координаты первой точки из всех вершин
                    normalized_vertex = [vertex[0] - first_point[0], vertex[1] - first_point[1]]
                    normalized_vertices.append(normalized_vertex)
                
                shape = np.array(normalized_vertices)
                for _ in range(quantity):
                    items.append(shape)

        else:
            i += 1

    return items


def visualize_solution(solution, items, H, S, W=200000, all_items=None):
    """
    Визуализирует решение упаковки фигур
    
    Args:
        solution: словарь с решением {'p', 'x', 's', ...} или {'status': 'NOT_OPTIMAL'}
        items: список фигур, которые пытались упаковать
        H: общая высота области
        S: количество строчек
        W: ширина области
        all_items: все доступные фигуры (опционально)
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Проверяем, есть ли валидное решение
    if solution.get('status') in ['NOT_OPTIMAL', 'INFEASIBLE'] or solution.get('objective_value') is None:
        ax.text(0.5, 0.5, f'Решение не найдено\nСтатус: {solution.get("status", "UNKNOWN")}', 
                transform=ax.transAxes, fontsize=16, ha='center', va='center',
                bbox=dict(facecolor='red', alpha=0.3))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_title('Решение не найдено')
        plt.tight_layout()
        plt.show()
        return
    
    # Вычисляем высоту одной строчки
    h = H / S
    
    # Определяем, какие фигуры были упакованы
    packed_items = items  # фигуры, которые были переданы на упаковку
    all_available_items = all_items if all_items is not None else items
    
    print(f"Упаковано фигур: {len([p for p in solution['p'] if p[0] == 1.0])} из {len(all_available_items)} доступных")
    print(f"Параметры: H={H}, S={S}, h={h:.1f}, W={W}")
    
    # Создаем коллекции для упакованных и неупакованных фигур
    packed_patches = []
    unpacked_patches = []
    packed_colors = []
    unpacked_colors = []
    
    # Визуализируем упакованные фигуры
    for i, item in enumerate(packed_items):
        if i >= len(solution['x']) or i >= len(solution['s']) or i >= len(solution['p']):
            continue
            
        # Проверяем, был ли предмет упакован (p[i][0] == 1.0)
        if solution['p'][i][0] == 1.0:
            x_pos = solution['x'][i][0]
            s_val = int(round(solution['s'][i][0]))
            y_pos = s_val * h
            
            # Создаем полигон с учетом смещения
            polygon_points = []
            for point in item:
                px = point[0] + x_pos
                py = point[1] + y_pos
                polygon_points.append([px, py])
            
            polygon = patches.Polygon(polygon_points, closed=True, alpha=0.8)
            packed_patches.append(polygon)
            packed_colors.append(i)
            
            # Подписываем упакованные фигуры
            ax.text(x_pos, y_pos + h/2, f'{i}', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.8))
        else:
            # Фигура не упакована - добавляем в список неупакованных
            x_pos = W * 1.1  # справа от рабочей области
            y_pos = (len(unpacked_patches) % 10) * h * 2  # вертикальное расположение
            
            polygon_points = []
            for point in item:
                px = point[0] + x_pos
                py = point[1] + y_pos
                polygon_points.append([px, py])
            
            polygon = patches.Polygon(polygon_points, closed=True, alpha=0.3, 
                                     edgecolor='red', facecolor='lightgray')
            unpacked_patches.append(polygon)
            unpacked_colors.append(i)
            
            ax.text(x_pos, y_pos + h/2, f'{i} (не упак.)', fontweight='bold',
                   color='red', bbox=dict(facecolor='white', alpha=0.8))
    
    # Визуализируем оставшиеся неупакованные фигуры (если переданы все доступные)
    if all_available_items is not None and len(all_available_items) > len(packed_items):
        for i in range(len(packed_items), len(all_available_items)):
            item = all_available_items[i]
            x_pos = W * 1.1  # справа от рабочей области
            y_pos = (len(unpacked_patches) % 10) * h * 2  # вертикальное расположение
            
            polygon_points = []
            for point in item:
                px = point[0] + x_pos
                py = point[1] + y_pos
                polygon_points.append([px, py])
            
            polygon = patches.Polygon(polygon_points, closed=True, alpha=0.3, 
                                     edgecolor='red', facecolor='lightgray')
            unpacked_patches.append(polygon)
            unpacked_colors.append(i)
            
            ax.text(x_pos, y_pos + h/2, f'{i} (не упак.)', fontweight='bold',
                   color='red', bbox=dict(facecolor='white', alpha=0.8))
    
    # Добавляем упакованные фигуры
    if packed_patches:
        p_packed = PatchCollection(packed_patches, cmap='tab10', alpha=0.7)
        p_packed.set_array(np.array(packed_colors) % 10)  # ограничиваем цвета
        ax.add_collection(p_packed)
    
    # Добавляем неупакованные фигуры
    if unpacked_patches:
        p_unpacked = PatchCollection(unpacked_patches, alpha=0.3, 
                                   edgecolor='red', facecolor='lightgray')
        ax.add_collection(p_unpacked)
    
    # Рисуем сетку строчек
    for s in range(0, S + 1, max(1, S // 20)):
        y_line = s * h
        ax.axhline(y=y_line, color='gray', linestyle='--', alpha=0.5)
        ax.text(W * 1.01, y_line + h/2, f'{s}', va='center', fontsize=8)
    
    # Настраиваем внешний вид
    x_limit = W * 1.2 if unpacked_patches else W * 1.05
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, H)
    ax.set_xlabel('X координата')
    ax.set_ylabel('Y координата')
    
    packed_count = len(packed_patches)
    total_count = len(all_available_items)
    title = f'Упаковка: {packed_count} из {total_count} фигур'
    if unpacked_patches:
        title += f' ({len(unpacked_patches)} не упаковано)'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', marker='s', linestyle='None',
              markersize=10, label='Упакованные фигуры'),
        Line2D([0], [0], color='red', marker='s', linestyle='None',
              markersize=10, label='Не упакованные фигуры', alpha=0.5)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Информация
    objective_value = solution.get('objective_value', 0)
    efficiency = (objective_value / (W * H)) * 100 if W * H > 0 else 0
    
    info_text = f'''Параметры упаковки:
Область: {W} × {H}
Строчек: {S} (высота: {h:.1f})
Упаковано: {packed_count} фигур
Площадь: {objective_value:.0f}
Эффективность: {efficiency:.1f}%'''
    
    ax.text(W * 1.01, H * 0.7, info_text, fontsize=9, 
            bbox=dict(facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()