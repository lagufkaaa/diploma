import numpy as np
import ortools
from ortools.linear_solver import pywraplp
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
from ..core.model import Model

class Test_Model:
    def __init__(self):
        pass
        
    def vis_simple_model(items, W, H, R, N, S, solution, all_items=None):
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
        
        # Высота одной строчки
        h = H / S
        
        packed_items = items
        all_available_items = all_items if all_items is not None else items
        
        print(f"Упаковано фигур: {len([p for p in solution['p'] if p[0] == 1.0])} из {len(all_available_items)} доступных")
        print(f"Параметры: H={H}, S={S}, h={h:.1f}, W={W}")
        
        packed_patches = []
        unpacked_patches = []
        packed_colors = []
        unpacked_colors = []
        
        # ------ нормализация по первой точке полигона ------
        def normalize_item_polygon(item_points):
            if item_points is None:
                return []

            # Превращаем в список Python, если это numpy-массив
            if isinstance(item_points, np.ndarray):
                item_points = item_points.tolist()

            if len(item_points) == 0:
                return []

            x0, y0 = item_points[0]
            return [(p[0] - x0, p[1] - y0) for p in item_points]
        # ---------------------------------------------------
        
        # Визуализируем упакованные фигуры
        for i, item in enumerate(packed_items):
            if i >= len(solution['x']) or i >= len(solution['s']) or i >= len(solution['p']):
                continue
                
            if solution['p'][i][0] == 1.0:
                x_pos = solution['x'][i][0]
                s_val = int(round(solution['s'][i][0]))
                y_pos = s_val * h

                # Нормализуем локальные координаты фигуры по первой точке
                norm_item = normalize_item_polygon(item)
                
                polygon_points = []
                for px_local, py_local in norm_item:
                    px = px_local + x_pos
                    py = py_local + y_pos
                    polygon_points.append([px, py])
                
                polygon = patches.Polygon(polygon_points, closed=True, alpha=0.8)
                packed_patches.append(polygon)
                packed_colors.append(i)
                
                ax.text(x_pos, y_pos + h/2, f'{i}', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8))
            else:
                # Не упакована — рисуем справа
                x_pos = W * 1.1
                y_pos = (len(unpacked_patches) % 10) * h * 2
                
                norm_item = normalize_item_polygon(item)
                
                polygon_points = []
                for px_local, py_local in norm_item:
                    px = px_local + x_pos
                    py = py_local + y_pos
                    polygon_points.append([px, py])
                
                polygon = patches.Polygon(polygon_points, closed=True, alpha=0.3, 
                                        edgecolor='red', facecolor='lightgray')
                unpacked_patches.append(polygon)
                unpacked_colors.append(i)
                
                ax.text(x_pos, y_pos + h/2, f'{i} (не упак.)', fontweight='bold',
                        color='red', bbox=dict(facecolor='white', alpha=0.8))
        
        # Оставшиеся неупакованные (если есть all_items)
        if all_available_items is not None and len(all_available_items) > len(packed_items):
            for i in range(len(packed_items), len(all_available_items)):
                item = all_available_items[i]
                x_pos = W * 1.1
                y_pos = (len(unpacked_patches) % 10) * h * 2
                
                norm_item = normalize_item_polygon(item)
                
                polygon_points = []
                for px_local, py_local in norm_item:
                    px = px_local + x_pos
                    py = py_local + y_pos
                    polygon_points.append([px, py])
                
                polygon = patches.Polygon(polygon_points, closed=True, alpha=0.3, 
                                        edgecolor='red', facecolor='lightgray')
                unpacked_patches.append(polygon)
                unpacked_colors.append(i)
                
                ax.text(x_pos, y_pos + h/2, f'{i} (не упак.)', fontweight='bold',
                        color='red', bbox=dict(facecolor='white', alpha=0.8))
        
        # Добавляем коллекции
        if packed_patches:
            p_packed = PatchCollection(packed_patches, cmap='tab10', alpha=0.7)
            p_packed.set_array(np.array(packed_colors) % 10)
            ax.add_collection(p_packed)
        
        if unpacked_patches:
            p_unpacked = PatchCollection(unpacked_patches, alpha=0.3, 
                                        edgecolor='red', facecolor='lightgray')
            ax.add_collection(p_unpacked)
        
        # Сетка строчек
        for s in range(0, S + 1, max(1, S // 20)):
            y_line = s * h
            ax.axhline(y=y_line, color='gray', linestyle='--', alpha=0.5)
            ax.text(W * 1.01, y_line + h/2, f'{s}', va='center', fontsize=8)
        
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
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', marker='s', linestyle='None',
                markersize=10, label='Упакованные фигуры'),
            Line2D([0], [0], color='red', marker='s', linestyle='None',
                markersize=10, label='Не упакованные фигуры', alpha=0.5)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
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
