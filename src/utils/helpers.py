# Список функций:
#     1. парсер данных
#     2. round_up_if_needed
#     3. polygon_to_path
import math as math
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPolygon

class util_polygon:
    def __init__(self):
        pass        
    
    def polygon_to_path(polygon):
        """Преобразует shapely Polygon в numpy массив точек (внешний контур + отверстия при необходимости)."""
        if not isinstance(polygon, Polygon):
            raise TypeError(f"[polygon_to_path] Ожидался Polygon, получен: {type(polygon).__name__}")

        # Внешний контур (без повторной конечной точки)
        exterior = np.array(polygon.exterior.coords[:-1], dtype=np.float64)

        # Если нужно также включать отверстия:
        interiors = [np.array(interior.coords[:-1], dtype=np.float64) for interior in polygon.interiors]

        return exterior, interiors
    
class util_encoding:
    def __init__(self):
        pass        

    def is_inside(point, poly):
        return poly.contains(Point(point)) or poly.touches(Point(point))
    
    def intersect_rows(y0, point1, point2): 
        y_min, y_max = min(point1[1], point2[1]), max(point1[1], point2[1])
        
        if y0 < y_min or y0 > y_max:
            return None
        
        if point1[1] == point2[1]:
            return None
        
        x = point1[0] + (y0 - point1[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])

        if y0 == point1[1]:
            return (point1[0], point1[1])
        if y0 == point2[1]:
            return (point2[0], point2[1])

        return (x, y0)
    
    def get_intersections_rows(Y, poly):
        exterior_path, interior_path = util_polygon.polygon_to_path(poly)
        # print(exterior_path)
        temp = []
        for y0 in Y:
            for i in range(len(exterior_path)):
                p1, p2 = exterior_path[i], exterior_path[(i + 1) % len(exterior_path)]
                point = util_encoding.intersect_rows(y0, p1, p2)
                if point and point not in temp:
                    temp.append(point)
                
            for i in range(len(interior_path)):
                p1, p2 = interior_path[i], interior_path[(i + 1) % len(interior_path)]
                point = util_encoding.intersect_rows(y0, p1, p2)
                if point and point not in temp:
                    temp.append(point)
                
        return temp
    
class util_NFP:
    def __init__(self):
        pass
    
    SCALE = 1e6
    
    def polygon_to_path_pycl(polygon):
        """Преобразует shapely Polygon в формат, подходящий для pyclipper (целочисленные координаты)."""
        if not isinstance(polygon, Polygon):
            raise TypeError(f"[polygon_to_path] Ожидался Polygon, получен: {type(polygon).__name__}")

        def to_int_coords(ring):
            return [(int(x * util_NFP.SCALE), int(y * util_NFP.SCALE)) for x, y in ring]

        outer = to_int_coords(polygon.exterior.coords[:-1])
        holes = [to_int_coords(interior.coords[:-1]) for interior in polygon.interiors]
        return outer, holes

class util_model:
    def __init__(self):
        pass

    def find_bounding_box_numpy(points):
        points_array = np.array(points)
        xs = points_array[:, 0]
        ys = points_array[:, 1]
        
        return {
            'min_x': np.min(xs),
            'max_x': np.max(xs),
            'min_y': np.min(ys),
            'max_y': np.max(ys),
        }
        
    def parse_items(file_path):
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

