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
        temp = []
        for y0 in Y:
            for i in range(exterior_path):
                p1, p2 = exterior_path[i], exterior_path[(i + 1) % len(exterior_path)]
                point = util_encoding.intersect_rows(y0, p1, p2)
                if point and point not in temp:
                    temp.append(point)
                
            for i in range(interior_path):
                p1, p2 = interior_path[i], interior_path[(i + 1) % len(interior_path)]
                point = util_encoding.intersect_rows(y0, p1, p2)
                if point and point not in temp:
                    temp.append(point)
                
        return temp