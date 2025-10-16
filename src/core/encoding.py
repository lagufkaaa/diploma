import math as math
import numpy as np
import shapely.geometry as geom

from ..utils.helpers import util_encoding as ue
from ..utils.helpers import util_polygon as up


class Encoding:
    def __init__(self):
        pass
    
    def seg_y(T, poly):
        exterior_path, interior_path = up.polygon_to_path(poly)
        seg = []

        for t1 in T:
            for t2 in T:
                # работаем ТОЛЬКО с парами на одной горизонтали
                if t1[1] == t2[1]:
                    # есть ли точки T между t1 и t2 на той же горизонтали?
                    between_same_row = [
                        ((t1[0] < p[0] < t2[0]) or (t2[0] < p[0] < t1[0])) and (p[1] == t1[1])
                        for p in T
                    ]

                    # если промежуточных точек нет — это кандидат на сегмент
                    if between_same_row.count(True) < 1:
                        y = t1[1]
                        px_min, px_max = min(t1[0], t2[0]), max(t1[0], t2[0])

                        # точка-середина для проверки принадлежности полигону
                        point = np.zeros(2)
                        point[0] = (px_min + px_max) / 2.0
                        point[1] = y

                        # если середина внутри полигона — добавляем горизонтальный сегмент
                        if ue.is_inside(point, poly):
                            seg.append([[px_min, y], [px_max, y]])

                        # если середина совпадает с вершиной контура — добавить вырожденный сегмент
                        paths = [exterior_path] + (interior_path or [])  # interior_path: список дыр или []
                        if any(np.all(point == p) for path in paths for p in path):
                            seg.append([point, point])

        return seg

    
    def encode_polygon(poly, S, H):
        if S <= 0: 
            raise ValueError("количество строк не может быть меньше 1")
        
        h = H/S
        Y = [h*i for i in range(S)]
        
        pass