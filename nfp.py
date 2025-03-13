import numpy as np
import math as math

from polygon import polygon

class nfp:
    def __init__(self, polygon1, polygon2):
        self.polygon = polygon1
        self.polygon = polygon2


    @staticmethod
    def orientation(p, q, r):
        """Определяет ориентацию трёх точек (p, q, r).
        Возвращает:
        0 -> точки коллинеарны
        1 -> поворот по часовой стрелке
        2 -> поворот против часовой стрелки
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if np.isclose(val, 0):  
            return 0  # Коллинеарны
        return 1 if val > 0 else 2  # 1 - по часовой, 2 - против часовой

    @staticmethod
    def on_segment(p, q, r):
        """Проверяет, лежит ли точка q на отрезке pr"""
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    
    def inter_bool(self, v1, v2):
        """Определяет, пересекаются ли два отрезка v1 и v2"""
        p1, q1 = np.array(v1[0]), np.array(v1[1])
        p2, q2 = np.array(v2[0]), np.array(v2[1])

        # Вычисляем ориентации
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        # Основной случай: отрезки пересекаются в общем случае
        if o1 != o2 and o3 != o4:
            return True

        # Специальные случаи: проверяем коллинеарные точки
        if o1 == 0 and self.on_segment(p1, p2, q1): return True
        if o2 == 0 and self.on_segment(p1, q2, q1): return True
        if o3 == 0 and self.on_segment(p2, p1, q2): return True
        if o4 == 0 and self.on_segment(p2, q1, q2): return True

        return False  # Не пересекаются
    
    def nfp(self, poly0, poly):
        nofit = []
        main_p = poly[0]

        for poi in poly0:
            new_poly = []
            fl = 0
            for p in poly:
                new_poly.append([p[0] + poi[0], p[1] + poi[1]])
            
            fl2 = 0
            for i in range(len(new_poly)):
                for j in range(len(poly0)):
                    if (not self.inter_bool([new_poly[i], new_poly[(i + 1)%len(poly)]], [poly0[j], poly0[(j + 1)%len(poly0)]])):
                        fl2 = 1

            if fl2 == 0:
                nofit.append([main_p[0] + poi[0], main_p[1] + poi[1]])

