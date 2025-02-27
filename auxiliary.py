import numpy as np
import matplotlib.pyplot as plt
import math as math

from __polygon import polygon
from encoding import encoding

eps = 1e-10

class auxiliary:
    def __init__(self, polygon):
        self.polygon = polygon
    
    @staticmethod
    def round_up_if_needed(x):
        return x if x == int(x) else math.ceil(x)
    
    def seg_x(self, X):
        P = self.polygon.points
        T = polygon.inter_T(self.polygon, X)

        seg = []
        points = []

        for t1 in T:
            for t2 in T:
                temp = [((t1[1] < p[1] < t2[1]) or (t2[1] < p[1] < t1[1])) and (t1[0] == p[0]) and (t2[0] == p[0]) for p in T]
                temp_filtered = [p if cond else False for p, cond in zip(T, temp)]
                # if t1[0] == t2[0] and t1[1] != t2[1]:
                #   print(t1, t2, temp_filtered)
                if t1[0] == t2[0] and temp.count(True) < 1:
                    poi = np.zeros(2)
                    poi[0] = t1[0]
                    py_min, py_max = min(t1[1], t2[1]), max(t1[1], t2[1])
                    poi[1] = (py_max + py_min) / 2

                    if polygon.ins(poi, P):
                        seg.append([[t1[0], py_min], [t1[0], py_max]])


                    # print(poi)
                    # print(P)
                    if any(np.all(poi == p) for p in P):
                        # print(poi)
                        # print(P)
                        points.append(poi)

        return seg, points
    
    def test(self, n = None, h = None):
        P = self.polygon.points
        polygon = np.vstack([P, P[0]])

        plt.plot(polygon[:, 0], polygon[:, 1], marker='o', linestyle='-')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Многоугольник из случайных точек")
        plt.grid(True)
        plt.show()

        x0 = 0
        # xn = max([P[i][0] for i in range(len(P))])
        # xn = 10
        xn = self.round_up_if_needed(max([P[i][0] for i in range(len(P))]))
        # print("xn =",  xn)
        if n:
            h = (xn - x0)/n
        else:
            n = int((xn - x0)/h)

        X = [(x0 + h*i) for i in range(n + 1)]

        T = polygon.inter_T(P, X)

        # # Отрисовка многоугольника
        # plt.plot(polygon[:, 0], polygon[:, 1], color='gray', marker='o', linestyle='-', alpha=1, zorder=1)

        # # Отрисовка вертикальной линии x = x1
        # for x1 in X:
        #   plt.axvline(x=x1, color='red', linestyle='--', alpha=0.3, zorder=2)

        # # Отрисовка точек пересечения поверх графика с увеличенным радиусом и полупрозрачностью
        # if T:
        #     T = np.array(T)
        #     plt.scatter(T[:, 0], T[:, 1], color='red', s=60, alpha=0.5, zorder=3)

        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.title("Многоугольник с точками пересечения")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # # print(P)

        # seg, points = seg_x(T, P)


        # if seg:
        #   for segment in seg:
        #       plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='blue', linestyle='-', linewidth=2, alpha=0.7)
        #   for point in points:
        #     plt.scatter(point[0], point[1], color='blue', alpha=0.7, s=60)
        #   plt.xlabel("X")
        #   plt.ylabel("Y")
        #   plt.title("Отрезки на основе точек пересечения")
        #   plt.grid(True)
        #   plt.show()

        # proj = projections(X, P)

        # # Отрисовка многоугольника
        # plt.plot(polygon[:, 0], polygon[:, 1], color='gray', marker='o', linestyle='-', alpha=1, zorder=1)

        # # Отрисовка вертикальной линии x = x1
        # for x1 in X:
        #   plt.axvline(x=x1, color='red', linestyle='--', alpha=0.3, zorder=2)

        # if proj:
        #     proj = np.array(proj)
        #     plt.scatter(proj[:, 0], proj[:, 1], color='red', s=60, alpha=0.5, zorder=3)

        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.title("Многоугольник с точками пересечения")
        # plt.legend()
        # plt.grid(True)
        # plt.show()


        cod_seg, cod_points = encoding.cod(X, P)

        # Отрисовка многоугольника
        plt.plot(polygon[:, 0], polygon[:, 1], color='gray', marker='o', linestyle='-', alpha=1, zorder=1)

        # Отрисовка вертикальной линии x = x1
        for x1 in X:
            plt.axvline(x=x1, color='red', linestyle='--', alpha=0.3, zorder=2)

        # Отрисовка точек пересечения поверх графика с увеличенным радиусом и полупрозрачностью
        if cod_points:
            cod_points = np.array(cod_points)
            plt.scatter(cod_points[:, 0], cod_points[:, 1], color='red', s=60, alpha=0.5, zorder=3)

        if cod_seg:
            for segment in cod_seg:
                plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='red', linestyle='-', linewidth=2, alpha=0.7)

        #   print(proj)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Многоугольник с точками пересечения")
        plt.legend()
        plt.grid(True)
        plt.show()
