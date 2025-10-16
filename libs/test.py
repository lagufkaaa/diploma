import math as math
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
from itertools import product
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union

import shapely.geometry as geom

from libs.auxiliary import Auxiliary
from libs.encoding import Encoding

from libs.nfp import NFP

class Test:
    def __init__(self):
        pass

    def test_vis(P):
        polygon = np.vstack([P, P[0]])

        plt.plot(polygon[:, 0], polygon[:, 1], marker='o', linestyle='-')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Многоугольник из случайных точек")
        plt.grid(True)
        plt.show()

    def test_encoding(P, n=None, h=None):
        polygon = np.vstack([P, P[0]])

        plt.plot(polygon[:, 0], polygon[:, 1], marker='o', linestyle='-')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Многоугольник из случайных точек")
        plt.grid(True)
        plt.show()

        x0 = min([P[i][0] for i in range(len(P))])
        # x0 = -1
        # xn = max([P[i][0] for i in range(len(P))])
        # xn = 10
        xn = max([P[i][0] for i in range(len(P))])
        # print("xn =",  xn)
        if n:
            h = (xn - x0) / n
        else:
            n = int((xn - x0) / h)

        X = [(x0 + h * i) for i in range(n + 1)]

        T = Auxiliary.inter_T(X, P)

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

        cod_seg, cod_points = Encoding.cod(X, P)

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
                plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='red', linestyle='-',
                         linewidth=2, alpha=0.7)

        #   print(proj)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Многоугольник с точками пересечения")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def test_model_encoding(onfp, n=None, h=None):
        # assert isinstance(onfp, geom.Polygon), f"polygon1 должен быть Polygon, но получен {type(onfp).__name__}"
        # P = NFP.polygon_to_path(onfp)
        print('t-1', type(onfp))
        print('t0', type(P))
        
        P = onfp
        print('t1', type(onfp))
        print('t2', type(P))
        polygon = np.vstack([P, P[0]])

        plt.plot(polygon[:, 0], polygon[:, 1], marker='o', linestyle='-')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Изначальный нфп")
        plt.grid(True)
        plt.show()
        
        

        x0 = min([P[i][0] for i in range(len(P))])
        # x0 = -1
        # xn = max([P[i][0] for i in range(len(P))])
        # xn = 10
        xn = max([P[i][0] for i in range(len(P))])
        # print("xn =",  xn)
        if n:
            h = (xn - x0) / n
        else:
            n = int((xn - x0) / h)

        X = [(x0 + h * i) for i in range(n + 1)]

        T = Auxiliary.inter_T(X, P)

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
        
        print('t3', type(onfp))

        cod_seg, cod_points = Encoding.cod_model(xn, n, P)

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
                plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='red', linestyle='-',
                        linewidth=2, alpha=0.7)

        #   print(proj)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Нарезка нфп")
        plt.legend()
        plt.grid(True)
        plt.show()


    def test_nfp(polygon1, polygon2, onfp, anchor_point):
        fig, ax = plt.subplots()

        x1, y1 = polygon1.exterior.xy
        ax.fill(x1, y1, alpha=0.5, fc='blue', label='Polygon 1')

        x2, y2 = polygon2.exterior.xy
        ax.fill(x2, y2, alpha=0.5, fc='red', label='Polygon 2')

        if isinstance(onfp, MultiPolygon):
            for poly in onfp.geoms:
                x_onfp, y_onfp = poly.exterior.xy
                ax.plot(x_onfp, y_onfp, color='gray', linewidth=2, label='ONFP')
        else:
            x_onfp, y_onfp = onfp.exterior.xy
            ax.plot(x_onfp, y_onfp, color='gray', linewidth=2, label='ONFP')

        ax.scatter(*anchor_point, color='black', marker='x', s=100, label='Anchor Point')

        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Outer No-Fit Polygon Visualization")
        plt.grid()
        plt.show()
