# Список функций:
#     1. Визуализация просто многоугольника 
#     2. Визуализация НФП? 

import math as math
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
from itertools import product
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union

import shapely.geometry as geom

from ..core.nfp import NFP

class Test_Polygon:
    def __init__(self):
        pass

    def vis_polygon(P):
        polygon = np.vstack([P, P[0]])

        plt.plot(polygon[:, 0], polygon[:, 1], marker='o', linestyle='-')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Многоугольник из случайных точек")
        plt.grid(True)
        plt.show()
        
class Test_NFP:
    def __init__(self):
        pass

    def vis_nfp(polygon1, polygon2, anchor_point):
        onfp = NFP.outer_no_fit_polygon(polygon1, polygon2, anchor_point)
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
