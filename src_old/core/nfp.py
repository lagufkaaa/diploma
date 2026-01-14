import math as math
import numpy as np
import pyclipper
from itertools import product
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union

from ..utils.helpers import util_NFP

class NFP:
    def __init__(self):
        pass
    
    SCALE = 1e6
    
    def minkowski_difference(polygon1, polygon2, anchor_point=(0, 0)):
        if not isinstance(polygon1, Polygon):
            raise TypeError(f"[minkowski_difference] polygon1 должен быть Polygon, получен {type(polygon1).__name__}")
        if not isinstance(polygon2, Polygon):
            raise TypeError(f"[minkowski_difference] polygon2 должен быть Polygon, получен {type(polygon2).__name__}")

        outer1, _ = util_NFP.polygon_to_path_pycl(polygon1)
        outer2, _ = util_NFP.polygon_to_path_pycl(polygon2)

        ax, ay = int(anchor_point[0] * NFP.SCALE), int(anchor_point[1] * NFP.SCALE)
        inv_outer1 = [(-x + ax, -y + ay) for x, y in outer1]

        result = pyclipper.MinkowskiSum(outer2, inv_outer1, True)

        polygons = [Polygon([(x / NFP.SCALE, y / NFP.SCALE) for x, y in path]) for path in result]
        polygons = [p.buffer(0) for p in polygons if p.is_valid and not p.is_empty]

        if not polygons:
            return Polygon()

        return unary_union(polygons)
    
    def outer_no_fit_polygon(polygon1, polygon2, anchor_point=(0, 0)):
        outer1, _ = util_NFP.polygon_to_path_pycl(polygon1)
        anchor_point = outer1[0]
        if not isinstance(polygon1, Polygon):
            raise TypeError(f"[outer_no_fit_polygon] polygon1 должен быть Polygon, получен {type(polygon1).__name__}")
        if not isinstance(polygon2, Polygon):
            raise TypeError(f"[outer_no_fit_polygon] polygon2 должен быть Polygon, получен {type(polygon2).__name__}")

        minkowski = NFP.minkowski_difference(polygon1, polygon2, anchor_point)
        base_polygon = polygon2.buffer(0)

        poly = unary_union([base_polygon, minkowski]).buffer(0)
        if getattr(poly, 'geom_type', None) == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda p: p.area)
        if not isinstance(poly, Polygon):
            poly = Polygon()
        return poly

