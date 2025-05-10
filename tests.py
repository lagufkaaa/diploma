import numpy as np
#git test comment 
from encoding import Encoding
from auxiliary import Auxiliary
from test import Test
from nfp import NFP
from model import parse, model_func

import shapely.geometry as geom
from shapely.geometry import Polygon

# m = 5

# poly = np.random.uniform(0, 10, (m, 2)) 
# poly[0][0] = 0

# Test.test_encoding(poly, n=25)

# n = 10
# h = 0.1

# poly1 = [
#     [0, 0],
#     [1, 0],
#     [1, 1],
#     [0, 1]
# ]

# P1 = Polygon(poly1)  # Правильный вызов конструктора

# polygon1 = Polygon([(0, 0), (0.5, 0), (0.5, 1)])
# polygon2 = Polygon([(0, 0), (1, 0), (1, 1), (3, 1), (3, 0), (4, 0), (4, 2), (0, 2)])
# anchor_point = (0, 0)

# assert isinstance(polygon1, geom.Polygon), f"polygon1 должен быть Polygon, но получен {type(polygon1).__name__}"
# assert isinstance(polygon2, geom.Polygon), f"polygon2 должен быть Polygon, но получен {type(polygon2).__name__}"

# onfp = NFP.outer_no_fit_polygon(polygon1, polygon2, anchor_point)

# Test.test_nfp(polygon1, polygon2, onfp, anchor_point)

# newonfp = NFP.polygon_to_path(onfp)

# Test.test_encoding(newonfp, n=25)

items = parse(".\\data_car_mats\\car_mats_10.txt")
# for item in items:
#     Test.test_vis(item)
Test.test_vis(items[0])

print(model_func(items, H = 10000, W = 10000, amount_rot = 1, n = 25))