import time

from shapely.geometry import Polygon
from src.tests.test_encoding import Test_Encoding
from src.tests.test_geometry import Test_Polygon, Test_NFP
from src.core.nfp import NFP
from src.utils.helpers import util_polygon
from src.utils.helpers import util_model
from src.tests.test_model import Test_Model
from src.core.model import Model

def testing_encoding(poly=None, n=None, h=None):
    if poly == None:
        arr1 = [(0, 0), (6, 0), (6, 5), (0, 5)]
        arr2 = ([(0, 0), (0.5, 0), (1, 1)])
        arr3 = ([(0, 0), (1, 0), (1, 1), (3, 1), (3, 0), (4, 0), (4, 2), (0, 2)])

        arr = arr1
        poly = Polygon(arr)

        Test_Polygon.vis_polygon(arr)
        
        # if h == None: h = 0.4
        # if n == None: n = 10

        Test_Encoding.vis_encoding(poly, n, h)
    else:
        arr, holes = util_polygon.polygon_to_path(poly)
        Test_Polygon.vis_polygon(arr)

        Test_Encoding.vis_encoding(poly, n, h)


def testing_nfp(arr1=None, arr2=None):
    if arr1 == None:     arr1 = ([(0, 0), (0.5, 0), (0.5, 1)])
    if arr2 == None:     arr2 = ([(0, 0), (1, 0), (1, 1), (3, 1), (3, 0), (4, 0), (4, 2), (0, 2)])
    
    anchor_point = arr1[0]
    
    Test_Polygon.vis_polygon(arr1)
    Test_Polygon.vis_polygon(arr2)
    
    poly1 = Polygon(arr1)
    poly2 = Polygon(arr2)
    
    Test_NFP.vis_nfp(poly1, poly2, anchor_point)

def testing_encoding_of_nfp(arr1=None, arr2=None, n=None, h=None):
    if arr1 == None:     arr1 = ([(0, 0), (0.5, 0), (0.5, 1)])
    if arr2 == None:     arr2 = ([(0, 0), (1, 0), (1, 1), (3, 1), (3, 0), (4, 0), (4, 2), (0, 2)])
    
    anchor_point = arr1[0]
    
    Test_Polygon.vis_polygon(arr1)
    Test_Polygon.vis_polygon(arr2)
    
    poly1 = Polygon(arr1)
    poly2 = Polygon(arr2)
    
    onfp = NFP.outer_no_fit_polygon(poly1, poly2, anchor_point)
    
    testing_encoding(onfp, n, h)
    
def testing_model(items, W, H, R, N, S):
    Test_Model.vis_simple_model(items, W, H, R, N, S)

# testing_encoding(h = 0.4)
# testing_nfp()

# testing_encoding_of_nfp(10)

items = util_model.parse_items(".\\data_car_mats\\test.txt")
# items = util_model.parse_items(".\\data_car_mats\\car_mats_1.txt")
# for item in items:
#     Test_Polygon.vis_polygon(item)
# Test_Polygon.vis_polygon(items[0])

mdl_items = items

# mdl_items = []
# for i in range( len(items)//5):
#     mdl_items.append(items[5*i])

# mdl_items = mdl_items[:2]
# for item in mdl_items:
#     Test_Polygon.vis_polygon(item)

N = len(mdl_items)
W=1000
H=1000
R=1
S=20

start_time = time.time()
result = Model.model_func(mdl_items, W, H, R, N, S)
end_time = time.time()
print(f"Время выполнения: {end_time - start_time:.2f} секунд")
print(result)

Test_Model.vis_simple_model(mdl_items, W, H, R, N, S, result)
