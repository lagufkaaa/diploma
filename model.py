import ortools
import numpy as np
from nfp import NFP
from shapely.geometry import Polygon
from encoding import Encoding


# items - np массивы точек многоугольников 
def model(items, H, W, amount_rot, n):
    h = (W)/n
    X = [(h*i) for i in range(n + 1)]

    ONFP = []
    for i in amount_items:
        ONFP.append([])
        for j in amount_items:
            if i < j:
                item1 = Polygon(items[i])
                item2 = Polygon(items[j])

            anchor_point = item1[0]

            ONFP[i].append(NFP.outer_no_fit_polygon(item1, item2, anchor_point))
            
    ONFP_COD = [Encoding.cod(X, NFP.polygon_to_path(onfp)) for onfp in ONFP]
    amount_items = len(items)

    M = 99999 
    # M = np.inf

    Areas = [i.area for i in items]

    used = np.zeros(amount_items, amount_rot)
    for i in range(amount_items):
        for j in range(amount_rot):
            used[i][j] = model.NewBoolVar(f'used_{i} rotation_{j}')

    X = np.zeros(amount_items, amount_rot)
    for i in range(amount_items):
        for j in range(amount_rot):
            X[i][j] = model.NewBoolVar(f'coords_{i} rotation_{j}')


    for i in range(amount_items):
        model.Add(sum(used[i]) <= 1)
        model.Add(sum(used[i]) >= 1)

    items_enc = []
    values = [Areas[i] * used[i] for i in range(amount_items)]
    model.Maximize(sum(values))
