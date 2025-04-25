import ortools
from nfp import NFP
from shapely.geometry import Polygon

def model(items, H, W, r):
    num = len(items)

    C = [i.area for i in items]

    used = []
    for i in range(num):
        for j in range(r):
            used.append(model.NewBoolVar(f'used_{i} rotation_{j}'))

    for i in range(num):
        model.Add(sum(used[i]) <= 1)
        model.Add(sum(used[i]) >= 1)

    values = [C[i] * used[i] for i in range(num)]
    model.Maximize(sum(values))
