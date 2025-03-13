import numpy as np
import math as math

from polygon import polygon

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
