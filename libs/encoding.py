import numpy as np
import matplotlib.pyplot as plt
import math as math

from libs.auxiliary import Auxiliary

eps = 1e-10

class Encoding:
    def __init__(self):
        pass
    
    def seg_x(T, P):
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

                    if Auxiliary.ins(poi, P):
                        seg.append([[t1[0], py_min], [t1[0], py_max]])
                    # print(poi)
                    # print(P)
                    if any(np.all(poi == p) for p in P):
                        # print(poi)
                        # print(P)
                        points.append(poi)

        return seg, points
    
    def elp(P, X, K_seg, K_points, mode):
        for i in range(len(P)):
            if mode == "inv":
                t1, t2 = P[(i + 1) % len(P)], P[i]
            else:
                t1, t2 = P[i], P[(i + 1) % len(P)]

            k = None
            for j in range(len(X) - 1):
                if X[j] <= t2[0] <= X[j + 1]:
                    k = j
                    break

            # print(k, t2)
            if k is None:
                continue

            if t2[0] not in X:
                left = Auxiliary.inter(X[k], t1, t2)
                right = Auxiliary.inter(X[k + 1], t1, t2)

                if left:
                    if not Auxiliary.ins([t2[0] + eps, t2[1]], P):
                        if left not in P:
                            K_seg.append([[left[0], min(t2[1], left[1])], [left[0], max(t2[1], left[1])]])
                            # print("left, t2, t1,", left, t2, t1, [[left[0], min(t2[1], left[1])], [left[0], max(t2[1], left[1])]])
                            K_points.append([X[k + 1], t2[1]])

                if right:
                    if not Auxiliary.ins([t2[0] - eps, t2[1]], P):
                        if right not in P:
                            K_seg.append([[right[0], min(t2[1], right[1])], [right[0], max(t2[1], right[1])]])
                            # print("right, t2, t1", right, t2, t1, [[right[0], min(t2[1], right[1])], [right[0], max(t2[1], right[1])]])
                            K_points.append([X[k], t2[1]])

                if not left and not right:
                    K_seg.append([[X[k], min(t1[1], t2[1])], [X[k], max(t1[1], t2[1])]])
                    K_seg.append([[X[k + 1], min(t1[1], t2[1])], [X[k + 1], max(t1[1], t2[1])]])
                    # print("t1, t2, left, right, k, X[k], X[k + 1]", t1, t2, left, right, k, X[k], X[k + 1])
                    # print([[X[k], min(t1[1], t2[1])], [X[k], max(t1[1], t2[1])]])
                    # print([[X[k + 1], min(t1[1], t2[1])], [X[k + 1], max(t1[1], t2[1])]])

    def merge_and_clean_segments(K_seg, K_points):
        merged_segments = []

        K_seg.sort(key=lambda seg: seg[0][1])

        for seg in K_seg:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last_seg = merged_segments[-1]
                if last_seg[0][0] == seg[0][0] and last_seg[1][1] >= seg[0][1]:
                    merged_segments[-1] = [[last_seg[0][0], min(last_seg[0][1], seg[0][1])],
                                        [last_seg[1][0], max(last_seg[1][1], seg[1][1])]]
                else:
                    merged_segments.append(seg)

        filtered_points = []
        for point in K_points:
            x, y = point
            inside_segment = any(seg[0][0] == x and seg[0][1] <= y <= seg[1][1] for seg in merged_segments)
            if not inside_segment and not any(np.all(point == p) for p in filtered_points):
                filtered_points.append(point)

        return merged_segments, filtered_points
    
    def cod(X, P):
        T = Auxiliary.inter_T(X, P)
        seg, points = Encoding.seg_x(T, P)

        K_seg = seg
        K_points = points

        Encoding.elp(P, X, K_seg, K_points, "for")
        Encoding.elp(P, X, K_seg, K_points, "inv")

        K_seg, K_points = Encoding.merge_and_clean_segments(K_seg, K_points)

        return K_seg, K_points
