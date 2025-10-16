import math as math
import numpy as np
import ast


class Auxiliary:
    def __init__(self):
        pass

    def round_up_if_needed(x):
        return x if x == int(x) else math.ceil(x)

    def inter(x1, t1, t2):
        x_min, x_max = min(t1[0], t2[0]), max(t1[0], t2[0])
        if x1 < x_min or x1 > x_max:
            return None

        if t1[0] == t2[0]:
            return None

        y = t1[1] + (x1 - t1[0]) * (t2[1] - t1[1]) / (t2[0] - t1[0])

        if x1 == t1[0]:
            return (t1[0], t1[1])

        if x1 == t2[0]:
            return (t2[0], t2[1])

        return (x1, y)

    def inter_T(X, P):
        T = []
        for x1 in X:
            for i in range(len(P)):
                t1, t2 = P[i], P[(i + 1) % len(P)]
                point = Auxiliary.inter(x1, t1, t2)
                if point and point not in T:
                    T.append(point)
        return T

    def ins(poi, P):
        count = 0
        count2 = 0
        for i in range(len(P)):
            t1, t2 = P[i], P[(i + 1) % len(P)]
            p = Auxiliary.inter(poi[0], t1, t2)
            if p and p[1] >= poi[1]:
                count += 1
            if p and p[1] <= poi[1]:
                count2 += 1
        return count % 2 == 1 or count2 % 2 == 1

    def projections(X, P):
        proj = []
        for p in P:
            if p[0] in X:
                proj.append(p)
                proj.append(p)
            j = None
            for i in range(len(X) - 1):
                if X[i] < p[0] < X[i + 1]:
                    j = i

            if j != None:
                proj.append([X[j], p[1]])
                proj.append([X[j + 1], p[1]])

            # print(p, j)
        return proj
    
    def find_bounding_box_numpy(points):
        points_array = np.array(points)
        xs = points_array[:, 0]
        ys = points_array[:, 1]
        
        return {
            'min_x': np.min(xs),
            'max_x': np.max(xs),
            'min_y': np.min(ys),
            'max_y': np.max(ys),
        }
    
    def reflect_over_yx(polygon):
        return [[point[1], point[0]] for point in polygon]
    
    def extract_points(enc):
        # 1) если пришла строка — распарсить
        if isinstance(enc, str):
            enc = ast.literal_eval(enc)

        pts = []

        # 2) рекурсивно обойти структуру и собрать пары [x, y]
        stack = [enc]
        while stack:
            x = stack.pop()
            if x is None:
                continue
            if isinstance(x, (list, tuple)):
                # чистая точка [x, y]
                if len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
                    pts.append([float(x[0]), float(x[1])])
                else:
                    # возможно, тут отрезки [[x1,y1],[x2,y2]] или просто вложенные списки
                    stack.extend(x)

        P = np.array(pts, dtype=float)

        if P.ndim != 2 or P.shape[1] != 2 or P.size == 0:
            raise ValueError(f"Не удалось извлечь точки вида (n,2). Получено: shape={getattr(P,'shape',None)}")

        return P

    
    