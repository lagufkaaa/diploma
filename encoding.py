import numpy as np
import math as math
import matplotlib.pyplot as plt

from __polygon import polygon
from auxiliary import auxiliary

eps = 1e-10

class encoding:
    def __init__(self, polygon):
        self.polygon = polygon
    
    def elp(self, X, K_seg, K_points, mode):
        P = self.polygon.points
        
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
            if not k:
                continue
            if t2[0] not in X:
                left = polygon.inter(X[k], t1, t2)
                right = polygon.inter(X[k + 1], t1, t2)

                if left:
                    if not polygon.ins([t2[0] + eps, t2[1]], P):
                        if left not in P:
                            K_seg.append([[left[0], min(t2[1], left[1])], [left[0], max(t2[1], left[1])]])
                            # print("left, t2, t1,", left, t2, t1, [[left[0], min(t2[1], left[1])], [left[0], max(t2[1], left[1])]])

                    K_points.append([X[k + 1], t2[1]])

                if right:
                    if not polygon.ins([t2[0] - eps, t2[1]], P):
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
    
    @staticmethod
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
    
    def cod(self, X):
        P = self.polygon.points
        T = polygon.inter_T(X, P)
        K_seg, K_points = auxiliary.seg_x(T, P)

        self.elp(P, X, K_seg, K_points, "for")
        self.elp(P, X, K_seg, K_points, "inv")

        K_seg, K_points = self.merge_and_clean_segments(K_seg, K_points)

        return K_seg, K_points

    def test(self, n = None, h = None):
        P = self.polygon.points
        poly = np.vstack([P, P[0]])

        plt.plot(poly[:, 0], poly[:, 1], marker='o', linestyle='-')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Многоугольник из случайных точек")
        plt.grid(True)
        plt.show()

        x0 = 0
        # xn = max([P[i][0] for i in range(len(P))])
        # xn = 10
        xn = auxiliary.round_up_if_needed(max([P[i][0] for i in range(len(P))]))
        # print("xn =",  xn)
        if n:
            h = (xn - x0)/n
        else:
            n = int((xn - x0)/h)

        X = [(x0 + h*i) for i in range(n + 1)]

        T = poly.inter_T(P, X)

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


        cod_seg, cod_points = self.cod(X, P)

        # Отрисовка многоугольника
        plt.plot(poly[:, 0], poly[:, 1], color='gray', marker='o', linestyle='-', alpha=1, zorder=1)

        # Отрисовка вертикальной линии x = x1
        for x1 in X:
            plt.axvline(x=x1, color='red', linestyle='--', alpha=0.3, zorder=2)

        # Отрисовка точек пересечения поверх графика с увеличенным радиусом и полупрозрачностью
        if cod_points:
            cod_points = np.array(cod_points)
            plt.scatter(cod_points[:, 0], cod_points[:, 1], color='red', s=60, alpha=0.5, zorder=3)

        if cod_seg:
            for segment in cod_seg:
                plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='red', linestyle='-', linewidth=2, alpha=0.7)

        #   print(proj)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Многоугольник с точками пересечения")
        plt.legend()
        plt.grid(True)
        plt.show()
