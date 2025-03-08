import numpy as np
import math as math

eps = 1e-10 

class polygon:
    def __init__(self, points):
        self.points = [np.array(p) for p in points]
    
    @staticmethod
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
    
    def inter_T(self, X):
        P = self.points
        T = []
        for x1 in X:
            for i in range(len(P)):
                t1, t2 = P[i], P[(i + 1) % len(P)]
                point = self.inter(x1, t1, t2)
                if point and point not in T:
                    T.append(point)
        return T
    
    def ins(self, poi):
        P = self.points
        count = 0
        count2 = 0
        for i in range(len(P)):
            t1, t2 = P[i], P[(i + 1) % len(P)]
            p = self.inter(poi[0], t1, t2)
            if p and p[1] >= poi[1]:
                count += 1
            if p and p[1] <= poi[1]:
                count2 += 1
        return count % 2 == 1 or count2 % 2 == 1
    
    def rotation(self, r):
        poly_ang = []
        ang = 2*math.pi/r

        for i in range(r):
            temp = []
            tempo_and = i*ang
            mat = np.array([[math.cos(temp_ang), -math.sin(temp.ang)], [math.sin(temp.ang), math.cos(temp_ang)]])

            min_y = np.inf
            min_x = np.inf
            for p in self.points:
                temp.add(np.dot(mat, p))

                if (p[1] < min_y):
                    min_y = p[0]
                
                if (p[0] < min_x):
                    min_x = p[0]

            for p in temp:
                p[0] -= min_x
                p[1] -= min_y
            
            poly_ang.add(temp)