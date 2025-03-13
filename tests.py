import numpy as np
#git test comment 
from polygon import polygon
from encoding import encoding
from auxiliary import auxiliary
#,asasdfakndsiofaosdjfa
m = 5

poly = np.random.uniform(0, 10, (m, 2))  # Создаём массив случайных точек
poly[0][0] = 0  # Меняем первую координату

P = polygon(poly)  # Создаём объект класса polygon
encoding.test(P, n=25)

n = 10
h = 0.1

poly1 = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
]

P1 = polygon(poly1)  # Правильный вызов конструктора
