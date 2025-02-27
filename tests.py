import numpy as np

from polygon import polygon
from encoding import encoding
from auxiliary import auxiliary
#,asasdfakndsiofaosdjfa
m = 5

P = np.zeros((m, 2))
P = np.random.uniform(0, 10, (m, 2))

P[0][0] = 0

P = polygon.__init__(P)

auxiliary.test(P, n = 25)

n = 10
h = 0.1

P1 = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
]

P1 = polygon.__init__(P1)

auxiliary.test(P1, n = n)