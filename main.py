N = [(lambda j=j: (i + j for i in range(10)))() for j in range(5)]

for g in N:
    print(list(g))
