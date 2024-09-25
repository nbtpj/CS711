import itertools
def psi(X_i, X_j):
    if X_i == X_j:
        return 2
    return 0.5

def lambda_(X_i, Y_i):
    if X_i == Y_i:
        return 4
    return 0.5
p = []
Y_1, Y_2, Y_3, Y_4 = 1, 0, 1, 0
for (X_1, X_2) in [(0,1), (1,0)]:
    p.append(0)
    for (X_3, X_4) in list(itertools.product([0,1], [0,1])):
        p[-1] += ((psi(X_1, X_2)*psi(X_1, X_3)*psi(X_2, X_4)*psi(X_3, X_4))
              *lambda_(X_1, Y_1)*lambda_(X_2, Y_2)*lambda_(X_3, Y_3)*lambda_(X_4, Y_4))
    print(p[-1])
print(p[0]/p[1])