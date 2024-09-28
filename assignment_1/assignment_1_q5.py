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
for i, (X_1, X_2) in enumerate([(0, 1), (1, 0)]):
    p.append(0)
    for (X_3, X_4) in list(itertools.product([0, 1], [0, 1])):
        p[-1] += ((psi(X_1, X_2) * psi(X_1, X_3) * psi(X_2, X_4) * psi(X_3, X_4))
                  * lambda_(X_1, Y_1) * lambda_(X_2, Y_2) * lambda_(X_3, Y_3) * lambda_(X_4, Y_4))
    print("V_n = \t" if not i else "V_d = \t", p[-1])
print("V = \t", p[0] / p[1])


# double check with library
import pyAgrum as gum

# Define the Markov Random Field (MRF)
mrf = gum.MarkovRandomField()

X = []
Y = []
for x in range(1, 5):
    X.append(mrf.add(gum.LabelizedVariable(f"X{x}", "", ['W', 'B'])))
    Y.append(mrf.add(gum.LabelizedVariable(f"Y{x}", "", ['W', 'B'])))

for pair in [["X1", "X2"], ["X1", "X3"], ["X2", "X4"], ["X3", "X4"]]:
    mrf.addFactor(pair)[:] = [[2.0, 0.5],
                              [0.5, 2.0]]
for x in range(1, 5):
    pair = [f"X{x}", f"Y{x}"]
    mrf.addFactor(pair)[:] = [[4.0, 0.5],
                              [0.5, 4.0]]

evidence = {'Y1': 'B', 'Y2': 'W', 'Y3': 'B', 'Y4': 'W'}

ie = gum.ShaferShenoyMRFInference(mrf)
ie.makeInference()
ie.setEvidence(evidence)
joint_posterior = ie.jointPosterior({X[0], X[1]})
prob_X1_W_X2_B = joint_posterior[{"X1": "W", "X2": "B"}]
prob_X1_B_X2_W = joint_posterior[{"X1": "B", "X2": "W"}]

print(f"P(X1=W, X2=B | evidence) = {prob_X1_W_X2_B:.4f}")
print(f"P(X1=B, X2=W | evidence) = {prob_X1_B_X2_W:.4f}")
print("V = ", prob_X1_W_X2_B / prob_X1_B_X2_W)
