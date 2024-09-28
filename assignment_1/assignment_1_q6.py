import itertools

import numpy as np

S = np.eye(4)
A = np.array([0, 1])
T_1 = np.array([
    [0.0, 0.3, 0.4, 0.3],
    [0.5, 0.0, 0.2, 0.3],
    [0.4, 0.3, 0.0, 0.3],
    [0.5, 0.25, 0.25, 0.0],
])
T_0 = np.eye(4)
p_0 = np.array([0.25, 0.3, 0.35, 0.1])

V = np.zeros((S.shape[0],))
pi = np.zeros((S.shape[0],))

F = np.array([
    [0.0, 12, 8, 6],
    [10, 0.0, 7, 9],
    [14, 9, 0.0, 5],
    [15, 7, 4, 0.0],
])
C = np.array([
    [0.0, 1.5, 2.0, 2.5],
    [1.0, 0.0, 1.8, 2.2],
    [1.8, 1.5, 0.0, 1.2],
    [2.0, 1.8, 1.0, 0.0],
])

gamma = 0.95
sasp = list(itertools.product(S, A, S))
batch_s = np.stack([i_[0] for i_ in sasp], axis=0).reshape(4, 2, 4, 4)
batch_a = np.stack([i_[1] for i_ in sasp], axis=0).reshape(4, 2, 4, )
batch_sp = np.stack([i_[2] for i_ in sasp], axis=0).reshape(4, 2, 4, 4)

transition = lambda s, a, s_prime: (a * (s @ T_1 * s_prime).sum(axis=-1)
                                    + (1 - a) * (s @ T_0 * s_prime).sum(axis=-1))
reward = lambda s, a, s_prime: (a * (s @ p_0) * (s @ F * s_prime).sum(axis=-1)
                                - (s @ C * s_prime).sum(axis=-1))

shaped_probs = transition(batch_s, batch_a, batch_sp)
shaped_rewards = reward(batch_s, batch_a, batch_sp)

# ------------------------- Policy Iteration ---------------------
print('--------------------- Policy Iteration ------------------------')


def print_v(V):
    V_txt = []
    for i, v in enumerate(V):
        V_txt.append(f'V(A_{i+1}) = {v:.1f}')
    print(';\t'.join(V_txt))


for p_iteration in range(1000):
    Q_sa = (shaped_probs * (shaped_rewards + gamma * V[None, None, :])).sum(axis=-1)
    arg_max_idx = np.argmax(Q_sa, axis=-1).reshape(-1)
    V = Q_sa[np.arange(4), arg_max_idx]
    pi = A[arg_max_idx]
    if p_iteration < 2:
        print('Policy Iteration ', p_iteration + 1)
        print_v(V)
print('Policy Iter solver: ')
print_v(V)

# --------------------- Linear Programing ------------------------
print('--------------------- Linear Programing ------------------------')
import cvxpy as cp

V = cp.Variable(S.shape[0])

objective = cp.Minimize(cp.sum(V))
constraints = []

for s, _ in enumerate(S):
    for a, _ in enumerate(A):
        Q_sa = shaped_probs[s, a] @ shaped_rewards[s, a] + gamma * (shaped_probs[s, a] @ V)
        constraints += [V[s] >= Q_sa]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GLPK, verbose=False)

print('LP solver: ')
print_v(V.value.reshape(-1))
