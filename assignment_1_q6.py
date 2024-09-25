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
T_0 = np.array([0.25, 0.3, 0.35, 0.1])

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
batch_s = np.stack([i_[0] for i_ in sasp], axis=0)
batch_a = np.stack([i_[1] for i_ in sasp], axis=0)
batch_sp = np.stack([i_[2] for i_ in sasp], axis=0)

transition = lambda s, a, s_prime: a.reshape(-1) * (s @ T_1 * s_prime).sum(axis=-1) + (1 - a).reshape(-1) * (
            s_prime * T_0).sum(axis=-1)
reward = lambda s, a, s_prime: a.reshape(-1) * (s @ F * s_prime).sum(axis=-1) - (s @ C * s_prime).sum(axis=-1)

transition_probs = transition(batch_s, batch_a, batch_sp)
rewards = reward(batch_s, batch_a, batch_sp)
shaped_probs = transition_probs.reshape(4, 2, 4, )
shaped_rewards = rewards.reshape(4, 2, 4, )

for value_iteration in range(10):
    Q_sa = (shaped_probs * (shaped_rewards + gamma * V[None, None, :])).sum(axis=-1)
    arg_max_idx = np.argmax(Q_sa, axis=-1).reshape(-1)
    V = Q_sa[np.arange(4), arg_max_idx]
    pi = A[arg_max_idx]
    # print(Q_sa)
    print(pi, V)

for s, a, sp, prob, r in zip(batch_s, batch_a, batch_sp, transition_probs, rewards):
    print(f'Pr({np.argmax(sp).item()}|{np.argmax(s).item()}, {a}) = {prob}\t R({np.argmax(sp).item()}|{np.argmax(s).item()}, {a}) = {r}')
