import numpy as np


def display_qs(array1, array2):
    for row1, row2 in zip(array1, array2):
        print(*row1, "\t|", *row2)


def display_v(array1, ):
    for row1 in array1:
        print(*row1)


beta = 1.0
T = np.array([
    [[0.6, 0.4],
     [0.3, 0.7], ],
    [[0.4, 0.6],
     [0.15, 0.85], ]
])  # A x S X S
R = np.array([
    [0, 1],  # reward for state = 1 (arm 2, 4 are still 0 because we hvnt witnessed the state yet)
    [0, 0],  # reward for state = 0
    [0, 1],
    [0, 0],
])
S = np.array(
    [
        [1, 0],  # state 0
        [0, 1],  # state 1
    ]
)

## By definition of V
## Initialize V for 4 arms, each arm has 2 state [0,1]
V_0 = np.zeros((4, 2))
print("V_0[0, 1] is")
display_v(V_0)
print('-' * 20)

## First iteration t=1; after reaching the state [1,0,1,0]
### Cal Qs of  4 arms, each arm has 2 state [0,1], each state has 2 actions [0, 1] (not pull/pull)
# Saying we are at a V update loop
R_pull = R_npull = R
# print(S @ T[1])
# print(((S @ T[1])[:, None, :] @ V_0.T).transpose().squeeze().shape)

Q_1_pull = R_pull + beta * ((S @ T[1])[:, None, :] @ V_0.T).transpose().squeeze()
Q_1_npull = R_npull + beta * ((S @ T[0])[:, None, :] @ V_0.T).transpose().squeeze()

V_1 = np.maximum(Q_1_npull, Q_1_npull)
print("Q_1_pull: \t Q_1_npull:")
display_qs(Q_1_pull, Q_1_npull)
print("V_1[0, 1] is")
display_v(V_1)

print('-' * 20)
# We repeat this with t=2
Q_2_pull = R_pull + beta * ((S @ T[1])[:, None, :] @ V_1.T).transpose().squeeze()
Q_2_npull = R_npull + beta * ((S @ T[0])[:, None, :] @ V_1.T).transpose().squeeze()

V_2 = np.maximum(Q_2_pull, Q_2_npull)
print("Q_2_pull: \t Q_2_npull:")
display_qs(Q_2_pull, Q_2_npull)
print("V_2[0, 1] is")
display_v(V_2)

print('-' * 20)
print(f"W[0,1] is \n{Q_2_pull - Q_2_npull}")
# display_v(Q_2_pull - Q_2_npull)
