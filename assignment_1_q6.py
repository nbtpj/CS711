import numpy as np

s = np.array([0, 0, 1, 0]).reshape(1, -1) # column vector
a = np.array([0, 1, 0, 0]).reshape(1, -1) # column vector

T = np.array([
    [0.0, 0.3, 0.4, 0.3],
    [0.5, 0.0, 0.2, 0.3],
    [0.4, 0.3, 0.0, 0.3],
    [0.5, 0.25, 0.25, 0.0],
])

print(s.T@a)