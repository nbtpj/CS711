import pandas as pd
import numpy as np

table = """s	a	s’	Pr	R
A	1	A	0.2	2
A	1	B	0.8	0
A	2	A	1	1
A	2	B	0	0
A	3	A	0.5	0
A	3	B	0.5	0
B	1	A	0.5	5
B	1	B	0.5	0
B	2	A	1	0
B	2	B	0	0
B	3	A	0.5	2
B	3	B	0.5	4"""
df = pd.DataFrame.from_records([line.split('\t') for line in table.split('\n')][1:],
                               columns=['s', 'a', 'sp', 'Pr', 'R'])
df['Pr'] = df['Pr'].astype(float)
df['R'] = df['R'].astype(float)
V = [{
    'A': 0,
    'B': 0,
}]
Q = []
S = ['A', 'B']
A = ['1', '2', '3']
for iter in range(20):
    Q.append({})
    for s in S:
        Q[-1][s] = {}
        for a in A:
            idx = (df[['s', 'a']] == [s, a]).values.all(axis=-1)
            V__1 = np.array([V[-1][sp] for sp in df['sp'][idx].values.tolist()])
            Q[-1][s][a] = float(df['Pr'][idx].values.T @ (df['R'][idx].values + V__1))
    V.append({s: max([Q[-1][s][a] for a in A]) for s in S})
    print(V[-1])