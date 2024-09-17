import numpy as np
import pandas as pd
import itertools
from tqdm.auto import trange, tqdm


def explain(row):
    new_row = {}

    return new_row


DATA_RANGE = {
    'Age': [0, 1, 2, ],
    'Risk Factors': [0, 1, 2, ],
    'COVID19 Status': [0, 1, 2, 3, ],
    'Cough': [0, 1, ],
    'Loss of Taste or Smell': [0, 1, ],
    'Tested Result': [0, 1, 2, ],
}

EDGE = {
    'Age': None,
    'Risk Factors': None,
    'COVID19 Status': ['Age', 'Risk Factors'],
    'Cough': ['COVID19 Status'],
    'Loss of Taste or Smell': ['COVID19 Status'],
    'Tested Result': ['COVID19 Status'],
}

NODE_ORDER = "Age,Risk Factors,COVID19 Status,Cough,Loss of Taste or Smell,Tested Result".split(',')


def decorate(table, caption, label):
    cols = table.columns.values.tolist()
    new_cols = [*cols[:2], *[col.split(' = ')[-1] for col in cols[2:]]]
    table.rename(columns=dict(zip(cols, new_cols)), inplace=True)
    latex_tab = table.to_latex(index=False,
                               float_format="{:.2f}".format,
                               )
    return latex_tab
#     table_type = "table" if len(cols) < 5 else "table*"
#     head = r"""\begin{""" + table_type + r"""}[!ht]
#     \centering
#     \adjustbox{max width=\linewidth}{"""
#     tail = r"""}
#     \caption{""" + caption + r"""}
#     \label{tab:""" + label + r"""}
# \end{""" + table_type + """}"""
#     return head + latex_tab + tail


def get_table(target: str, df: pd.DataFrame) -> dict:
    """
    Find the joint distribution table for the given column [target]; on the evidence record [df]
    """
    data_range = DATA_RANGE[target]
    parents = EDGE[target]
    target_matched_rows = {val: (df[target] == val).values for val in data_range}
    tables = {val: {} for val in data_range}
    if parents is not None:  # conditional probability
        parents_ranges = [DATA_RANGE[parent] for parent in parents]
        combinations = list(itertools.product(*parents_ranges))  # find all parent value combinations
        cols = [f"{', '.join(parents)} = {comb}" for comb in combinations]
        for col, comb_val in zip(cols, combinations):  # for each parent conditions
            conditional_rows = (df[parents] == comb_val).values.all(axis=-1)  # rows satisfying all parent conditions
            n_con_rows = max(1, conditional_rows.sum())  # number of rows satisfying all parent conditions
            for val in data_range:
                conditional_target_rows = np.logical_and(target_matched_rows[val], conditional_rows)
                n_con_target_rows = conditional_target_rows.sum()  # number of rows satisfying all parent conditions and also the target value
                tables[val][col] = n_con_target_rows / n_con_rows

    else:
        for val in data_range:  # directly calculate the unconditional probability
            tables[val]["Non-condition"] = target_matched_rows[val].sum() / len(df)
    return tables


def round_prob(col, decimal=2):
    """
    Format the probability for printing, such that all prob will be formatted "{:.[decimal]f};
    while the sum of values is 1.
    """
    rounded = np.round(col, decimal)
    rounded.iloc[-1] += 1.0 - rounded.sum()
    return rounded


def generate_sample(tables: dict, evidences: dict = None) -> tuple:
    sampled_values = {}
    weight = 1.0

    for target in NODE_ORDER:
        parents = EDGE[target]
        if parents is None:
            condition = "Non-condition"
        else:
            comb = tuple([sampled_values[parent] for parent in parents])
            condition = f"{', '.join(parents)} = {comb}"
        if evidences is None or target not in evidences.keys():
            vals = DATA_RANGE[target]
            probs = [tables[target][val][condition] for val in vals]
            sampled_values[target] = np.random.choice(vals, p=probs)
        else:
            target_val = evidences[target]
            sampled_values[target] = target_val
            weight *= tables[target][target_val][condition]
    return sampled_values, weight


def MPA(tables: dict, evidences: dict) -> tuple:
    """ Brute Force Algorithm by generating all possible values and corresponding probabilities"""
    # P(evidences) is constant, then we don't care it
    Xn = tuple(set(NODE_ORDER) - set(evidences.keys()))
    Xn_vals = [DATA_RANGE[X] for X in Xn]
    combinations = list(itertools.product(*Xn_vals))  # find all Xn value combinations
    comb_probs = []
    for comb in tqdm(combinations, desc="Searching in all possible combinations"):
        vals = dict(zip(Xn, comb))
        vals.update(evidences)
        prob = 1.0
        for target in NODE_ORDER:
            target_value = vals[target]
            parents = EDGE[target]
            if parents is None:
                condition = "Non-condition"
            else:
                comb = tuple([vals[parent] for parent in parents])
                condition = f"{', '.join(parents)} = {comb}"
            prob *= tables[target][target_value][condition]
        comb_probs.append(prob)
    max_idx = np.argmax(comb_probs)
    full_estimation = pd.DataFrame.from_records(
        [{**dict(zip(Xn, comb)), "prob": comb_prob} for comb, comb_prob in zip(combinations, comb_probs)])
    return dict(zip(Xn, combinations[max_idx])), comb_probs[max_idx], full_estimation


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    df = pd.read_csv('./simulated_data.csv')
    columns = "Age,Risk Factors,COVID19 Status,Cough,Loss of Taste or Smell,Tested Result".split(',')

    print("""Requirement (i): Print out all node tables""")
    TABLES = {}
    for col in columns:
        TABLES[col] = get_table(col, df)
        print("Node: {} ".format(col))
        node_table = df.from_dict(TABLES[col]).T
        node_table.index.name = f"{col} Value"
        node_table = node_table.apply(round_prob)
        node_table = node_table.reset_index().reset_index(drop=True)

        latex_tab = decorate(node_table,
                             caption="Distribution Table of {}".format(col),
                             label=col[:4].lower())
        print(latex_tab)
        print('-' * 20)

    print("""Requirement (ii): calculate P(Risk Factors | Loss of Taste or Smell = 1, Cough = 0) \
    using likelihood weighting by generating 100000 samples""")
    samples = []
    target = 'Risk Factors'
    evidences = {'Loss of Taste or Smell': 1, 'Cough': 0}
    for _ in trange(int(1e5)):
        vals, w = generate_sample(tables=TABLES, evidences=evidences)
        samples.append({target: vals[target], 'w': w})
    distribution = df.from_records(samples).groupby(target)['w'].sum().reset_index()
    distribution['w'] = distribution['w'] / distribution['w'].sum()
    distribution['w'] = round_prob(distribution['w'], 2)
    distribution.rename(columns={'w': 'P(Risk Factors | Loss of Taste or Smell = 1, Cough = 0)'})
    print(distribution.to_latex(index=False,
                                float_format="{:.2f}".format,
                                ))

    print("""Requirement (iii): What is the most probable explanation for someone who has cough and is \
    35 years of age?""")
    evidences = {"Cough": 1, "Age": 1}
    _map, prob, full_estimation = MPA(tables=TABLES, evidences=evidences)
    print("MAP({}) = {}".format(evidences, _map), "with the probability of {:.2f}%".format(prob * 100))
    print("The full estimation is the table below:")
    print(full_estimation.to_latex(index=False,
                                   float_format="{:.4f}".format,
                                   ))
