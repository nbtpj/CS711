import numpy as np
import pandas as pd
import itertools


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


def get_joint_distribution(target: str, df: pd.DataFrame):
    data_range = DATA_RANGE[target]
    parents = EDGE[target]
    target_matched_rows = {val: (df[target] == val).values for val in data_range}
    tables = {f"{target} = {val}": {} for val in data_range}
    if parents is not None:
        parents_ranges = [DATA_RANGE[parent] for parent in parents]
        combinations = list(itertools.product(*parents_ranges))
        cols = [f"{', '.join(parents)} = {comb}" for comb in combinations]
        for col, comb_val in zip(cols, combinations):
            # filter by parent conditions
            conditional_rows = (df[parents] == comb_val).values.all(axis=-1)
            n_con_rows = conditional_rows.sum()
            if n_con_rows == 0:
                for val in data_range:
                    tables[f"{target} = {val}"][col] = 0
            else:
                for val in data_range:
                    conditional_target_rows = np.logical_and(target_matched_rows[val], conditional_rows)
                    n_con_target_rows = conditional_target_rows.sum()
                    tables[f"{target} = {val}"][col] = n_con_target_rows / n_con_rows

    else:
        for val in data_range:
            tables[f"{target} = {val}"]["Nan"] = target_matched_rows[val].sum() / len(df)
    return tables


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    df = pd.read_csv('./simulated_data.csv')
    columns = "Age,Risk Factors,COVID19 Status,Cough,Loss of Taste or Smell,Tested Result".split(',')
    for column in columns:
        print(df.from_dict(get_joint_distribution(column, df)).T)
