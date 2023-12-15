# Importing libraries
import pandas as pd
import numpy as np

def import_dataset(path, column_names):
    return pd.read_csv(path, sep=" ", header=None, names=column_names)

def transform_variables(df_in, columns_to_dummify, columns_to_apply_ln):
    df = _apply_ln_and_drop_original(df_in, columns_to_apply_ln)
    df_out = _dummify_columns(df, columns_to_dummify)
    return df_out

def _apply_ln_and_drop_original(df, columns):
    for column in columns:
        df[f'ln_{column}'] = np.log(df[column])
        df.drop(columns=column, inplace=True)
    return df

def _dummify_columns(df, columns):
    return pd.get_dummies(df, columns=columns)

def pop_variable(df, column_to_pop):
    """
    Returns X, y
    Where X is the initial dataset without the popped column
    And y is the popped column as a Series
    """
    if column_to_pop in df.columns:
        y = df.pop(column_to_pop)
    return df, y

def convert_to_float(df):
    return df.astype(float)

def normalize_dataset(df):
    df -= df.mean()
    df /= df.std()
    return df