# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_ordered_correlation_matrix(df, column_to_sort_by):
    corr_matrix = _get_correlation_matrix(df)
    desired_column_order = _get_desired_column_order(corr_matrix, column_to_sort_by)
    ordered_corr_matrix = _sort_correlation_matrix(corr_matrix, desired_column_order)
    return ordered_corr_matrix

def _get_correlation_matrix(df, num_decimals=2):
    return round(df.corr(), num_decimals)

def _get_desired_column_order(corr_matrix, column_to_sort_by):
    variables_ordered_by_corr = _get_variables_ordered_by_corr(corr_matrix, column_to_sort_by)
    return _get_each_variable_position(variables_ordered_by_corr)

def _get_variables_ordered_by_corr(corr_matrix, column_to_sort_by):
    return corr_matrix[column_to_sort_by].sort_values(ascending=False).index

    # Returns a dict with variable_name: desired_position
def _get_each_variable_position(variables_ordered_by_corr): 
    desired_column_order = {}
    for i, variable in enumerate(variables_ordered_by_corr):
        desired_column_order.update({variable: i})
    return desired_column_order

def _sort_correlation_matrix(corr_matrix, desired_column_order):
    corr_matrix.reset_index(inplace=True)
    corr_matrix['sort_order'] = corr_matrix['index'].map(desired_column_order)
    corr_matrix.sort_values('sort_order', inplace=True)
    corr_matrix.set_index('index', inplace=True)
    corr_matrix.drop('sort_order', axis=1, inplace=True)
    corr_matrix = corr_matrix.reindex(columns=desired_column_order)
    return corr_matrix

def plot_correlation_matrix(corr_matrix, result_path):
    corr_matrix = corr_matrix.round(decimals=1)
    fig, ax = plt.subplots(figsize=(8,8))         # Sample figsize in inches
    sns.heatmap(corr_matrix, annot=True, vmin=-1,vmax=1, cmap=sns.color_palette("coolwarm", as_cmap=True));
    plt.ylabel('Variables');
    fig.savefig(fname=f'{result_path}corr_matrix.png', bbox_inches='tight')
    plt.close()
