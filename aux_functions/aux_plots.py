import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def plot_boxplot_histogram(series, var_name):
    """
    Plot a black and white boxplot and histogram side by side with annotations for a given pandas Series.

    Parameters:
    - series (pd.Series): The data series to plot.
    - var_name (str): The name of the variable (used for labeling axes).

    Returns:
    - fig (matplotlib.figure.Figure): The matplotlib figure object that can be saved to an image.
    """

    # Set the style of the visualization
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_palette("gray")

    # Set up the matplotlib figure
    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))

    # Create the boxplot on the left
    sns.boxplot(x=series, ax=ax_box, color='gray')
    ax_box.set_xlabel(var_name)
    ax_box.set_ylabel('Frequency')

    # Add annotations for the boxplot
    stats = series.describe()
    ax_box.annotate(f'Min: {stats["min"]:.1f}', xy=(stats["min"], 0), xytext=(-25, 70),
                    textcoords='offset points', fontsize=8, color='darkblue')
    ax_box.annotate(f'25%: {stats["25%"]:.2f}', fontsize=8, xy=(stats["25%"], 0), xytext=(-25, 140),
                    textcoords='offset points',color='darkblue')
    ax_box.annotate(f'50%: {stats["50%"]:.2f}', xy=(stats["50%"], 0), xytext=(-25,160),
                    textcoords='offset points', fontsize=8, color='darkblue')
    ax_box.annotate(f'75%: {stats["75%"]:.2f}', xy=(stats["75%"], 0), xytext=(-25, 140),
                    textcoords='offset points', fontsize=8, color='darkblue')
    ax_box.annotate(f'Max: {stats["max"]:.2f}', xy=(stats["max"], 0), xytext=(-25, 70),
                    textcoords='offset points', fontsize=8, color='darkblue')

    ax_box.annotate(f'Outliers: ', xy=(stats["max"], 0), xytext=(-50, -80),
                    textcoords='offset points', fontsize=8, color='darkblue')
    # Identify outliers and annotate them
    outliers = series[~series.between(stats["25%"] - 1.5 * (stats["75%"] - stats["25%"]),
                                      stats["75%"] + 1.5 * (stats["75%"] - stats["25%"]))]
    sorted_outliers = outliers.sort_values()

    i = 0
    for idx in sorted_outliers.index:
        ax_box.annotate(f'ID {idx}: {outliers[idx]:.2f}', (stats["max"], 0), xytext=(-50, -90-i),
                        textcoords='offset points',
                        fontsize=8, color='darkblue')
        i = i+10

    # Create the histogram on the right
    sns.histplot(series, ax=ax_hist, kde=False, color='gray')
    ax_hist.set_xlabel(var_name)
    ax_hist.set_ylabel('Frequency')

    # Draw a vertical line for the mean
    mean_value = series.mean()
    ax_hist.axvline(mean_value, color='darkblue', linestyle='--')
    ax_hist.annotate(f'Mean: {mean_value:.2f}', xy=(mean_value+0.5, 2), xytext=(0, 0),
                     textcoords='offset points', fontsize=8, color='darkblue')

    # Set a single title for the figure
    plt.suptitle(f'Boxplot and Histogram of {var_name}', fontsize=14, fontweight='bold')

    # Tight layout to ensure there's space for the figure title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Despine the plots
    sns.despine(left=True, bottom=True)

    # After saving the figure or at the end of the function, add:
    plt.close(fig)

    # Return the figure object
    return fig

def plot_scatterplot(df, x_axis, y_axis) -> None:
    """
    Displays a scatter plot for two variables from a DataFrame.
    
    Parameters:
    - df : DataFrame containing the data to plot.
    - x_axis : str, name of the column to be used as the x-axis.
    - y_axis : str, name of the column to be used as the y-axis.
    
    Returns:
    None, shows a matplotlib scatter plot.
    """
    sns.set_theme(style="whitegrid", rc={"axes.grid": False})
    plt.figure(figsize=(6, 4))
    
    sns.scatterplot(data=df, x=x_axis, y=y_axis, color='steelblue')
    sns.despine()
    
    plt.title(f'{y_axis} vs {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()
    plt.show()

def plot_distribution_actual_predicted(y_test: np.ndarray, y_prediction: np.ndarray) -> None:
    """
    Plots the distribution of actual vs. predicted values using Kernel Density Estimate (KDE) plots.

    Parameters:
    y_test : np.ndarray
        The true values of the target variable from the test dataset.
    y_prediction : np.ndarray
        The predicted values of the target variable from the model.

    Returns:
    None
    """
    sns.set_theme(style="whitegrid")  # Sets the theme for a cleaner look
    plt.figure(figsize=(6, 4))  # Sets a standard figure size
    
    # KDE plot for predicted values
    sns.kdeplot(y_prediction, color='blue', label='Predicted Values')
    
    # KDE plot for actual values
    sns.kdeplot(y_test, color='darkred', label='Actual Values')
    
    plt.title('Distribution of Actual vs Predicted Values')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()  # If you still want to show the legend
    plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    plt.show()  # Displays the plot]
