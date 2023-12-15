import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def test_pearson_correlation(df, covariate, target):
    return (stats.pearsonr(np.array(df[covariate]), np.array(df[target])))


def apply_linear_regression(df: pd.DataFrame, target: str) -> tuple:
    """
    Fits a Linear Regression model to the provided dataframe on the specified target variable.
    
    Parameters:
    df : pd.DataFrame
        The dataframe containing the dataset with features and target variable.
    target : str
        The name of the target variable in the dataframe.
    
    Returns:
    tuple
        A tuple containing the true values and the predicted values by the model.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the dataframe.")

    y = df[target]
    x = df.drop(columns=[target])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    LR = LinearRegression()
    LR.fit(x_train, y_train)
    y_prediction = LR.predict(x_test)
    
    return y_test, y_prediction

def evaluate_linear_regression(y_test: np.ndarray, y_prediction: np.ndarray) -> tuple:
    """
    Evaluates the performance of a Linear Regression model by calculating R-squared
    and Root Mean Squared Error.
    
    Parameters:
    y_test : np.ndarray
        The true values of the target variable from the test dataset.
    y_prediction : np.ndarray
        The predicted values of the target variable from the Linear Regression model.
    
    Returns:
    tuple
        A tuple containing the R-squared value and Root Mean Squared Error.
    """
    r2 = r2_score(y_test, y_prediction)
    mse = mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse)
    return r2, rmse