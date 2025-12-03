"""
This module contains helper functions for mathematical operations
Author: Osvaldo Burastero
"""

import numpy as np
import pandas as pd

from collections        import Counter
from scipy              import stats

from ..utils.constants  import Tref_cst
from scipy.signal       import savgol_filter

temperature_to_kelvin  = lambda T: T + 273.15 if np.max(T) < 270 else T
temperature_to_celsius = lambda T: T - 273.15 if np.max(T) > 270 else T
shift_temperature      = lambda T: temperature_to_kelvin(T) - Tref_cst

def solve_one_root_quadratic(a,b,c):
    """
    Obtain one root of the quadratic equation of the form ax^2 + bx + c = 0.

    Parameters
    ----------
    a : float
        Coefficient of x^2
    b : float
        Coefficient of x
    c : float
        Constant term

    Returns
    -------
    float
        One root of the quadratic equation
    """
    return 2*c / (-b - np.sqrt(b**2 - 4*a*c))


def solve_one_root_depressed_cubic(p,q):

    """
    Obtain one root of the depressed cubic equation of the form x^3 + p x + q = 0.

    Parameters
    ----------
    p : float
        Coefficient of x
    q : float
        Constant term

    Returns
    -------
    float
        One real root of the cubic equation
    """

    delta = np.sqrt((q**2/4) + (p**3/27))

    return np.cbrt(-q/2+delta) + np.cbrt(-q/2-delta)


def is_evenly_spaced(x, tol = 1e-4):
    """
    Check if x is evenly spaced within a given tolerance.

    Parameters
    ----------
    x : array-like
        x data
    tol : float, optional
        Tolerance for considering spacing equal (default: 1e-4)

    Returns
    -------
    bool
        True if x is evenly spaced, False otherwise
    """

    diffs = np.diff(x)
    return np.all(np.abs(diffs - diffs[0]) < tol)


def first_derivative_savgol(x, y, window_length=5, polyorder=4):

    """
    Estimate the first derivative using Savitzky-Golay filtering.

    Parameters
    ----------
    x : array-like
        x data (must be evenly spaced)
    y : array-like
        y data
    window_length : int, optional
        Length of the filter window, in temperature units (default: 5)
    polyorder : int, optional
        Order of the polynomial used to fit the samples (default: 4)

    Returns
    -------
    numpy.ndarray
        First derivative of y with respect to x

    Notes
    -----
    This function will raise a ValueError if `x` is not evenly spaced.
    """

    # Check if x is evenly spaced
    if not is_evenly_spaced(x):
        raise ValueError("x must be evenly spaced for Savitzky-Golay filter.")

    # Calculate spacing (assuming uniform x)
    dx = np.mean(np.diff(x))
    odd_n_data_points_window_len = np.ceil(window_length / dx) // 2 * 2 + 1

    if polyorder >= odd_n_data_points_window_len:
        polyorder = int(odd_n_data_points_window_len - 1)

    # Apply Savitzky-Golay filter for first derivative
    dydx = savgol_filter(y, window_length=odd_n_data_points_window_len, polyorder=polyorder, deriv=1,mode="nearest")

    return dydx


def relative_errors(params,cov):
    """
    Calculate the relative errors of the fitted parameters.

    Parameters
    ----------
    params : numpy.ndarray
        Fitted parameters
    cov : numpy.ndarray
        Covariance matrix of the fitted parameters

    Returns
    -------
    numpy.ndarray
        Relative errors of the fitted parameters (in percent)
    """

    error = np.sqrt(np.diag(cov))
    rel_error = np.abs(error / params) * 100

    return rel_error


def find_line_outliers(m,b,x,y,sigma=2.5):
    """
    Find outliers in a linear fit using the sigma rule.

    Parameters
    ----------
    m : float
        Slope of the line
    b : float
        Intercept of the line
    x : array-like
        x data
    y : array-like
        y data
    sigma : float, optional
        Number of standard deviations to use for outlier detection (default: 2.5)

    Returns
    -------
    numpy.ndarray
        Indices of the outliers
    """

    # Calculate the residuals
    residuals = y - (m * x + b)

    # Calculate the standard deviation of the residuals
    std_residuals = np.std(residuals)

    # Calculate the mean of the residuals
    mean_residuals = np.mean(residuals)

    # Identify outliers
    outliers = np.where(np.abs(residuals - mean_residuals) > sigma * std_residuals)[0]

    return outliers


def residuals_squares_sum(y_true,y_pred):

    """
    Calculate the residual sum of squares.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Residual sum of squares
    """

    # Convert to numpy arrays if it is a list
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    rss = np.sum((y_true - y_pred)**2)

    return rss


def r_squared(y_true, y_pred):
    """
    Calculate the R-squared value for a regression model.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        R-squared value
    """

    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - ss_res / ss_total


def adjusted_r2(r2, n, p):
    """
    Calculate the adjusted R-squared value for a regression model.

    Parameters
    ----------
    r2 : float
        R-squared value
    n : int
        Number of observations
    p : int
        Number of predictors

    Returns
    -------
    float
        Adjusted R-squared value
    """

    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def compute_aic(y_true, y_pred, k):
    """
    Compute the Akaike Information Criterion (AIC) for a regression model.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    k : int
        Number of parameters in the model

    Returns
    -------
    float
        AIC value
    """

    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return n * np.log(rss / n) + 2 * k


def compare_akaikes(akaikes_1, akaikes_2, akaikes_3, akaikes_4, denaturant_concentrations):
    model_names = ['Linear - Linear', 'Linear - Quadratic',
                   'Quadratic - Linear', 'Quadratic - Quadratic']

    akaikes_df = pd.DataFrame({
        'Model': model_names})

    i = 0
    for a1, a2, a3, a4 in zip(akaikes_1, akaikes_2, akaikes_3, akaikes_4):
        # Create a new column with the Akaike values
        # The name is the denaturant concentration

        # Compute delta AIC
        min_aic = np.min([a1, a2, a3, a4])
        a1 = a1 - min_aic
        a2 = a2 - min_aic
        a3 = a3 - min_aic
        a4 = a4 - min_aic

        akaikes_df[str(i) + '_' + str(denaturant_concentrations[i])] = [a1, a2, a3, a4]
        i += 1

    # Find the best model for each denaturant concentration
    best_models_ids = []
    for i in range(len(denaturant_concentrations)):

        # Get the column with the Akaike values
        aic_col = akaikes_df.iloc[:, i + 1].to_numpy()

        # Find index that sort them from min to max a numpy array
        sorted_idx = np.argsort(aic_col)

        first_model_id = np.arange(4)[sorted_idx][0]
        second_model_id = np.arange(4)[sorted_idx][1]
        third_model_id = np.arange(4)[sorted_idx][2]
        fourth_model_id = np.arange(4)[sorted_idx][3]

        best_models_ids.append(first_model_id)

        # Compare the AIC value of the second model to the first one
        if aic_col[second_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(second_model_id)

        # Compare the AIC value of the third model to the first one
        if aic_col[third_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(third_model_id)

        # Compare the AIC value of the fourth model to the first one
        if aic_col[fourth_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(fourth_model_id)

    # Print the overall best model
    best_model_all = Counter(best_models_ids).most_common(1)[0][0]
    return model_names[best_model_all]


def rss_p(rrs0, n, p, alfa):

    """
    Given the residuals of the best fitted model,
    compute the desired residual sum of squares for a 1-alpha confidence interval.
    This is used to compute asymmetric confidence intervals for the fitted parameters.

    Parameters
    ----------
    rrs0 : float
        Residual sum of squares of the model with the best fit
    n : int
        Number of data points
    p : int
        Number of parameters
    alfa : float
        Desired significance level (alpha)

    Returns
    -------
    float
        Residual sum of squares for the desired confidence interval
    """

    critical_value = stats.f.ppf(q=1 - alfa, dfn=1, dfd=n - p)

    return rrs0 * (1 + critical_value / (n - p))


def get_rss(y, y_fit):

    """
    Compute the residual sum of squares.

    Parameters
    ----------
    y : array-like
        Observed values
    y_fit : array-like
        Fitted values

    Returns
    -------
    float
        Residual sum of squares
    """

    residuals = y - y_fit
    rss       = np.sum(residuals ** 2)

    return rss


def get_desired_rss(y, y_fit, p,alpha=0.05):

    """
    Given the observed and fitted data, find the residual sum of squares required for a 1-alpha confidence interval.

    Parameters
    ----------
    y : array-like
        Observed values or list of arrays
    y_fit : array-like
        Fitted values or list of arrays
    p : int
        Number of parameters
    alpha : float, optional
        Desired significance level (default: 0.05)

    Returns
    -------
    float
        Residual sum of squares corresponding to the desired confidence interval
    """

    # If y is of type list, convert it to a numpy array by concatenating
    if isinstance(y, list):
        y = np.concatenate(y,axis=0)
    # If y_fit is of type list, convert it to a numpy array by concatenating
    if isinstance(y_fit, list):
        y_fit = np.concatenate(y_fit,axis=0)

    n = len(y)

    rss = get_rss(y, y_fit)

    return rss_p(rss, n, p, alpha)