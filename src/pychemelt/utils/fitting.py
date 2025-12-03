"""
This module contains helper functions to fit unfolding data
Author: Osvaldo Burastero

The functions description is below:

fit_line_robust - to fit a line to xy data
fit_line_robust_quadratic - to fit a quadratic to xy data
fit_exponential_robust - to fit an exponential to xy data

compare_linear_to_quadratic - fit xy data to linear and quadratic models and compare the results

fit_thermal_unfolding - to fit unfolding curves with a shared Tm, and DH but different baselines and slopes.
No denaturant concentration is taken into account

fit_tc_unfolding_single_slopes - to fit unfolding curves with shared Tm, DH, Cp and m.
The curves can still have different baseline and slopes

fit_tc_unfolding_shared_slopes_many_signals - to fit unfolding curves with shared Tm, DH, Cp and m.
The slope terms are shared. The intercepts can be different.

fit_tc_unfolding_many_signals - to fit unfolding curves with shared Tm, DH, Cp and m.
The slope terms are shared. The intercepts are defined by global parameters.

"""

import numpy as np
from scipy.optimize     import curve_fit
from scipy.optimize     import least_squares
from scipy.stats import f as f_dist

from ..utils.math import get_rss
from ..utils.math import temperature_to_kelvin

def fit_line_robust(x,y):

    """
    Fit a line to the data using robust fitting

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    m : float
        Slope of the fitted line
    b : float
        Intercept of the fitted line
    """

    # convert x and y to numpy arrays, if they are lists
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    def linear_model(x,params):
        m,b = params
        return m * x + b

    p0 = np.polyfit(x, y, 1)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: linear_model(x, params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1
    )

    m, b = res_robust.x

    return m, b

def fit_quadratic_robust(x,y):

    """
    Fit a quadratic equation to the data using robust fitting

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    a : float
        Quadratic coefficient of the fitted polynomial
    b : float
        Linear coefficient of the fitted polynomial
    c : float
        Constant coefficient of the fitted polynomial
    """

    # convert x and y to numpy arrays, if they are lists
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    def model(x,params):
        a,b,c = params
        return a*np.square(x) + b*x + c

    p0 = np.polyfit(x, y, 2)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: model(x, params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1
    )

    a,b,c = res_robust.x

    return a,b,c

def fit_exponential_robust(x,y):

    """
    Fit an exponential function to the data using robust fitting.

    Notes
    -----
    Temperatures should be shifted to the reference (Tref) before calling this function.

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    a : float
        Baseline
    c : float
        Pre-exponential factor
    alpha : float
        Exponential factor
    """

    # convert x and y to numpy arrays, if they are lists
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    def model(x,a,c,alpha):

        return a + c * np.exp(-alpha * x)

    # Initial parameter estimation by grid search

    rss = np.inf

    alpha_seq = np.logspace(-6, -1, 32)

    p0 = np.array( [np.min(y), np.min(y)/2])
    best_alpha = alpha_seq[0]

    max_y_abs = np.max(np.abs(y))

    low_bounds = [0, -3*max_y_abs]

    high_bounds = [(np.min(y) + np.max(y) / 2), 3*max_y_abs]

    for alpha in alpha_seq:

        try:

            def fit_fx(a,c):

                return a + c * np.exp(-alpha * x)

            params, cov = curve_fit(
                fit_fx,
                x,
                y,
                p0=p0,
                bounds=(low_bounds, high_bounds))

            pred =  fit_fx(x, *params)

            rss_curr = get_rss(y, pred)

            if rss_curr < rss:

                p0 = params
                rss = rss_curr
                best_alpha = alpha
        except:

            pass

    p0 = p0.tolist() + [best_alpha]

    low_bounds.append(best_alpha/10)
    high_bounds.append(best_alpha*10)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: model(x, *params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1,
        bounds=(low_bounds, high_bounds),
    )

    a,c,alpha = res_robust.x

    return a,c,alpha

def compare_linear_to_quadratic(x,y):

    """
    Compare the linear and quadratic fits to the data using an F-test.

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    bool
        True if the linear model is statistically preferable to the quadratic model
    """

    m, b       = fit_line_robust(x, y)
    y_pred_lin = m * x + b

    a,b,c     = fit_quadratic_robust(x, y)
    y_pred_quad = a * x ** 2 + b * x + c

    # Residual sums
    rss_lin = np.sum((y - y_pred_lin) ** 2)
    rss_quad = np.sum((y - y_pred_quad) ** 2)

    # R² and Adjusted R²
    n = len(x)
    p_lin = 1
    p_quad = 2

    # F-test
    numerator   = (rss_lin - rss_quad) / (p_quad - p_lin)
    denominator = rss_quad / (n - (p_quad + 1))
    f_stat = numerator / denominator
    p_value = 1 - f_dist.cdf(f_stat, dfn=p_quad - p_lin, dfd=n - (p_quad + 1))

    # True if linear model is better
    return p_value > 0.05

def fit_thermal_unfolding(
    list_of_temperatures, 
    list_of_signals,
    initial_parameters,
    low_bounds, 
    high_bounds,
    signal_fx, 
    Cp,
    fit_slopes,
    list_of_oligomer_conc=None):

    """
    Fit the thermal unfolding profile of many curves at the same time.

    This performs global fitting of shared thermodynamic parameters with per-curve baselines.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        List of temperature arrays for each dataset
    list_of_signals : list of array-like
        List of signal arrays for each dataset
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Function to calculate the signal based on the parameters
    Cp : float
        Heat capacity change (passed to `signal_fx`)
    fit_slopes : dict
        Dictionary indicating which baseline slope/quadratic terms should be fitted
    list_of_oligomer_conc : list, optional
        List of oligomer concentrations for each dataset (if applicable)

    Returns
    -------
    global_fit_params : numpy.ndarray
        Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix of the fitted parameters
    predicted_lst : list of numpy.ndarray
        Predicted signals for each dataset based on the fitted parameters
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    fit_slope_native = fit_slopes["fit_slope_native"]
    fit_slope_unfolded = fit_slopes["fit_slope_unfolded"]
    fit_quadratic_native = fit_slopes["fit_quadratic_native"]
    fit_quadratic_unfolded = fit_slopes["fit_quadratic_unfolded"]

    def thermal_unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single slopes, folded
            - Single slopes, unfolded
            - Single quadratic coefficients, folded
            - Single quadratic coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        n_datasets = len(list_of_temperatures)
        Tm, dh     = args[:2]  # Temperature of melting, Enthalpy of unfolding

        intercepts_folded   = args[2:(2 + n_datasets)]
        intercepts_unfolded = args[(2 + n_datasets):(2 + n_datasets * 2)]

        id_param_init = (2 + n_datasets * 2)
        n_params      = n_datasets

        if fit_slope_native:
            slopes_folded = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        if fit_slope_unfolded:
            slopes_unfolded = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        if fit_quadratic_native:
            quadratic_folded = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        if fit_quadratic_unfolded:
            quadratic_unfolded = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        signal = []

        for i, T in enumerate(list_of_temperatures):

            b_n, b_u = intercepts_folded[i], intercepts_unfolded[i]

            k_n = 0 if not fit_slope_native       else slopes_folded[i]
            k_u = 0 if not fit_slope_unfolded     else slopes_unfolded[i]
            q_n = 0 if not fit_quadratic_native   else quadratic_folded[i]
            q_u = 0 if not fit_quadratic_unfolded else quadratic_unfolded[i]

            c = 0 if list_of_oligomer_conc is None else list_of_oligomer_conc[i]

            y = signal_fx(T, Tm, dh, b_n, k_n, b_u, k_u, Cp,q_n,q_u,c)
            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        thermal_unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
        )

    predicted = thermal_unfolding(1,*global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init+n])
        init += n

    return global_fit_params, cov, predicted_lst

def fit_thermal_unfolding_exponential(
    list_of_temperatures,
    list_of_signals,
    initial_parameters,
    low_bounds,
    high_bounds,
    signal_fx,
    Cp=0,
    list_of_oligomer_conc=None):

    """
    Fit the thermal unfolding profile of many curves at the same time using exponential baselines.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        List of temperature arrays for each dataset
    list_of_signals : list of array-like
        List of signal arrays for each dataset
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Function to calculate the signal based on the parameters
    Cp : float, optional
        Heat capacity change (default: 0)
    list_of_oligomer_conc : list, optional
        List of oligomer concentrations for each dataset (if applicable)

    Returns
    -------
    global_fit_params : numpy.ndarray
        Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix of the fitted parameters
    predicted_lst : list of numpy.ndarray
        Predicted signals for each dataset based on the fitted parameters
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    def thermal_unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single pre exponential terms, folded
            - Single pre exponential terms, unfolded
            - Single exponential terms, folded
            - Single exponential terms, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        n_datasets = len(list_of_temperatures)
        Tm, dh     = args[:2]  # Temperature of melting, Enthalpy of unfolding

        intercepts_folded   = args[2:(2 + n_datasets)]
        intercepts_unfolded = args[(2 + n_datasets):(2 + n_datasets * 2)]

        id_param_init = (2 + n_datasets * 2)
        n_params      = n_datasets

        pre_exp_folded = args[id_param_init:(id_param_init+n_params)]
        id_param_init += n_params

        pre_exp_unfolded = args[id_param_init:(id_param_init+n_params)]
        id_param_init += n_params

        exp_folded = args[id_param_init:(id_param_init+n_params)]
        id_param_init += n_params

        exp_unfolded = args[id_param_init:(id_param_init+n_params)]
        id_param_init += n_params

        signal = []

        for i, T in enumerate(list_of_temperatures):

            a_n = intercepts_folded[i]
            a_u = intercepts_unfolded[i]

            c_n     = pre_exp_folded[i]
            c_u     = pre_exp_unfolded[i]
            alpha_n = exp_folded[i]
            alpha_u = exp_unfolded[i]

            c = 0 if list_of_oligomer_conc is None else list_of_oligomer_conc[i]

            y = signal_fx(T, Tm, dh, a_n, c_n, alpha_n, a_u, c_u, alpha_u, Cp,c)

            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        thermal_unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
        )

    predicted = thermal_unfolding(1,*global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init+n])
        init += n

    return global_fit_params, cov, predicted_lst

def fit_tc_unfolding_single_slopes(
    list_of_temperatures, 
    list_of_signals,
    denaturant_concentrations,
    initial_parameters,
    low_bounds, 
    high_bounds,
    signal_fx,
    fit_slopes,
    list_of_oligomer_conc = None,
    fit_m1                = False,
    cp_value              = None,
    tm_value              = None,
    dh_value              = None):

    """
    Fit thermochemical unfolding curves sharing thermodynamic parameters while allowing different baselines.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    denaturant_concentrations : list
        Denaturant concentrations (one per dataset)
    initial_parameters : array-like
        Initial guess for parameters
    low_bounds : array-like
        Lower bounds for parameters
    high_bounds : array-like
        Upper bounds for parameters
    signal_fx : callable
        Signal model function
    fit_slopes : dict
        Dict indicating which baseline terms to fit
    list_of_oligomer_conc : list, optional
        Oligomer concentrations per dataset
    fit_m1 : bool, optional
        Whether to fit temperature dependence of m-value
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters

    Returns
    -------
    global_fit_params : numpy.ndarray
    cov : numpy.ndarray
    predicted_lst : list of numpy.ndarray
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_datasets = len(list_of_temperatures)

    fit_slope_native = fit_slopes["fit_slope_native"]
    fit_slope_unfolded = fit_slopes["fit_slope_unfolded"]
    fit_quadratic_native = fit_slopes["fit_quadratic_native"]
    fit_quadratic_unfolded = fit_slopes["fit_quadratic_unfolded"]

    def unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Global Cp0
            - Global m0
            - Global m1 (if fit_m1 is True)
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single slopes, folded
            - Single slopes, unfolded
            - Single quadratic coefficients, folded
            - Single quadratic coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        id_param_init = 0

        if tm_value is None:

            Tm = args[0]  # Temperature of melting
            id_param_init += 1

        else:

            Tm = tm_value

        if dh_value is None:

            DHm = args[id_param_init]  # Enthalpy of unfolding
            id_param_init += 1

        else:

            DHm = dh_value

        if cp_value is None:

            Cp0 = args[id_param_init]  # Cp0
            id_param_init += 1

        else:

            Cp0 = cp_value

        m0 = args[id_param_init]  # m0
        id_param_init += 1

        if fit_m1:
            
            m1 = args[id_param_init]
            id_param_init += 1

        else:

            m1 = 0

        # First filter, verify that DG is not lower than 0 at 5C
        # In other words, we do not have cold denaturation at 5C

        """
        Tfive = temperature_to_kelvin(5)
        TmK   = temperature_to_kelvin(Tm)

        DGfive = DHm * (1 - Tfive / TmK) + Cp0 * (Tfive - TmK - Tfive * np.log(Tfive / TmK))
        """

        intercepts_folded   = args[id_param_init:(id_param_init + n_datasets)]
        intercepts_unfolded = args[(id_param_init + n_datasets):(id_param_init + n_datasets * 2)]

        id_param_init = (id_param_init + n_datasets * 2)
        n_params      = n_datasets

        if fit_slope_native:
            slopes_folded = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        if fit_slope_unfolded:
            slopes_unfolded = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        if fit_quadratic_native:
            quadratic_folded = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        if fit_quadratic_unfolded:
            quadratic_unfolded = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        signal = []

        for i, T in enumerate(list_of_temperatures):

            b_n, b_u = intercepts_folded[i], intercepts_unfolded[i]

            k_n = 0 if not fit_slope_native       else slopes_folded[i]
            k_u = 0 if not fit_slope_unfolded     else slopes_unfolded[i]
            q_n = 0 if not fit_quadratic_native   else quadratic_folded[i]
            q_u = 0 if not fit_quadratic_unfolded else quadratic_unfolded[i]

            c = 0 if list_of_oligomer_conc is None else list_of_oligomer_conc[i]

            d = denaturant_concentrations[i]

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                b_n, k_n, 0, q_n,
                b_u, k_u, 0, q_u, c
            )

            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
    )

    predicted = unfolding(1,*global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init+n])
        init += n

    return global_fit_params, cov, predicted_lst


def fit_tc_unfolding_shared_slopes_many_signals(
    list_of_temperatures,
    list_of_signals,
    signal_ids,
    denaturant_concentrations,
    initial_parameters,
    low_bounds, 
    high_bounds,
    signal_fx,
    fit_slopes,
    list_of_oligomer_conc = None,
    fit_m1                = False,
    cp_value              = None,
    tm_value              = None,
    dh_value              = None):

    """
    Fit thermochemical unfolding curves for multiple signal types sharing thermodynamic parameters and slopes.

    Parameters
    ----------
    list_of_temperatures : list of array-like
    list_of_signals : list of array-like
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    denaturant_concentrations : list
        Denaturant concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
    fit_slopes : dict
        Dict indicating which baseline terms to fit
    list_of_oligomer_conc : list, optional
        Oligomer concentrations per dataset
    fit_m1 : bool, optional
        Whether to fit temperature dependence of m-value
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters

    Returns
    -------
    global_fit_params : numpy.ndarray
    cov : numpy.ndarray
    predicted_lst : list of numpy.ndarray
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_signals = np.max(signal_ids) + 1

    n_datasets = len(list_of_temperatures)

    fit_slope_native = fit_slopes["fit_slope_native"]
    fit_slope_unfolded = fit_slopes["fit_slope_unfolded"]
    fit_quadratic_native = fit_slopes["fit_quadratic_native"]
    fit_quadratic_unfolded = fit_slopes["fit_quadratic_unfolded"]

    def unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Global Cp0
            - Global m0
            - Global m1 (if fit_m1 is True)
            - Single intercepts, folded - one per signal
            - Single intercepts, unfolded - one per signal
            - Single slopes, folded - one per signal
            - Single slopes, unfolded - one per signal
            - Single quadratic coefficients, folded - one per signal
            - Single quadratic coefficients, unfolded - one per signal

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """
        
        id_param_init = 0

        if tm_value is None:

            Tm = args[0]  # Temperature of melting
            id_param_init += 1
        
        else:

            Tm = tm_value

        if dh_value is None:

            DHm = args[id_param_init]  # Enthalpy of unfolding
            id_param_init += 1
        
        else:

            DHm = dh_value

        if cp_value is None:

            Cp0 = args[id_param_init]  # Cp0
            id_param_init += 1
        
        else:

            Cp0 = cp_value

        m0 = args[id_param_init]  # m0
        id_param_init += 1

        if fit_m1:
            m1 = args[id_param_init]
            id_param_init += 1
        else:
            m1 = 0

        # First filter, verify that DG is not lower than 0 at 5C
        # In other words, we do not have cold denaturation at 5C

        """
        Tfive = temperature_to_kelvin(5)
        TmK   = temperature_to_kelvin(Tm)

        DGfive = DHm * (1 - Tfive / TmK) + Cp0 * (Tfive - TmK - Tfive * np.log(Tfive / TmK))

        if DGfive < 0:

            return np.zeros(len(all_signal))
        """

        intercepts_folded   = args[id_param_init:(id_param_init + n_datasets)]
        intercepts_unfolded = args[(id_param_init + n_datasets):(id_param_init + n_datasets * 2)]

        id_param_init = (id_param_init + n_datasets * 2)

        if fit_slope_native:
            k_n_s = args[id_param_init:(id_param_init+n_signals)]
            id_param_init += n_signals
        else:
            k_n_s = [0]*n_signals

        if fit_slope_unfolded:
            k_u_s = args[id_param_init:(id_param_init+n_signals)]
            id_param_init += n_signals
        else:
            k_u_s = [0]*n_signals

        if fit_quadratic_native:
            q_n_s = args[id_param_init:(id_param_init+n_signals)]
            id_param_init += n_signals
        else:
            q_n_s = [0]*n_signals

        if fit_quadratic_unfolded:
            q_u_s = args[id_param_init:(id_param_init+n_signals)]
            id_param_init += n_signals
        else:
            q_u_s = [0]*n_signals

        signal = []

        for i, T in enumerate(list_of_temperatures):

            b_n, b_u = intercepts_folded[i], intercepts_unfolded[i]

            c = 0 if list_of_oligomer_conc is None else list_of_oligomer_conc[i]

            d = denaturant_concentrations[i]

            k_n = k_n_s[signal_ids[i]]
            k_u = k_u_s[signal_ids[i]]
            q_n = q_n_s[signal_ids[i]]
            q_u = q_u_s[signal_ids[i]]

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                b_n, k_n, 0, q_n,
                b_u, k_u, 0, q_u, c
            )

            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
    )

    predicted = unfolding(1,*global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init+n])
        init += n

    return global_fit_params, cov, predicted_lst

def fit_tc_unfolding_many_signals_slow(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        denaturant_concentrations,
        initial_parameters,
        low_bounds, high_bounds,
        signal_fx,
        fit_slope_native_temp=True,
        fit_slope_unfolded_temp=True,
        fit_slope_native_den=True,
        fit_slope_unfolded_den=True,
        fit_quadratic_native=False,
        fit_quadratic_unfolded=False,
        oligomer_concentrations=None,
        fit_m1=False,
        model_scale_factor=False,
        scale_factor_exclude_ids = []):

    """
    Fit thermochemical unfolding curves for many signals (slow variant).

    Parameters
    ----------
    list_of_temperatures : list of array-like
        List of temperature arrays for each dataset
    list_of_signals : list of array-like
        List of signal arrays for each dataset
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    denaturant_concentrations : list
        Denaturant concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
    fit_slope_native_temp : bool, optional
        Whether to fit the temperature slope of the native baseline (per-signal)
    fit_slope_unfolded_temp : bool, optional
        Whether to fit the temperature slope of the unfolded baseline (per-signal)
    fit_slope_native_den : bool, optional
        Whether to fit the denaturant slope of the native baseline (per-signal)
    fit_slope_unfolded_den : bool, optional
        Whether to fit the denaturant slope of the unfolded baseline (per-signal)
    fit_quadratic_native : bool, optional
        Whether to fit a quadratic temperature term for the native baseline (per-signal)
    fit_quadratic_unfolded : bool, optional
        Whether to fit a quadratic temperature term for the unfolded baseline (per-signal)
    oligomer_concentrations : list, optional
        Oligomer concentrations per dataset (used by oligomeric models)
    fit_m1 : bool, optional
        Whether to include and fit temperature dependence of the m-value (m1)
    model_scale_factor : bool, optional
        If True, include a per-denaturant concentration scale factor to account for intensity differences
    scale_factor_exclude_ids : list, optional
        IDs of scale factors to exclude / fix to 1 (useful to avoid fitting trivial factors)

    Returns
    -------
    global_fit_params : numpy.ndarray
        Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix of the fitted parameters
    predicted_lst : list of numpy.ndarray
        Predicted signals for each dataset based on the fitted parameters
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_signals = np.max(signal_ids) + 1

    nr_den = int(len(denaturant_concentrations) / n_signals)

    if len(scale_factor_exclude_ids) > 0 and model_scale_factor:
        # Sort them in ascending order to avoid issues when inserting
        scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

    # Find if highest concentration of denaturant has a higher signal or not
    if model_scale_factor:

        den_conc_simple = denaturant_concentrations[:nr_den]

        # Find the index that sorts them in descending order from highest to lowest
        sort_indeces = np.argsort(den_conc_simple)[::-1]

        signal_first = list_of_signals[:nr_den]

        signal_sort = [signal_first[i] for i in sort_indeces]

        higher_den_equal_higher_signal = signal_sort[0][0] > signal_sort[-1][0]

    def unfolding(dummyVariable, *args):

        Tm, DHm, Cp0, m0 = args[:4]  # Enthalpy of unfolding, Temperature of melting, Cp0, m0, m1

        id_param_init = 4 + fit_m1
        m1 = args[4] if fit_m1 else 0

        # First filter, verify that DG is not lower than 0 at 5C
        # In other words, we do not have cold denaturation at 5C
        """
        Tfive = temperature_to_kelvin(5)
        TmK   = temperature_to_kelvin(Tm)

        DGfive = DHm * (1 - Tfive / TmK) + Cp0 * (Tfive - TmK - Tfive * np.log(Tfive / TmK))

        if DGfive < 0:

            return np.zeros(len(all_signal))
        """

        a_Ns = args[id_param_init:id_param_init+n_signals]
        a_Us = args[id_param_init+n_signals:id_param_init+2*n_signals]

        id_param_init = id_param_init+2*n_signals
        if fit_slope_native_temp:
            b_Ns = args[id_param_init:id_param_init+n_signals]
            id_param_init += n_signals
        else:
            b_Ns = [0] * n_signals

        if fit_slope_unfolded_temp:
            b_Us = args[id_param_init:id_param_init+n_signals]
            id_param_init += n_signals
        else:
            b_Us = [0] * n_signals

        if fit_slope_native_den:
            c_Ns = args[id_param_init:id_param_init+n_signals]
            id_param_init += n_signals
        else:
            c_Ns = [0] * n_signals

        if fit_slope_unfolded_den:
            c_Us = args[id_param_init:id_param_init+n_signals]
            id_param_init += n_signals
        else:
            c_Us = [0] * n_signals

        if fit_quadratic_native:
            d_Ns = args[id_param_init:id_param_init+n_signals]
            id_param_init += n_signals
        else:
            d_Ns = [0] * n_signals

        if fit_quadratic_unfolded:
            d_Us = args[id_param_init:id_param_init+n_signals]
            id_param_init += n_signals
        else:
            d_Us = [0] * n_signals

        if model_scale_factor:
            # One per denaturant concentration
            factors = args[id_param_init:id_param_init + (nr_den - len(scale_factor_exclude_ids))]
            
            for id_ex in scale_factor_exclude_ids:
                factors = np.insert(factors, id_ex, 1)

            # Repeat the list so have the same length as list_of_temperatures, equal to denaturant concentration * number of signals
            factors = np.tile(factors, n_signals)

            id_param_init += nr_den

        signal = []

        for i, T in enumerate(list_of_temperatures):

            a_N = a_Ns[signal_ids[i]]
            b_N = b_Ns[signal_ids[i]]
            c_N = c_Ns[signal_ids[i]]
            d_N = d_Ns[signal_ids[i]]

            a_U = a_Us[signal_ids[i]]
            b_U = b_Us[signal_ids[i]]
            c_U = c_Us[signal_ids[i]]
            d_U = d_Us[signal_ids[i]]

            d = denaturant_concentrations[i]

            c = 0 if oligomer_concentrations is None else oligomer_concentrations[i]

            d_factor = 1

            d = d * d_factor

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                a_N, b_N, c_N, d_N,
                a_U, b_U, c_U, d_U, c
            )

            scale_factor = 1 if not model_scale_factor else factors[i]

            y = y * scale_factor

            signal.append(y)

        # Second filter, verify that higher_den_equal_higher_signal is same in the raw and fitted signal
        if model_scale_factor:

            signal_first = signal[:nr_den]

            signal_sort = [signal_first[i] for i in sort_indeces]

            pred_higher_den_equal_higher_signal = signal_sort[0][0] > signal_sort[-1][0]

            if pred_higher_den_equal_higher_signal != higher_den_equal_higher_signal:

                return np.zeros(len(all_signal))

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters,
        bounds=(low_bounds, high_bounds))

    predicted = unfolding(1,*global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init+n])
        init += n

    return global_fit_params, cov, predicted_lst


def fit_tc_unfolding_many_signals(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        denaturant_concentrations,
        initial_parameters,
        low_bounds, high_bounds,
        signal_fx,
        fit_slope_native_temp=True,
        fit_slope_unfolded_temp=True,
        fit_slope_native_den=True,
        fit_slope_unfolded_den=True,
        fit_quadratic_native=False,
        fit_quadratic_unfolded=False,
        oligomer_concentrations=None,
        fit_m1=False,
        model_scale_factor=False,
        scale_factor_exclude_ids=[],
        cp_value=None
):
    """
    Fit thermochemical unfolding curves for many signals (optimized variant).

    Parameters
    ----------
    list_of_temperatures : list of array-like
    list_of_signals : list of array-like
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    denaturant_concentrations : list
        Denaturant concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
    fit_slope_native_temp : bool, optional
        Whether to fit the temperature slope of the native baseline (per-signal)
    fit_slope_unfolded_temp : bool, optional
        Whether to fit the temperature slope of the unfolded baseline (per-signal)
    fit_slope_native_den : bool, optional
        Whether to fit the denaturant slope of the native baseline (per-signal)
    fit_slope_unfolded_den : bool, optional
        Whether to fit the denaturant slope of the unfolded baseline (per-signal)
    fit_quadratic_native : bool, optional
        Whether to fit a quadratic temperature term for the native baseline (per-signal)
    fit_quadratic_unfolded : bool, optional
        Whether to fit a quadratic temperature term for the unfolded baseline (per-signal)
    oligomer_concentrations : list, optional
        Oligomer concentrations per dataset (used by oligomeric models)
    fit_m1 : bool, optional
        Whether to include and fit temperature dependence of the m-value (m1)
    model_scale_factor : bool, optional
        If True, include a per-denaturant concentration scale factor to account for intensity differences
    scale_factor_exclude_ids : list, optional
        IDs of scale factors to exclude / fix to 1
    cp_value : float or None, optional
        If provided, Cp is fixed to this value and not fitted

    Returns
    -------
    global_fit_params : numpy.ndarray
    cov : numpy.ndarray
    predicted_lst : list of numpy.ndarray
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_signals = np.max(signal_ids) + 1

    nr_den = int(len(denaturant_concentrations) / n_signals)

    if len(scale_factor_exclude_ids) > 0 and model_scale_factor:
        # Sort them in ascending order to avoid issues when inserting
        scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

    # Find if highest concentration of denaturant has a higher signal or not
    """
    if model_scale_factor:
        den_conc_simple = denaturant_concentrations[:nr_den]

        # Find the index that sorts them in descending order from highest to lowest
        sort_indeces = np.argsort(den_conc_simple)[::-1]

        signal_first = list_of_signals[:nr_den]

        signal_sort = [signal_first[i] for i in sort_indeces]

        higher_den_equal_higher_signal = signal_sort[0][0] > signal_sort[-1][0]
    """

    def unfolding(dummyVariable, *args):

        if cp_value is not None:

            Cp0 = cp_value
            Tm, DHm, m0 = args[:3]  # Enthalpy of unfolding, Temperature of melting, m0, m1
        
        else:

            Tm, DHm, Cp0, m0 = args[:4]  # Enthalpy of unfolding, Temperature of melting, Cp0, m0, m1

        id_param_init = 3 + fit_m1 + (cp_value is None)

        m1 = args[id_param_init] if fit_m1 else 0

        # First filter, verify that DG is not lower than 0 at 5C
        # In other words, we do not have cold denaturation at 5C
        """
        Tfive = temperature_to_kelvin(5)
        TmK   = temperature_to_kelvin(Tm)

        DGfive = DHm * (1 - Tfive / TmK) + Cp0 * (Tfive - TmK - Tfive * np.log(Tfive / TmK))

        if DGfive < 0:
            print('error')
            return np.zeros(len(all_signal))
        """

        a_Ns = args[id_param_init:id_param_init + n_signals]
        a_Us = args[id_param_init + n_signals:id_param_init + 2 * n_signals]

        id_param_init = id_param_init + 2 * n_signals

        if fit_slope_native_temp:
            b_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            b_Ns = [0] * n_signals

        if fit_slope_unfolded_temp:

            b_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            b_Us = [0] * n_signals

        if fit_slope_native_den:

            c_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            c_Ns = [0] * n_signals

        if fit_slope_unfolded_den:

            c_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            c_Us = [0] * n_signals

        if fit_quadratic_native:

            d_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            d_Ns = [0] * n_signals

        if fit_quadratic_unfolded:
            d_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            d_Us = [0] * n_signals

        if model_scale_factor:
            # One per denaturant concentration
            factors = args[id_param_init:id_param_init + (nr_den - len(scale_factor_exclude_ids))]

            for id_ex in scale_factor_exclude_ids:
                factors = np.insert(factors, id_ex, 1)

            # Repeat the list so have the same length as list_of_temperatures, equal to denaturant concentration * number of signals
            factors = np.tile(factors, n_signals)

            id_param_init += nr_den

        signal = []

        for i, T in enumerate(list_of_temperatures):

            a_N = a_Ns[signal_ids[i]]
            b_N = b_Ns[signal_ids[i]]
            c_N = c_Ns[signal_ids[i]]
            d_N = d_Ns[signal_ids[i]]

            a_U = a_Us[signal_ids[i]]
            b_U = b_Us[signal_ids[i]]
            c_U = c_Us[signal_ids[i]]
            d_U = d_Us[signal_ids[i]]

            d = denaturant_concentrations[i]

            c = 0 if oligomer_concentrations is None else oligomer_concentrations[i]

            d_factor = 1

            d = d * d_factor

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                a_N, b_N, c_N, d_N,
                a_U, b_U, c_U, d_U, c
            )

            scale_factor = 1 if not model_scale_factor else factors[i]

            y = y * scale_factor

            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters,
        bounds=(low_bounds, high_bounds))

    predicted = unfolding(1, *global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n

    return global_fit_params, cov, predicted_lst


def evaluate_need_to_refit(
        global_fit_params,
        high_bounds,
        low_bounds,
        p0,
        fit_m1=False,
        check_cp=True,
        check_dh=True,
        check_tm=True,
        fixed_cp=False):

    """
    Check and expand parameter bounds when fitted parameters are too close to boundaries.

    Parameters
    ----------
    global_fit_params : array-like
        Fitted parameters
    high_bounds : array-like
        Upper bounds
    low_bounds : array-like
        Lower bounds
    p0 : array-like
        Initial guess for parameters
    fit_m1 : bool, optional
    check_cp, check_dh, check_tm : bool, optional
    fixed_cp : bool, optional

    Returns
    -------
    re_fit : bool
        True if a refit is recommended after bounds expansion
    p0 : array-like
        Updated initial parameters
    low_bounds : array-like
        Updated lower bounds
    high_bounds : array-like
        Updated upper bounds
    """

    re_fit = False

    # Check the Tm boundary - upper
    tm_diff = high_bounds[0] - global_fit_params[0]
    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        high_bounds[0] = global_fit_params[0] + 12
        p0[0] = global_fit_params[0] + 5
        re_fit = True

    # Check the Tm boundary - lower
    tm_diff = global_fit_params[0] - low_bounds[0]
    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        low_bounds[0] = global_fit_params[0] - 12
        p0[0] = global_fit_params[0] - 5
        re_fit = True

    # Check the Dh boundary
    dh_diff = high_bounds[1] - global_fit_params[1]
    # Expand the boundary if the Dh is too close to the boundary
    if dh_diff < 20 and check_dh:
        high_bounds[1] = global_fit_params[1] + 80
        p0[1] = global_fit_params[1] + 50
        re_fit = True

    id_next = 2
    if not fixed_cp:

        # Check the Cp boundary
        cp_diff = high_bounds[2] - global_fit_params[2]
        # Expand the boundary if the Cp is too close to the boundary
        if cp_diff < 0.25 and check_cp:
            high_bounds[2] = global_fit_params[2] + 1
            p0[2] = global_fit_params[2] + 0.5
            re_fit = True
        
        id_next += 1

    # Check the m-value boundary
    m_diff = high_bounds[id_next] - global_fit_params[id_next]
    # Expand the boundary if the m-value is too close to the boundary
    if m_diff < 0.5:
        high_bounds[id_next] = global_fit_params[id_next] + 2
        p0[id_next] = global_fit_params[id_next] + 0.5
        re_fit = True

    # Evaluate if m1 is fitted
    id_start = id_next + 1
    if fit_m1:

        m1_diff = high_bounds[id_start] - global_fit_params[id_start]
        # Expand the boundary if the m-value is too close to the boundary
        if m1_diff < 0.1:
            high_bounds[id_start] = global_fit_params[id_start] + 1
            re_fit = True

        m1_diff = global_fit_params[id_start] - low_bounds[id_start]
        # Expand the boundary if the m-value is too close to the boundary
        if m1_diff < 0.1:
            low_bounds[id_start] = global_fit_params[id_start] - 1
            re_fit = True

        id_start += 1

    # Evaluate all the other parameters
    for i in range(id_start,len(global_fit_params)):

        diff_to_high = high_bounds[i] - global_fit_params[i]
        diff_to_low  = global_fit_params[i] - low_bounds[i]

        if diff_to_high < 0.5:
            high_bounds[i] = high_bounds[i] + 50
            re_fit = True

        if diff_to_low < 0.5:
            low_bounds[i] = low_bounds[i] - 50
            re_fit = True

    return re_fit, p0, low_bounds, high_bounds