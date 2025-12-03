"""
This module contains helper functions to process data
Author: Osvaldo Burastero
"""
import re
import numpy as np
from collections import Counter

from .math import shift_temperature, relative_errors
from .fitting import fit_line_robust, fit_quadratic_robust, fit_thermal_unfolding, fit_exponential_robust, fit_thermal_unfolding_exponential

from .signals import signal_two_state_t_unfolding_monomer, signal_two_state_t_unfolding_monomer_exponential


def expand_temperature_list(temp_lst,signal_lst):

    """
    Expand the temperature list to match the length of the signal list.
    Args:
        temp_lst (list): List of temperatures
        signal_lst (list): List of signals
    Returns:
        list: Expanded temperature list
    """

    if len(temp_lst) < len(signal_lst):
        temp_lst = [temp_lst[0] for _ in signal_lst]

    return temp_lst

def delete_words_appearing_more_than_five_times(strings):
    """
    Deletes words that appear more than 5 times from a list of strings.

    Args:
        strings (list): List of strings.

    Returns:
        list: List of strings with frequent words removed.
    """
    all_words = " ".join(strings).split()
    word_counts = Counter(all_words)
    words_to_remove = {word for word, count in word_counts.items() if count > 5}
    cleaned_strings = [
        " ".join(word for word in string.split() if word not in words_to_remove)
        for string in strings
    ]
    return cleaned_strings

def remove_letter_number_combinations(text):
    """
    Removes any combination of a single letter followed by one or two digits (e.g., A1, B10, D5) from the input string.

    Args:
        text (str): The input string from which patterns should be removed.

    Returns:
        str: The cleaned string with all matching patterns removed.
    """
    # Pattern: one letter (case-insensitive) followed by 1 or 2 digits, as a whole word
    pattern = r'\b[A-Za-z]\d{1,2}\b'
    cleaned_text = re.sub(pattern, '', text)
    # Optionally remove extra spaces left behind
    return re.sub(r'\s{2,}', ' ', cleaned_text).strip()

def remove_numbers_after_letter(text):
    """
    Removes all numbers coming after a letter until an underscore or space appears.

    Args:
        text (str): The input string.

    Returns:
        str: The cleaned string.
    """

    pattern = r'(?<=[A-Za-z])\d+(?=[_\s])'

    return re.sub(pattern, '', text)

def remove_non_numeric_char(input_string):
    """
    Remove all non-numeric characters except dots from a string.
    Args:
        input_string (str): Input string
    Returns:
        str: String with non-numeric characters (except dots) removed
    """

    return re.sub(r'[^\d.]', '', input_string)

def clean_conditions_labels(conditions):
    """
    Clean the conditions labels by removing unwanted characters and patterns.

    Args:
        conditions (list): List of condition strings.

    Returns:
        list: List of cleaned condition strings.
    """
    conditions = [text.replace("_", " ") for text in conditions]
    conditions = delete_words_appearing_more_than_five_times(conditions)
    conditions = [remove_letter_number_combinations(text) for text in conditions]
    conditions = [remove_numbers_after_letter(text)       for text in conditions]
    conditions = [remove_non_numeric_char(text)           for text in conditions]

    # Try to convert to float or return 0
    for i, text in enumerate(conditions):
        try:
            conditions[i] = float(text)
        except ValueError:
            conditions[i] = 0.0

    return conditions

def subset_signal_by_temperature(signal_lst, temp_lst, min_temp, max_temp):
    """
    Subset the signal and temperature lists based on the specified temperature range.

    Args:
        signal (list): List of signal values.
        temp (list): List of temperature values.
        min_temp (float): Minimum temperature for subsetting.
        max_temp (float): Maximum temperature for subsetting.
    Returns:
        tuple: Tuple containing the subsetted signal and temperature lists.
    """

    # Limit the signal to the temperature range
    subset_signal = [s[np.logical_and(t >= min_temp, t <= max_temp)] for s,t in zip(signal_lst,temp_lst)]
    subset_temp   = [t[np.logical_and(t >= min_temp, t <= max_temp)] for t in temp_lst]

    return subset_signal, subset_temp

def guess_Tm_from_derivative(temp_lst, deriv_lst, x1, x2):

    t_melting_init = []

    for sd,t in zip(deriv_lst,temp_lst):

        min_t = np.min(t)
        max_t = np.max(t)

        # max_t - min_t can't be lower than x2
        if (max_t - min_t) < x2:
            raise ValueError('The temperature range is too small to estimate the Tm. ' \
            'Please increase the range or decrease x2.')

        der_temp_init = sd[np.logical_and(t < min_t + x2, t > min_t + x1)]
        der_temp_end  = sd[np.logical_and(t < max_t - x1, t > max_t - x2)]

        med_init = np.median(der_temp_init, axis=0)
        med_end  = np.median(der_temp_end,  axis=0)

        mid_value = (med_init + med_end) / 2
        mid_value = mid_value * np.where(mid_value > 0, 1, -1)

        der_temp  = sd[np.logical_and(t > min_t + x1, t < max_t - x1)]
        temp_temp = t[np.logical_and(t > min_t + x1, t < max_t - x1)]

        der_temp = np.add(der_temp, mid_value)

        max_der = np.abs(np.max(der_temp, axis=0))
        min_der = np.abs(np.min(der_temp, axis=0))

        idx = np.argmax(der_temp) if max_der > min_der else np.argmin(der_temp)

        t_melting_init.append(temp_temp[idx])

    return t_melting_init

def estimate_signal_baseline_params(
   signal_lst,
   temp_lst,
   window_range_native=12,
   window_range_unfolded=12,     
   poly_order_native=1,
   poly_order_unfolded=1):
        
    """
    Estimate the baseline parameters for the sample
    Args:
        window_range_native (int): Range of the window (in degrees) to estimate the baselines and slopes of the native state
        window_range_unfolded (int): Range of the window (in degrees) to estimate the baselines and slopes of the unfolded state
        poly_order_native (int): Order of the polynomial to fit the native state baseline (0, 1 or 2)
        poly_order_unfolded (int): Order of the polynomial to fit the unfolded state baseline (0, 1 or 2)
    Returns:
        bNs (list): List of native state baselines
        bUs (list): List of unfolded state baselines
        kNs (list): List of native state slopes
        kUs (list): List of unfolded state slopes

    """

    bNs  = []
    bUs  = []
    kNs  = []
    kUs  = []
    qNs  = []
    qUs  = []

    for s,t in zip(signal_lst,temp_lst):

        signal_native = s[t < np.min(t) + window_range_native]
        temp_native   = t[t < np.min(t) + window_range_native]

        # Shift temperature to be centered at Tref !!! defined in constants.py
        temp_native = shift_temperature(temp_native)

        signal_denat  = s[t > np.max(t) - window_range_unfolded]
        temp_denat    = t[t > np.max(t) - window_range_unfolded]

        # Shift temperature to be centered at Tref !!! defined in constants.py
        temp_denat = shift_temperature(temp_denat)

        if poly_order_native == 0:
            bN = np.median(signal_native)
            bNs.append(bN)

        if poly_order_unfolded == 0:
            bU = np.median(signal_denat)
            bUs.append(bU)

        if poly_order_native == 1:

            kN, bN = fit_line_robust(temp_native,signal_native)

            bNs.append(bN)
            kNs.append(kN)

        if poly_order_unfolded == 1:

            kU, bU = fit_line_robust(temp_denat,signal_denat)

            bUs.append(bU)
            kUs.append(kU)

        if poly_order_native == 2:

            qN, kN, bN = fit_quadratic_robust(temp_native,signal_native)

            bNs.append(bN)
            kNs.append(kN)
            qNs.append(qN)

        if poly_order_unfolded == 2:

            qU, kU, bU = fit_quadratic_robust(temp_denat,signal_denat)

            bUs.append(bU)
            kUs.append(kU)
            qUs.append(qU) 

    return bNs, bUs, kNs, kUs, qNs, qUs


def estimate_signal_baseline_params_exponential(
        signal_lst,
        temp_lst,
        window_range_native=12,
        window_range_unfolded=12):
    """
    Estimate the baseline parameters for the sample using exponential functions
    Args:
        window_range_native (int): Range of the window (in degrees) to estimate the baselines and slopes of the native state
        window_range_unfolded (int): Range of the window (in degrees) to estimate the baselines and slopes of the unfolded state

    Returns:
        aNs (list): List of native state baselines
        aUs (list): List of unfolded state baselines
        cNs (list): List of native state pre exponential factor
        cUs (list): List of unfolded state  pre exponential factor
        alphaNs (list): List of native state exponential factors
        alphaUs (list): List of unfolded state exponential factors

    """

    aNs = []
    aUs = []
    cNs = []
    cUs = []
    alphaNs = []
    alphaUs = []

    for s, t in zip(signal_lst, temp_lst):

        signal_native = s[t < np.min(t) + window_range_native]
        temp_native = t[t < np.min(t) + window_range_native]

        # Shift temperature to be centered at Tref !!! defined in constants.py
        temp_native = shift_temperature(temp_native)

        signal_denat = s[t > np.max(t) - window_range_unfolded]
        temp_denat = t[t > np.max(t) - window_range_unfolded]

        # Shift temperature to be centered at Tref !!! defined in constants.py
        temp_denat = shift_temperature(temp_denat)

        aN, cN, alphaN = fit_exponential_robust(temp_native, signal_native)

        aU, cU, alphaU = fit_exponential_robust(temp_denat, signal_denat)

        aNs.append(aN)
        cNs.append(cN)
        alphaNs.append(alphaN)

        aUs.append(aU)
        cUs.append(cU)
        alphaUs.append(alphaU)

    return aNs, aUs, cNs, cUs, alphaNs, alphaUs

def fit_local_thermal_unfolding_to_signal_lst(
    signal_lst,
    temp_lst,
    t_melting_init,
    bNs,
    bUs,
    kNs,
    kUs,
    qNs,
    qUs,
    poly_order_native=1,
    poly_order_unfolded=1):
    
    predicted_lst = []
    Tms           = []
    dHs           = []

    fit_slopes_dic = {
        'fit_slope_native': poly_order_native > 0,
        'fit_slope_unfolded': poly_order_unfolded > 0,
        'fit_quadratic_native': poly_order_native > 1,
        'fit_quadratic_unfolded': poly_order_unfolded > 1
    }

    i = 0
    for s,t in zip(signal_lst,temp_lst):

        p0 = np.array([t_melting_init[i], 85.0, bNs[i], bUs[i]])

        if poly_order_native > 0:
            p0 = np.concatenate([p0,[kNs[i]]])
        if poly_order_unfolded > 0:
            p0 = np.concatenate([p0,[kUs[i]]])

        if poly_order_native == 2:
            p0 = np.concatenate([p0,[qNs[i]]])
        if poly_order_unfolded == 2:
            p0 = np.concatenate([p0,[qUs[i]]])

        low_bounds  = p0.copy()
        high_bounds = p0.copy()

        low_bounds[2:]  = [x / 100 - 5 if x > 0 else 100 * x - 5 for x in low_bounds[2:]]
        high_bounds[2:] = [100 * x + 5 if x > 0 else x / 100 + 5 for x in high_bounds[2:]]

        low_bounds[0]  = np.min(t)
        high_bounds[0] = np.max(t) + 15

        low_bounds[1]  = 10
        high_bounds[1] = 500

        params, cov, predicted = fit_thermal_unfolding(
            [t], [s],
            p0, low_bounds, high_bounds,
            signal_two_state_t_unfolding_monomer,
            0,
            fit_slopes=fit_slopes_dic)

        rel_errors = relative_errors(params, cov)

        if rel_errors[0] < 50 and rel_errors[1] < 50:
            Tms.append(params[0])
            dHs.append(params[1])

        predicted_lst.append(predicted[0])

        i += 1

    return Tms, dHs, predicted_lst


def fit_local_thermal_unfolding_to_signal_lst_exponential(
        signal_lst,
        temp_lst,
        t_melting_init,
        aNs,
        aUs,
        cNs,
        cUs,
        alphaNs,
        alphaUs):

    predicted_lst = []
    Tms = []
    dHs = []

    i = 0
    for s, t in zip(signal_lst, temp_lst):

        p0 = np.array([
            t_melting_init[i],
            85.0,
            aNs[i],
            aUs[i],
            cNs[i],
            cUs[i],
            alphaNs[i],
            alphaUs[i]
        ])

        low_bounds = p0.copy()
        high_bounds = p0.copy()

        max_s_abs = np.max(np.abs(s))

        low_bounds[0] = np.min(t)
        high_bounds[0] = np.max(t) + 15

        low_bounds[1]  = 10
        high_bounds[1] = 500

        low_bounds[2]   = 0
        high_bounds[2]  = np.min(s)

        low_bounds[3]   = 0
        high_bounds[3]  = np.min(s)

        low_bounds[4]  = -3 * max_s_abs
        high_bounds[4] =  3 * max_s_abs

        low_bounds[5]  = -3 * max_s_abs
        high_bounds[5] =  3 * max_s_abs

        low_bounds[6]  = 0
        high_bounds[6] = 0.1

        low_bounds[7]  = 0
        high_bounds[7] = 0.1

        params, cov, predicted = fit_thermal_unfolding_exponential(
            [t], [s],
            p0, low_bounds, high_bounds,
            signal_two_state_t_unfolding_monomer_exponential,
            0)

        rel_errors = relative_errors(params, cov)

        if rel_errors[0] < 50 and rel_errors[1] < 50:
            Tms.append(params[0])
            dHs.append(params[1])

        predicted_lst.append(predicted[0])

        i += 1

    return Tms, dHs, predicted_lst

def re_arrange_predictions(predicted_lst, n_signals, n_denaturants):

    """
    Re-arrange the flattened predictions to match the original signal list with sublists
    Args:
        predicted_lst (list): Flattened list of predicted signals of length n_signals * n_denaturants
        n_signals (int): Number of signals
        n_denaturants (int): Number of denaturants
    Returns:
        list: Re-arranged list of predicted signals to be of length n_signals with sublists of length n_denaturants
    """

    data = []

    for i in range(n_signals):

        data_i = predicted_lst[i*n_denaturants:(i+1)*n_denaturants]
        data.append(data_i)

    return data

def re_arrange_params(params,n_signals):

    """
    Re arrange the flattened parameters to be a list with sublists, as many sublists  as n_signals
    Args:
        params (list): Flattened list of parameters
        n_signals (int): Number of signals
    Returns:
        list: Re-arranged list of parameters to be of length n_signals with sublists
    """

    n_params = int(len(params) / n_signals)

    params_arranged = []

    for i in range(n_signals):

        params_i = params[i*n_params:(i+1)*n_params]
        params_arranged.append(params_i)

    return params_arranged

def subset_data(data,max_points):

    """
    Args:
        data (np.ndarray): Input data array
        max_points (int): Maximum number of points to keep
    Returns:
        np.ndarray: Subsetted data array
    """

    # Remove one every two points until the number of points is less than max_points
    do_remove = len(data) >= max_points

    while do_remove:
        data = data[::2]
        do_remove = len(data) >= max_points

    return data




