"""
This module contains helper functions to obtain the signal, given certain parameters
Author: Osvaldo Burastero
"""

from ..utils.rates     import *
from ..utils.fractions import *

def signal_two_state_tc_unfolding_monomer(
        T,D,DHm,Tm,Cp0,m0,m1,
        a_N, b_N, c_N, d_N,
        a_U, b_U, c_U, d_U,extra_arg=None):

    """
    Ref: Louise Hamborg et al., 2020. Global analysis of protein stability by temperature and chemical
    denaturation

    Args:
        T (np.ndarray): temperature
        D (np.ndarray): denaturant agent concentration
        DHm (float): variation of enthalpy between the two considered states at Tm
        Tm (float): temperature at which the equilibrium constant equals one
        Cp0 (float): variation of calorific capacity between the two states
        m0 (float): m-value at the reference temperature (Tref)
        m1 (float): variation of calorific capacity between the two states
        a_N (float): intercept of the native state signal
        b_N (float): slope of the native state signal  - temperature
        c_N (float): slope of the native state signal  - denaturant agent concentration
        d_N (float): quadratic term of the native state signal - temperature
        a_U (float): intercept of the unfolded state signal
        b_U (float): slope of the unfolded state signal - temperature
        c_U (float): slope of the unfolded state signal - denaturant agent concentration
        d_U (float): quadratic term of the unfolded state signal - temperature
        extra_arg (None): not used but required

    Returns:
        np.ndarray: Signal at the given temperatures and denaturant agent concentration, given the parameters
    """

    K   = eq_constant_termochem(T,D,DHm,Tm,Cp0,m0,m1)
    fn  = fn_two_state_monomer(K)
    fu  = 1 - fn
    dT   = shift_temperature(T)

    # Baseline signals (with quadratic dependence on temperature)
    S_native   = a_N + b_N * dT + c_N * D + d_N * (dT**2)
    S_unfolded = a_U + b_U * dT + c_U * D + d_U * (dT**2)

    return  fn*(S_native) + fu*(S_unfolded)

def signal_two_state_t_unfolding_monomer(
        T,Tm,dHm,bN,kN,bU,kU,
        Cp=0,qN=0,qU=0,extra_arg=None):

    """
    N ⇔ U

    Args:
        T (np.ndarray): temperature
        Tm (float): temperature at which the equilibrium constant equals one
        dHm (float): variation of enthalpy between the two considered states at Tm
        bN (float): intercept of the native state signal  - temperature
        kN (float): slope of the native state signal
        bU (float): intercept of the unfolded state signal  - temperature
        kU (float): slope of the unfolded state signal
        extra_arg (None): not used but required
        Cp (float): variation of calorific capacity between the two states
        qN (float): quadratic dependence for the native state
        qU (float): quadratic dependence for the unfolded state
    Returns:
        np.ndarray: Signal at the given temperatures, given the parameters
    """

    K   = eq_constant_thermo(T,dHm,Tm,Cp)
    fn  = fn_two_state_monomer(K)
    fu  = 1 - fn

    dT  = shift_temperature(T)

    S_native   = bN + kN * dT + qN * (dT**2)
    S_unfolded = bU + kU * dT + qU * (dT**2)

    return fn*(S_native) + fu*(S_unfolded)


def signal_two_state_t_unfolding_monomer_exponential(
        T,Tm,dHm,aN,cN,alpha_n,aU,cU,
        alpha_u=0,Cp=0,extra_arg=None):

    """
    N ⇔ U

    We use an exponential function to model the signal dependence on temperature

    Args:
        T (np.ndarray): temperature
        Tm (float): temperature at which the equilibrium constant equals one
        dHm (float): variation of enthalpy between the two considered states at Tm
        aN (float): intercept of the native state signal
        cN (float): pre exponential factor of the native state signal
        alpha_n (float): exponential factor of the native state signal
        aU (float): intercept of the unfolded state signal
        cU (float): pre exponential factor of the unfolded state signal
        alpha_u (float): exponential factor of the unfolded state signal

    Returns:
        np.ndarray: Signal at the given temperatures, given the parameters
    """

    K   = eq_constant_thermo(T,dHm,Tm,Cp)
    fn  = fn_two_state_monomer(K)
    fu  = 1 - fn

    dT  = shift_temperature(T)

    S_native   = aN + cN * np.exp(-alpha_n * dT)

    S_unfolded = aU + cU * np.exp(-alpha_u * dT)

    return fn*(S_native) + fu*(S_unfolded)