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

    Parameters
    ----------
    T : array-like
        Temperature
    D : array-like
        Denaturant agent concentration
    DHm : float
        Variation of enthalpy between the two considered states at Tm
    Tm : float
        Temperature at which the equilibrium constant equals one
    Cp0 : float
        Variation of calorific capacity between the two states
    m0 : float
        m-value at the reference temperature (Tref)
    m1 : float
        Variation of m-value with temperature
    a_N, b_N, c_N, d_N : float
        Parameters describing the native-state baseline (intercept, temp slope, denaturant slope, quadratic term)
    a_U, b_U, c_U, d_U : float
        Parameters describing the unfolded-state baseline (intercept, temp slope, denaturant slope, quadratic term)
    extra_arg : None, optional
        Not used but present for API compatibility with oligomeric models

    Returns
    -------
    numpy.ndarray
        Signal at the given temperatures and denaturant agent concentration, given the parameters
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
    Two-state temperature unfolding (monomer).

    Parameters
    ----------
    T : array-like
        Temperature
    Tm : float
        Temperature at which the equilibrium constant equals one
    dHm : float
        Variation of enthalpy between the two considered states at Tm
    bN : float
        Intercept of the native state signal
    kN : float
        Slope of the native state signal
    bU : float
        Intercept of the unfolded state signal
    kU : float
        Slope of the unfolded state signal
    Cp : float, optional
        Variation of heat capacity between the two states (default: 0)
    qN : float, optional
        Quadratic dependence coefficient for the native state (default: 0)
    qU : float, optional
        Quadratic dependence coefficient for the unfolded state (default: 0)
    extra_arg : None, optional
        Not used but present for API compatibility

    Returns
    -------
    numpy.ndarray
        Signal at the given temperatures, given the parameters
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
    Two-state temperature unfolding (monomer) with exponential baselines.

    Parameters
    ----------
    T : array-like
        Temperature
    Tm : float
        Temperature at which the equilibrium constant equals one
    dHm : float
        Variation of enthalpy between the two considered states at Tm
    aN : float
        Intercept of the native state signal
    cN : float
        Pre-exponential factor of the native state signal
    alpha_n : float
        Exponential factor of the native state signal
    aU : float
        Intercept of the unfolded state signal
    cU : float
        Pre-exponential factor of the unfolded state signal
    alpha_u : float, optional
        Exponential factor of the unfolded state signal (default: 0)
    Cp : float, optional
        Heat capacity change (default: 0)
    extra_arg : None, optional
        Not used but present for API compatibility

    Returns
    -------
    numpy.ndarray
        Signal at the given temperatures, given the parameters
    """

    K   = eq_constant_thermo(T,dHm,Tm,Cp)
    fn  = fn_two_state_monomer(K)
    fu  = 1 - fn

    dT  = shift_temperature(T)

    S_native   = aN + cN * np.exp(-alpha_n * dT)

    S_unfolded = aU + cU * np.exp(-alpha_u * dT)

    return fn*(S_native) + fu*(S_unfolded)