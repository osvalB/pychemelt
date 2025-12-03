"""
This module contains helper functions to obtain equilibrium constants
Author: Osvaldo Burastero

Useful references for unfolding models:
    - Rumfeldt, Jessica AO, et al. "Conformational stability and folding mechanisms of dimeric proteins." Progress in biophysics and molecular biology 98.1 (2008): 61-84.
    - Bedouelle, Hugues. "Principles and equations for measuring and interpreting protein stability: From monomer to tetramer." Biochimie 121 (2016): 29-37.
    - Mazurenko, Stanislav, et al. "Exploration of protein unfolding by modelling calorimetry data from reheating." Scientific reports 7.1 (2017): 16321.

All thermodynamic parameters are used in kcal mol units

Unfolding functions for monomers have an argument called 'extra_arg' that is not used.
This is because unfolding functions for oligomers require the protein concentration in that position

"""

from ..utils.math import *
from ..utils.constants    import *

def eq_constant_thermo(T,DH1,T1,Cp):

    """
    T1 is the temperature at which ΔG(T) = 0
    ΔH1, the variation of enthalpy between the two considered states at T1
    Cp the variation of calorific capacity between the two states

    Parameters
    ----------
    T : array-like
        Temperature (°C or K)
    DH1 : float
        Variation of enthalpy between the two considered states at T1 (kcal/mol)
    T1 : float
        Temperature at which the equilibrium constant equals one (°C or K)
    Cp : float
        Variation of heat capacity between the two states (kcal/mol/K)

    Returns
    -------
    numpy.ndarray
        Equilibrium constant at the given temperature
    """

    T  = temperature_to_kelvin(T)
    T1 = temperature_to_kelvin(T1)

    DG = DH1*(1 - T/T1) - Cp*(T1 - T + T*np.log(T/T1))
    K  = np.exp(-DG / (R_gas * T))

    return K


def arrhenius(T, Tf, Ea):
    """
    Arrhenius equation: defines dependence of reaction rate constant k on temperature.
    In this version of the equation we use Tf (a temperature of k=1) to avoid specifying a pre-exponential constant A.

    Parameters
    ----------
    T : array-like
        Temperature (°C or K)
    Tf : float
        Reference temperature at which the reaction rate constant equals 1 (°C or K)
    Ea : float
        Activation energy (kcal/mol)

    Returns
    -------
    numpy.ndarray
        Reaction rate constant at the given temperature
    """

    T  = temperature_to_kelvin(T)
    Tf = temperature_to_kelvin(Tf)

    return np.exp(-Ea / R_gas * (1 / T - 1 / Tf))


def eq_constant_termochem(T,D,DHm,Tm,Cp0,m0,m1):

    """
    Ref: Louise Hamborg et al., 2020. Global analysis of protein stability by temperature and chemical
    denaturation

    Parameters
    ----------
    T : array-like
        Temperature (°C or K)
    D : float
        Denaturant concentration (M)
    DHm : float
        Enthalpy change at Tm (kcal/mol)
    Tm : float
        Melting temperature where ΔG = 0 (°C or K)
    Cp0 : float
        Heat capacity change (kcal/mol/K)
    m0 : float
        m-value at the reference temperature
    m1 : float
        Temperature dependence of the m-value

    Returns
    -------
    numpy.ndarray
        Equilibrium constant at a certain temperature and denaturant agent concentration
    """

    T   = temperature_to_kelvin(T)
    Tm  = temperature_to_kelvin(Tm)

    DT  = shift_temperature(T)

    DG   = DHm*(1 - T/Tm) + Cp0*(T - Tm - T*np.log(T/Tm)) - D*(m0 + m1*DT)

    DG_RT = -DG / (R_gas * T)

    K     = np.exp(DG_RT)

    return K