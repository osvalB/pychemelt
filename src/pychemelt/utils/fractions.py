"""
This module contains helper functions to obtain the amount of folded/intermediate/unfolded (etc.) protein
Author: Osvaldo Burastero
"""

from ..utils.math import *

def fn_two_state_monomer(K):
    """
    Given the equilibrium constant K of N <-> U, return the fraction of folded protein.

    Parameters
    ----------
    K : float
        Equilibrium constant of the reaction N <-> U

    Returns
    -------
    float
        Fraction of folded protein
    """
    return (1/(1 + K))

def fu_two_state_dimer(K,C):
    """
    Given the equilibrium constant K of N2 <-> 2U and the concentration of dimer equivalent C,
    return the fraction of unfolded protein.

    Parameters
    ----------
    K : float
        Equilibrium constant of the reaction N2 <-> 2U
    C : float
        Concentration of dimer equivalent

    Returns
    -------
    float
        Fraction of unfolded protein
    """

    return solve_one_root_quadratic(4*C, K, -K)
