"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    n_b = params.n_background
    T_b = params.temperature_background

    n = n_b + params.alpha * af.cos(0.5 * q1)
    T = T_b + 0 * q1

    v_bulk = 0 * params.alpha * af.sin(0.5 * q1)

    f = n * (m / (2 * np.pi * k * T))**(1 / 2) \
          * af.exp(-m * (v1 - v_bulk)**2 / (2 * k * T)) \
    
    af.eval(f)
    return (f)

# def initialize_E(q1, q2, params):
    
#     E1 = 2 * af.sum(params.charge) * params.alpha * af.sin(0.5 * q1)
#     E2 = 0 * q1**0
#     E3 = 0 * q1**0

#     af.eval(E1, E2, E3)
#     return(E1, E2, E3)

def initialize_A_phi(q1, q2, params):
    
    A1 = 0 * q1**0
    A2 = 0 * q1**0
    A3 = 0 * q1**0

    phi = 4 * af.sum(params.charge) * params.alpha * af.cos(0.5 * q1)

    af.eval(A1, A2, A3, phi)
    return(A1, A2, A3, phi)

def initialize_B(q1, q2, params):

    B1 = 0 * q1**0
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
