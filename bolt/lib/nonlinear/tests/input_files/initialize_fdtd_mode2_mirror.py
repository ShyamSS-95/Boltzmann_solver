import numpy as np
import arrayfire as af

def initialize_B(q1, q2, params):

    dt = params.dt

    B1 = 0 * q1**0
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)

def initialize_A_phi(q1, q2, params):

    A1  = 0 * q1**0
    A2  = 0 * q1**0
    A3  = af.cos(2 * np.pi * q1) * af.cos(4 * np.pi * q2) / np.sqrt(20)
    phi = 0 * q1**0
    af.eval(A1, A2, A3, phi)
    return(A1, A2, A3, phi)
