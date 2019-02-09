import numpy as np
import arrayfire as af

def initialize_E(q1, q2, params):
    
    E1 = 0 * q1
    E2 = 0 * q1
    E3 = 0 * q1

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_A3_B3(q1, q2, params):

    A3 = af.sin(2 * np.pi * q1) * af.sin(4 * np.pi * q2) / np.sqrt(20)
    B3 = 0 * q1**0

    af.eval(A3, B3)
    return(A3, B3)
