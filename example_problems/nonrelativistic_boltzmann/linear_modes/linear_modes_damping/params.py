import numpy as np
import arrayfire as af

fields_type       = 'electrostatic'
fields_initialize = 'fft'
fields_solver     = 'fdtd'

solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [-10]

# Initial Conditions used in initialize:
rho_background         = 1
temperature_background = 1

p1_bulk_background = 0
p2_bulk_background = 0
p3_bulk_background = 0

pert_real = 0.01
pert_imag = 0.02

k_q1 = 2 * np.pi
k_q2 = 0 * np.pi

EM_fields_enabled = False
source_enabled    = True

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0.01 * p1**0 * q1**0)