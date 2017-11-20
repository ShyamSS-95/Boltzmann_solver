import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'fft'

# Can be defined as 'electrostatic' and 'fdtd'
fields_solver = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'minmod'

riemann_solver = 'upwind-flux'

# Restart Simulation:
t_restart = 0 # 0 when the simulation is to be started from t = 0

# Intervals of time for which the date is dumped to file:
t_dump_moments = 0.01
t_dump_f       = 1.0

# Timestepping parameters:
cfl_number = 0.9
t_final    = 2.5

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = 0

p_dim       = 3
num_devices = 4

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return(0 * q1**0 * p1**0)