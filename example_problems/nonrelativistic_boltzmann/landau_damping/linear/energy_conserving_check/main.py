import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.linear.linear_solver import linear_solver
from bolt.lib.utils.fft_funcs import ifft2

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.sources as sources
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         sources,
                         moments
                        )

N_g = system.N_ghost

# Declaring the solver object which will evolve the defined physical system:
nls = nonlinear_solver(system)

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end + domain.p2_end + domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)
n_data_nls = np.zeros([time_array.size])

# Storing data at time t = 0:
n_data_nls[0] = af.max(nls.compute_moments('density')[:, :, N_g:-N_g, N_g:-N_g])

print('Error in Density')
print(af.sum(af.abs(   nls.compute_moments('density') 
                     - (params.n_background + params.alpha * af.cos(0.5 * nls.q1_center))
                   )
            )
     )

print('Error in Velocity:')
print(af.sum(af.abs(   nls.compute_moments('mom_v1_bulk') / nls.compute_moments('density')  
                     - (params.alpha * af.sin(0.5 * nls.q1_center))
                   )
            )
     )

# Variables to track the change in mass and energy of the system:
mass_variation   = np.zeros(time_array.size)
energy_variation = np.zeros(time_array.size)

for time_index, t0 in enumerate(time_array[1:]):

    # Getting total mass and energy of the system:
#     n  = nls.compute_moments('density', f = nls.f_n_plus_half)
#     E  = nls.compute_moments('energy')
#     E1 = nls.fields_solver.yee_grid_EM_fields[0]

#     kinetic_energy  = np.sum(E)
#     electric_energy = np.sum(E1**2 + af.shift(E1, 0, 0, -1)**2) * params.eps / 4

#     mass_variation[time_index]   = np.sum(n)
#     energy_variation[time_index] = kinetic_energy + electric_energy

    print('Computing For Time =', t0)
    nls.strang_timestep(dt)

#     print(mass_variation[time_index])
#     print(energy_variation[time_index])

    n_data_nls[time_index + 1] = af.max(nls.compute_moments('density', nls.f_n_plus_half)[:, :, N_g:-N_g, N_g:-N_g])

import pylab as pl
pl.plot(time_array, n_data_nls)
pl.show()
