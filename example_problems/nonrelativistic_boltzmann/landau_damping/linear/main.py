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
n_data     = np.zeros([time_array.size])

print('Error in Density')
print(af.sum(af.abs(   nls.compute_moments('density') 
                     - (params.n_background + params.alpha * af.cos(0.5 * nls.q1_center))
                   )
            )
     )

E1_lc = nls.fields_solver.yee_grid_EM_fields[0]
E1_rc = af.shift(E1_lc, 0, 0, -1)

initial_mass           = af.sum(nls.compute_moments('density')[:, :, N_g:-N_g, N_g]) * nls.dq1
initial_kinetic_energy = af.sum(nls.compute_moments('energy')[:, :, N_g:-N_g, N_g]) * nls.dq1
initial_em_energy      = 0.25 * af.sum(  E1_lc[:, :, N_g:-N_g, N_g]**2 
                                       + E1_rc[:, :, N_g:-N_g, N_g]**2
                                      ) * nls.dq1
initial_total_energy   = initial_kinetic_energy + initial_em_energy

for time_index, t0 in enumerate(time_array[1:]):

    n_data[time_index] = af.max(nls.compute_moments('density')[:, :, N_g:-N_g, N_g:-N_g])
    print('Computing For Time =', t0)
    nls.strang_timestep(dt)

E1_lc = nls.fields_solver.yee_grid_EM_fields[0]
E1_rc = af.shift(E1_lc, 0, 0, -1)

final_mass           = af.sum(nls.compute_moments('density')[:, :, N_g:-N_g, N_g]) * nls.dq1
final_kinetic_energy = af.sum(nls.compute_moments('energy')[:, :, N_g:-N_g, N_g]) * nls.dq1
final_em_energy      = 0.25 * af.sum(  E1_lc[:, :, N_g:-N_g, N_g]**2 
                                     + E1_rc[:, :, N_g:-N_g, N_g]**2
                                    ) * nls.dq1

final_total_energy = final_kinetic_energy + final_em_energy

print('Change in Mass:', (final_mass - initial_mass))
print('Change in KE:', (final_kinetic_energy - initial_kinetic_energy))
print('Change in EME:', (final_em_energy - initial_em_energy))
print('Change in TE:', (final_total_energy - initial_total_energy))

import pylab as pl
pl.plot(time_array[:-1], n_data[:-1])
pl.savefig('plot.png', bbox_inches = 'tight')
