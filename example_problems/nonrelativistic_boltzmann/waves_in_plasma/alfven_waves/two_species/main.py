import arrayfire as af
import numpy as np
import pylab as pl
import math
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.nonlinear.finite_volume.df_dt_fvm import df_dt_fvm

import domain
import boundary_conditions
import initialize
import params

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.sources as sources
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 16, 10 #10, 14
pl.rcParams['figure.dpi']      = 80
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 30
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1

# Time parameters:
dt_fvm = params.N_cfl * dq1 \
                      / max(domain.p1_end + domain.p2_end + domain.p3_end) # joining elements of the list

dt_fdtd = params.N_cfl * dq1 \
                       / params.c # lightspeed

dt        = min(dt_fvm, dt_fdtd)
params.dt = dt

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         sources,
                         moments
                        )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
N_g = nls.N_ghost

print('Minimum Value of f_e:', af.min(nls.f[:, 0]))
print('Minimum Value of f_i:', af.min(nls.f[:, 1]))

print('Error in density_e:', af.mean(af.abs(nls.compute_moments('density')[:, 0] - 1)))
print('Error in density_i:', af.mean(af.abs(nls.compute_moments('density')[:, 1] - 1)))

v2_bulk = nls.compute_moments('mom_v2_bulk') / nls.compute_moments('density')
v3_bulk = nls.compute_moments('mom_v3_bulk') / nls.compute_moments('density')


v2_bulk_i =   params.amplitude * -4.801714581503802e-15 * af.cos(params.k_q1 * nls.q1_center) \
            - params.amplitude *  0.6363571202013185 * af.sin(params.k_q1 * nls.q1_center)

v2_bulk_e =   params.amplitude * -4.85722573273506e-15 * af.cos(params.k_q1 * nls.q1_center) \
            - params.amplitude * 0.10249033165518363 * af.sin(params.k_q1 * nls.q1_center)

v3_bulk_i =   params.amplitude * 0.6363571202013188 * af.cos(params.k_q1 * nls.q1_center) \
            - params.amplitude * 1.8041124150158794e-16  * af.sin(params.k_q1 * nls.q1_center)

v3_bulk_e =   params.amplitude * 0.10249033165518295 * af.cos(params.k_q1 * nls.q1_center) \
            - params.amplitude * -3.885780586188048e-16 * af.sin(params.k_q1 * nls.q1_center)

print('Error in v2_bulk_e:', af.mean(af.abs((v2_bulk[:, 0] - v2_bulk_e) / v2_bulk_e)))
print('Error in v2_bulk_i:', af.mean(af.abs((v2_bulk[:, 1] - v2_bulk_i) / v2_bulk_i)))
print('Error in v3_bulk_e:', af.mean(af.abs((v3_bulk[:, 0] - v3_bulk_e) / v3_bulk_e)))
print('Error in v3_bulk_i:', af.mean(af.abs((v3_bulk[:, 1] - v3_bulk_i) / v3_bulk_i)))

if(params.t_restart == 0):
    time_elapsed = 0
    nls.dump_distribution_function('dump_f/t=0.000')
    nls.dump_moments('dump_moments/t=0.000')
    nls.dump_EM_fields('dump_fields/t=0.000')

else:
    time_elapsed = params.t_restart
    nls.load_distribution_function('dump_f/t=' + '%.3f'%time_elapsed)
    nls.load_EM_fields('dump_fields/t=' + '%.3f'%time_elapsed)

# Checking that the file writing intervals are greater than dt:
assert(params.dt_dump_f > dt)
assert(params.dt_dump_moments > dt)
assert(params.dt_dump_fields > dt)

E1_lc = nls.fields_solver.yee_grid_EM_fields[0]
E2_cb = nls.fields_solver.yee_grid_EM_fields[1]
E3_cc = nls.fields_solver.yee_grid_EM_fields[2]

B1_cb = nls.fields_solver.yee_grid_EM_fields[3]
B2_lc = nls.fields_solver.yee_grid_EM_fields[4]
B3_lb = nls.fields_solver.yee_grid_EM_fields[5]

E1_rc = af.shift(E1_lc, 0, 0, -1)
E2_ct = af.shift(E1_lc, 0, 0,  0, -1)

B1_ct = af.shift(E1_lc, 0, 0,  0, -1)
B2_rc = af.shift(E1_lc, 0, 0, -1)

B3_rb = af.shift(E1_lc, 0, 0, -1)
B3_lt = af.shift(E1_lc, 0, 0,  0, -1)
B3_rt = af.shift(E1_lc, 0, 0, -1, -1)

initial_mass           = af.sum(nls.compute_moments('density')[:, :, N_g:-N_g, N_g]) * nls.dq1
initial_kinetic_energy = af.sum(nls.compute_moments('energy')[:, :, N_g:-N_g, N_g]) * nls.dq1
initial_e_energy       = 0.25 * af.sum(  E1_lc[:, :, N_g:-N_g, N_g]**2 
                                       + E1_rc[:, :, N_g:-N_g, N_g]**2
                                       + E2_cb[:, :, N_g:-N_g, N_g]**2
                                       + E2_ct[:, :, N_g:-N_g, N_g]**2
                                       + 2 * E3_cc[:, :, N_g:-N_g, N_g]**2
                                      ) * nls.dq1

initial_m_energy       = 0.125 * af.sum(  2 * B1_cb[:, :, N_g:-N_g, N_g]**2 
                                        + 2 * B1_ct[:, :, N_g:-N_g, N_g]**2
                                        + 2 * B2_lc[:, :, N_g:-N_g, N_g]**2
                                        + 2 * B2_rc[:, :, N_g:-N_g, N_g]**2
                                        + B3_lb[:, :, N_g:-N_g, N_g]**2
                                        + B3_lt[:, :, N_g:-N_g, N_g]**2
                                        + B3_rb[:, :, N_g:-N_g, N_g]**2
                                        + B3_rt[:, :, N_g:-N_g, N_g]**2
                                       ) * nls.dq1

initial_em_energy    = initial_e_energy + initial_m_energy
initial_total_energy = initial_kinetic_energy + initial_em_energy

# E  = return_moment_to_be_plotted('energy', moments_n)
# Jx = return_moment_to_be_plotted('J1', moments_n2)

# E1 = return_field_to_be_plotted('E1', fields_n)
# E2 = return_field_to_be_plotted('E2', fields_n)
# E3 = return_field_to_be_plotted('E3', fields_n)

# B1_n_plus_half = return_field_to_be_plotted('B1', fields_n)
# B2_n_plus_half = return_field_to_be_plotted('B2', fields_n)
# B3_n_plus_half = return_field_to_be_plotted('B3', fields_n)

# B1_n_minus_half = return_field_to_be_plotted('B1', fields_n_minus_one)
# B2_n_minus_half = return_field_to_be_plotted('B2', fields_n_minus_one)
# B3_n_minus_half = return_field_to_be_plotted('B3', fields_n_minus_one)

# kinetic_energy_initial  = np.sum(E) * dq1 * dq2
# electric_energy_initial = np.sum(E1**2 + E2**2 + E3**2) * params.eps / 2 * dq1 * dq2
# magnetic_energy_initial = np.sum(  B1_n_minus_half * B1_n_plus_half 
#                                  + B2_n_minus_half * B2_n_plus_half 
#                                  + B3_n_minus_half * B3_n_plus_half
#                                 ) / (2 * params.mu) * dq1 * dq2


while(abs(time_elapsed - params.t_final) > 1e-5):
    
    nls.strang_timestep(dt)
    time_elapsed += dt

    if(params.dt_dump_moments != 0):

        # We step by delta_dt to get the values at dt_dump
        delta_dt =   (1 - math.modf(time_elapsed/params.dt_dump_moments)[0]) \
                   * params.dt_dump_moments

        if(delta_dt<dt):
            nls.strang_timestep(delta_dt)
            time_elapsed += delta_dt
            nls.dump_moments('dump_moments/t=' + '%.3f'%time_elapsed)
            nls.dump_EM_fields('dump_fields/t=' + '%.3f'%time_elapsed)

    if(math.modf(time_elapsed/params.dt_dump_f)[0] < 1e-5):
        nls.dump_distribution_function('dump_f/t=' + '%.3f'%time_elapsed)

    PETSc.Sys.Print('Computing For Time =', time_elapsed / params.t0, "|t0| units(t0)")

E1_lc = nls.fields_solver.yee_grid_EM_fields[0]
E2_cb = nls.fields_solver.yee_grid_EM_fields[1]
E3_cc = nls.fields_solver.yee_grid_EM_fields[2]

B1_cb = nls.fields_solver.yee_grid_EM_fields[3]
B2_lc = nls.fields_solver.yee_grid_EM_fields[4]
B3_lb = nls.fields_solver.yee_grid_EM_fields[5]

E1_rc = af.shift(E1_lc, 0, 0, -1)
E2_ct = af.shift(E1_lc, 0, 0,  0, -1)

B1_ct = af.shift(E1_lc, 0, 0,  0, -1)
B2_rc = af.shift(E1_lc, 0, 0, -1)

B3_rb = af.shift(E1_lc, 0, 0, -1)
B3_lt = af.shift(E1_lc, 0, 0,  0, -1)
B3_rt = af.shift(E1_lc, 0, 0, -1, -1)

final_mass           = af.sum(nls.compute_moments('density')[:, :, N_g:-N_g, N_g]) * nls.dq1
final_kinetic_energy = af.sum(nls.compute_moments('energy')[:, :, N_g:-N_g, N_g]) * nls.dq1
final_e_energy       = 0.25 * af.sum(  E1_lc[:, :, N_g:-N_g, N_g]**2 
                                     + E1_rc[:, :, N_g:-N_g, N_g]**2
                                     + E2_cb[:, :, N_g:-N_g, N_g]**2
                                     + E2_ct[:, :, N_g:-N_g, N_g]**2
                                     + 2 * E3_cc[:, :, N_g:-N_g, N_g]**2
                                    ) * nls.dq1

final_m_energy = 0.125 * af.sum(  2 * B1_cb[:, :, N_g:-N_g, N_g]**2 
                                + 2 * B1_ct[:, :, N_g:-N_g, N_g]**2
                                + 2 * B2_lc[:, :, N_g:-N_g, N_g]**2
                                + 2 * B2_rc[:, :, N_g:-N_g, N_g]**2
                                + B3_lb[:, :, N_g:-N_g, N_g]**2
                                + B3_lt[:, :, N_g:-N_g, N_g]**2
                                + B3_rb[:, :, N_g:-N_g, N_g]**2
                                + B3_rt[:, :, N_g:-N_g, N_g]**2
                               ) * nls.dq1

final_em_energy    = final_e_energy + final_m_energy
final_total_energy = final_kinetic_energy + final_em_energy

print('Change in Mass:', (final_mass - initial_mass))
print('Change in KE:', (final_kinetic_energy - initial_kinetic_energy))
print('Change in EE:', (final_e_energy - initial_e_energy))
print('Change in ME:', (final_m_energy - initial_m_energy))
print('Change in EME:', (final_em_energy - initial_em_energy))
print('Change in TE:', (final_total_energy - initial_total_energy))
