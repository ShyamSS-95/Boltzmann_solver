import arrayfire as af
import numpy as np
import petsc4py
import sys
import h5py
import pylab as pl

petsc4py.init(sys.argv)

from bolt.lib.physical_system import physical_system
from bolt.lib.linear_solver.linear_solver import linear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms

import bolt.src.nonrelativistic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.nonrelativistic_boltzmann.moment_defs as moment_defs

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
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

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moment_defs
                        )

# Declaring a linear system object which will evolve the defined physical system:
ls  = linear_solver(system)

# Time parameters:
dt      = 0.001
t_final = 4.999

time_array = np.arange(0, t_final + dt, dt)

# Initializing Array used in storing the data:
rho_hat_data = af.constant(0, time_array.size, ls.N_q1, dtype = af.Dtype.c64)
rho_data     = np.zeros_like(time_array)

k        = 2 * np.pi * np.fft.fftfreq(ls.N_q1, ls.dq1)[:int(ls.N_q1/2)]
omega    = 2 * np.pi * np.fft.fftfreq(time_array.size, dt)[:int(time_array.size/2)]
k, omega = np.meshgrid(k, omega)

for time_index, t0 in enumerate(time_array):
    print('Computing For Time =', t0)

    n = ls.compute_moments('density')
    rho_hat_data[time_index, :] = af.reorder(af.fft(n-1)[:, 0])
    rho_data[time_index]        = af.max(n)
    ls.RK2_timestep(dt)

rho_hat_hat = af.fft(rho_hat_data)

# pl.contourf(omega, k, (np.array(rho_hat_hat).real)[:int(time_array.size/2),:int(ls.N_q1/2)] ,100)
# pl.xlabel(r'$\omega$')
# pl.ylabel(r'$k$')
# pl.title(r'$\Re(\hat{\hat{\rho}})$')
# pl.colorbar()
# pl.savefig('plot1.png')
# pl.clf()

# pl.contourf(omega, k, (np.array(rho_hat_hat).imag)[:int(time_array.size/2),:int(ls.N_q1/2)] ,100)
# pl.xlabel(r'$\omega$')
# pl.ylabel(r'$k$')
# pl.title(r'$\Im(\hat{\hat{\rho}})$')
# pl.colorbar()
# pl.savefig('plot2.png')
# pl.clf()

# pl.plot(time_array, rho_data)
# pl.xlabel(r'$t$')
# pl.ylabel(r'$\rho$')
# pl.savefig('plot3.png')

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('rho_hat_hat', data = rho_hat_hat[:int(time_array.size/2),:int(ls.N_q1/2)])
h5f.create_dataset('rho', data = rho_data)
h5f.create_dataset('omega', data = omega)
h5f.create_dataset('k', data = k)
h5f.close()
