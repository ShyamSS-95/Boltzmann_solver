import cks.initialize as initialize
import cks.evolve as evolve
from cks.poisson_solvers import fft_poisson
from cks.boundary_conditions.periodic import periodic_x, periodic_y
# import matplotlib as mpl
# mpl.use("Agg")
import pylab as pl
import arrayfire as af
import numpy as np
import params

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

print(af.info())

config = initialize.set(params)

x     = initialize.calculate_x(config)
vel_x = initialize.calculate_vel_x(config)
y     = initialize.calculate_y(config)
vel_y = initialize.calculate_vel_y(config)

dv_x = af.sum(vel_x[0, 0, 0, 1] - vel_x[0, 0, 0, 0])
dv_y = af.sum(vel_y[0, 0, 1, 0] - vel_y[0, 0, 0, 0])

f_initial    = initialize.f_initial(config)
f_background = initialize.f_background(config)
time_array   = initialize.time_array(config)

normalization = af.sum(f_background) * dv_x * dv_y/(x.shape[0] * x.shape[1])
f_initial     = f_initial/normalization

class args:
    pass

args.config = config
args.f      = f_initial
args.vel_x  = vel_x
args.vel_y  = vel_y
args.x      = x
args.y      = y

dx = af.sum(x[0, 1, 0, 0] - x[0, 0, 0, 0])
dy = af.sum(y[1, 0, 0, 0] - y[0, 0, 0, 0])

N_x = config.N_x
N_y = config.N_y

N_ghost_x = config.N_ghost_x
N_ghost_y = config.N_ghost_y

E_x_local, E_y_local = fft_poisson(config.charge_particle*evolve.calculate_density(args)\
                                   [N_ghost_y:-N_ghost_y - 1, N_ghost_x:-N_ghost_x - 1], \
                                   dx, \
                                   dy 
                                  )

E_x_local = af.join(0, E_x_local, E_x_local[0])
E_x_local = af.join(1, E_x_local, E_x_local[:, 0])

E_y_local = af.join(0, E_y_local, E_y_local[0])
E_y_local = af.join(1, E_y_local, E_y_local[:, 0])

E_x = af.constant(0, N_y + 2*N_ghost_y, N_x + 2*N_ghost_x, dtype=af.Dtype.c64)
E_y = af.constant(0, N_y + 2*N_ghost_y, N_x + 2*N_ghost_x, dtype=af.Dtype.c64)

E_x[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] = E_x_local

E_x                                             = periodic_x(config, E_x)
E_x                                             = periodic_y(config, E_x)

E_y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] = E_y_local
E_y                                             = periodic_x(config, E_y)
E_y                                             = periodic_y(config, E_y)

E_x = af.real(E_x)
E_y = af.real(E_y)

# E_x and E_y obtained above are calculated at (i, j)
# All CK quantities have been calculated at (i, j)
# We interpolate in the following set of lines to obtain at:
# B_x_fdtd --> (i, j + 1/2)
# B_y_fdtd --> (i + 1/2, j)
# B_z_fdtd --> (i + 1/2, j + 1/2)

# E_x_fdtd --> (i + 1/2, j)
# E_y_fdtd --> (i, j + 1/2)
# E_z_fdtd --> (i, j)

E_x_fdtd = 0.5 * (E_x + af.shift(E_x, 0, -1))
E_x_fdtd = periodic_x(config, E_x_fdtd)
E_x_fdtd = periodic_y(config, E_x_fdtd)

E_y_fdtd = 0.5 * (E_y + af.shift(E_y, -1, 0))
E_y_fdtd = periodic_x(config, E_y_fdtd)
E_y_fdtd = periodic_y(config, E_y_fdtd)

args.E_x = E_x_fdtd
args.E_y = E_y_fdtd
args.B_z = af.constant(0, E_x.shape[0], E_x.shape[1], dtype = af.Dtype.f64)
args.B_x = af.constant(0, E_x.shape[0], E_x.shape[1], dtype = af.Dtype.f64)
args.B_y = af.constant(0, E_x.shape[0], E_x.shape[1], dtype = af.Dtype.f64)
args.E_z = af.constant(0, E_x.shape[0], E_x.shape[1], dtype = af.Dtype.f64)

data, f_final = evolve.time_integration(args, time_array)

pl.plot(time_array, data - 1)
pl.xlabel('Time')
pl.ylabel(r'$MAX(\delta \rho(x))$')
pl.savefig('plot.png')
