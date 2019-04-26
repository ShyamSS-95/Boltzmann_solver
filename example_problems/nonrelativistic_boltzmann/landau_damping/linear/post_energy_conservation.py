import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

from post import return_moment_to_be_plotted, return_field_to_be_plotted, determine_min_max, q1, q2, dq1, dq2

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 12
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

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(dq1, dq2) \
                  / max(domain.p1_end + domain.p2_end + domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)

kinetic_energy_initial  = 0
electric_energy_initial = 0
magnetic_energy_initial = 0
total_energy_initial    = 0

kinetic_energy_data  = np.zeros([time_array.size])
electric_energy_data = np.zeros([time_array.size])
magnetic_energy_data = np.zeros([time_array.size])
total_energy_data    = np.zeros([time_array.size])
JE_data              = np.zeros([time_array.size])

kinetic_energy_error  = np.zeros([time_array.size])
electric_energy_error = np.zeros([time_array.size])
magnetic_energy_error = np.zeros([time_array.size])
total_energy_error    = np.zeros([time_array.size])

for time_index, t0 in enumerate(time_array[1:]):

    h5f       = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
    moments_n = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    h5f        = h5py.File('dump_moments2/t=%.3f'%(t0) + '.h5', 'r')
    moments_n2 = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    # Gives the electric fields at n and magnetic fields at (n + 1/2) 
    h5f      = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
    fields_n = np.swapaxes(h5f['EM_fields'][:], 0, 1)
    h5f.close()

    # NOTE: t0 corresponds to time_array[time_index + 1] since we have enumerate(time_array[1:])
    # Gives the electric fields at (n-1) and magnetic fields at (n - 1/2)
    h5f                = h5py.File('dump_fields/t=%.3f'%(time_array[time_index]) + '.h5', 'r')
    fields_n_minus_one = np.swapaxes(h5f['EM_fields'][:], 0, 1)
    h5f.close()

    E  = return_moment_to_be_plotted('energy', moments_n)
    Jx = return_moment_to_be_plotted('J1', moments_n2)

    E1 = return_field_to_be_plotted('E1', fields_n)
    E2 = return_field_to_be_plotted('E2', fields_n)
    E3 = return_field_to_be_plotted('E3', fields_n)

    B1_n_plus_half = return_field_to_be_plotted('B1', fields_n)
    B2_n_plus_half = return_field_to_be_plotted('B2', fields_n)
    B3_n_plus_half = return_field_to_be_plotted('B3', fields_n)

    B1_n_minus_half = return_field_to_be_plotted('B1', fields_n_minus_one)
    B2_n_minus_half = return_field_to_be_plotted('B2', fields_n_minus_one)
    B3_n_minus_half = return_field_to_be_plotted('B3', fields_n_minus_one)

    if(time_index == 0):
        
        kinetic_energy_initial  = np.sum(E) * dq1 * dq2
        electric_energy_initial = np.sum(E1**2 + np.roll(E1, -1)**2) * params.eps / 4 * dq1 * dq2 #np.sum(E1**2 + E2**2 + E3**2) * params.eps / 2 * dq1 * dq2
        magnetic_energy_initial = np.sum(  B1_n_minus_half * B1_n_plus_half 
                                         + B2_n_minus_half * B2_n_plus_half 
                                         + B3_n_minus_half * B3_n_plus_half
                                        ) / (2 * params.mu) * dq1 * dq2

        total_energy_initial    =   kinetic_energy_initial \
                                  + electric_energy_initial \
                                  + magnetic_energy_initial 

    kinetic_energy  = np.sum(E) * dq1 * dq2
    electric_energy = np.sum(E1**2 + np.roll(E1, -1)**2) * params.eps / 4 * dq1 * dq2 #np.sum(E1**2 + E2**2 + E3**2) * params.eps / 2 * dq1 * dq2
    magnetic_energy = np.sum(  B1_n_minus_half * B1_n_plus_half 
                             + B2_n_minus_half * B2_n_plus_half 
                             + B3_n_minus_half * B3_n_plus_half
                            ) / (2 * params.mu) * dq1 * dq2

    total_energy    =   kinetic_energy \
                      + electric_energy \
                      + magnetic_energy 

    kinetic_energy_data[time_index + 1]  = kinetic_energy
    electric_energy_data[time_index + 1] = electric_energy
    magnetic_energy_data[time_index + 1] = magnetic_energy
    total_energy_data[time_index + 1]    = total_energy
    JE_data[time_index + 1]              = np.sum(0.5 * (Jx * E1 + np.roll(Jx, -1) * np.roll(E1, -1)) * dq1 * dq2)

    kinetic_energy_error[time_index]  = abs(kinetic_energy - kinetic_energy_initial)
    electric_energy_error[time_index] = abs(electric_energy - electric_energy_initial)
    magnetic_energy_error[time_index] = abs(magnetic_energy - magnetic_energy_initial)
    total_energy_error[time_index]    = abs(total_energy - total_energy_initial)

# import h5py
# h5f = h5py.File('data_512.h5', 'w')
# h5f.create_dataset('kinetic_energy', data = kinetic_energy_data[1:])
# h5f.create_dataset('electric_energy', data = electric_energy_data[1:])
# h5f.create_dataset('magnetic_energy', data = magnetic_energy_data[1:])
# h5f.create_dataset('time', data = time_array[1:])
# h5f.close()

# h5f = h5py.File('data_128.h5', 'r')
# ke1 = h5f['kinetic_energy'][:]
# ee1 = h5f['electric_energy'][:]
# me1 = h5f['magnetic_energy'][:]
# t   = h5f['time'][:]
# h5f.close()

# te1 = ke1 + ee1 + me1

# h5f = h5py.File('data_512.h5', 'r')
# ke2 = h5f['kinetic_energy'][:]
# ee2 = h5f['electric_energy'][:]
# me2 = h5f['magnetic_energy'][:]
# h5f.close()

# te2 = ke2 + ee2 + me2

# pl.plot(t / params.t0, abs(ke1 - ke2), label = 'Kinetic Energy')
# pl.plot(t / params.t0, abs(ee1 - ee2), label = 'Electric Energy')
# pl.plot(t / params.t0, abs(te1 - te2), label = 'Total Energy')
# pl.plot(t / params.t0, abs(me1 - me2), label = 'Magnetic Energy')
# pl.semilogy(t / params.t0, abs(ke1 - ke1[0]), label = r'KE Energy($N_v = 128$)')
# pl.semilogy(t / params.t0, abs(ke2 - ke2[0]), label = r'KE Energy($N_v = 512$)')

# pl.semilogy(t / params.t0, abs(ee1 - ee1[0]), label = r'EM Energy($N_v = 128$)')
# pl.semilogy(t / params.t0, abs(ee2 - ee2[0]), label = r'EM Energy($N_v = 512$)')

# pl.semilogy(t / params.t0, abs(te1 - te1[0]), label = r'Total Energy($N_v = 128$)')
# pl.semilogy(t / params.t0, abs(te2 - te2[0]), label = r'Total Energy($N_v = 512$)')

# # # # pl.ylabel('Total Energy')
# pl.legend(fontsize = 20, bbox_to_anchor = (1, 1))
# pl.xlabel(r'Time($\omega_p^{-1}$)')
# pl.savefig('plot.png', bbox_inches = 'tight')

# pl.semilogy(time_array / params.t0, abs(kinetic_energy_error + electric_energy_error + magnetic_energy), label = r'$|$KE(t) - KE(t = 0)$|$')
# pl.plot(time_array[1:-1] / params.t0, kinetic_energy_error[1:-1], label = r'Kinetic Energy $(t)$ - Kinetic Energy$(t=0)$')
pl.subplot(311)
pl.plot(time_array[1:] / params.t0, kinetic_energy_data[1:], label = r'$\int 0.5 v^2 f dv dx$')
pl.legend(fontsize = 20)
pl.subplot(312)
pl.plot(time_array[1:] / params.t0, electric_energy_data[1:], label = r'$\int \frac{E(i)^2 + E(i + 1)^2}{4} dx$')
pl.legend(fontsize = 20)
pl.subplot(313)
pl.plot(time_array[1:] / params.t0, JE_data[1:], label = r'$\int J(i+1/2) \left(\frac{E(i) + E(i + 1)}{2}\right) dx$')
# pl.plot(time_array[1:] / params.t0, kinetic_energy_data[1:], label = 'Kinetic Energy')
# pl.plot(time_array[1:] / params.t0, total_energy_data[1:], label = 'Kinetic Energy')

# pl.plot(time_array[1:-1] / params.t0, magnetic_energy_error[1:-1] + electric_energy_error[1:-1], label = r'EM Energy $(t)$ - EM Energy$(t=0)$')
# pl.plot(time_array[1:-1] / params.t0, total_energy_error[1:-1], label = r'Total Energy $(t)$ - Total Energy$(t = 0)$')
# pl.ylabel('Total Energy')
pl.legend(fontsize = 20)
pl.xlabel(r'Time($\omega_p^{-1}$)')
pl.savefig('plot.png', bbox_inches = 'tight')
