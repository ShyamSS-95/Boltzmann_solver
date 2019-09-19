import numpy as np
from petsc4py import PETSc
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

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



def return_f(file_name, N_q1, N_q2, N_p1, N_p2, N_p3):
    """
    Returns the distribution function in the desired format of 
    (q1, q2, p1, p2, p3), when the file that is written by the 
    dump_distribution_function is passed as the argument

    Parameters
    ----------

    file_name : str
                Pass the name of the file that needs to be read as
                a string. This should be the file that is written by
                dump_distribution_function
    """
    da_f  = PETSc.DMDA().create([N_q1, N_q2], 
                             dof=(N_p1*N_p2*N_p3), stencil_width=0
                           )
    f_vec = da_f.createGlobalVec()

    # When written using the routine dump_distribution_function, 
    # distribution function gets written in the format (q2, q1, p1 * p2 * p3 * Ns)
    viewer = PETSc.Viewer().createBinary(file_name, 
                                         PETSc.Viewer.Mode.READ, 
                                        )

    f_vec.load(viewer)
    f = da_f.getVecArray(f_vec) # [N_q1, N_q2, N_p1*N_p2*N_p3*N_s]

    f = np.array(f[:]).reshape([N_q1, N_q2, N_p3, N_p2, N_p1])

    return f

# Checking the errors
def check_convergence():
    N     = np.array([128, 192, 256, 384, 512]) #2**np.arange(7, 10)
    error = np.zeros(N.size)
    
    for i in range(N.size):

        nls_f = return_f('dump_files/nlsf_' + str(N[i]) + '.bin', N[i], 3, N[i], 1, 1)
        ls_f = return_f('dump_files/lsf_' + str(N[i]) + '.bin', N[i], 3, N[i], 1, 1)

        error[i] = np.mean(abs(nls_f - ls_f))

    print(error)
    poly = np.polyfit(np.log10(N), np.log10(error), 1)
    print(poly)

    pl.loglog(N, error, 'o-', label = 'Numerical')
    pl.loglog(N, error[0]*128**2/N**2, '--', color = 'black', 
              label = r'$O(N^{-2})$'
             )
    pl.legend(loc = 'best')
    pl.ylabel('Error')
    pl.xlabel('$N$')
    pl.savefig('convergence_plot.svg')
    pl.savefig('convergence_plot.png')

    assert(abs(poly[0] + 2)<0.25)
