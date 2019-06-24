import arrayfire as af
from .df_dt_fvm import df_dt_fvm
from bolt.lib.utils.broadcasted_primitive_operations import multiply
from bolt.lib.nonlinear.temporal_evolution import operator_splitting_methods as split

def timestep_fvm(self, dt):
    """
    Evolves the system defined using FVM. It does so by integrating 
    the function df_dt using an RK2 stepping scheme. After the initial 
    evaluation at the midpoint, we evaluate the currents(J^{n+0.5}) and 
    pass it to the FDTD algo when an electrodynamic case needs to be evolved.
    The FDTD algo updates the field values, which are used at the next
    evaluation of df_dt.

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    if(self.physical_system.params.energy_conserving == True):
        self.f_n_plus_half.center = self.f_n_plus_half.center + df_dt_fvm(self.f_n.center, False, self) * dt
        af.eval(self.f_n_plus_half.center)

        # These would be applied to f_n_plus_half by setting the at_n flag to False
        self._communicate_f(False)
        self._apply_bcs_f(False)

        self.f_n.center = self.f_n + multiply(df_dt_fvm(self.f_n_plus_half.center, True, self), 1 / self.p_center_squared) * dt
        af.eval(self.f_n.center)

    else:
        f_initial          = self.f_n_plus_half.center.copy()
        self.f_n_plus_half = self.f_n_plus_half + df_dt_fvm(self.f_n_plus_half, False, self) * (dt / 2)
    
        self._communicate_f(False)
        self._apply_bcs_f(False)

        self.f_n_plus_half = f_initial + df_dt_fvm(self.f_n_plus_half, False, self) * dt

def update_for_instantaneous_collisions(self, dt):
    
    self.f = self._source(self.f, self.time_elapsed,
                          self.q1_center, self.q2_center,
                          self.p1_center, self.p2_center, self.p3_center, 
                          self.compute_moments, 
                          self.physical_system.params, 
                          True
                         )

    return

def op_fvm(self, dt):
    if(self.performance_test_flag == True):
        tic = af.time()
    
    if(self.physical_system.params.energy_conserving == True):
        # These would be applied to f_n by setting the at_n flag to True
        self._communicate_f(True)
        self._apply_bcs_f(True)
    else:
        self._communicate_f(False)
        self._apply_bcs_f(False)

    if(self.physical_system.params.instantaneous_collisions == True):
        split.strang(self, timestep_fvm, update_for_instantaneous_collisions, dt)
    else:
        timestep_fvm(self, dt)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fvm_solver += toc - tic
    
    return
