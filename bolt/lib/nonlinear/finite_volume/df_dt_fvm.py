import arrayfire as af

# Importing Riemann solver used in calculating fluxes:
from .riemann import riemann_solver
from .reconstruct import reconstruct
from bolt.lib.utils.broadcasted_primitive_operations import multiply
from bolt.lib.nonlinear.communicate import communicate_fields

"""
Equation to solve:
When solving only for q-space:
df/dt + d(C_q1 * f)/dq1 + d(C_q2 * f)/dq2 = C[f]
Grid convention considered:

                 (i+1/2, j+1)
             X-------o-------X
             |               |
             |               |
  (i, j+1/2) o       o       o (i+1, j+1/2)
             | (i+1/2, j+1/2)|
             |               |
             X-------o-------X
                 (i+1/2, j)

Using the finite volume method in q-space:
d(f_{i+1/2, j+1/2})/dt  = ((- (C_q1 * f)_{i + 1, j + 1/2} + (C_q1 * f)_{i, j + 1/2})/dq1
                           (- (C_q2 * f)_{i + 1/2, j + 1} + (C_q2 * f)_{i + 1/2, j})/dq2
                           +  C[f_{i+1/2, j+1/2}]
                          )
The same concept is extended to p-space as well.                          
"""

def get_f_cell_edges_q(f, self):

    # Giving shorter name reference:
    reconstruction_in_q = self.physical_system.params.reconstruction_method_in_q

    f_left_plus_eps, f_right_minus_eps = reconstruct(self, f, 2, reconstruction_in_q)
    f_bot_plus_eps, f_top_minus_eps    = reconstruct(self, f, 3, reconstruction_in_q)

    # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
    f_left_minus_eps = af.shift(f_right_minus_eps, 0, 0, 1)
    # Extending the same to bot:
    f_bot_minus_eps  = af.shift(f_top_minus_eps,   0, 0, 0, 1)

    # Nicer variables for passing the arguments:
    args_left_center = (self.time_elapsed, self.q1_left_center, self.q2_left_center,
                        self.p1_center, self.p2_center, self.p3_center, 
                        self.physical_system.params
                       )

    args_center_bot = (self.time_elapsed, self.q1_center_bot, self.q2_center_bot,
                       self.p1_center, self.p2_center, self.p3_center, 
                       self.physical_system.params
                      )                       

    # af.broadcast used to perform batched operations on arrays of different sizes:
    self._C_q1 = af.broadcast(self._C_q, *args_left_center)[0]
    self._C_q2 = af.broadcast(self._C_q, *args_center_bot)[1]

    self.f_q1_left_q2_center = riemann_solver(self, f_left_minus_eps, 
                                              f_left_plus_eps, self._C_q1
                                             )

    self.f_q1_center_q2_bot  = riemann_solver(self, f_bot_minus_eps, 
                                              f_bot_plus_eps, self._C_q2
                                             )

    return

def df_dt_fvm(f, at_n, self, term_to_return = 'all'):
    """
    Returns the expression for df/dt which is then 
    evolved by a timestepper.

    Parameters
    ----------

    f : af.Array
        Array of the distribution function at which is used
        to evaluate the fluxes.
    """ 
    
    # Giving shorter name reference:
    reconstruction_in_p = self.physical_system.params.reconstruction_method_in_p

    # Initializing df_dt
    df_dt = 0

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        # This method is used to set C_q1, C_q2, f_q1_left_q2_center, f_q1_center_q2_bot:
        get_f_cell_edges_q(f, self)

        left_flux = multiply(self._C_q1, self.f_q1_left_q2_center)
        bot_flux  = multiply(self._C_q2, self.f_q1_center_q2_bot)

        right_flux = af.shift(left_flux, 0, 0, -1)
        top_flux   = af.shift(bot_flux,  0, 0,  0, -1)
        
        df_dt += - (right_flux - left_flux) / self.dq1 \
                 - (top_flux   - bot_flux ) / self.dq2 \

        if(    self.physical_system.params.source_enabled == True 
           and self.physical_system.params.instantaneous_collisions == False
           and self.physical_system.params.energy_conserving == False
          ):
            df_dt += self._source.source_term(self.f, self.time_elapsed, 
                                              self.q1_center, self.q2_center,
                                              self.p1_center, self.p2_center, self.p3_center, 
                                              self.compute_moments, self.fields_solver,
                                              self.physical_system.params
                                             )

        elif(    self.physical_system.params.source_enabled == True 
             and self.physical_system.params.instantaneous_collisions == False
             and self.physical_system.params.energy_conserving == False
            ):
            df_dt += self._source.source_term_energy_conserving(self.f, self.time_elapsed, 
                                                                self.q1_center, self.q2_center,
                                                                self.p1_center, self.p2_center, self.p3_center,
                                                                self.compute_moments, self.fields_solver,
                                                                self.physical_system.params
                                                                )


    if(    self.physical_system.params.solver_method_in_p == 'FVM' 
       and self.physical_system.params.fields_enabled == True
      ):
        if(self.physical_system.params.fields_type == 'electrostatic'):
            if(self.physical_system.params.fields_solver == 'fft'):
                rho = multiply(self.physical_system.params.charge,
                               self.compute_moments('density', f = f)
                              )

                self.fields_solver.compute_electrostatic_fields(rho)

        if(self.physical_system.params.fields_type == 'electrodynamic'):
            J1 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v1_bulk', f = self.f_q1_left_q2_center)
                         ) # (i, j + 1/2)

            J2 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v2_bulk', f = self.f_q1_center_q2_bot)
                         ) # (i + 1/2, j)

            J3 = multiply(self.physical_system.params.charge, 
                          self.compute_moments('mom_v3_bulk', f = f)
                         ) # (i + 1/2, j + 1/2)

            self.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, at_n, self.dt)

        # Nicer variables for passing the arguments:
        args_p_left = (self.time_elapsed, self.q1_center, self.q2_center,
                        self.p1_left, self.p2_left, self.p3_left, self.fields_solver,
                        self.physical_system.params, 'left_center'
                        )

        args_p_bottom = (self.time_elapsed, self.q1_center, self.q2_center,
                            self.p1_bottom, self.p2_bottom, self.p3_bottom, self.fields_solver,
                            self.physical_system.params, 'center_bottom'
                        )                       

        args_p_back = (self.time_elapsed, self.q1_center, self.q2_center,
                        self.p1_back, self.p2_back, self.p3_back,  self.fields_solver,
                        self.physical_system.params
                        )                       

        if(at_n == True):
            # Getting C_p at q1_left_q2_center:
            self._C_p1_left_at_q1_left_q2_center \
            = af.broadcast(self._C_p, *args_p_left)[0]
            # Getting C_p at q1_center_q2_bot:
            self._C_p2_bot_at_q1_center_q2_bot \
            = af.broadcast(self._C_p, *args_p_bottom)[1]
            # Getting C_p at q1_center_q2_center:
            self._C_p3_back_at_q1_center_q2_center \
            = af.broadcast(self._C_p, *args_p_back)[2]
        
        # Converting all variables to p-expanded:(p1, p2, p3, s * q1 * q2)
        self._C_p1_left_at_q1_left_q2_center   = self._convert_to_p_expanded(self._C_p1_left_at_q1_left_q2_center)
        self._C_p2_bot_at_q1_center_q2_bot     = self._convert_to_p_expanded(self._C_p2_bot_at_q1_center_q2_bot)
        self._C_p3_back_at_q1_center_q2_center = self._convert_to_p_expanded(self._C_p3_back_at_q1_center_q2_center)

        self.f_q1_left_q2_center = self._convert_to_p_expanded(self.f_q1_left_q2_center)
        self.f_q1_center_q2_bot  = self._convert_to_p_expanded(self.f_q1_center_q2_bot)
        f_q1_center_q2_center    = self._convert_to_p_expanded(f)

        # Variation of p1 is along axis 0:
        f_p1_left_plus_eps_at_q1_left_q2_center, f_p1_right_minus_eps_at_q1_left_q2_center \
        = reconstruct(self, self.f_q1_left_q2_center, 0, reconstruction_in_p)

        # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
        f_p1_left_minus_eps_at_q1_left_q2_center = af.shift(f_p1_right_minus_eps_at_q1_left_q2_center, 1)

        f_p1_left_at_q1_left_q2_center = riemann_solver(self, f_p1_left_minus_eps_at_q1_left_q2_center, 
                                                        f_p1_left_plus_eps_at_q1_left_q2_center, 
                                                        self._C_p1_left_at_q1_left_q2_center
                                                        )

        # Variation of p2 is along axis 1:
        f_p2_bot_plus_eps_at_q1_center_q2_bot, f_p2_top_minus_eps_at_q1_center_q2_bot \
        = reconstruct(self, self.f_q1_center_q2_bot, 1, reconstruction_in_p)

        # f_bot_minus_eps of i-th cell is f_top_minus_eps of the (i-1)th cell
        f_p2_bot_minus_eps_at_q1_center_q2_bot  = af.shift(f_p2_top_minus_eps_at_q1_center_q2_bot, 0, 1)

        f_p2_bot_at_q1_center_q2_bot  = riemann_solver(self, f_p2_bot_minus_eps_at_q1_center_q2_bot, 
                                                        f_p2_bot_plus_eps_at_q1_center_q2_bot, 
                                                        self._C_p2_bot_at_q1_center_q2_bot
                                                        )

        # Variation of p3 is along axis 2:
        f_p3_back_plus_eps_at_q1_center_q2_center, f_p3_front_minus_eps_at_q1_center_q2_center \
        = reconstruct(self, f_q1_center_q2_center, 2, reconstruction_in_p)

        # f_back_minus_eps of i-th cell is f_front_minus_eps of the (i-1)th cell
        f_p3_back_minus_eps_at_q1_center_q2_center = af.shift(f_p3_front_minus_eps_at_q1_center_q2_center, 0, 0, 1)

        f_p3_back_at_q1_center_q2_center = riemann_solver(self, f_p3_back_minus_eps_at_q1_center_q2_center, 
                                                            f_p3_back_plus_eps_at_q1_center_q2_center, 
                                                            self._C_p3_back_at_q1_center_q2_center
                                                            )
        
        # For flux along p1 at q1_left_q2_center:
        flux_p1_left_at_q1_left_q2_center \
        = multiply(self._C_p1_left_at_q1_left_q2_center, f_p1_left_at_q1_left_q2_center)

        flux_p1_right_at_q1_left_q2_center \
        = af.shift(flux_p1_left_at_q1_left_q2_center, -1)

        d_flux_p1_at_q1_left_q2_center_dp1 \
        = multiply(self._convert_to_q_expanded(  flux_p1_right_at_q1_left_q2_center \
                                                - flux_p1_left_at_q1_left_q2_center
                                                ),
                    1 / self.dp1
                    )    

        # For flux along p2 at q1_center_q2_bot:
        flux_p2_bot_at_q1_center_q2_bot \
        = multiply(self._C_p2_bot_at_q1_center_q2_bot, f_p2_bot_at_q1_center_q2_bot)

        flux_p2_top_at_q1_center_q2_bot \
        = af.shift(flux_p2_bot_at_q1_center_q2_bot, 0, -1)

        d_flux_p2_at_q1_center_q2_bot_dp2 \
        = multiply(self._convert_to_q_expanded(  flux_p2_top_at_q1_center_q2_bot \
                                                - flux_p2_bot_at_q1_center_q2_bot
                                                ),
                    1 / self.dp2
                    )    

        # For flux along p3 at q1_center_q2_center:
        flux_p3_back_at_q1_center_q2_center \
        = multiply(self._C_p3_back_at_q1_center_q2_center, f_p3_back_at_q1_center_q2_center)

        flux_p3_front_at_q1_center_q2_center \
        = af.shift(flux_p3_back_at_q1_center_q2_center,  0,  0, -1)

        d_flux_p3_at_q1_center_q2_center_dp3 \
        = multiply(self._convert_to_q_expanded(  flux_p3_front_at_q1_center_q2_center \
                                                - flux_p3_back_at_q1_center_q2_center
                                                ),
                    1 / self.dp3
                    )

        d_flux_p1_at_q1_right_q2_center_dp1 = af.shift(d_flux_p1_at_q1_left_q2_center_dp1, 0, 0, -1)
        d_flux_p2_at_q1_center_q2_top_dp2   = af.shift(d_flux_p2_at_q1_center_q2_bot_dp2, 0, 0, 0, -1)


        d_flux_p1_dp1 = 0.5 * (  d_flux_p1_at_q1_left_q2_center_dp1 
                                + d_flux_p1_at_q1_right_q2_center_dp1
                                )

        d_flux_p2_dp2 = 0.5 * (  d_flux_p2_at_q1_center_q2_bot_dp2
                                + d_flux_p2_at_q1_center_q2_top_dp2
                                )

        d_flux_p3_dp3 = d_flux_p3_at_q1_center_q2_center_dp3

        # J1 = multiply(self.physical_system.params.charge,
        #               self.compute_moments('mom_v1_bulk', f = self._convert_to_q_expanded(self.f_q1_left_q2_center))
        #              ) # (i, j + 1/2)

        # E1, E2, E3, B1, B2, B3 = self.fields_solver.get_fields('left_center')
        # import pylab as pl
        # pl.style.use('latexplot')
        # pl.plot(af.flat(self.compute_moments('energy', f = self._convert_to_q_expanded(d_flux_p1_at_q1_left_q2_center_dp1))[0, 0, :, 0]), label = r'$\int \frac{eE|v|^2}{2} \frac{\partial f}{\partial v} dv$')
        # pl.plot(-af.flat((E1 * J1)[0, 0, :, 0]), '--', color = 'black', label = r'$-J \cdot E$')
        # pl.legend(bbox_to_anchor = (1, 1))
        # pl.savefig('plot.png', bbox_inches = 'tight')
        # pl.show()

        # \int 0.5 * |v|^2 * F * df/dv
        # print(af.sum(self.compute_moments('energy', f = self._convert_to_q_expanded(d_flux_p1_at_q1_left_q2_center_dp1))))

        # -J.E
        # print(-af.sum(E1 * J1))
        # print(abs(-af.sum(E1 * J1) - af.sum(self.compute_moments('energy', f = self._convert_to_q_expanded(d_flux_p1_at_q1_left_q2_center_dp1)))))

        df_dt += -(d_flux_p1_dp1 + d_flux_p2_dp2 + d_flux_p3_dp3)
            
    if(term_to_return == 'd_flux_p1_dp1'):
        af.eval(d_flux_p1_dp1)
        return(d_flux_p1_dp1)

    elif(term_to_return == 'd_flux_p2_dp2'):
        af.eval(d_flux_p2_dp2)
        return(d_flux_p2_dp2)

    if(term_to_return == 'd_flux_p3_dp3'):
        af.eval(d_flux_p3_dp3)
        return(d_flux_p3_dp3)

    else:
        af.eval(df_dt)
        return(df_dt)
