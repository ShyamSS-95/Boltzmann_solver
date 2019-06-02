#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def apply_shearing_box_bcs_f(self, boundary, at_n):
    """
    Applies the shearing box boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.

    at_n - bool
           When toggled to true, the boundary conditions are applied onto f_n.
           Otherwise it uses f_n_plus_half
    """

    N_g = self.N_ghost
    q     = self.physical_system.params.q 
    omega = self.physical_system.params.omega
    
    L_q1  = self.q1_end - self.q1_start
    L_q2  = self.q2_end - self.q2_start

    if(boundary == 'left'):
        sheared_coordinates = self.q2_center[:, :, :N_g] - q * omega * L_q1 * self.time_elapsed
        
        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)
        if(at_n):
            self.f_n[:, :, :N_g] = af.reorder(af.approx2(af.reorder(self.f_n[:, :, :N_g], 2, 3, 0, 1),
                                                         af.reorder(self.q1_center[:, :, :N_g], 2, 3, 0, 1),
                                                         af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                         af.INTERP.BICUBIC_SPLINE,
                                                         xp = af.reorder(self.q1_center[:, :, :N_g], 2, 3, 0, 1),
                                                         yp = af.reorder(self.q2_center[:, :, :N_g], 2, 3, 0, 1)
                                                        ),
                                              2, 3, 0, 1
                                             )
        else:
            self.f_n_plus_half[:, :, :N_g] = af.reorder(af.approx2(af.reorder(self.f_n_plus_half[:, :, :N_g], 2, 3, 0, 1),
                                                                   af.reorder(self.q1_center[:, :, :N_g], 2, 3, 0, 1),
                                                                   af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                                   af.INTERP.BICUBIC_SPLINE,
                                                                   xp = af.reorder(self.q1_center[:, :, :N_g], 2, 3, 0, 1),
                                                                   yp = af.reorder(self.q2_center[:, :, :N_g], 2, 3, 0, 1)
                                                                  ),
                                                        2, 3, 0, 1
                                                       )

    elif(boundary == 'right'):
        sheared_coordinates = self.q2_center[:, :, -N_g:] + q * omega * L_q1 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)
        if(at_n):
            self.f_n[:, :, -N_g:] = af.reorder(af.approx2(af.reorder(self.f_n[:, :, -N_g:], 2, 3, 0, 1),
                                                        af.reorder(self.q1_center[:, :, -N_g:], 2, 3, 0, 1),
                                                        af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                        af.INTERP.BICUBIC_SPLINE,
                                                        xp = af.reorder(self.q1_center[:, :, -N_g:], 2, 3, 0, 1),
                                                        yp = af.reorder(self.q2_center[:, :, -N_g:], 2, 3, 0, 1)
                                                        ),
                                                2, 3, 0, 1
                                            )
        else:
            self.f_n_plus_half[:, :, -N_g:] = af.reorder(af.approx2(af.reorder(self.f_n_plus_half[:, :, -N_g:], 2, 3, 0, 1),
                                                        af.reorder(self.q1_center[:, :, -N_g:], 2, 3, 0, 1),
                                                        af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                        af.INTERP.BICUBIC_SPLINE,
                                                        xp = af.reorder(self.q1_center[:, :, -N_g:], 2, 3, 0, 1),
                                                        yp = af.reorder(self.q2_center[:, :, -N_g:], 2, 3, 0, 1)
                                                        ),
                                                2, 3, 0, 1
                                            )

    elif(boundary == 'bottom'):

        sheared_coordinates = self.q1_center[:, :, :, :N_g] - q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        if(at_n):
            self.f_n[:, :, :, :N_g] = af.reorder(af.approx2(af.reorder(self.f_n[:, :, :, :N_g], 2, 3, 0, 1),
                                                            af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                            af.reorder(self.q2_center[:, :, :, :N_g], 2, 3, 0, 1),
                                                            af.INTERP.BICUBIC_SPLINE,
                                                            xp = af.reorder(self.q1_center[:, :, :, :N_g], 2, 3, 0, 1),
                                                            yp = af.reorder(self.q2_center[:, :, :, :N_g], 2, 3, 0, 1)
                                                        ),
                                                2, 3, 0, 1
                                                )

        else:
            self.f_n_plus_half[:, :, :, :N_g] = af.reorder(af.approx2(af.reorder(self.f_n_plus_half[:, :, :, :N_g], 2, 3, 0, 1),
                                                            af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                            af.reorder(self.q2_center[:, :, :, :N_g], 2, 3, 0, 1),
                                                            af.INTERP.BICUBIC_SPLINE,
                                                            xp = af.reorder(self.q1_center[:, :, :, :N_g], 2, 3, 0, 1),
                                                            yp = af.reorder(self.q2_center[:, :, :, :N_g], 2, 3, 0, 1)
                                                        ),
                                                2, 3, 0, 1
                                                )

    elif(boundary == 'top'):

        sheared_coordinates = self.q1_center[:, :, :, -N_g:] + q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )
        
        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        if(at_n):
            self.f_n[:, :, :, -N_g:] = af.reorder(af.approx2(af.reorder(self.f_n[:, :, :, -N_g:], 2, 3, 0, 1),
                                                            af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                            af.reorder(self.q2_center[:, :, :, -N_g:], 2, 3, 0, 1),
                                                            af.INTERP.BICUBIC_SPLINE,
                                                            xp = af.reorder(self.q1_center[:, :, :, -N_g:], 2, 3, 0, 1),
                                                            yp = af.reorder(self.q2_center[:, :, :, -N_g:], 2, 3, 0, 1)
                                                            ),
                                                2, 3, 0, 1
                                                )
        else:
            self.f_n_plus_half[:, :, :, -N_g:] = af.reorder(af.approx2(af.reorder(self.f_n_plus_half[:, :, :, -N_g:], 2, 3, 0, 1),
                                                            af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                            af.reorder(self.q2_center[:, :, :, -N_g:], 2, 3, 0, 1),
                                                            af.INTERP.BICUBIC_SPLINE,
                                                            xp = af.reorder(self.q1_center[:, :, :, -N_g:], 2, 3, 0, 1),
                                                            yp = af.reorder(self.q2_center[:, :, :, -N_g:], 2, 3, 0, 1)
                                                            ),
                                                2, 3, 0, 1
                                                )


    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_dirichlet_bcs_f(self, boundary, at_n):
    """
    Applies Dirichlet boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.

    at_n: bool
          When toggled to true, the boundary conditions are applied onto f_n.
          Otherwise it uses f_n_plus_half
    """

    N_g = self.N_ghost
    
    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        if(at_n):
            velocity_q1, velocity_q2 = \
                af.broadcast(self._C_q_n, self.time_elapsed, 
                             self.q1_center, self.q2_center,
                             self.p1_center, self.p2_center, self.p3_center,
                             self.physical_system.params
                            )
        else:
            velocity_q1, velocity_q2 = \
                af.broadcast(self._C_q_n_plus_half, self.time_elapsed, 
                             self.q1_center, self.q2_center,
                             self.p1_center, self.p2_center, self.p3_center,
                             self.physical_system.params
                            )

    else:
        velocity_q1, velocity_q2 = \
            af.broadcast(self._A_q, self.time_elapsed, 
                         self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center,
                         self.physical_system.params
                        )

    if(velocity_q1.elements() == self.N_species * self.N_p1 * self.N_p2 * self.N_p3):
        # If velocity_q1 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, 1, Nq1, Nq2)
        velocity_q1 = af.tile(velocity_q1, 1, 1,
                              self.f_n.shape[2],
                              self.f_n.shape[3]
                             )

    if(velocity_q2.elements() == self.N_species * self.N_p1 * self.N_p2 * self.N_p3):
        # If velocity_q2 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, 1, Nq1, Nq2)
        velocity_q2 = af.tile(velocity_q2, 1, 1,
                              self.f_n.shape[2],
                              self.f_n.shape[3]
                             )

    # Arguments that are passing to the called functions:
    if(at_n):
        args = (self.f_n, self.time_elapsed, self.q1_center, self.q2_center,
                self.p1_center, self.p2_center, self.p3_center, 
                self.physical_system.params
               )
    else:
        args = (self.f_n_plus_half, self.time_elapsed, self.q1_center, self.q2_center,
                self.p1_center, self.p2_center, self.p3_center, 
                self.physical_system.params
               )

    if(boundary == 'left'):
        f_left = self.boundary_conditions.f_left(*args)
        # Only changing inflowing characteristics:
        if(at_n):
            f_left = af.select(velocity_q1>0, f_left, self.f_n)
            self.f_n[:, :, :N_g] = f_left[:, :, :N_g]
        else:
            f_left = af.select(velocity_q1>0, f_left, self.f_n_plus_half)
            self.f_n_plus_half[:, :, :N_g] = f_left[:, :, :N_g]

    elif(boundary == 'right'):
        f_right = self.boundary_conditions.f_right(*args)
        # Only changing inflowing characteristics:
        if(at_n):
            f_right = af.select(velocity_q1<0, f_right, self.f_n)
            self.f_n[:, :, -N_g:] = f_right[:, :, -N_g:]
        else:
            f_right = af.select(velocity_q1<0, f_right, self.f_n_plus_half)
            self.f_n_plus_half[:, :, -N_g:] = f_right[:, :, -N_g:]

    elif(boundary == 'bottom'):
        f_bottom = self.boundary_conditions.f_bottom(*args)
        # Only changing inflowing characteristics:
        if(at_n):
            f_bottom = af.select(velocity_q2>0, f_bottom, self.f_n)
            self.f_n[:, :, :, :N_g] = f_bottom[:, :, :, :N_g]
        else:
            f_bottom = af.select(velocity_q2>0, f_bottom, self.f_n_plus_half)
            self.f_n_plus_half[:, :, :, :N_g] = f_bottom[:, :, :, :N_g]

    elif(boundary == 'top'):
        f_top = self.boundary_conditions.f_top(*args)
        # Only changing inflowing characteristics:
        if(at_n):
            f_top = af.select(velocity_q2<0, f_top, self.f_n)
            self.f_n[:, :, :, -N_g:] = f_top[:, :, :, -N_g:]
        else:
            f_top = af.select(velocity_q2<0, f_top, self.f_n_plus_half)
            self.f_n_plus_half[:, :, :, -N_g:] = f_top[:, :, :, -N_g:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_mirror_bcs_f(self, boundary):
    """
    Applies mirror boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.

    at_n: bool
          When toggled to true, the boundary conditions are applied onto f_n.
          Otherwise it uses f_n_plus_half
    """

    N_g = self.N_ghost

    if(boundary == 'left'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        if(at_n):
            self.f_n[:, :, :N_g] = af.flip(self.f_n[:, :, N_g:2 * N_g], 2)
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p1
            self.f_n[:, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f_n), 
                                                    0
                                                )
                                        )[:, :, :N_g]

        else:
            self.f_n_plus_half[:, :, :N_g] = af.flip(self.f_n_plus_half[:, :, N_g:2 * N_g], 2)
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p1
            self.f_n_plus_half[:, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f_n_plus_half), 
                                                    0
                                                )
                                        )[:, :, :N_g]

    elif(boundary == 'right'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        if(at_n):
            self.f_n[:, :, -N_g:] = af.flip(self.f_n[:, :, -2 * N_g:-N_g], 2)
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p1
            self.f_n[:, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f_n), 
                                                    0
                                                )
                                        )[:, :, -N_g:]

        else:
            self.f_n_plus_half[:, :, -N_g:] = af.flip(self.f_n_plus_half[:, :, -2 * N_g:-N_g], 2)
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p1
            self.f_n_plus_half[:, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f_n_plus_half), 
                                                    0
                                                )
                                        )[:, :, -N_g:]

    elif(boundary == 'bottom'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        if(at_n):
            self.f_n[:, :, :, :N_g] = af.flip(self.f_n[:, :, :, N_g:2 * N_g], 3)
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p2
            self.f_n[:, :, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f_n), 
                                                    1
                                                )
                                        )[:, :, :, :N_g]

        else:
            self.f_n_plus_half[:, :, :, :N_g] = af.flip(self.f_n_plus_half[:, :, :, N_g:2 * N_g], 3)
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p2
            self.f_n_plus_half[:, :, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f_n_plus_half), 
                                                    1
                                                )
                                        )[:, :, :, :N_g]

    elif(boundary == 'top'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        if(at_n):
            self.f_n[:, :, :, -N_g:] = af.flip(self.f_n[:, :, :, -2 * N_g:-N_g], 3)
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p2
            self.f_n[:, :, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f_n), 
                                                    1
                                                )
                                        )[:, :, :, -N_g:]
        else:
            self.f_n_plus_half[:, :, :, -N_g:] = af.flip(self.f_n_plus_half[:, :, :, -2 * N_g:-N_g], 3)
            # The points in the ghost zone need to have direction 
            # of velocity reversed as compared to the physical zones 
            # they are mirroring. To do this we flip the axis that 
            # contains the variation in p2
            self.f_n_plus_half[:, :, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f_n_plus_half), 
                                                    1
                                                )
                                        )[:, :, :, -N_g:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_bcs_f(self, at_n):
    """
    Applies boundary conditions to the distribution function as specified by 
    the user in params.

    Parameters
    ----------
    at_n: bool
          When toggled to true, the boundary conditions are applied onto f_n.
          Otherwise it uses f_n_plus_half
    """

    if(self.performance_test_flag == True):
        tic = af.time()

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    # Obtaining the end coordinates for the local zone
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    # If local zone includes the left physical boundary:
    if(i_q1_start == 0):

        if(self.boundary_conditions.in_q1_left == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'left', at_n)

        elif(self.boundary_conditions.in_q1_left == 'mirror'):
            apply_mirror_bcs_f(self, 'left', at_n)            

        elif(self.boundary_conditions.in_q1_left == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'left', at_n)            
            apply_dirichlet_bcs_f(self, 'left', at_n)
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q1_left == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q1_left == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'left', at_n)

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')
    
    # If local zone includes the right physical boundary:
    if(i_q1_end == self.N_q1 - 1):

        if(self.boundary_conditions.in_q1_right == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'right', at_n)

        elif(self.boundary_conditions.in_q1_right == 'mirror'):
            apply_mirror_bcs_f(self, 'right', at_n)
        
        elif(self.boundary_conditions.in_q1_right == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'right', at_n)            
            apply_dirichlet_bcs_f(self, 'right', at_n)
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q1_right == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q1_right == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'right', at_n)

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the bottom physical boundary:
    if(i_q2_start == 0):

        if(self.boundary_conditions.in_q2_bottom == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'bottom', at_n)

        elif(self.boundary_conditions.in_q2_bottom == 'mirror'):
            apply_mirror_bcs_f(self, 'bottom', at_n)            

        elif(self.boundary_conditions.in_q2_bottom == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'bottom', at_n)            
            apply_dirichlet_bcs_f(self, 'bottom', at_n)
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q2_bottom == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q2_bottom == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'bottom', at_n)

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the top physical boundary:
    if(i_q2_end == self.N_q2 - 1):

        if(self.boundary_conditions.in_q2_top == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'top', at_n)

        elif(self.boundary_conditions.in_q2_top == 'mirror'):
            apply_mirror_bcs_f(self, 'top', at_n)
        
        elif(self.boundary_conditions.in_q2_top == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'top', at_n)            
            apply_dirichlet_bcs_f(self, 'top', at_n)
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q2_top == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q2_top == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'top', at_n)

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    if(at_n):
        af.eval(self.f_n)
    else:
        af.eval(self.f_n_plus_half)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_f += toc - tic
   
    return
