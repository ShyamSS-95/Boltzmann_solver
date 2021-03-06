#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def apply_shearing_box_bcs_f(self, boundary):
    """
    Applies the shearing box boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    N_g_q = self.N_ghost_q
    q     = self.physical_system.params.q 
    omega = self.physical_system.params.omega
    
    L_q1  = self.q1_end - self.q1_start
    L_q2  = self.q2_end - self.q2_start

    if(boundary == 'left'):
        sheared_coordinates = self.q2_center[:, :, :N_g_q] - q * omega * L_q1 * self.time_elapsed
        
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

        self.f[:, :, :N_g_q] = af.reorder(af.approx2(af.reorder(self.f[:, :, :N_g_q], 2, 3, 0, 1),
                                                     af.reorder(self.q1_center[:, :, :N_g_q], 2, 3, 0, 1),
                                                     af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                     af.INTERP.BICUBIC_SPLINE,
                                                     xp = af.reorder(self.q1_center[:, :, :N_g_q], 2, 3, 0, 1),
                                                     yp = af.reorder(self.q2_center[:, :, :N_g_q], 2, 3, 0, 1)
                                                    ),
                                          2, 3, 0, 1
                                         )
        
    elif(boundary == 'right'):
        sheared_coordinates = self.q2_center[:, :, -N_g_q:] + q * omega * L_q1 * self.time_elapsed

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

        self.f[:, :, -N_g_q:] = af.reorder(af.approx2(af.reorder(self.f[:, :, -N_g_q:], 2, 3, 0, 1),
                                                      af.reorder(self.q1_center[:, :, -N_g_q:], 2, 3, 0, 1),
                                                      af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                      af.INTERP.BICUBIC_SPLINE,
                                                      xp = af.reorder(self.q1_center[:, :, -N_g_q:], 2, 3, 0, 1),
                                                      yp = af.reorder(self.q2_center[:, :, -N_g_q:], 2, 3, 0, 1)
                                                     ),
                                            2, 3, 0, 1
                                           )

    elif(boundary == 'bottom'):

        sheared_coordinates = self.q1_center[:, :, :, :N_g_q] - q * omega * L_q2 * self.time_elapsed

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

        self.f[:, :, :, :N_g_q] = af.reorder(af.approx2(af.reorder(self.f[:, :, :, :N_g_q], 2, 3, 0, 1),
                                                        af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                        af.reorder(self.q2_center[:, :, :, :N_g_q], 2, 3, 0, 1),
                                                        af.INTERP.BICUBIC_SPLINE,
                                                        xp = af.reorder(self.q1_center[:, :, :, :N_g_q], 2, 3, 0, 1),
                                                        yp = af.reorder(self.q2_center[:, :, :, :N_g_q], 2, 3, 0, 1)
                                                       ),
                                             2, 3, 0, 1
                                            )

    elif(boundary == 'top'):

        sheared_coordinates = self.q1_center[:, :, :, -N_g_q:] + q * omega * L_q2 * self.time_elapsed

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

        self.f[:, :, :, -N_g_q:] = af.reorder(af.approx2(af.reorder(self.f[:, :, :, -N_g_q:], 2, 3, 0, 1),
                                                         af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                         af.reorder(self.q2_center[:, :, :, -N_g_q:], 2, 3, 0, 1),
                                                         af.INTERP.BICUBIC_SPLINE,
                                                         xp = af.reorder(self.q1_center[:, :, :, -N_g_q:], 2, 3, 0, 1),
                                                         yp = af.reorder(self.q2_center[:, :, :, -N_g_q:], 2, 3, 0, 1)
                                                        ),
                                              2, 3, 0, 1
                                             )

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_dirichlet_bcs_f(self, boundary):
    """
    Applies Dirichlet boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    N_g_q = self.N_ghost_q
    
    if(self._A_q1.elements() ==   (self.N_p1 + 2 * self.N_ghost_p) 
                                * (self.N_p2 + 2 * self.N_ghost_p) 
                                * (self.N_p3 + 2 * self.N_ghost_p)
      ):
        # If A_q1 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, Nq1, Nq2)
        A_q1 = af.tile(self._A_q1, 1, 1,
                       self.f.shape[2],
                       self.f.shape[3]
                      )

    if(self._A_q2.elements() ==   (self.N_p1 + 2 * self.N_ghost_p) 
                                * (self.N_p2 + 2 * self.N_ghost_p) 
                                * (self.N_p3 + 2 * self.N_ghost_p)
      ):
        # If A_q2 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, Nq1, Nq2)
        A_q2 = af.tile(self._A_q2, 1, 1, 
                       self.f.shape[2],
                       self.f.shape[3]
                      )

    # Arguments that are passing to the called functions:
    args = (self.time_elapsed, self.q1_center, self.q2_center,
            self.p1_center, self.p2_center, self.p3_center, 
            self.physical_system.params
           )

    if(boundary == 'left'):
        f_left = self.boundary_conditions.f_left(self.f, *args)
        # Only changing inflowing characteristics:
        f_left = af.select(A_q1>0, f_left, self.f)
        self.f[:, :, :N_g_q] = f_left[:, :, :N_g_q]

    elif(boundary == 'right'):
        f_right = self.boundary_conditions.f_right(self.f, *args)
        # Only changing inflowing characteristics:
        f_right = af.select(A_q1<0, f_right, self.f)
        self.f[:, :, -N_g_q:] = f_right[:, :, -N_g_q:]

    elif(boundary == 'bottom'):
        f_bottom = self.boundary_conditions.f_bottom(self.f, *args)
        # Only changing inflowing characteristics:
        f_bottom = af.select(A_q2>0, f_bottom, self.f)
        self.f[:, :, :, :N_g_q] = f_bottom[:, :, :, :N_g_q]

    elif(boundary == 'top'):
        f_top = self.boundary_conditions.f_top(self.f, *args)
        # Only changing inflowing characteristics:
        f_top = af.select(A_q2<0, f_top, self.f)
        self.f[:, :, :, -N_g_q:] = f_top[:, :, :, -N_g_q:]

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
    """

    N_g_q = self.N_ghost_q

    if(boundary == 'left'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :, :N_g_q] = af.flip(self.f[:, :, N_g_q:2 * N_g_q], 2)
        
        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        self.f[:, :, :N_g_q] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                0
                                               )
                                       )[:, :, :N_g_q]

    elif(boundary == 'right'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, :, -N_g_q:] = af.flip(self.f[:, :, -2 * N_g_q:-N_g_q], 2)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        self.f[:, :, -N_g_q:] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                0
                                               )
                                       )[:, :, -N_g_q:]

    elif(boundary == 'bottom'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :, :, :N_g_q] = af.flip(self.f[:, :, :, N_g_q:2 * N_g_q], 3)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        self.f[:, :, :, :N_g_q] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                1
                                               )
                                       )[:, :, :, :N_g_q]

    elif(boundary == 'top'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, :, :, -N_g_q:] = af.flip(self.f[:, :, :, -2 * N_g_q:-N_g_q], 3)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        self.f[:, :, :, -N_g_q:] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                1
                                               )
                                       )[:, :, :, -N_g_q:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_bcs_f(self):
    """
    Applies boundary conditions to the distribution function as specified by 
    the user in params.
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
            apply_dirichlet_bcs_f(self, 'left')

        elif(self.boundary_conditions.in_q1_left == 'mirror'):
            apply_mirror_bcs_f(self, 'left')            

        elif(self.boundary_conditions.in_q1_left == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'left')            
            apply_dirichlet_bcs_f(self, 'left')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q1_left == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q1_left == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'left')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')
    
    # If local zone includes the right physical boundary:
    if(i_q1_end == self.N_q1 - 1):

        if(self.boundary_conditions.in_q1_right == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'right')

        elif(self.boundary_conditions.in_q1_right == 'mirror'):
            apply_mirror_bcs_f(self, 'right')
        
        elif(self.boundary_conditions.in_q1_right == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'right')            
            apply_dirichlet_bcs_f(self, 'right')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q1_right == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q1_right == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'right')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the bottom physical boundary:
    if(i_q2_start == 0):

        if(self.boundary_conditions.in_q2_bottom == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'bottom')

        elif(self.boundary_conditions.in_q2_bottom == 'mirror'):
            apply_mirror_bcs_f(self, 'bottom')            

        elif(self.boundary_conditions.in_q2_bottom == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'bottom')            
            apply_dirichlet_bcs_f(self, 'bottom')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q2_bottom == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q2_bottom == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'bottom')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the top physical boundary:
    if(i_q2_end == self.N_q2 - 1):

        if(self.boundary_conditions.in_q2_top == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'top')

        elif(self.boundary_conditions.in_q2_top == 'mirror'):
            apply_mirror_bcs_f(self, 'top')
        
        elif(self.boundary_conditions.in_q2_top == 'mirror+dirichlet'):
            apply_mirror_bcs_f(self, 'top')            
            apply_dirichlet_bcs_f(self, 'top')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(self.boundary_conditions.in_q2_top == 'periodic'):
            pass

        elif(self.boundary_conditions.in_q2_top == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'top')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_f += toc - tic
   
    return

