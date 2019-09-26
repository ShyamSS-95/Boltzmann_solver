#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

"""Contains the function which returns the Source/Sink term."""

import numpy as np
import arrayfire as af

from bolt.lib.utils.broadcasted_primitive_operations import multiply, add

# Using af.broadcast, since v1, v2, v3 are of size (1, 1, Nv1*Nv2*Nv3)
# All moment quantities are of shape (Nq1, Nq2)
# By wrapping with af.broadcast, we can perform batched operations
# on arrays of different sizes.
@af.broadcast
def f0(v1, v2, v3, n, T, v1_bulk, v2_bulk, v3_bulk, params):
    """
    Return the Local MB distribution.
    Parameters:
    -----------
    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)
    
    n: Computed density at the current state.
       shape:(1, N_s, N_q1, N_q2)

    T: Computed temperature at the current state.
       shape:(1, N_s, N_q1, N_q2)

    v1_bulk: Computed bulk velocity in the q1 direction at the current state.
             shape:(1, N_s, N_q1, N_q2)

    v2_bulk: Computed bulk velocity in the q2 direction at the current state.
             shape:(1, N_s, N_q1, N_q2)

    v3_bulk: Computed bulk velocity in the q3 direction at the current state.
             shape:(1, N_s, N_q1, N_q2)

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function

    """
    m = params.mass
    k = params.boltzmann_constant

    if (params.p_dim == 3):
        f0 = n * (m / (2 * np.pi * k * T))**(3 / 2)  \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v3 - v3_bulk)**2 / (2 * k * T))

    elif (params.p_dim == 2):
        f0 = n * (m / (2 * np.pi * k * T)) \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
               * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T))

    else:
        f0 = n * af.sqrt(m / (2 * np.pi * k * T)) \
               * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T))

    af.eval(f0)
    return (f0)

def source_term_energy_conserving(f_left, f_bot, f_center, t, 
                                  q1, q2, v1, v2, v3, moments, 
                                  fields_solver, params
                                 ):
    """
    Parameters:
    -----------
    f_left   : Distribution function array at left-center
        shape:(N_v, N_s, N_q1, N_q2)

    f_bot    : Distribution function array at center-bot
        shape:(N_v, N_s, N_q1, N_q2)

    f_center : Distribution function array at center of cell
        shape:(N_v, N_s, N_q1, N_q2)
    
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    moments : The compute moments object that can be used to get moment values

    fields_solver : The required EM fields can be obtained from this object

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    """
    n = moments('density', f_center)
    e = params.charge
    m = params.mass

    # Floor used to avoid 0/0 limit:
    eps = 1e-30

    v1_bulk = moments('mom_v1_bulk', f_center) / (n + eps)
    v2_bulk = moments('mom_v2_bulk', f_center) / (n + eps)
    v3_bulk = moments('mom_v3_bulk', f_center) / (n + eps)

    T = (1 / params.p_dim) * (  2 * multiply(moments('energy', f_center), m)
                                  - multiply(n, m) * v1_bulk**2
                                  - multiply(n, m) * v2_bulk**2
                                  - multiply(n, m) * v3_bulk**2
                             ) / (n + eps) + eps

    f_MB = f0(v1, v2, v3, n, T, v1_bulk, v2_bulk, v3_bulk, params)
    tau  = params.tau(q1, q2, v1, v2, v3)

    C_f = -(f_center - f_MB) / tau

    E1_lc, E2_lc, E3_lc, B1_lc, B2_lc, B3_lc = fields_solver.get_fields('left_center')
    E1_cb, E2_cb, E3_cb, B1_cb, B2_cb, B3_cb = fields_solver.get_fields('center_bottom')
    E1_cc, E2_cc, E3_cc, B1_cc, B2_cc, B3_cc = fields_solver.get_fields('center')

    # At right center:
    E1_rc   = af.shift(E1_lc,  0, 0, -1)
    E2_rc   = af.shift(E2_lc,  0, 0, -1)
    E3_rc   = af.shift(E3_lc,  0, 0, -1)
    B1_rc   = af.shift(B1_lc,  0, 0, -1)
    B2_rc   = af.shift(B2_lc,  0, 0, -1)
    B3_rc   = af.shift(B3_lc,  0, 0, -1)
    f_right = af.shift(f_left, 0, 0, -1)

    # At center top:
    E1_ct = af.shift(E1_cb, 0, 0, 0, -1)
    E2_ct = af.shift(E2_cb, 0, 0, 0, -1)
    E3_ct = af.shift(E3_cb, 0, 0, 0, -1)
    B1_ct = af.shift(B1_cb, 0, 0, 0, -1)
    B2_ct = af.shift(B2_cb, 0, 0, 0, -1)
    B3_ct = af.shift(B3_cb, 0, 0, 0, -1)
    f_top = af.shift(f_bot, 0, 0, 0, -1)

    # Source terms arising from formulation:
    #   e(E_x + v_y B_z - v_z B_y)v_x f / m
    # + e(E_y + v_z B_x - v_x B_z)v_y f / m
    # + e(E_z + v_x B_y - v_y B_x)v_z f / m
    src_p1 =   multiply(multiply(e/m, add(E1_lc, 
                                          multiply(v2, B3_lc) - multiply(v3, B2_lc)
                                         )
                                ), v1) * f_left \
             + multiply(multiply(e/m, add(E1_rc, 
                                          multiply(v2, B3_rc) - multiply(v3, B2_rc)
                                         )), v1) * f_right

    src_p1 = src_p1 / 2

    src_p2 =   multiply(multiply(e/m, add(E2_cb, 
                                          multiply(v3, B1_cb) - multiply(v1, B3_cb)
                                         )
                                ), v2) * f_bot \
             + multiply(multiply(e/m, add(E2_ct, 
                                          multiply(v3, B1_ct) - multiply(v1, B3_ct)
                                         )), v2) * f_top
    src_p2 = src_p2 / 2
    src_p3 = multiply(multiply(e/m, add(E3_cc, multiply(v1, B2_cc) - multiply(v2, B1_cc))), v3) * f_center

    return(C_f + src_p1 + src_p2 + src_p3)

def source_term(f, t, q1, q2, v1, v2, v3, moments, 
                fields_solver, params
               ):
    """
    Parameters:
    -----------
    f : Distribution function array
        shape:(N_v, N_s, N_q1, N_q2)
    
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    moments : The compute moments object that can be used to get moment values

    fields_solver : The required EM fields can be obtained from this object

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    """
    n = moments('density', f)
    m = params.mass

    # Floor used to avoid 0/0 limit:
    eps = 1e-30

    v1_bulk = moments('mom_v1_bulk', f) / (n + eps)
    v2_bulk = moments('mom_v2_bulk', f) / (n + eps)
    v3_bulk = moments('mom_v3_bulk', f) / (n + eps)

    T = (1 / params.p_dim) * (  2 * multiply(moments('energy', f), m)
                                  - multiply(n, m) * v1_bulk**2
                                  - multiply(n, m) * v2_bulk**2
                                  - multiply(n, m) * v3_bulk**2
                             ) / (n + eps) + eps

    f_MB = f0(v1, v2, v3, n, T, v1_bulk, v2_bulk, v3_bulk, params)
    tau  = params.tau(q1, q2, v1, v2, v3)

    C_f = -(f - f_MB) / tau

    return(C_f)
