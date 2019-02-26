#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file ensures that the timestagerring of all
variable involved in FDTD has been carried out
correctly.
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear.fields.fields import fields_solver
from bolt.lib.physical_system import physical_system

from input_files import domain
from input_files import params
from input_files import initialize_fdtd_mode1 as initialize
from input_files import boundary_conditions

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

def test_fdtd_timestaggering():

    N         = np.random.randint(32, 512)
    # We are using dt = 0
    params.dt = 0

    domain.N_q1 = int(N)
    domain.N_q2 = int(N)
    domain.N_p1 = 1
    domain.N_p2 = 1
    domain.N_p3 = 1

    N_g    = domain.N_ghost
    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize,
                             advection_terms,
                             collision_operator.BGK,
                             moments
                            )

    fs = fields_solver(system, 
                       af.constant(0, 1, 1, 
                                   domain.N_q1 + 2 * N_g, 
                                   domain.N_q2 + 2 * N_g, 
                                   dtype = af.Dtype.f64
                                  )
                      )

    J1 = J2 = J3 = 0 * fs.q1_center**0

    # In this check, we have set all initial field values
    # (E^n, B^{n+1/2}) = (3, 3)
    # Since dt = 0, there would be no change in these field values:
    # (E^{n+1}, B^{n+3/2}) = (3, 3)
    # We need to still set the EM_fields_at_n_plus_half since this holds
    # the value of B^{n-1/2} of the previous timestep at the begining of the 
    # evolve_electrodynamic_fields routine. These have been set to 1

    # We are verifying that E^n = 3
    #                       B^n = 0.5 * (B^{n-1/2} + B^{n+1/2}) = 2

    #                       E^{n+1/2} = 0.5 * (E^n + E^{n+1}) = 3
    #                       B^{n+1/2} = 3
    fs.yee_grid_EM_fields_at_n_plus_half = af.constant(0, 6, 1, 
                                                       domain.N_q1 + 2 * N_g, 
                                                       domain.N_q2 + 2 * N_g, 
                                                       dtype = af.Dtype.f64
                                                      )
    # Setting magnetic field values to 1
    fs.yee_grid_EM_fields_at_n_plus_half[3:] = 1


    fs.yee_grid_EM_fields = af.constant(3, 6, 1, 
                                        domain.N_q1 + 2 * N_g, 
                                        domain.N_q2 + 2 * N_g, 
                                        dtype = af.Dtype.f64
                                       )

    fs.cell_centered_EM_fields_at_n_plus_half = af.constant(0, 6, 1, 
                                                            domain.N_q1 + 2 * N_g, 
                                                            domain.N_q2 + 2 * N_g, 
                                                            dtype = af.Dtype.f64
                                                           )
    # Setting magnetic field values to 1
    fs.cell_centered_EM_fields_at_n_plus_half[3:] = 1

    fs.cell_centered_EM_fields = af.constant(3, 6, 1, 
                                             domain.N_q1 + 2 * N_g, 
                                             domain.N_q2 + 2 * N_g, 
                                             dtype = af.Dtype.f64
                                            )

    fs.evolve_electrodynamic_fields(J1, J2, J3, params.dt)

    assert(np.sum(abs(  np.array(af.sum(af.sum(fs.yee_grid_EM_fields, 2), 3) / (N + 2 * N_g)**2).ravel() 
                      - np.array([3, 3, 3, 3, 3, 3])
                     )) < 5e-14)
    assert(np.sum(abs(  np.array(af.sum(af.sum(fs.yee_grid_EM_fields_at_n, 2), 3) / (N + 2 * N_g)**2).ravel() 
                      - np.array([3, 3, 3, 2, 2, 2])
                     )) < 5e-14)
    assert(np.sum(abs(  np.array(af.sum(af.sum(fs.yee_grid_EM_fields_at_n_plus_half, 2), 3) / (N + 2 * N_g)**2).ravel() 
                      - np.array([3, 3, 3, 3, 3, 3])
                     )) < 5e-14)

    assert(np.sum(abs(  np.array(af.sum(af.sum(fs.cell_centered_EM_fields, 2), 3) / (N + 2 * N_g)**2).ravel() 
                      - np.array([3, 3, 3, 3, 3, 3])
                     )) < 5e-14)
    assert(np.sum(abs(  np.array(af.sum(af.sum(fs.cell_centered_EM_fields_at_n, 2), 3) / (N + 2 * N_g)**2).ravel() 
                      - np.array([3, 3, 3, 2, 2, 2])
                     )) < 5e-14)
    assert(np.sum(abs(  np.array(af.sum(af.sum(fs.cell_centered_EM_fields_at_n_plus_half, 2), 3) / (N + 2 * N_g)**2).ravel() 
                      - np.array([3, 3, 3, 3, 3, 3])
                     )) < 5e-14)
