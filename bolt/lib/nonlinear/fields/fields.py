#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing dependencies:
import arrayfire as af
import numpy as np
from petsc4py import PETSc

from .. import communicate
from .boundaries import apply_bcs_fields

from .electrostatic.fft import fft_poisson
from .electrodynamic.fdtd_explicit import fdtd

from bolt.lib.utils.calculate_q import \
    calculate_q_center, calculate_q_left_center, \
    calculate_q_center_bot, calculate_q_left_bot

class fields_solver(object):
  
    # Computing divB on cell edges
    def compute_divB(self):
        
        B1 = self.yee_grid_EM_fields[3] # (i + 1/2, j)
        B2 = self.yee_grid_EM_fields[4] # (i, j + 1/2)

        B1_minus_q1 = af.shift(B1, 0, 0, 1) # (i - 1/2, j)
        B2_minus_q2 = af.shift(B2, 0, 0, 0, 1) # (i, j - 1/2)

        divB = (B1 - B1_minus_q1) / self.dq1 + (B2 - B2_minus_q2) / self.dq2 # (i, j)
        return(divB)

    # Computing divE at cell centers
    def compute_divE(self):
      
        communicate.communicate_fields(self, True)
        
        E1 = self.yee_grid_EM_fields[0] # (i, j + 1/2)
        E2 = self.yee_grid_EM_fields[1] # (i + 1/2, j)

        E1_plus_q1 = af.shift(E1, 0, 0, -1)    # (i + 1, j + 1/2)
        E2_plus_q2 = af.shift(E2, 0, 0, 0, -1) # (i + 1/2, j + 1)

        divE = (E1_plus_q1 - E1) / self.dq1 + (E2_plus_q2 - E2) / self.dq2 # (i + 1/2, j + 1/2)
        return(divE)

    def check_maxwells_contraint_equations(self, rho):

        N_g = self.N_g

        PETSc.Sys.Print("Initial Maxwell's Constraints:")
        mean_divB = af.mean(af.abs(self.compute_divB()[:, :, N_g:-N_g, N_g:-N_g]))
        PETSc.Sys.Print('MEAN(|divB|) =', mean_divB)

        rho_by_eps     = rho / self.params.eps
        rho_b          = af.mean(rho_by_eps) # background
        divE           = self.compute_divE()
        mean_gauss_law = af.mean(af.abs((divE- rho_by_eps + rho_b)[:, :, N_g:-N_g, N_g:-N_g]))
        PETSc.Sys.Print('MEAN(|divE-rho|) =', mean_gauss_law)
        # Appropriately raising exceptions:
        # Checking for ∇.B = 0
        try:
            assert(mean_divB < 1e-10)
        except:
            raise SystemExit('divB contraint isn\'t preserved')

        # Checking for ∇.E = ρ/ε:
        # TODO: Need to look into this further:
        # try:
        #     assert(mean_gauss_law < 1e-7)
        # except:
        #     raise SystemExit('divE - rho/epsilon contraint isn\'t preserved')

    def __init__(self, physical_system, rho_initial, performance_test_flag = False):
        """
        Constructor for the fields_solver object, which takes in the physical system
        object and initial charge density as the input. 
        
        Additionally, a performance test flag is also passed which when true 
        stores time which is consumed by each of the major solver routines.
        This proves particularly useful in analyzing performance bottlenecks 
        and obtaining benchmarks.

        Parameters:
        -----------
        physical_system: The defined physical system object which holds
                         all the simulation information

        rho_initial: af.Array
                     The initial charge density array that's passed to an electrostatic 
                     solver for initialization

        performance_test_flag: bool
                               When set to true, the time elapsed in routines for the
                               fields solver, inter-processor communication of field values 
                               and application of boundary conditions for the EM_fields
                               is measured.
        """
        self.N_q1 = physical_system.N_q1
        self.N_q2 = physical_system.N_q2
        
        self.N_g = physical_system.N_ghost

        self.dq1 = physical_system.dq1
        self.dq2 = physical_system.dq2

        self.q1_start = physical_system.q1_start
        self.q2_start = physical_system.q2_start

        self._comm = physical_system.mpi_communicator

        self.boundary_conditions = physical_system.boundary_conditions
        self.params              = physical_system.params
        self.initialize          = physical_system.initial_conditions

        self.performance_test_flag   = performance_test_flag
        
        self.time_fieldsolver        = 0
        self.time_apply_bcs_fields   = 0
        self.time_communicate_fields = 0

        self.time_elapsed = 0
        
        petsc_bc_in_q1 = 'ghosted'
        petsc_bc_in_q2 = 'ghosted'

        # Only for periodic boundary conditions or shearing-box boundary conditions 
        # do the boundary conditions passed to the DA need to be changed. PETSc
        # automatically handles the application of periodic boundary conditions when
        # running in parallel. For shearing box boundary conditions, an interpolation
        # operation needs to be applied on top of the periodic boundary conditions.
        # In all other cases, ghosted boundaries are used.
        
        if(   self.boundary_conditions.in_q1_left == 'periodic'
           or self.boundary_conditions.in_q1_left == 'shearing-box'
          ):
            petsc_bc_in_q1 = 'periodic'

        if(   self.boundary_conditions.in_q2_bottom == 'periodic'
           or self.boundary_conditions.in_q2_bottom == 'shearing-box'
          ):
            petsc_bc_in_q2 = 'periodic'

        nproc_in_q1 = PETSc.DECIDE  
        nproc_in_q2 = PETSc.DECIDE

        # Since shearing boundary conditions require interpolations which are non-local:
        if(self.boundary_conditions.in_q2_bottom == 'shearing-box'):
            nproc_in_q1 = 1
        
        if(self.boundary_conditions.in_q1_left == 'shearing-box'):
            nproc_in_q2 = 1

        # This DA object is used in the communication routines for the
        # EM field quantities. A DOF of 6 is taken so that the communications,
        # and application of B.C's may be carried out in a single call among
        # all the field quantities(E1, E2, E3, B1, B2, B3)
        self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof           = 6,
                                              stencil_width = self.N_g,
                                              boundary_type = (petsc_bc_in_q1,
                                                               petsc_bc_in_q2
                                                              ),
                                              proc_sizes    = (nproc_in_q1, 
                                                               nproc_in_q2
                                                              ),
                                              stencil_type  = 1,
                                              comm          = self._comm
                                             )


        # Obtaining start coordinates for the local zone
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_fields.getCorners()

        self.q1_center, self.q2_center = \
            calculate_q_center(self.q1_start + i_q1_start * self.dq1, 
                               self.q2_start + i_q2_start * self.dq2,
                               N_q1_local, N_q2_local, self.N_g,
                               self.dq1, self.dq2
                              )

        self.q1_left_center, self.q2_left_center = \
            calculate_q_left_center(self.q1_start + i_q1_start * self.dq1, 
                                    self.q2_start + i_q2_start * self.dq2,
                                    N_q1_local, N_q2_local, self.N_g,
                                    self.dq1, self.dq2
                                   )

        self.q1_center_bot, self.q2_center_bot = \
            calculate_q_center_bot(self.q1_start + i_q1_start * self.dq1, 
                                   self.q2_start + i_q2_start * self.dq2,
                                   N_q1_local, N_q2_local, self.N_g,
                                   self.dq1, self.dq2
                                  )

        self.q1_left_bot, self.q2_left_bot = \
            calculate_q_left_bot(self.q1_start + i_q1_start * self.dq1, 
                                 self.q2_start + i_q2_start * self.dq2,
                                 N_q1_local, N_q2_local, self.N_g,
                                 self.dq1, self.dq2
                                )

        # The following global and local vectors are used in
        # the communication routines for EM fields
        self._glob_fields  = self._da_fields.createGlobalVec()
        self._local_fields = self._da_fields.createLocalVec()

        self._glob_fields_array  = self._glob_fields.getArray()
        self._local_fields_array = self._local_fields.getArray()

        PETSc.Object.setName(self._glob_fields, 'EM_fields')
        
        # Alternating upon each call to get_fields for FVM:
        # This ensures that the fields are staggerred correctly in time:
        self.at_n = True

        # Summing for all species:
        rho_initial = af.sum(rho_initial, 1)
        self._initialize(rho_initial)
        self.check_maxwells_contraint_equations(rho_initial)

    def initialize_electric_fields(self):

        if('initialize_E' in dir(self.initialize)):
            
            E1 = self.initialize.initialize_E(self.q1_left_center,
                                              self.q2_left_center,
                                              self.params
                                             )[0]

            E2 = self.initialize.initialize_E(self.q1_center_bot,
                                              self.q2_center_bot,
                                              self.params
                                             )[1]

            E3 = self.initialize.initialize_E(self.q1_center,
                                              self.q2_center,
                                              self.params
                                             )[2]

        elif('initialize_A_phi' in dir(self.initialize)):

            A1 = self.initialize.initialize_A_phi(self.q1_center_bot,
                                                  self.q2_center_bot,
                                                  self.params
                                                 )[0]

            A2 = self.initialize.initialize_A_phi(self.q1_left_center,
                                                  self.q2_left_center,
                                                  self.params
                                                 )[1]

            A3 = self.initialize.initialize_A_phi(self.q1_left_bot,
                                                  self.q2_left_bot,
                                                  self.params
                                                 )[2]

            phi = self.initialize.initialize_A_phi(self.q1_center,
                                                   self.q2_center,
                                                   self.params
                                                  )[3]

            dA3_dq2 = (af.shift(A3, 0, 0,  0, -1) - A3) / self.dq2
            dA3_dq1 = (af.shift(A3, 0, 0, -1,  0) - A3) / self.dq1
            dA2_dq1 = (af.shift(A2, 0, 0, -1,  0) - A2) / self.dq1
            dA1_dq2 = (af.shift(A1, 0, 0,  0, -1) - A1) / self.dq2

            dphi_dq1 = -(af.shift(phi, 0, 0, 1, 0) - phi) / self.dq1
            dphi_dq2 = -(af.shift(phi, 0, 0, 0, 1) - phi) / self.dq2

            E1 =  dA3_dq2 - dphi_dq1
            E2 = -dA3_dq1 - dphi_dq2
            E3 = dA2_dq1 - dA1_dq2

        else:
            raise NotImplementedError('Initialization method for electric fields not valid/found')

        self.yee_grid_EM_fields[:3] = af.join(0, E1, E2, E3)
        af.eval(self.yee_grid_EM_fields)
        return

    def initialize_magnetic_fields(self, update_yee_grid_fields = True):

        if(    'initialize_B' in dir(self.initialize)
           and 'initialize_A3_B3' in dir(self.initialize)
          ):
            raise Exception('Can\'t use both initialize_B and initialize_A3_B3!')
        
        if('initialize_B' in dir(self.initialize)):
            
            B1 = self.initialize.initialize_B(self.q1_center_bot,
                                              self.q2_center_bot,
                                              self.params
                                             )[0]

            B2 = self.initialize.initialize_B(self.q1_left_center,
                                              self.q2_left_center,
                                              self.params
                                             )[1]

            B3 = self.initialize.initialize_B(self.q1_left_bot,
                                              self.q2_left_bot,
                                              self.params
                                             )[2]

        elif('initialize_A' in dir(self.initialize)):

            A1 = self.initialize.initialize_A(self.q1_left_center,
                                              self.q2_left_center,
                                              self.params
                                             )[0]

            A2 = self.initialize.initialize_A(self.q1_center_bot,
                                              self.q2_center_bot,
                                              self.params
                                             )[1]

            A3 = self.initialize.initialize_A(self.q1_center,
                                              self.q2_center,
                                              self.params
                                             )[2]

            B1 =  (af.shift(A3, 0, 0, 0, -1) - A3) / self.dq2
            B2 = -(af.shift(A3, 0, 0,-1,  0) - A3) / self.dq1

            dA2_dq1 = (af.shift(A2, 0, 0,-1,  0) - A2) / self.dq1
            dA1_dq2 = (af.shift(A1, 0, 0, 0, -1) - A1) / self.dq2
            B3      = dA2_dq1 - dA1_dq2

        elif('initialize_A3_B3' in dir(self.initialize)):

            A3 = self.initialize.initialize_A3_B3(self.q1_center,
                                                  self.q2_center,
                                                  self.params
                                                 )[0]

            # NOTE: This is currently throwing a segfault on Savio:
            # A3_plus_q2 = af.shift(A3, 0, 0,  0, -1)
            # A3_plus_q1 = af.shift(A3, 0, 0, -1,  0)

            # HACK: Change to numpy and change back:
            A3_minus_q2 = af.to_array(np.roll(np.array(A3), 1, 3))
            A3_minus_q1 = af.to_array(np.roll(np.array(A3), 1, 2))

            B1 =  (A3 - A3_minus_q2) / self.dq2 #  dA3_dq2
            B2 = -(A3 - A3_minus_q1) / self.dq1 # -dA3_dq1
            B3 = self.initialize.initialize_A3_B3(self.q1_left_bot,
                                                  self.q2_left_bot,
                                                  self.params
                                                 )[1]

        else:
            raise NotImplementedError('Initialization method for magnetic fields not valid/found')

        if(update_yee_grid_fields == True):
            self.yee_grid_EM_fields[3:] = af.join(0, B1, B2, B3)
            af.eval(self.yee_grid_EM_fields)
            return

        else:
            return(af.join(0, B1, B2, B3))

    def _initialize(self, rho_initial):
        """
        Called when the solver object is declared. This function is
        used to initialize the EM field quantities using the options 
        as provided by the user.

        Parameters
        ----------

        rho_initial : af.Array
                     The initial charge density array that's passed to an electrostatic 
                     solver for initialization
        """
        # Obtaining start coordinates for the local zone
        # Additionally, we also obtain the size of the local zone
        ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_fields.getCorners()

        # Following quantities are cell-centered (i + 0.5, j + 0.5):
        # Electric fields are defined at the n-th timestep:
        # Magnetic fields are defined at the (n-1/2)-th timestep:
        self.cell_centered_EM_fields = af.constant(0, 6, 1, 
                                                   N_q1_local + 2 * self.N_g,
                                                   N_q2_local + 2 * self.N_g,
                                                   dtype=af.Dtype.f64
                                                  )

        # Field values at n-th timestep:
        self.cell_centered_EM_fields_at_n = af.constant(0, 6, 1, 
                                                        N_q1_local + 2 * self.N_g,
                                                        N_q2_local + 2 * self.N_g,
                                                        dtype=af.Dtype.f64
                                                       )

        # Field values at (n+1/2)-th timestep:
        self.cell_centered_EM_fields_at_n_plus_half = af.constant(0, 6, 1, 
                                                                  N_q1_local + 2 * self.N_g,
                                                                  N_q2_local + 2 * self.N_g,
                                                                  dtype=af.Dtype.f64
                                                                 )

        # Declaring the arrays which store data on the yee grid for FDTD:
        self.yee_grid_EM_fields = af.constant(0, 6, 1, 
                                              N_q1_local + 2 * self.N_g,
                                              N_q2_local + 2 * self.N_g,
                                              dtype=af.Dtype.f64
                                             )

        # Field values at n-th timestep on the Yee Grid:
        self.yee_grid_EM_fields_at_n = af.constant(0, 6, 1, 
                                                   N_q1_local + 2 * self.N_g,
                                                   N_q2_local + 2 * self.N_g,
                                                   dtype=af.Dtype.f64
                                                  )

        # Field values at (n+1/2)-th timestep on the Yee grid:
        self.yee_grid_EM_fields_at_n_plus_half = af.constant(0, 6, 1, 
                                                             N_q1_local + 2 * self.N_g,
                                                             N_q2_local + 2 * self.N_g,
                                                             dtype=af.Dtype.f64
                                                            )

        if (self.params.fields_initialize == 'fft'):
            fft_poisson(self, rho_initial)
            communicate.communicate_fields(self)
            apply_bcs_fields(self)
            
            # Transferring the information about the electric field to the Yee-grid:
            self.cell_centered_grid_to_yee_grid('E')

        elif (self.params.fields_initialize == 'user-defined'):
            self.initialize_electric_fields()
            self.initialize_magnetic_fields()
            self.yee_grid_to_cell_centered_grid()

        elif (self.params.fields_initialize == 'fft + user-defined magnetic fields'):
            fft_poisson(self, rho_initial)
            communicate.communicate_fields(self)
            apply_bcs_fields(self)

            # Transferring the information about the electric field to the Yee-grid:
            self.cell_centered_grid_to_yee_grid('E')
            # Initialize on Yee grid:
            self.initialize_magnetic_fields()
            # Transferring the information about the magnetic field to the cell centered grid:
            self.yee_grid_to_cell_centered_grid('B')

        else:
            raise NotImplementedError('Method not valid/not implemented')

        # At t = 0, we take the value of B_{0} = B{1/2}:
        self.cell_centered_EM_fields_at_n = self.cell_centered_EM_fields.copy()
        self.cell_centered_EM_fields_at_n_plus_half = self.cell_centered_EM_fields.copy()
        self.yee_grid_EM_fields_at_n = self.yee_grid_EM_fields.copy()
        self.yee_grid_EM_fields_at_n_plus_half = self.yee_grid_EM_fields.copy()

        return

    def cell_centered_grid_to_yee_grid(self, fields_to_transform = None):
        """
        Making use of the fields_to_transform option, one can convert
        the magnetic fields or the electric fields alone. The default
        option is to convert both when the option is passed as None
        """
        E1 = self.cell_centered_EM_fields[0]
        E2 = self.cell_centered_EM_fields[1]
        E3 = self.cell_centered_EM_fields[2]

        B1 = self.cell_centered_EM_fields[3]
        B2 = self.cell_centered_EM_fields[4]
        B3 = self.cell_centered_EM_fields[5]

        if(fields_to_transform == 'E' or fields_to_transform == None):
            self.yee_grid_EM_fields[0] = 0.5 * (E1 + af.shift(E1, 0, 0, 1, 0))  # (i, j+1/2)
            self.yee_grid_EM_fields[1] = 0.5 * (E2 + af.shift(E2, 0, 0, 0, 1))  # (i+1/2, j)
            self.yee_grid_EM_fields[2] = E3  # (i+1/2, j+1/2)

        if(fields_to_transform == 'B' or fields_to_transform == None):
            self.yee_grid_EM_fields[3] = 0.5 * (B1 + af.shift(B1, 0, 0, 0, 1)) # (i+1/2, j) 
            self.yee_grid_EM_fields[4] = 0.5 * (B2 + af.shift(B2, 0, 0, 1, 0)) # (i, j+1/2)
            self.yee_grid_EM_fields[5] = 0.25 * (  B3 
                                                 + af.shift(B3, 0, 0, 1, 0)
                                                 + af.shift(B3, 0, 0, 0, 1) 
                                                 + af.shift(B3, 0, 0, 1, 1)
                                                ) # (i, j)

        af.eval(self.yee_grid_EM_fields)
        return

    def yee_grid_to_cell_centered_grid(self, fields_to_transform = None):
        """
        Making use of the optional fields_to_transform option, one can convert
        the magnetic fields or the electric fields alone. The default
        option is to convert both when the option is passed as None
        """
        E1_yee = self.yee_grid_EM_fields[0] # (i, j + 1/2)
        E2_yee = self.yee_grid_EM_fields[1] # (i + 1/2, j)
        E3_yee = self.yee_grid_EM_fields[2] # (i + 1/2, j + 1/2)

        B1_yee = self.yee_grid_EM_fields[3] # (i + 1/2, j)
        B2_yee = self.yee_grid_EM_fields[4] # (i, j + 1/2)
        B3_yee = self.yee_grid_EM_fields[5] # (i, j)

        # Interpolating at the (i + 1/2, j + 1/2) point of the grid:

        if(fields_to_transform == 'E' or fields_to_transform == None):
            self.cell_centered_EM_fields[0] = 0.5 * (E1_yee + af.shift(E1_yee, 0, 0, -1,  0))
            self.cell_centered_EM_fields[1] = 0.5 * (E2_yee + af.shift(E2_yee, 0, 0,  0, -1))
            self.cell_centered_EM_fields[2] = E3_yee

        if(fields_to_transform == 'B' or fields_to_transform == None):
            self.cell_centered_EM_fields[3] = 0.5 * (B1_yee + af.shift(B1_yee, 0, 0,  0, -1))
            self.cell_centered_EM_fields[4] = 0.5 * (B2_yee + af.shift(B2_yee, 0, 0, -1,  0))
            self.cell_centered_EM_fields[5] = 0.25 * (  B3_yee 
                                                      + af.shift(B3_yee, 0, 0,  0, -1)
                                                      + af.shift(B3_yee, 0, 0, -1,  0)
                                                      + af.shift(B3_yee, 0, 0, -1, -1)
                                                     )

        af.eval(self.cell_centered_EM_fields)
        return

    def compute_electrostatic_fields(self, rho):

        # Summing for all species:
        rho = af.sum(rho, 1)
        if (self.params.fields_solver == 'fft'):
            fft_poisson(self, rho)
            communicate.communicate_fields(self)
            apply_bcs_fields(self)

        # ADD SNES BELOW

    def evolve_electrodynamic_fields(self, J1, J2, J3, dt):
        """
        Evolve the fields using FDTD.
        
        Updates:
        self_cell_centered_EM_fields[:3] => (E_x^n,     E_y^n,     E_z^n) --->
                                            (E_x^{n+1}, E_y^{n+1}, E_z^{n+1})

        self_cell_centered_EM_fields[3:] => (B_x^{n+1/2}, B_y^{n+1/2}, B_z^{n+1/2}) --->
                                            (B_x^{n+3/2}, B_y^{n+3/2}, B_z^{n+3/2})

        self.cell_centered_EM_fields_at_n => (E_x^{n-1}, E_y^{n-1}, E_z^{n-1}
                                              B_x^{n-1}, B_y^{n-1}, B_z^{n-1}) --->
                                             (E_x^{n}, E_y^{n}, E_z^{n}
                                              B_x^{n}, B_y^{n}, B_z^{n})

        self.cell_centered_EM_fields_at_n_plus_half => (E_x^{n-1/2}, E_y^{n-1/2}, E_z^{n-1/2}
                                                        B_x^{n-1/2}, B_y^{n-1/2}, B_z^{n-1/2}) --->
                                                       (E_x^{n+1/2}, E_y^{n+1/2}, E_z^{n+1/2}
                                                        B_x^{n+1/2}, B_y^{n+1/2}, B_z^{n+1/2})

        Parameters
        ----------

        J1 : af.Array
             Array which contains the J1 current for each species at (n + 1/2)        
              
        J2 : af.Array
             Array which contains the J2 current for each species at (n + 1/2).        
        
        J3 : af.Array
             Array which contains the J3 current for each species at (n + 1/2).        
        
        dt: double
            Timestep size
        """
        self.J1 = af.sum(J1, 1)
        self.J2 = af.sum(J2, 1)
        self.J3 = af.sum(J3, 1)

        # E_at_n holds (E_x^n , E_y^n, E_z^n)
        # cell_centered_EM_fields[:3] => (E_x^n , E_y^n, E_z^n)
        E_at_n = self.cell_centered_EM_fields[:3].copy()
        # Getting it for the yee grid:
        E_yee_at_n = self.yee_grid_EM_fields[:3].copy()

        self.cell_centered_EM_fields_at_n[:3] = E_at_n
        # Doing the same for Yee Grid fields:
        self.yee_grid_EM_fields_at_n[:3] = self.yee_grid_EM_fields[:3].copy()
 
        # B_at_n_minus_half holds (B_x^{n-1/2} , B_y^{n-1/2}, B_z^{n-1/2})
        # cell_centered_EM_fields_at_n_plus_half[3:] => (B_x^{n-1/2} , B_y^{n-1/2}, B_z^{n-1/2})
        # ^ NOTE: This is because cell_centered_EM_fields_at_n_plus_half has not been updated for
        # this timestep, and holds the information for the previous timestep:
        B_at_n_minus_half = self.cell_centered_EM_fields_at_n_plus_half[3:].copy()

        # cell_centered_EM_fields[3:] => (B_x^{n+1/2} , B_y^{n+1/2}, B_z^{n+1/2})
        B_at_n_plus_half = self.cell_centered_EM_fields[3:].copy()

        # cell_centered_EM_fields_at_n[3:] => (B_x^n , B_y^n, B_z^n)
        B_at_n = 0.5 * (B_at_n_minus_half + B_at_n_plus_half)
        self.cell_centered_EM_fields_at_n[3:] = B_at_n
        # Doing the same for Yee Grid fields:
        self.yee_grid_EM_fields_at_n[3:] = 0.5 * (  self.yee_grid_EM_fields_at_n_plus_half[3:] 
                                                  + self.yee_grid_EM_fields[3:]
                                                 )

        # Now updating cell_centered_EM_fields_at_n_plus_half for this timestep:
        # cell_centered_EM_fields_at_n_plus_half[3:] => (B_x^{n+1/2} , B_y^{n+1/2}, B_z^{n+1/2})
        self.cell_centered_EM_fields_at_n_plus_half[3:] = B_at_n_plus_half
        # Doing the same for Yee Grid fields:
        self.yee_grid_EM_fields_at_n_plus_half[3:] = self.yee_grid_EM_fields[3:].copy()

        # Evolving:
        # cell_centered_EM_fields[:3] => (E_x^n ,     E_y^n,     E_z^n) --> 
        #                                (E_x^{n+1} , E_y^{n+1}, E_z^{n+1})
        # cell_centered_EM_fields[3:] => (B_x^{n+1/2} , B_y^{n+1/2}, B_z^{n+1/2}) -->
        #                                (B_x^{n+3/2} , B_y^{n+3/2}, B_z^{n+3/2})
        fdtd(self, dt)
        self.yee_grid_to_cell_centered_grid()

        # cell_centered_EM_fields[:3] => (E_x^{n+1} ,  E_y^{n+1},   E_z^{n+1})
        E_at_n_plus_one = self.cell_centered_EM_fields[:3]
        
        # cell_centered_EM_fields_at_n_plus_half[:3] => (E_x^{n+1/2}, E_y^{n+1/2}, E_z^{n+1/2})
        E_at_n_plus_half = 0.5 * (E_at_n + E_at_n_plus_one)
        self.cell_centered_EM_fields_at_n_plus_half[:3] = E_at_n_plus_half
        self.yee_grid_EM_fields_at_n_plus_half[:3] = 0.5 * (  E_yee_at_n
                                                            + self.yee_grid_EM_fields[:3]
                                                           )

        # Update time elapsed:
        self.time_elapsed += dt
        return

    def get_fields(self, fields_location):
        """
        Returns the fields value as held by the
        solver in it's current state.
        """
        if(self.params.fields_solver != 'fdtd'):

            E1 = self.cell_centered_EM_fields[0]
            E2 = self.cell_centered_EM_fields[1]
            E3 = self.cell_centered_EM_fields[2]

            B1 = self.cell_centered_EM_fields[3]
            B2 = self.cell_centered_EM_fields[4]
            B3 = self.cell_centered_EM_fields[5]

        else:

            if(fields_location == 'center'):

                if(self.at_n == True):

                    E1 = self.cell_centered_EM_fields_at_n[0]
                    E2 = self.cell_centered_EM_fields_at_n[1]
                    E3 = self.cell_centered_EM_fields_at_n[2]

                    B1 = self.cell_centered_EM_fields_at_n[3]
                    B2 = self.cell_centered_EM_fields_at_n[4]
                    B3 = self.cell_centered_EM_fields_at_n[5]

                else:

                    E1 = self.cell_centered_EM_fields_at_n_plus_half[0]
                    E2 = self.cell_centered_EM_fields_at_n_plus_half[1]
                    E3 = self.cell_centered_EM_fields_at_n_plus_half[2]

                    B1 = self.cell_centered_EM_fields_at_n_plus_half[3]
                    B2 = self.cell_centered_EM_fields_at_n_plus_half[4]
                    B3 = self.cell_centered_EM_fields_at_n_plus_half[5]

            # Gets fields at the center of the left face: i.e (i, j + 1/2)
            elif(fields_location == 'left_center'):

                if(self.at_n == True):

                    E1 = self.yee_grid_EM_fields_at_n[0] # (i, j + 1/2)
                    
                    E2 = 0.25 * (  self.yee_grid_EM_fields_at_n[1] # (i + 1/2, j)
                                 + af.shift(self.yee_grid_EM_fields_at_n[1], 0, 0, 1, 0)  # (i - 1/2, j)
                                 + af.shift(self.yee_grid_EM_fields_at_n[1], 0, 0, 1, -1) # (i - 1/2, j+1)
                                 + af.shift(self.yee_grid_EM_fields_at_n[1], 0, 0, 0, -1) # (i + 1/2, j+1)
                                )

                    E3 = 0.5 * (  self.yee_grid_EM_fields_at_n[2] # (i + 1/2, j + 1/2) 
                                + af.shift(self.yee_grid_EM_fields_at_n[2], 0, 0, 1) # (i - 1/2, j + 1/2)
                               )

                    B1 = 0.25 * (  self.yee_grid_EM_fields_at_n[3] # (i + 1/2, j)
                                 + af.shift(self.yee_grid_EM_fields_at_n[3], 0, 0, 1, 0)  # (i - 1/2, j)
                                 + af.shift(self.yee_grid_EM_fields_at_n[3], 0, 0, 1, -1) # (i - 1/2, j+1)
                                 + af.shift(self.yee_grid_EM_fields_at_n[3], 0, 0, 0, -1) # (i + 1/2, j+1)
                                )

                    B2 = self.yee_grid_EM_fields_at_n[4] # (i, j + 1/2)

                    B3 = 0.5 * (  self.yee_grid_EM_fields_at_n[5] # (i, j) 
                                + af.shift(self.yee_grid_EM_fields_at_n[5], 0, 0, 0, -1) # (i, j + 1)
                               )

                else:

                    E1 = self.yee_grid_EM_fields_at_n_plus_half[0] # (i, j + 1/2)
                    
                    E2 = 0.25 * (  self.yee_grid_EM_fields_at_n_plus_half[1] # (i + 1/2, j)
                                 + af.shift(self.yee_grid_EM_fields_at_n_plus_half[1], 0, 0, 1, 0)  # (i - 1/2, j)
                                 + af.shift(self.yee_grid_EM_fields_at_n_plus_half[1], 0, 0, 1, -1) # (i - 1/2, j+1)
                                 + af.shift(self.yee_grid_EM_fields_at_n_plus_half[1], 0, 0, 0, -1) # (i + 1/2, j+1)
                                )

                    E3 = 0.5 * (  self.yee_grid_EM_fields_at_n_plus_half[2] # (i + 1/2, j + 1/2) 
                                + af.shift(self.yee_grid_EM_fields_at_n_plus_half[2], 0, 0, 1) # (i - 1/2, j + 1/2)
                               )

                    B1 = 0.25 * (  self.yee_grid_EM_fields_at_n_plus_half[3] # (i + 1/2, j)
                                 + af.shift(self.yee_grid_EM_fields_at_n_plus_half[3], 0, 0, 1, 0)  # (i - 1/2, j)
                                 + af.shift(self.yee_grid_EM_fields_at_n_plus_half[3], 0, 0, 1, -1) # (i - 1/2, j+1)
                                 + af.shift(self.yee_grid_EM_fields_at_n_plus_half[3], 0, 0, 0, -1) # (i + 1/2, j+1)
                                )

                    B2 = self.yee_grid_EM_fields_at_n_plus_half[4] # (i, j + 1/2)

                    B3 = 0.5 * (  self.yee_grid_EM_fields_at_n_plus_half[5] # (i, j) 
                                + af.shift(self.yee_grid_EM_fields_at_n_plus_half[5], 0, 0, 0, -1) # (i, j + 1)
                               )

            # Gets fields at the center of the bottom face: i.e (i+1/2, j)
            elif(fields_location == 'center_bottom'):

                if(self.at_n == True):

                    E1 = 0.25 * (  self.yee_grid_EM_fields_at_n[0] # (i, j+1/2)
                                 + af.shift(self.yee_grid_EM_fields_at_n[0], 0, 0,  0, 1)  # (i, j-1/2)
                                 + af.shift(self.yee_grid_EM_fields_at_n[0], 0, 0, -1, 0) # (i + 1, j+1/2)
                                 + af.shift(self.yee_grid_EM_fields_at_n[0], 0, 0, -1, -1) # (i + 1, j-1/2)
                                )

                    E2 = self.yee_grid_EM_fields_at_n[1] # (i + 1/2, j)
                    
                    E3 = 0.5 * (  self.yee_grid_EM_fields_at_n[2] # (i + 1/2, j + 1/2) 
                                + af.shift(self.yee_grid_EM_fields_at_n[2], 0, 0, 0, 1) # (i + 1/2, j - 1/2)
                               )

                    B1 = self.yee_grid_EM_fields_at_n[3] # (i + 1/2, j)
                    
                    B2 = 0.25 * (  self.yee_grid_EM_fields_at_n[4] # (i, j+1/2)
                                 + af.shift(self.yee_grid_EM_fields_at_n[4], 0, 0,  0, 1)  # (i, j-1/2)
                                 + af.shift(self.yee_grid_EM_fields_at_n[4], 0, 0, -1, 0) # (i + 1, j+1/2)
                                 + af.shift(self.yee_grid_EM_fields_at_n[4], 0, 0, -1, -1) # (i + 1, j-1/2)
                                )
                    
                    B3 = 0.5 * (  self.yee_grid_EM_fields_at_n[5] #(i, j)
                                + af.shift(self.yee_grid_EM_fields_at_n[5], 0, 0, -1, 0) # (i + 1, j)
                               )

                else:

                    E1 = 0.25 * (  self.cell_centered_EM_fields_at_n_plus_half[0] # (i, j+1/2)
                                 + af.shift(self.cell_centered_EM_fields_at_n_plus_half[0], 0, 0,  0, 1)  # (i, j-1/2)
                                 + af.shift(self.cell_centered_EM_fields_at_n_plus_half[0], 0, 0, -1, 0) # (i + 1, j+1/2)
                                 + af.shift(self.cell_centered_EM_fields_at_n_plus_half[0], 0, 0, -1, -1) # (i + 1, j-1/2)
                                )

                    E2 = self.cell_centered_EM_fields_at_n_plus_half[1] # (i + 1/2, j)
                    
                    E3 = 0.5 * (  self.cell_centered_EM_fields_at_n_plus_half[2] # (i + 1/2, j + 1/2) 
                                + af.shift(self.cell_centered_EM_fields_at_n_plus_half[2], 0, 0, 0, 1) # (i + 1/2, j - 1/2)
                               )

                    B1 = self.cell_centered_EM_fields_at_n_plus_half[3] # (i + 1/2, j)
                    
                    B2 = 0.25 * (  self.cell_centered_EM_fields_at_n_plus_half[4] # (i, j+1/2)
                                 + af.shift(self.cell_centered_EM_fields_at_n_plus_half[4], 0, 0,  0, 1)  # (i, j-1/2)
                                 + af.shift(self.cell_centered_EM_fields_at_n_plus_half[4], 0, 0, -1, 0) # (i + 1, j+1/2)
                                 + af.shift(self.cell_centered_EM_fields_at_n_plus_half[4], 0, 0, -1, -1) # (i + 1, j-1/2)
                                )
                    
                    B3 = 0.5 * (  self.cell_centered_EM_fields_at_n_plus_half[5] #(i, j)
                                + af.shift(self.cell_centered_EM_fields_at_n_plus_half[5], 0, 0, -1, 0) # (i + 1, j)
                               )

        return(E1, E2, E3, B1, B2, B3)
