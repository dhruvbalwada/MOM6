!> Calculation of horizontal density flux for use in thickness diffuse 
module MOM_meso_sfn_ANN
  
! This file is part of MOM6. See LICENSE.md for the license.
use MOM_grid,          only : ocean_grid_type
use MOM_verticalGrid,  only : verticalGrid_type
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_file_parser,   only : get_param, log_version, param_file_type
use MOM_unit_scaling,  only : unit_scale_type
use MOM_diag_mediator, only : post_data, register_diag_field
use MOM_domains,       only : create_group_pass, do_group_pass, group_pass_type, &
                              start_group_pass, complete_group_pass
use MOM_domains,       only : To_North, To_East
use MOM_domains,       only : pass_var, CORNER
use MOM_ANN,           only : ANN_init, ANN_apply_array_sio, ANN_end, ANN_CS
use MOM_isopycnal_slopes,      only : calc_isoneutral_slopes
use MOM_variables,             only : thermo_var_ptrs

implicit none ; private

#include <MOM_memory.h>

public :: MOM_meso_sfn_ANN_init, MOM_meso_sfn_ANN_compute, MOM_meso_sfn_ANN_end

!> Control structure for meso-scale streamfunction ANN parameterization
type, public :: MESO_SFN_ANN_CS; private
  logical :: initialized = .false. !< If true, the module has been initialized.
  logical :: debug !< if true, write verbose checksums for debugging purposes. 

  real :: ann_coeff  !< Coefficient to multiply the ANN output by.
  real    :: kappa_smooth        !< Vertical diffusivity used to interpolate more sensible values
                                 !! of T & S into thin layers [H Z T-1 ~> m2 s-1 or kg m-1 s-1]
end type MESO_SFN_ANN_CS

contains 

! Algorithm:
! - The main procedure from here can be called from MOM_thickness_diffuse.F90
! It returns 3D arrays of the Sfn_unlim_u and Sfn_unlim_v, which in 
! thickness diffuse are carried around as 2D sections in horizontal and depth. 
! - In the thickness diffuse, these 3D arrays will be used to select the appropriate 2D slices. 
! These 3D arrays will be on u and v point, but also the interface. 
! 
! In this module a few things need to happen:
! 1. The density fluxes needs to be computed from the ANN at the center points and on the
!   interface. Note that the ANN outputs the density flux (not the streamfunction)
! 2. This is going to require us to move the u,v gradients to intefaces. 
! 3. The density fluxes computed as drdx are already on interfaces in the thickness diffuse, but we will have 
!   to figure out how to 
! NOTE: When we trained the model everything was at the same vertical level, but I am not sure if we
! can access the same vertical depth in this layered model. So some smoothness assumption will be implicit here. 


!> Calculate the meso-scale streamfunction ANN parameterization

subroutine MOM_meso_sfn_ANN_compute(h, e, sfn_u, sfn_v, G, GV, US, tv, CS, dt)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(unit_scale_type),                      intent(in)    :: US     !< A dimensional unit scaling type
  type(thermo_var_ptrs),                      intent(in)    :: tv     !< Thermodynamics structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)    :: h      !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1),  intent(in)    :: e      !< Layer thickness [H ~> m or kg m-2]
  type(MESO_SFN_ANN_CS),                intent(inout) :: CS !< Control structure for thickness_flux_ann
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(out)   :: sfn_u  !< Meso-scale streamfunction on u-points [L2 T-1 ~> m2 s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(out)   :: sfn_v  !< Meso-scale streamfunction on v-points [L2 T-1 ~> m2 s-1]
  real,                                      intent(in)    :: dt     !< Model time step [T ~> s]
  

  
  ! Local variables
  integer :: i, j, k, is, ie, js, je, nz

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: drdx_u, drdz_u !< Zonal density gradient at u-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: drdy_v, drdz_v !< Meridional density gradient at v-points [R L-1 ~> kg m-4]
  ! These next 2 probably don't get used. 
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: slope_x !< Isopycnal slope in x-direction at u-points [L L-1 ~> nondim]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: slope_y !< Isopycnal slope in y-direction at v-points [L L-1 ~> nondim]

  !real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: drdx_c !< Zonal density gradient at center points [R L-1 ~> kg m-4]
  !real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: drdy_c !< Meridional density gradient at center points [R L-1 ~> kg m-4]
  
  !real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: upsilon_x, upsilon_y !< Stream function components at center points [L2 T-1 ~> m2 s-1] 
  real :: mag_grad !< Magnitude of density gradient at center points [R L-1 ~> kg m-4]
  real :: Kappa !< Dimensional diffusivity [L2 T-1 ~> m2 s-1]
  logical :: use_stanley

  use_stanley = .false. ! Not using Stanley smoothing here.
  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Kappa = 1000.0 ! Temporary value for testing [m2 s-1]
  slope_x(:,:,:) = 0.0
  slope_y(:,:,:) = 0.0
  
  sfn_u(:,:,:) = 0.0
  sfn_v(:,:,:) = 0.0

  ! Compute rho gradients 
  call calc_isoneutral_slopes(G, GV, US, h, e, tv, dt*CS%kappa_smooth, use_stanley, slope_x, slope_y, &
                              drdx_u=drdx_u, drdy_v=drdy_v, drdz_u=drdz_u, drdz_v=drdz_v)
  
  ! Interpolate the rho gradients to the center point 
  ! Compute the streamfunction  
  

  do k=2, nz
    do j=js,je ; do i=is-1,ie
      mag_grad = sqrt((US%Z_to_L*drdx_u(i,j,k))**2 + drdz_u(i,j,k)**2)
      sfn_u(I,j,k) = (-Kappa * drdx_u(i,j,k))/mag_grad * G%dy_Cu(I,j)
    enddo ; enddo
    do j=js-1,je ; do i=is,ie
      mag_grad = sqrt((US%Z_to_L*drdy_v(i,j,k))**2 + drdz_v(i,j,k)**2)
      sfn_v(i,J,k) = (-Kappa * drdy_v(i,j,k))/mag_grad * G%dx_Cv(i,J)
    enddo ; enddo
  enddo
  
  ! Interpolate the streamfunction to the edges
  ! do k=1, nz+1
  !  !> Interpolate streamfunction to u, v points
  !   ! > We multiply the ann coeff at this later stage, so the unchanged F can be used for diagnostics
  !   do j=js,je ; do i=is-1,ie
  !     sfn_u(I,j,k) = 0.5 * (upsilon_x(i,j,k) + upsilon_x(i+1,j,k)) * G%dyCu(I,j) * G%mask2dCu(I,j) * CS%ann_coeff
  !   enddo ; enddo
  !   do j=js-1,je ; do i=is,ie
  !     sfn_v(i,J,k) = 0.5 * (upsilon_y(i,j,k) + upsilon_y(i,j+1,k)) * G%dxCv(i,J) * G%mask2dCv(i,J) * CS%ann_coeff
  !   enddo ; enddo 
  ! enddo


end subroutine MOM_meso_sfn_ANN_compute


!> Initializes the meso-scale streamfunction ANN parameterization
!! 
subroutine MOM_meso_sfn_ANN_init(Time, G, GV, US, param_file, diag, CS)
  type(time_type),         intent(in) :: Time    !< Current model time
  type(ocean_grid_type),   intent(in) :: G       !< Ocean grid structure
  type(verticalGrid_type), intent(in) :: GV      !< Vertical grid structure
  type(unit_scale_type),   intent(in) :: US      !< A dimensional unit scaling type
  type(param_file_type),   intent(in) :: param_file !< Parameter file handles
  type(diag_ctrl), target, intent(inout) :: diag !< Diagnostics control structure
  type(thickness_flux_ann_CS), intent(inout) :: CS !< Control structure for thickness_flux_ann

  ! Local variables 
  character(len=40) :: mdl = "MOM_meso_sfn_ANN" ! This is module's name
# include "version_variable.h"  

  ! We don't need to check if use is true, because this is only called if it is.
  call get_param(param_file, mdl, "meso_sfn_ann_coeff", CS%ann_coeff, &
                      "Coefficient to multiply the mesoscale streamfunction ANN output by", default=1.0, units="nondim")

  call get_param(param_file, mdl, "KD_SMOOTH", CS%kappa_smooth, &
                 "A diapycnal diffusivity that is used to interpolate "//&
                 "more sensible values of T & S into thin layers.", &
                 units="m2 s-1", default=1.0e-6, scale=GV%m2_s_to_HZ_T)
end subroutine MOM_meso_sfn_ANN_init


!> Finalizes the meso-scale streamfunction ANN parameterization
!! 
subroutine MOM_meso_sfn_ANN_end(CS)
  type(MESO_SFN_ANN_CS), intent(inout) :: CS !< Control structure

  ! Deallocate anything that needs to be. 

end subroutine MOM_meso_sfn_ANN_end

end module MOM_meso_sfn_ANN