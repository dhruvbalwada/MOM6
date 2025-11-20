!> Calculation of horizontal density flux for use in thickness diffuse 
module MOM_meso_sfn_ANN
  
! This file is part of MOM6. See LICENSE.md for the license.
use MOM_grid,          only : ocean_grid_type
use MOM_verticalGrid,  only : verticalGrid_type
use MOM_diag_mediator,         only : post_data, query_averaging_enabled, diag_ctrl
use MOM_diag_mediator,         only : register_diag_field, safe_alloc_ptr, time_type
use MOM_diag_mediator,         only : diag_update_remap_grids
use MOM_file_parser,   only : get_param, log_version, param_file_type
use MOM_unit_scaling,  only : unit_scale_type
use MOM_diag_mediator, only : post_data, register_diag_field
use MOM_domains,       only : create_group_pass, do_group_pass, group_pass_type, &
                              start_group_pass, complete_group_pass
use MOM_domains,       only : To_North, To_East
use MOM_domains,       only : pass_var, CORNER
use MOM_ANN,           only : ANN_init, ANN_apply_array_sio, ANN_end, ANN_CS
use MOM_ANN,           only : ANN_apply_vector_orig
use MOM_isopycnal_slopes,      only : calc_isoneutral_slopes
use MOM_variables,             only : thermo_var_ptrs
use MOM_error_handler,        only : MOM_error,  FATAL

implicit none ; private

#include <MOM_memory.h>

public :: meso_sfn_ANN_init, meso_sfn_ANN_compute, meso_sfn_ANN_end

!> Control structure for meso-scale streamfunction ANN parameterization
type, public :: MESO_SFN_ANN_CS; private
  logical :: initialized = .false. !< If true, the module has been initialized.
  logical :: debug !< if true, write verbose checksums for debugging purposes. 

  real :: ann_coeff  !< Coefficient to multiply the ANN output by.
  real    :: kappa_smooth        !< Vertical diffusivity used to interpolate more sensible values
                                 !! of T & S into thin layers [H Z T-1 ~> m2 s-1 or kg m-1 s-1]
  character(len=40) :: meso_sfn_ann_model_type !< Type of ANN model (e.g. options GM_simple, GM_ann, GM_rotated_ann, nondim_ann).
  integer :: ann_window !< Size of the window used in the ANN model.

  type(ANN_CS) :: ann_rho_flux !< ANN instance for off-diagonal and diagonal stress
  character(len=200) :: ann_file_rho_flux !< Path to netcdf file with ANN

  type(diag_ctrl), pointer :: diag => NULL() !< structure used to regulate timing of diagnostics
  !! Diagnostic identifiers 
  integer :: id_drdx_u, id_drdy_v !< Diagnostic ids for density gradients at u and v points.
  integer :: id_drdz_u, id_drdz_v !< Diagnostic ids for density gradients at u and v points.
  integer :: id_drdx_c, id_drdy_c
  integer :: id_Fx_c, id_Fy_c
  integer :: id_Fx_u, id_Fy_v
  integer :: id_sfn_u, id_sfn_v !< Diagnostic ids for streamfunction at u and v points.


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

subroutine meso_sfn_ANN_compute(h, e, sfn_u, sfn_v, G, GV, US, tv, CS, dt, u, v)
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
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(in)    :: u      !< Zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(in)    :: v      !< Meridional velocity [L T-1 ~>

  ! Local variables
  integer :: i, j, k, is, ie, js, je, nz, shift, stencil_points, ii, jj

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: drdx_u, drdz_u !< Zonal density gradient at u-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: drdy_v, drdz_v !< Meridional density gradient at v-points [R L-1 ~> kg m-4]
  ! These next 2 probably don't get used. 
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: slope_x !< Isopycnal slope in x-direction at u-points [L L-1 ~> nondim]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: slope_y !< Isopycnal slope in y-direction at v-points [L L-1 ~> nondim]
  
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: Fx_u !< Zonal density flux at u-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: Fy_v !< Meridional density flux at v-points [R L-1 ~> kg m-4]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: drdx_c !< Zonal density gradient at center points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: drdy_c !< Meridional density gradient at center points [R L-1 ~> kg m-4]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: Fx_c !< Zonal density flux at center points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: Fy_c !< Meridional density flux at center points [R L-1 ~> kg m-4]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV))  :: dudx, dvdy, dudy, dvdx   ! components of the velocity gradient tensor on the i,j points (cell-center)

  real, allocatable :: drdx_local(:,:), drdy_local(:,:)
  real, allocatable :: dudx_local(:,:), dudy_local(:,:), dvdx_local(:,:), dvdy_local(:,:)
  real, allocatable :: sh_xx_local(:,:), sh_xy_local(:,:), vort_local(:,:)
  real :: vel_grad_mag, rho_grad_mag
  real, allocatable :: x(:) !< Input vector to the ANN
  real, allocatable :: y(:) !< Output vector from the ANN

  
  !real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: upsilon_x, upsilon_y !< Stream function components at center points [L2 T-1 ~> m2 s-1] 
  real :: mag_grad !< Magnitude of density gradient at center points [R L-1 ~> kg m-4]
  real :: Kappa !< Dimensional diffusivity [L2 T-1 ~> m2 s-1]
  logical :: use_stanley

  use_stanley = .false. ! Not using Stanley smoothing here.
  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  ! Allocate the local stencil variables
  allocate(drdx_local(CS%ann_window, CS%ann_window), drdy_local(CS%ann_window, CS%ann_window), &
           dudx_local(CS%ann_window, CS%ann_window), dudy_local(CS%ann_window, CS%ann_window), &
           dvdx_local(CS%ann_window, CS%ann_window), dvdy_local(CS%ann_window, CS%ann_window), &
           sh_xx_local(CS%ann_window, CS%ann_window), sh_xy_local(CS%ann_window, CS%ann_window), &
           vort_local(CS%ann_window, CS%ann_window))

  shift = (CS%ann_window-1)/2
  stencil_points = CS%ann_window * CS%ann_window

  if (CS%meso_sfn_ann_model_type == "nondim_rhoF_ann") then
    allocate(x(stencil_points*5), y(2))
  else
    allocate(x(2), y(2))
  endif
  
  slope_x(:,:,:) = 0.0
  slope_y(:,:,:) = 0.0
  
  sfn_u(:,:,:) = 0.0
  sfn_v(:,:,:) = 0.0

  Fx_u(:,:,:) = 0.0
  Fy_v(:,:,:) = 0.0
  Fx_c(:,:,:) = 0.0
  Fy_c(:,:,:) = 0.0
 
  drdx_u(:,:,:) = 0.0
  drdy_v(:,:,:) = 0.0
  drdz_u(:,:,:) = 0.0
  drdz_v(:,:,:) = 0.0
  drdx_c(:,:,:) = 0.0
  drdy_c(:,:,:) = 0.0


  ! Compute rho gradients 
  call calc_isoneutral_slopes(G, GV, US, h, e, tv, dt*CS%kappa_smooth, use_stanley, slope_x, slope_y, &
                              drdx_u=drdx_u, drdy_v=drdy_v, drdz_u=drdz_u, drdz_v=drdz_v, halo=3)
  
  
  ! Interpolate the rho gradients to the center point 
  call center_grad_rho(drdx_u, drdy_v, drdx_c, drdy_c, G, GV, CS)

  ! Compute velocity gradients at center points 
  call vel_gradients(u, v, G, GV, dudx, dudy, dvdx, dvdy, CS)

  ! Post diagnostics
  if (CS%id_drdx_u > 0) call post_data(CS%id_drdx_u, drdx_u, CS%diag)
  if (CS%id_drdy_v > 0) call post_data(CS%id_drdy_v, drdy_v, CS%diag)

  if (CS%id_drdz_u > 0) call post_data(CS%id_drdz_u, drdz_u, CS%diag)
  if (CS%id_drdz_v > 0) call post_data(CS%id_drdz_v, drdz_v, CS%diag)

  if (CS%id_drdx_c > 0) call post_data(CS%id_drdx_c, drdx_c, CS%diag)
  if (CS%id_drdy_c > 0) call post_data(CS%id_drdy_c, drdy_c, CS%diag)
  ! Compute the streamfunction  
  
  ! Simple GM (non-ANN) implementation for testing
  if (CS%meso_sfn_ann_model_type == "GM_simple") then
    Kappa = 1000.0 ! Temporary value for testing [m2 s-1]
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
    return
  endif 
  
  ! We need a separate type of structure for handling the ANN cases, 
  ! because the ANN works at center points.

  ! compute the density fluxes at center points using the ANN
  do k = 1, nz+1
    do j = js-1, je+1 ; do i = is-1, ie+1
      ! ANN-based GM implementation (where the A matrix is like Kappa)
      if (CS%meso_sfn_ann_model_type == "GM_ann") then
        x(1) = drdx_c(i,j,k)
        x(2) = drdy_c(i,j,k)

        ! TODO: Call the ANN 
        call ANN_apply_vector_orig(x,y, CS%ann_rho_flux)
        
        ! temp fix that works like GM
        ! use in place of above ANN call for testing
        !y(1) = -1000.0 * x(1)
        !y(2) = -1000.0 * x(2)
        ! End temp fix
      else if (CS%meso_sfn_ann_model_type == "nondim_rhoF_ann") then
        drdx_local(:,:) = drdx_c(i-shift:i+shift,j-shift:j+shift,k)
        drdy_local(:,:) = drdy_c(i-shift:i+shift,j-shift:j+shift,k)
        ! Take the velocity gradients below the interface k
        dudx_local(:,:) = dudx(i-shift:i+shift,j-shift:j+shift,k) 
        dudy_local(:,:) = dudy(i-shift:i+shift,j-shift:j+shift,k)
        dvdx_local(:,:) = dvdx(i-shift:i+shift,j-shift:j+shift,k)
        dvdy_local(:,:) = dvdy(i-shift:i+shift,j-shift:j+shift,k)

        ! Compute the strain rate tensor components and vorticity
        sh_xx_local(:,:) = dudx_local(:,:) - dvdy_local(:,:)
        sh_xy_local(:,:) = dudy_local(:,:) + dvdx_local(:,:)
        vort_local(:,:)  = dvdx_local(:,:) - dudy_local(:,:)

        ! Compute the magnitude of the velocity gradient tensor for the local stencil
        rho_grad_mag = 0.0
        vel_grad_mag = 0.0
        do jj=1, CS%ann_window
          do ii=1, CS%ann_window
            rho_grad_mag = rho_grad_mag + drdx_local(ii,jj)**2 + drdy_local(ii,jj)**2
            vel_grad_mag = vel_grad_mag + sh_xx_local(ii,jj)**2 + sh_xy_local(ii,jj)**2 + vort_local(ii,jj)**2
          enddo
        enddo
        rho_grad_mag = sqrt(rho_grad_mag) + 1e-30
        vel_grad_mag = sqrt(vel_grad_mag) + 1e-30

        ! Normalize inputs 
        drdx_local(:,:) = drdx_local(:,:) / rho_grad_mag
        drdy_local(:,:) = drdy_local(:,:) / rho_grad_mag

        sh_xx_local(:,:) = sh_xx_local(:,:)/ vel_grad_mag
        sh_xy_local(:,:) = sh_xy_local(:,:)/ vel_grad_mag
        vort_local(:,:) = vort_local(:,:)/ vel_grad_mag

        ! Prepare input vector for ANN
        x(1:stencil_points) = RESHAPE(drdx_local, (/stencil_points/))
        x(stencil_points+1:2*stencil_points) = RESHAPE(drdy_local, (/stencil_points/))
        x(2*stencil_points+1:3*stencil_points) = RESHAPE(sh_xx_local, (/stencil_points/))
        x(3*stencil_points+1:4*stencil_points) = RESHAPE(sh_xy_local, (/stencil_points/))
        x(4*stencil_points+1:5*stencil_points) = RESHAPE(vort_local, (/stencil_points/))

        ! Call the ANN
        call ANN_apply_vector_orig(x,y, CS%ann_rho_flux)

        ! Dimensionalize the output
        y(:) = y(:) * rho_grad_mag * vel_grad_mag * G%areaT(i,j)
      else 
        call MOM_error(FATAL, "meso_sfn_ANN_compute: Unknown meso_sfn_ann_model_type "//&
                       trim(CS%meso_sfn_ann_model_type))
      end if

      Fx_c(i,j,k) = - y(1) !* G%mask2dT(i,j) 
      Fy_c(i,j,k) = - y(2) !* G%mask2dT(i,j) 
      ! I think the minus sign is needed because our friend Pavel
      ! defined the fluxes as - u'rho' instead of u'rho'.

    enddo ; enddo
  end do
  
  ! Now we need to interpolate the density fluxes to u and v points.
  call center2uv(Fx_c, Fy_c, Fx_u, Fy_v, G, GV)


  do k=2, nz
    do j=js,je ; do i=is-1,ie
      mag_grad = sqrt( (US%Z_to_L*drdx_u(i,j,k))**2 + drdz_u(i,j,k)**2 )
      sfn_u(I,j,k) = (Fx_u(i,j,k))/mag_grad * G%dy_Cu(I,j) * G%OBCmaskCu(I,j)
    enddo ; enddo
    do j=js-1,je ; do i=is,ie
      mag_grad = sqrt( (US%Z_to_L*drdy_v(i,j,k))**2 + drdz_v(i,j,k)**2 )
      sfn_v(i,J,k) = (Fy_v(i,j,k))/mag_grad * G%dx_Cv(i,J) * G%OBCmaskCv(i,J)
    enddo ; enddo
  enddo
  ! Is the OBC mask similar to the regular one?

  if (CS%id_Fx_c > 0) call post_data(CS%id_Fx_c, Fx_c, CS%diag)
  if (CS%id_Fy_c > 0) call post_data(CS%id_Fy_c, Fy_c, CS%diag)

  if (CS%id_Fx_u > 0) call post_data(CS%id_Fx_u, Fx_u, CS%diag)
  if (CS%id_Fy_v > 0) call post_data(CS%id_Fy_v, Fy_v, CS%diag)

  if (CS%id_sfn_u > 0) call post_data(CS%id_sfn_u, sfn_u, CS%diag)
  if (CS%id_sfn_v > 0) call post_data(CS%id_sfn_v, sfn_v, CS%diag)


end subroutine meso_sfn_ANN_compute

subroutine center_grad_rho(drdx_u, drdy_v, drdx_c, drdy_c, G, GV, CS)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(MESO_SFN_ANN_CS),                intent(inout) :: CS !< Control structure for thickness_flux_ann
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(in)    :: drdx_u !< Zonal density gradient at u-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(in)    :: drdy_v !< Meridional density gradient at v-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1),  intent(inout)   :: drdx_c !< Zonal density gradient at center points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1),  intent(inout)   :: drdy_c !< Meridional density gradient at center points [R L-1 ~> kg m-4]

  integer :: i, j, k, is, ie, js, je, nz, shift

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  shift = (CS%ann_window-1)/2

  do k=1, nz+1
    do j=js-shift-1,je+shift+1 ; do i=is-shift-1,ie+shift+1
      drdx_c(i,j,k) = 0.5 * (drdx_u(i-1,j,k) * G%mask2dCu(i-1,j) + drdx_u(i,j,k) * G%mask2dCu(i,j)) * G%mask2dT(i,j)
      drdy_c(i,j,k) = 0.5 * (drdy_v(i,j-1,k) * G%mask2dCv(i,j-1) + drdy_v(i,j,k) * G%mask2dCv(i,j)) * G%mask2dT(i,j)
    enddo ; enddo
  enddo

end subroutine center_grad_rho

subroutine center2uv(var1_c, var2_c, var1_u, var2_v, G, GV)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1),  intent(in)    :: var1_c !< Variable at center points
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1),  intent(in)    :: var2_c !< Variable at center points
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(inout)   :: var1_u !< Variable at u points
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(inout)   :: var2_v !< Variable at v points

  integer :: i, j, k, is, ie, js, je, nz

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke


  do k=1, nz+1
    do j=js,je ; do i=is-1,ie
      var1_u(I,j,k) = 0.5 * (var1_c(i,j,k) * G%mask2dT(i,j) + var1_c(i+1,j,k) * G%mask2dT(i+1,j)) * G%mask2dCu(I,j)
    enddo ; enddo
    do j=js-1,je ; do i=is,ie
      var2_v(i,J,k) = 0.5 * (var2_c(i,j,k) * G%mask2dT(i,j) + var2_c(i,j+1,k) * G%mask2dT(i,j+1)) * G%mask2dCv(i,J)
    enddo ; enddo 
  enddo

end subroutine center2uv


!> Calculates the velocity gradients at the center points in 3D.
subroutine vel_gradients(u, v, G, GV, dudx, dudy, dvdx, dvdy, CS)
  type(ocean_grid_type),                     intent(in)    :: G   !< Ocean grid structure
  type(verticalGrid_type),                   intent(in)    :: GV  !< The ocean's vertical grid structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)),intent(in)    :: u   !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)),intent(in)    :: v   !< The meridional velocity [L T-1 ~> m s-1].
  ! Center points 
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(out) :: dudx, dvdy, dudy, dvdx   ! components of the velocity gradient tensor on the i,j points (cell-center)
  type(MESO_SFN_ANN_CS), intent(in) :: CS !< Control structure for thickness_flux_ann
      
  ! Corner points
  real, dimension(SZIB_(G), SZJB_(G),SZK_(GV)) :: dudy_q, dvdx_q   
  integer :: is, ie, js, je
  !integer :: Isq, Ieq, Jsq, Jeq
  integer :: nz
  integer :: i, j, k
  integer :: shift
  
  ! Line 407 of MOM_hor_visc.F90
  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  !Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  shift = (CS%ann_window-1)/2

  do k=1, nz
    ! Copy code from MOM_hor_visc.F90
    ! Calculate some velocity gradients at center points directly
    do j=js-shift-1,je+shift+1 ; do i=is-shift-1,ie+shift+1 ! has halo 2 ! loops over c points
      dudx(i,j,k) = G%IdxT(i,j)* (u(I,j,k) * G%mask2dCu(I,j)   - u(I-1,j,k) * G%mask2dCu(I-1,j)) * G%mask2dT(i,j)
      dvdy(i,j,k) = G%IdyT(i,j)* (v(i,J,k) * G%mask2dCv(i,J)   - v(i,J-1,k) * G%mask2dCv(i,J-1)) * G%mask2dT(i,j)
      ! the above masking ensures no-flow condition. 
    enddo ; enddo

    ! Calculate velocity gradients at corner points 
    ! loops over q points (we don't use the soft convention of MOM6 for do loop indices here)
    do j=js-shift-2,je+shift+1 ; do i=is-shift-2,ie+shift+1
      dvdx_q(I,J,k) = G%IdxBu(I,J)*(v(i+1,J,k)  - v(i,J,k) ) * G%mask2dBu(I,J)
      dudy_q(I,J,k) = G%IdyBu(I,J)*(u(I,j+1,k)  - u(I,j,k) ) * G%mask2dBu(I,J)
      ! 
    enddo ; enddo

    ! interpolate corner grads to center points 
    do j = js-shift-1, je+shift+1; do i = is-shift-1, ie+shift+1
      dvdx(i,j,k) =  0.25 * (dvdx_q(I,J,k) + dvdx_q(I-1,J,k) + dvdx_q(I,J-1,k) + dvdx_q(I-1,J-1,k)) * G%mask2dT(i,j) 
      dudy(i,j,k) =  0.25 * (dudy_q(I,J,k) + dudy_q(I-1,J,k) + dudy_q(I,J-1,k) + dudy_q(I-1,J-1,k)) * G%mask2dT(i,j) 
    enddo; enddo 
  enddo
end subroutine vel_gradients

!> Initializes the meso-scale streamfunction ANN parameterization
!! 
subroutine meso_sfn_ANN_init(Time, G, GV, US, param_file, diag, CS)
  type(time_type),         intent(in) :: Time    !< Current model time
  type(ocean_grid_type),   intent(in) :: G       !< Ocean grid structure
  type(verticalGrid_type), intent(in) :: GV      !< Vertical grid structure
  type(unit_scale_type),   intent(in) :: US      !< A dimensional unit scaling type
  type(param_file_type),   intent(in) :: param_file !< Parameter file handles
  type(diag_ctrl), target, intent(inout) :: diag !< Diagnostics control structure
  type(MESO_SFN_ANN_CS), intent(inout) :: CS !< Control structure for meso sfn ann

  ! Local variables 
  character(len=40) :: mdl = "meso_sfn_ANN" ! This is module's name
# include "version_variable.h"  

  CS%diag => diag

  ! We don't need to check if use is true, because this is only called if it is.
  call get_param(param_file, mdl, "meso_sfn_ann_coeff", CS%ann_coeff, &
                      "Coefficient to multiply the mesoscale streamfunction ANN output by", default=1.0, units="nondim")

  call get_param(param_file, mdl, "KD_SMOOTH", CS%kappa_smooth, &
                 "A diapycnal diffusivity that is used to interpolate "//&
                 "more sensible values of T & S into thin layers.", &
                 units="m2 s-1", default=1.0e-6, scale=GV%m2_s_to_HZ_T)
  call get_param(param_file, mdl, "meso_sfn_ann_type", CS%meso_sfn_ann_model_type, &
                      "Type of ANN model (e.g. options GM_simple, GM_ann).", default="GM_simple")
  

  if (CS%meso_sfn_ann_model_type /= "GM_simple") then
    call get_param(param_file, mdl, "meso_sfn_ann_window", CS%ann_window, &
                        "Number of horizontal grid points to use in the thickness flux ANN window", default=1)
    call get_param(param_file, mdl, "meso_sfn_ann_file", CS%ann_file_rho_flux, &
                 "ANN parameters for prediction of density fluxes (netcdf)", &
                 default="INPUT/rho_flux.nc")
    call ANN_init(CS%ann_rho_flux, CS%ann_file_rho_flux)
  endif

  ! Register diagnostic fields
  CS%id_drdx_u = register_diag_field('ocean_model', 'meso_sfn_drdx_u', diag%axesCui, Time, &
           'Zonal density gradient used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_drdy_v = register_diag_field('ocean_model', 'meso_sfn_drdy_v', diag%axesCvi, Time, &
           'Meridional density gradient used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_drdz_u = register_diag_field('ocean_model', 'meso_sfn_drdz_u', diag%axesCui, Time, &
           'Vertical density gradient at u points used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_drdz_v = register_diag_field('ocean_model', 'meso_sfn_drdz_v', diag%axesCvi, Time, &
           'Vertical density gradient at v points used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_drdx_c = register_diag_field('ocean_model', 'meso_sfn_drdx_c', diag%axesTi, Time, &
           'Zonal density gradient at center points used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_drdy_c = register_diag_field('ocean_model', 'meso_sfn_drdy_c', diag%axesTi, Time, &
           'Meridional density gradient at center points used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_Fx_c = register_diag_field('ocean_model', 'meso_sfn_flux_x_c', diag%axesTi, Time, &
           'Zonal density flux at center points used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_Fy_c = register_diag_field('ocean_model', 'meso_sfn_flux_y_c', diag%axesTi, Time, &
           'Meridional density flux at center points used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_Fx_u = register_diag_field('ocean_model', 'meso_sfn_flux_x_u', diag%axesCui, Time, &
           'Zonal density flux at u points used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_Fy_v = register_diag_field('ocean_model', 'meso_sfn_flux_y_v', diag%axesCvi, Time, &
           'Meridional density flux at v points used in meso sfn', &
           'kg m-4', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_sfn_u = register_diag_field('ocean_model', 'meso_sfn_unlim_u', diag%axesCui, Time, &
           'Meso-scale streamfunction at u points', &
           'm2 s-1', conversion=US%L_to_m**2*US%s_to_T)
  CS%id_sfn_v = register_diag_field('ocean_model', 'meso_sfn_unlim_v', diag%axesCvi, Time, &
           'Meso-scale streamfunction at v points', &
           'm2 s-1', conversion=US%L_to_m**2*US%s_to_T)
end subroutine meso_sfn_ANN_init


!> Finalizes the meso-scale streamfunction ANN parameterization
!! 
subroutine meso_sfn_ANN_end(CS)
  type(MESO_SFN_ANN_CS), intent(inout) :: CS !< Control structure

  ! Deallocate anything that needs to be. 

  
  if (CS%meso_sfn_ann_model_type /= "GM_simple") then
    call ANN_end(CS%ann_rho_flux)
  endif

end subroutine meso_sfn_ANN_end

end module MOM_meso_sfn_ANN