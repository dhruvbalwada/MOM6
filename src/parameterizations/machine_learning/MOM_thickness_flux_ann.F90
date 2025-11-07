!> Thickness flux using ANN 
module MOM_thickness_flux_ann

! This file is part of MOM6. See LICENSE.md for the license.
use MOM_ann,                   only : ann_init, ann, ann_cs
use MOM_grid,                  only : ocean_grid_type
use MOM_verticalGrid,          only : verticalGrid_type
use MOM_unit_scaling,          only : unit_scale_type
use MOM_file_parser,           only : get_param, log_version, param_file_type
use MOM_diag_mediator,         only : post_data, query_averaging_enabled, diag_ctrl
use MOM_diag_mediator,         only : register_diag_field, safe_alloc_ptr, time_type
use MOM_diag_mediator,         only : diag_update_remap_grids
use MOM_domains,               only : pass_var, CORNER, pass_vector
use MOM_interface_heights,     only : find_eta, thickness_to_dz
use MOM_variables,             only : thermo_var_ptrs, cont_diag_ptrs
use MOM_isopycnal_slopes,      only : calc_isoneutral_slopes

implicit none ; private

#include <MOM_memory.h>

public thickness_flux_ann_init, thickness_flux_ann_end, thickness_flux_ann_full

!> Control structure/type for thickness flux ANN
type, public :: THICKNESS_FLUX_ANN_CS ; private
  logical :: initialized = .false. !< If true, the module has been initialized.
  !logical :: thickness_flux_ann  !< If true, use thickness fluxes are computed using ANN.
  logical :: debug !< if true, write verbose checksums for debugging purposes. 

  type(ann_cs) :: ann_cs !< ANN control structure.
  integer :: thickness_ann_num_layers ! number of layers
  character(len=200) :: thickness_ann_NNfile   ! The name of netcdf file having neural network shape function
  character(len=200) :: thickness_ann_model_type ! The type of model to use for thickness fluxes
  real :: ann_coeff  !< Coefficient to multiply the ANN output by.
  real :: FGR_ANN !< The filter to grid ratio for ANN, we have different filter scales and dx
  integer :: ann_window  !< Number of horizontal grid points to use in the ANN window.
  logical :: decompose_h_gradients !< If true, decompose the h gradients into steady and transient parts.
  real    :: h_min_mask !< Minimum thickness for the mask
  real    :: h_mask_width !< Width of the mask for the h gradients
  logical :: h_mask_flux !< If true, mask regions where layer very thin.
  logical :: remove_barotropic_flux !< If true, remove the barotropic component from the thickness fluxes.
  logical :: limit_upslope_flow !< If true, limit the upslope flow.
  real :: kappa_smooth    !< A diffusivity for smoothing T/S in vanished layers [H Z T-1 ~> m2 s-1 or kg m-1 s-1]
  real    :: slope_max           !< Slopes steeper than slope_max are limited in some way [Z L-1 ~> nondim]
  real    :: delta_BL_top, delta_BL_bottom !< The delta for the boundary layer up and down [Z ~> m]
  logical :: mask_boundary_layers !< If true, mask the boundary layers in the ANN output.

  type(diag_ctrl), pointer :: diag => NULL() !< structure used to regulate timing of diagnostics

  !! Diagnostic identifier
  integer :: id_dhdx, id_dhdy, id_dhbardx, id_dhbardy
  integer :: id_h_mask, id_boundary_mask 
  integer :: id_Fx, id_Fy, id_uhTrANN, id_vhTrANN
  integer :: id_dudx, id_dudy, id_dvdx, id_dvdy
  integer :: id_slope_x, id_slope_y !< IDs for the slopes
  integer :: id_sfn_unlim_x, id_sfn_unlim_y

end type THICKNESS_FLUX_ANN_CS

contains 

!> Calculates the parameterized thickness fluxes for use in the continuity equation.
!> Returns the fluxes at the u,v points.
subroutine thickness_flux_ann_full(h, u, v, uhTrANN, vhTrANN, G, GV, US, tv, CS, dt)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(unit_scale_type),                      intent(in)    :: US     !< A dimensional unit scaling type
  type(thermo_var_ptrs),                      intent(in)    :: tv     !< Thermodynamics structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)    :: h      !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(in)    :: u      !< Zonal velocity [L T-1 ~> m s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(in)    :: v      !< Meridional velocity [L T-1 ~> m s-1]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(out)   :: uhTrANN   !< Zonal ANN h transport u*h*dy [L2 H T-1 ~> m3 s-1 or kg s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(out)   :: vhTrANN   !< Meridional ANN h transport v*h*dx [L2 H T-1 ~> m3 s-1 or kg s-1]
  type(thickness_flux_ann_CS),                intent(inout) :: CS !< Control structure for thickness_flux_ann
  real,                                       intent(in)    :: dt     !< Time increment [T ~> s]
  ! Local variables
  integer :: i, j, k, is, ie, js, je, nz, shift, stencil_points, ii, jj
  integer :: Nin ! number of input variable types to the ANN (will be multiplied by the window size later )
  integer :: Nout = 2! number of output variable types from the ANN
  ! Variables for the gradients
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dhdx, dhdy
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dudx, dudy
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dvdx, dvdy
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: h_mask
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dhbardx, dhbardy

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1)    :: Sfn_unlim_u, slope_x !< Zonal transport streamfunction without limiters
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1)    :: Sfn_unlim_v, slope_y !< Meridional transport streamfunction without limiters
  
  
  ! Variables for the ANN output 
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: FxC, FyC

  ! Variables for the local stencil 
  real, allocatable :: dhdx_local(:,:), dhdy_local(:,:), dudx_local(:,:), dudy_local(:,:), dvdx_local(:,:), dvdy_local(:,:)
  real, allocatable :: x(:) !To-Do: make this adjustable based on window size.
  real, allocatable :: y(:), y_rot(:)
  real :: vel_grad_mag, h_grad_mag
  real :: tol_vel_grad, tol_h_grad
  real :: NGM_C
  integer :: nk_linear  ! The number of layers over which the streamfunction goes to 0.
  logical :: use_EOS    ! If true, density is calculated from T & S using an equation of state.

  use_EOS = associated(tv%eqn_of_state)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  !Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! Calculate the extra points that grid needs to be extended to, using the ANN window
  ! done as (ann_window-1)/2 as integer
  shift = (CS%ann_window-1)/2
  stencil_points = CS%ann_window * CS%ann_window

  uhTrANN(:,:,:) = 0.0
  vhTrANN(:,:,:) = 0.0

  slope_x(:,:,:) = 0.0
  slope_y(:,:,:) = 0.0
  
  Sfn_unlim_u(:,:,:) = 0.0
  Sfn_unlim_v(:,:,:) = 0.0

  nk_linear = max(GV%nkml, 1)
  !write(*,*) "nk_linear", nk_linear

  ! Allocate the local stencil variables
  allocate(dhdx_local(CS%ann_window, CS%ann_window), dhdy_local(CS%ann_window, CS%ann_window), &
           dudx_local(CS%ann_window, CS%ann_window), dudy_local(CS%ann_window, CS%ann_window), &
           dvdx_local(CS%ann_window, CS%ann_window), dvdy_local(CS%ann_window, CS%ann_window))

  if (CS%thickness_ann_model_type == "GM_ann" .or. CS%thickness_ann_model_type == "GM_rotated_ann") then
    Nin = 2
  else if (CS%thickness_ann_model_type == "nondim_ann") then
    Nin = 6
  endif

  allocate(x(stencil_points*Nin), y(Nout), y_rot(Nout))

  !> Calculates the h and u gradients in full 3D domain
  call vel_gradients(u, v, G, GV, dudx, dudy, dvdx, dvdy, CS)

  ! Calculate the h gradients in full 3D domain
  if (use_EOS) then
    call h_gradients_EOS(h, G, GV, US, dhdx, dhdy, CS, tv, dt, slope_x, slope_y)
  else
    call h_gradients(h, G, GV, dhdx, dhdy, CS)
  endif

  if (CS%decompose_h_gradients) then
    call calculate_h_mask(h, h_mask, G, GV, CS)
    call decompose_h_gradients(dhdx, dhdy, dhbardx, dhbardy, h_mask, G, GV, CS)
  endif
  

  tol_vel_grad = 1.0e-30 * US%T_to_s
  tol_h_grad = 1.0e-30 * US%Z_to_m / US%L_to_m
  
  !do k=1, nz
  do k=1, nz
   !> Rotation, local normalize etc
   !write(*,*) "k", k, shift, stencil_points
    !> Calculate the fluxes at center points 
    do j=js-1,je+1 ; do i=is-1,ie+1

      if (CS%thickness_ann_model_type == "GM_ann") then
        ! To test code with a simple GM function incorporated as ANN
        x(1) = dhdx(i,j,k)
        x(2) = dhdy(i,j,k)
        call ann(x, y, CS%ann_cs)

      else if (CS%thickness_ann_model_type == "NGM_non_ann") then
        ! To test code with a simple NGM function incorporated in the ANN module
        ! but would have to be non-ann since the function is not ANN 
        y(1) = dudx(i,j,k)*dhdx(i,j,k) + dudy(i,j,k)*dhdy(i,j,k) 
        y(2) = dvdx(i,j,k)*dhdx(i,j,k) + dvdy(i,j,k)*dhdy(i,j,k)

        write(*,*) "NGM coefficient has not been properly set."
        NGM_C = 1.0
        y(:) = y(:) * NGM_C * G%areaT(i,j)
     
      else if (CS%thickness_ann_model_type == "GM_rotated_ann") then 
        ! To test code with a simple GM function incorporated in rotated form
        dhdx_local(:,:) = dhdx(i-shift:i+shift,j-shift:j+shift,k)
        dhdy_local(:,:) = dhdy(i-shift:i+shift,j-shift:j+shift,k)

        call rotate_all_inputs(CS%ann_window, dhdx_local, dhdy_local)

        ! Compute the magnitude of the velocity gradient tensor for the local stencil
        
        h_grad_mag = 0.0
        do jj=1, CS%ann_window
          do ii=1, CS%ann_window
            h_grad_mag = h_grad_mag + dhdx_local(ii,jj)**2 + dhdy_local(ii,jj)**2
          enddo
        enddo
        h_grad_mag = sqrt(h_grad_mag) + tol_h_grad
        !write(*,*) "h_grad_mag", h_grad_mag, "at i,j,k", i, j, k
        
        dhdx_local(:,:) = dhdx_local(:,:) / h_grad_mag
        dhdy_local(:,:) = dhdy_local(:,:) / h_grad_mag

        x(1:stencil_points)                    = RESHAPE(dhdx_local, (/stencil_points/))
        x(stencil_points+1:2*stencil_points)   = RESHAPE(dhdy_local, (/stencil_points/))

        call ann(x, y_rot, CS%ann_cs)

        call rotate_outputs(dhdx(i,j,k), dhdy(i,j,k), y_rot, y)

        y(:) = y(:) * h_grad_mag
      

      else if (CS%thickness_ann_model_type == "nondim_ann") then
        ! To test the code with non-dim ANN
        ! Start : Code to work with specific ANN 
        
        !write(*,*) "i,j,k", i, j, k
        !write(*,*) "shift", shift

        ! Get the data on the local stencil
        dhdx_local(:,:) = dhdx(i-shift:i+shift,j-shift:j+shift,k)
        dhdy_local(:,:) = dhdy(i-shift:i+shift,j-shift:j+shift,k)
        dudx_local(:,:) = dudx(i-shift:i+shift,j-shift:j+shift,k)
        dudy_local(:,:) = dudy(i-shift:i+shift,j-shift:j+shift,k)
        dvdx_local(:,:) = dvdx(i-shift:i+shift,j-shift:j+shift,k)
        dvdy_local(:,:) = dvdy(i-shift:i+shift,j-shift:j+shift,k)

        !if (i == 10 .and. j == 10 .and. k == 1) then
        !  write(*,*) 'dudx', dudx(i,j)
        !  write(*,*) 'dudx_reshaped', RESHAPE(dudx_local, (/stencil_points/))
        !endif
        ! Rotation to grad h coordinates (always around center point)
        call rotate_all_inputs(CS%ann_window, dhdx_local, dhdy_local, dudx_local, dudy_local, dvdx_local, dvdy_local)

        ! Compute the magnitude of the velocity gradient tensor for the local stencil
        
        h_grad_mag = 0.0
        vel_grad_mag = 0.0
        do jj=1, CS%ann_window
          do ii=1, CS%ann_window
            h_grad_mag = h_grad_mag + dhdx_local(ii,jj)**2 + dhdy_local(ii,jj)**2
            vel_grad_mag = vel_grad_mag + dudx_local(ii,jj)**2 + dvdx_local(ii,jj)**2 + dudy_local(ii,jj)**2 + dvdy_local(ii,jj)**2
          enddo
        enddo
        h_grad_mag = sqrt(h_grad_mag) + tol_h_grad
        vel_grad_mag = sqrt(vel_grad_mag) + tol_vel_grad

        !write(*,*) "h_grad_mag", h_grad_mag, "vel_grad_mag", vel_grad_mag
        ! Non-dim the local gradients
        
        dudx_local(:,:) = dudx_local(:,:) / vel_grad_mag
        dudy_local(:,:) = dudy_local(:,:) / vel_grad_mag
        dvdx_local(:,:) = dvdx_local(:,:) / vel_grad_mag
        dvdy_local(:,:) = dvdy_local(:,:) / vel_grad_mag
        
        dhdx_local(:,:) = dhdx_local(:,:) / h_grad_mag
        dhdy_local(:,:) = dhdy_local(:,:) / h_grad_mag

        ! General ANN with vel and h gradients as input, with stencils. 
        ! On 12 March 2025, the data was arranged as following in X 
        ! du/dx, dv/dx, du/dy, dv/dy, dh/dx, dh/dy
        ! for each of these variables the arrangement is: 
        ! Doing the tranposing here to ensure that the data arrangement is correct. 
        !write(*,*) "dudx shape", SHAPE(dudx_local),", dudx values: ", dudx_local
        x(1:stencil_points)                    = RESHAPE(TRANSPOSE(dudx_local), (/stencil_points/))
        x(stencil_points+1:2*stencil_points)   = RESHAPE(TRANSPOSE(dvdx_local), (/stencil_points/))
        x(2*stencil_points+1:3*stencil_points) = RESHAPE(TRANSPOSE(dudy_local), (/stencil_points/))
        x(3*stencil_points+1:4*stencil_points) = RESHAPE(TRANSPOSE(dvdy_local), (/stencil_points/))
        x(4*stencil_points+1:5*stencil_points) = RESHAPE(TRANSPOSE(dhdx_local), (/stencil_points/))
        x(5*stencil_points+1:6*stencil_points) = RESHAPE(TRANSPOSE(dhdy_local), (/stencil_points/))

        !write(*,*) "x dudx", x(1:stencil_points)

        call ann(x, y_rot, CS%ann_cs)    

        ! Rotate back
        call rotate_outputs(dhdx(i,j,k), dhdy(i,j,k), y_rot, y)
        
        ! Multiply by the norm factors.
        y(:) = y(:) * h_grad_mag * ( h(i,j,k) ** 2 / (h(i,j,k) + CS%h_min_mask)**2 ) * vel_grad_mag * G%areaT(i,j) * CS%FGR_ANN * CS%FGR_ANN
        ! End : Code to work with specific ANN
      else
        write(*,*) "thickness_ann_model_type not recognized"
        stop
      endif
      
      if (CS%h_mask_flux) then
        FxC(i,j,k) = y(1) * G%mask2dT(i,j) * h_mask(i,j,k)
        FyC(i,j,k) = y(2) * G%mask2dT(i,j) * h_mask(i,j,k)
      else
        FxC(i,j,k) = y(1) * G%mask2dT(i,j) 
        FyC(i,j,k) = y(2) * G%mask2dT(i,j) 
      end if

    enddo ; enddo
  enddo
  
  ! > vertically smooth the fluxes to remove edge effects
  ! > Sort of trying to do a hacky version of FGNV 2010
  ! with the  knowledge that smoothing fluxes is not the same
  ! as smoothing streamfunction, but it is better than nothing.
  if (CS%mask_boundary_layers) then
    ! Smooth boundary layer fluxes
    call smooth_fluxes_vertically(h, FxC, FyC, G, GV, CS, tv, US)
  end if


  do k=1, nz
   !> Interpolate fluxes to u, v points
    ! > We multiply the ann coeff at this later stage, so the unchanged F can be used for diagnostics
    do j=js,je ; do i=is-1,ie
      uhTrANN(I,j,k) = 0.5 * (FxC(i,j,k) + FxC(i+1,j,k)) * G%dyCu(I,j) * G%mask2dCu(I,j) * CS%ann_coeff
    enddo ; enddo
    do j=js-1,je ; do i=is,ie
      vhTrANN(i,J,k) = 0.5 * (FyC(i,j,k) + FyC(i,j+1,k)) * G%dxCv(i,J) * G%mask2dCv(i,J) * CS%ann_coeff
    enddo ; enddo 

  enddo

  

  ! Convert the fluxes to stream function 
  do k=nz,2,-1
    do j=js,je ; do i=is-1,ie
      Sfn_unlim_u(I,j,k) = Sfn_unlim_u(I,j,k+1) + uhTrANN(I,j,k) 
    enddo ; enddo
    do j=js-1,je ; do i=is,ie
      Sfn_unlim_v(i,J,k) = Sfn_unlim_v(i,J,k+1) + vhTrANN(i,J,k)
    enddo ; enddo
  enddo

  ! only for testing
  ! if (CS%thickness_ann_model_type == "GM_ann") then
  !   do k=nz,2,-1
  !     do j=js,je ; do i=is-1,ie
  !       Sfn_unlim_u(I,j,k) = -1000. *slope_x(I,j,k) * G%dyCu(I,j) * G%mask2dCu(I,j) !Sfn_unlim_u(I,j,k+1) + uhTrANN(I,j,k) 
  !     enddo ; enddo
  !     do j=js-1,je ; do i=is,ie
  !       Sfn_unlim_v(i,J,k) = -1000. *slope_y(i,J,k) * G%dxCv(i,J) * G%mask2dCv(i,J) !Sfn_unlim_v(i,J,k+1) + vhTrANN(i,J,k)
  !     enddo ; enddo
  !   enddo
  ! end if  


  ! Apply flux limiters
  if (CS%limit_upslope_flow) then
    call upslope_limiter(h, Sfn_unlim_u, Sfn_unlim_v, G, GV, US, CS, tv)
    !call upslope_limiter_old(h, uhTrANN, vhTrANN, G, GV, US, CS, tv)
  endif

  if (CS%id_sfn_unlim_x > 0) call post_data(CS%id_sfn_unlim_x, Sfn_unlim_u, CS%diag)
  if (CS%id_sfn_unlim_y > 0) call post_data(CS%id_sfn_unlim_y, Sfn_unlim_v, CS%diag)


   !> Apply the no- BT flow condition (for 2 layers)
  ! There are many ways to do this, and we made simplest seeming adhoc choice.
  if (CS%remove_barotropic_flux) then
    do k =1,nk_linear
      Sfn_unlim_u(:,:,k) = 0.0
      Sfn_unlim_v(:,:,k) = 0.0
    enddo
    !uhTrANN(:,:,1) = - uhTrANN(:,:,2) 
    !vhTrANN(:,:,1) = - vhTrANN(:,:,2)
  endif

  ! TODO : This is where to include the FGNV smoothing 

  ! Convert the stream function back to fluxes 
  ! (for testing some older code)
  ! do k=nz,1,-1
  !   do j=js,je ; do I=is-1,ie
  !     uhTrANN(I,j,k) = Sfn_unlim_u(I,j,k) - Sfn_unlim_u(I,j,k+1)
  !     !Sfn_unlim_u(I,j,k) = Sfn_unlim_u(I,j,k+1) + uhTrANN(I,j,k) 
  !   enddo ; enddo
  !   do J=js-1,je ; do i=is,ie
  !     vhTrANN(i,J,k) = Sfn_unlim_v(i,J,k) - Sfn_unlim_v(i,J,k+1)
  !     !Sfn_unlim_v(i,J,k) = Sfn_unlim_v(i,J,k+1) + vhTrANN(i,J,k)
  !   enddo ; enddo
  ! enddo


  ! Compute transport from flux limited streamfunction and 
  !> Apply the squeeze limiter to the streamfunction

  ! Seems like a good idea to call this unconditionally.
  !call squeeze_limiter_old(h, uhTrANN, vhTrANN, dt, G, Gv, CS)
  call squeeze_limiter(h, uhTrANN, vhTrANN, Sfn_unlim_u, Sfn_unlim_v, slope_x, slope_y, dt, G, Gv, CS, tv)

  
  !if (CS%id_dhdx > 0) write(*,*) 'Here', dhdx(2,2,2)
  if (CS%id_dhdx > 0) call post_data(CS%id_dhdx, dhdx, CS%diag)
  if (CS%id_dhdy > 0) call post_data(CS%id_dhdy, dhdy, CS%diag)
  
  if (CS%id_slope_x > 0) call post_data(CS%id_slope_x, slope_x, CS%diag)
  if (CS%id_slope_y > 0) call post_data(CS%id_slope_y, slope_y, CS%diag)

  if (CS%id_dhbardx > 0 .and. CS%decompose_h_gradients) call post_data(CS%id_dhbardx, dhbardx, CS%diag)
  if (CS%id_dhbardy > 0 .and. CS%decompose_h_gradients) call post_data(CS%id_dhbardy, dhbardy, CS%diag)
  if (CS%id_h_mask > 0 .and. CS%decompose_h_gradients) call post_data(CS%id_h_mask, h_mask, CS%diag)

  if (CS%id_Fx > 0) call post_data(CS%id_Fx, FxC, CS%diag)
  if (CS%id_Fy > 0) call post_data(CS%id_Fy, FyC, CS%diag)
  if (CS%id_uhTrANN > 0) call post_data(CS%id_uhTrANN, uhTrANN, CS%diag)
  if (CS%id_vhTrANN > 0) call post_data(CS%id_vhTrANN, vhTrANN, CS%diag)

  if (CS%id_dudx > 0) call post_data(CS%id_dudx, dudx, CS%diag)
  if (CS%id_dudy > 0) call post_data(CS%id_dudy, dudy, CS%diag)
  if (CS%id_dvdx > 0) call post_data(CS%id_dvdx, dvdx, CS%diag)
  if (CS%id_dvdy > 0) call post_data(CS%id_dvdy, dvdy, CS%diag)



end subroutine thickness_flux_ann_full

! > Rotate the outputs back to the original frame.
subroutine rotate_outputs(dhdx, dhdy, Fvec_rot, Fvec_xy)
  real, intent(in) :: dhdx, dhdy
  real, dimension(2), intent(in) :: Fvec_rot
  real, dimension(2), intent(out) :: Fvec_xy

  real :: frame_vec_x, frame_vec_y
  real :: mag_frame_vec
  real :: T_hat_i, T_hat_j
  real :: N_hat_i, N_hat_j
  real :: R_11, R_12, R_21, R_22 ! Rotation matrix

  frame_vec_x = dhdx
  frame_vec_y = dhdy

  call calc_rotation_matrix(frame_vec_x, frame_vec_y, R_11, R_12, R_21, R_22)

  ! note that here we pass in R_transpose, since the function will multiply the vector by (R_transpose)_transpose = R
  call rotate_vector(R_11, R_21, R_12, R_22, Fvec_rot(1), Fvec_rot(2), Fvec_xy(1), Fvec_xy(2))

end subroutine rotate_outputs

! > Rotates the inputs into the flow dependent coordiante frame. 
subroutine rotate_all_inputs(stencil_width, dhdx, dhdy, dudx, dudy, dvdx, dvdy)
  integer, intent(in) :: stencil_width
  real, dimension(stencil_width, stencil_width), intent(inout) :: dhdx, dhdy
  real, dimension(stencil_width, stencil_width), optional, intent(inout) :: dudx, dudy, dvdx, dvdy
  real, dimension(stencil_width, stencil_width) :: dhdx_rot, dhdy_rot, dudx_rot, dudy_rot, dvdx_rot, dvdy_rot
  integer :: mid_point
  real :: frame_vec_x, frame_vec_y
  real :: mag_frame_vec
  real :: T_hat_i, T_hat_j
  real :: N_hat_i, N_hat_j
  real :: R_11, R_12, R_21, R_22 ! Rotation matrix
  integer :: i, j

  mid_point = (stencil_width-1)/2 + 1

  frame_vec_x = dhdx(mid_point, mid_point)
  frame_vec_y = dhdy(mid_point, mid_point)

  call calc_rotation_matrix(frame_vec_x, frame_vec_y, R_11, R_12, R_21, R_22)

  ! Rotate the gradients
  do j=1, stencil_width ; do i=1, stencil_width
    call rotate_vector(R_11, R_12, R_21, R_22, dhdx(i,j), dhdy(i,j), dhdx_rot(i,j), dhdy_rot(i,j))
    if (present(dudx)) then
      call rotate_tensor(R_11, R_12, R_21, R_22, dudx(i,j), dudy(i,j), dvdx(i,j), dvdy(i,j), dudx_rot(i,j), dudy_rot(i,j), dvdx_rot(i,j), dvdy_rot(i,j))
    endif
  enddo ; enddo

  ! Copy the rotated gradients back
  dhdx(:,:) = dhdx_rot(:,:)
  dhdy(:,:) = dhdy_rot(:,:)
  if (present(dudx)) then
    dudx(:,:) = dudx_rot(:,:)
    dudy(:,:) = dudy_rot(:,:)
    dvdx(:,:) = dvdx_rot(:,:)
    dvdy(:,:) = dvdy_rot(:,:)
  endif

end subroutine rotate_all_inputs

! Takes as input R components and a vector (v) and rotates it, as v_rot = R_transpose*v
subroutine rotate_vector(R_11, R_12, R_21, R_22, V1, V2, Vrot1, Vrot2)
  real, intent(in) :: R_11, R_12, R_21, R_22 ! Rotation matrix
  real, intent(in) :: V1, V2 ! Vector components to be rotated
  real, intent(out) :: Vrot1, Vrot2 ! Rotated vector components

  Vrot1 = R_11 * V1 + R_21 * V2
  Vrot2 = R_12 * V1 + R_22 * V2
  
end subroutine rotate_vector

subroutine rotate_tensor(R_11, R_12, R_21, R_22, T11, T12, T21, T22, T11_rot, T12_rot, T21_rot, T22_rot)
  real, intent(in) :: R_11, R_12, R_21, R_22 ! Rotation matrix
  real, intent(in) :: T11, T12, T21, T22 ! Tensor components
  real, intent(out) :: T11_rot, T12_rot, T21_rot, T22_rot ! Rotated tensor components

  real :: C11, C12, C21, C22

  ! C = R_transpose (T)
  call two_by_two_matrix_multiplication(R_11, R_21, R_12, R_22, T11, T12, T21, T22, C11, C12, C21, C22)

  ! T_rot = C * R
  call two_by_two_matrix_multiplication(C11, C12, C21, C22, R_11, R_12, R_21, R_22, T11_rot, T12_rot, T21_rot, T22_rot)
  
end subroutine rotate_tensor

subroutine two_by_two_matrix_multiplication(A_11, A_12, A_21, A_22, B_11, B_12, B_21, B_22, C_11, C_12, C_21, C_22)
  real, intent(in) :: A_11, A_12, A_21, A_22
  real, intent(in) :: B_11, B_12, B_21, B_22
  real, intent(out) :: C_11, C_12, C_21, C_22

  C_11 = A_11 * B_11 + A_12 * B_21
  C_12 = A_11 * B_12 + A_12 * B_22
  C_21 = A_21 * B_11 + A_22 * B_21
  C_22 = A_21 * B_12 + A_22 * B_22

end subroutine two_by_two_matrix_multiplication
  
subroutine calc_rotation_matrix(frame_vec_x, frame_vec_y, R_11, R_12, R_21, R_22)
  real, intent(in) :: frame_vec_x, frame_vec_y
  real, intent(out) :: R_11, R_12, R_21, R_22

  real :: mag_frame_vec
  real :: T_hat_i, T_hat_j
  real :: N_hat_i, N_hat_j

  mag_frame_vec = sqrt(frame_vec_x**2 + frame_vec_y**2) + 1.0e-30

  T_hat_i = frame_vec_x / mag_frame_vec
  T_hat_j = frame_vec_y / mag_frame_vec

  N_hat_i = -T_hat_j
  N_hat_j = T_hat_i

  R_11 = T_hat_i
  R_12 = N_hat_i
  R_21 = T_hat_j
  R_22 = N_hat_j

end subroutine calc_rotation_matrix

!> Smooth fluxes to remove edge effects.
subroutine smooth_fluxes_vertically(h, FxC, FyC, G, GV, CS, tv, US)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(thermo_var_ptrs),                      intent(in)    :: tv    !< Thermodynamics structure
  type(unit_scale_type),                      intent(in)    :: US     !< A dimensional unit scaling type
  type(thickness_flux_ann_CS),                intent(in)    :: CS !< Control structure for thickness_flux_ann
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(inout) :: FxC
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(inout) :: FyC
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)    :: h      !< Layer thickness [H ~> m or kg m-2]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)     :: e    ! The interface heights relative to mean sea level [Z ~> m]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV))     :: mask_diag   ! The interface heights relative to mean sea level [Z ~> m]
  real :: z, mask
  real :: D
  real :: delta_BL_top, delta_BL_bottom, delta_ml, s
  integer :: is, ie, js, je, nk_linear
  !integer :: Isq, Ieq, Jsq, Jeq
  integer :: nz
  integer :: i, j, k

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  call find_eta(h, tv, G, GV, US, e, halo_size=3)

  nk_linear = max(GV%nkml, 1) ! what is this variable?

  s = 50.0 
  delta_BL_top =  CS%delta_BL_top + s
  delta_BL_bottom = CS%delta_BL_bottom + s 
   
  do k=1,nz
    do j=js-1,je+1 ; do i=is-1,ie+1
      D = e(i,j,1) - e(i,j,nz+1) ! Mean sea level height, positive number
      
      ! If we want to use boundary layer depth
      !delta_BL_top = e(i,j,nk_linear)
      !delta_ml = e(i,j,1) - e(i,j,nk_linear + 1) + s  ! positive number, thickness of the mixed layer
      delta_ml = delta_BL_top

      z = 0.5*(e(i,j,k) + e(i,j,k+1)) - e(i,j,nz+1)  ! distance above the bottom interface, positive number

      if (D > ( delta_ml + delta_BL_bottom)) then
        mask = 0.5 * (tanh((z - delta_BL_bottom)/s) - tanh((z - ( D - delta_ml))/s))
      else
        mask = 0.0  ! completely mask very thin water columns
      end if

      !mask(i,j,k) = 0.5 * (tanh((z(i,j,k) - delta)/s) - tanh((z(i,j,k) - (H(i,j) - delta))/s))
      FxC(i,j,k) = FxC(i,j,k) * mask
      FyC(i,j,k) = FyC(i,j,k) * mask

      mask_diag(i,j,k) = mask
    enddo ; enddo
  enddo ! end k loop

  if (CS%id_boundary_mask > 0) call post_data(CS%id_boundary_mask, mask_diag, CS%diag)

end subroutine smooth_fluxes_vertically

!> Calculate thickness gradients when EOS is used.
subroutine h_gradients_EOS(h, G, GV, US, dhdx, dhdy, CS, tv, dt, slope_x, slope_y)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(unit_scale_type),                      intent(in)    :: US     !< A dimensional unit scaling type
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)    :: h      !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(out)   :: dhdx, dhdy      !< components of the h gradients on the i,j points (cell-center)
  type(thickness_flux_ann_CS),                intent(in)    :: CS !< Control structure for thickness_flux_ann
  type(thermo_var_ptrs),                      intent(in)    :: tv    !< Thermodynamics structure
  real,                                       intent(in)    :: dt     !< Time increment [T ~> s]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(out)    :: slope_x !< Zonal isoneutral slope [Z L-1 ~> nondim]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(out)    :: slope_y !< Meridional isoneutral slope [Z L-1 ~> nondim]
  
  ! Local variables
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)) :: dhdx_u
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)) :: dhdy_v
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)     :: e    ! The interface heights relative to mean sea level [Z ~> m]
   
  logical :: use_stanley
  integer :: is, ie, js, je
  !integer :: Isq, Ieq, Jsq, Jeq
  integer :: nz
  integer :: i, j, k
  integer :: shift

  use_stanley = .false.
  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  call find_eta(h, tv, G, GV, US, e, halo_size=3)
  call calc_isoneutral_slopes(G, GV, US, h, e, tv, dt*CS%kappa_smooth, use_stanley, &
                                  slope_x, slope_y, halo=2)
  ! Calculate the extra points that grid needs to be extended to, using the ANN window
  ! done as (ann_window-1)/2 as integer
  shift = (CS%ann_window-1)/2

  do k=1, nz
    ! Calculate the x-gradients at u points
    ! I don't follow the MOM6 soft convention for loops (as it seemed a bit confusing with these shifts)
    do j=js-shift-1, je+shift+1 ; do i=is-shift-2, ie+shift+1 ! extra points needed in the x direction since we interpolate to center
      dhdx_u(I,j,k) =  (slope_x(i,j,k) - slope_x(i,j,k+1)) * G%mask2dCu(I,j)
    enddo ; enddo
    ! Calculate the y-gradients at v points
    do j=js-shift-2, je+shift+1 ; do i=is-shift-1, ie+shift+1 ! extra points needed in the y direction since we interpolate to center
      dhdy_v(i,J,k) =  (slope_y(i,j,k) - slope_y(i,j,k+1)) * G%mask2dCv(i,J)
    enddo ; enddo
    ! Interpolate the gradients to the center points
    ! We need these at +/- shift points because that is the local domain that the ANN will use.
    do j=js-shift-1, je+shift+1 ; do i=is-shift-1, ie+shift+1
      dhdx(i,j,k) = 0.5 * (dhdx_u(I,j,k) + dhdx_u(I-1,j,k)) * G%mask2dT(i,j)
      dhdy(i,j,k) = 0.5 * (dhdy_v(i,J,k) + dhdy_v(i,J-1,k)) * G%mask2dT(i,j)
    enddo ; enddo
  enddo ! end k loop                                

end subroutine h_gradients_EOS

!> Calculates the thickness gradients in each layer at the center points in 3D. 
subroutine h_gradients(h, G, GV, dhdx, dhdy, CS)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in) :: h      !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(out) :: dhdx, dhdy      !< components of the h gradients on the i,j points (cell-center)
  type(thickness_flux_ann_CS), intent(in) :: CS !< Control structure for thickness_flux_ann

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)) :: dhdx_u
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)) :: dhdy_v 
  integer :: is, ie, js, je
  !integer :: Isq, Ieq, Jsq, Jeq
  integer :: nz
  integer :: i, j, k
  integer :: shift

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  !Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! Calculate the extra points that grid needs to be extended to, using the ANN window
  ! done as (ann_window-1)/2 as integer
  shift = (CS%ann_window-1)/2

  do k=1, nz
    ! Calculate the x-gradients at u points
    ! I don't follow the MOM6 soft convention for loops (as it seemed a bit confusing with these shifts)
    do j=js-shift-1, je+shift+1 ; do i=is-shift-2, ie+shift+1 ! extra points needed in the x direction since we interpolate to center
      dhdx_u(I,j,k) = G%IdxCu(i,j) * (h(i+1,j,k) - h(i,j,k)) * G%mask2dCu(I,j)
    enddo ; enddo
    ! Calculate the y-gradients at v points
    do j=js-shift-2, je+shift+1 ; do i=is-shift-1, ie+shift+1 ! extra points needed in the y direction since we interpolate to center
      dhdy_v(i,J,k) = G%IdyCv(i,J) * (h(i,j+1,k) - h(i,j,k)) * G%mask2dCv(i,J)
    enddo ; enddo
    ! Interpolate the gradients to the center points
    ! We need these at +/- shift points because that is the local domain that the ANN will use.
    do j=js-shift-1, je+shift+1 ; do i=is-shift-1, ie+shift+1
      dhdx(i,j,k) = 0.5 * (dhdx_u(I,j,k) + dhdx_u(I-1,j,k)) * G%mask2dT(i,j)
      dhdy(i,j,k) = 0.5 * (dhdy_v(i,J,k) + dhdy_v(i,J-1,k)) * G%mask2dT(i,j)
    enddo ; enddo
  enddo ! end k loop

end subroutine h_gradients

!> Decompose the h gradient into a steady and transient part.
!> The steady part is the part that is due to topography. 
subroutine decompose_h_gradients(dhdx, dhdy, dhbardx, dhbardy, h_mask, G, GV, CS)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(inout)    :: dhdx, dhdy !< components of the h gradients on the i,j points (cell-center)
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(out)   :: dhbardx, dhbardy !< components of the h gradients on the i,j points (cell-center)
  type(thickness_flux_ann_CS), intent(in) :: CS !< Control structure for thickness_flux_ann
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in) :: h_mask !< Mask based on the thickness

  integer :: i, j, k, is, ie, js, je, nz, shift

  real :: e_bottom(SZI_(G),SZJ_(G))
  real :: de_bottomdx_u(SZIB_(G),SZJ_(G)), de_bottomdy_v(SZI_(G),SZJB_(G))
  real :: de_bottomdx(SZI_(G),SZJ_(G)), de_bottomdy(SZI_(G),SZJ_(G))

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  shift = (CS%ann_window-1)/2

  do j=js-shift-2, je+shift+2 ; do i=is-shift-2, ie+shift+2
    e_bottom(i,j) = - G%bathyT(i,j)
  enddo ; enddo

  ! Calculate the x-gradients at u points
  do j=js-shift-1, je+shift+1 ; do i=is-shift-2, ie+shift+1 ! extra points needed in the x direction since we interpolate to center
    de_bottomdx_u(I,j) = G%IdxCu(i,j) * (e_bottom(i+1,j) - e_bottom(i,j)) * G%mask2dCu(I,j)
  enddo ; enddo
  ! Calculate the y-gradients at v points
  do j=js-shift-2, je+shift+1 ; do i=is-shift-1, ie+shift+1 ! extra points needed in the y direction since we interpolate to center
    de_bottomdy_v(i,J) = G%IdyCv(i,J) * (e_bottom(i,j+1) - e_bottom(i,j)) * G%mask2dCv(i,J)
  enddo ; enddo

  ! Interpolate the gradients to the center points
  ! We need these at +/- shift points because that is the local domain that the ANN will use.
  do j=js-shift-1, je+shift+1 ; do i=is-shift-1, ie+shift+1
    de_bottomdx(i,j) = 0.5 * (de_bottomdx_u(I,j) + de_bottomdx_u(I-1,j)) * G%mask2dT(i,j)
    de_bottomdy(i,j) = 0.5 * (de_bottomdy_v(i,J) + de_bottomdy_v(i,J-1)) * G%mask2dT(i,j)
  enddo ; enddo

  ! For bottom layer
  do j=js-shift-1, je+shift+1 ; do i=is-shift-1, ie+shift+1
    dhbardx(i,j,nz) =  - de_bottomdx(i,j) * h_mask(i,j,nz)
    dhbardy(i,j,nz) =  - de_bottomdy(i,j) * h_mask(i,j,nz)
  enddo ; enddo
  ! all other layers
  do k = 1, nz-1 ! 
    do j=js-shift-1, je+shift+1 ; do i=is-shift-1, ie+shift+1
      dhbardx(i,j,k) =  - de_bottomdx(i,j) * (1 - h_mask(i,j,k+1)) * h_mask(i,j,k)
      dhbardy(i,j,k) =  - de_bottomdy(i,j) * (1 - h_mask(i,j,k+1)) * h_mask(i,j,k)
    enddo ; enddo
  enddo

  dhdx(:,:,:) = dhdx(:,:,:) - dhbardx(:,:,:)
  dhdy(:,:,:) = dhdy(:,:,:) - dhbardy(:,:,:)

end subroutine decompose_h_gradients

!> Calculate a thickness based mask. 
subroutine calculate_h_mask(h, h_mask, G, GV, CS)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)   :: h      !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(out) :: h_mask !< Mask based on the thickness
  type(thickness_flux_ann_CS), intent(in) :: CS !< Control structure for thickness_flux_ann

  integer :: i, j, k, is, ie, js, je, nz, shift
  !real :: h_min

  shift = (CS%ann_window-1)/2

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  !h_min = 1.0 
  ! This is not a universal choice. 
  ! We should have a better way to set this.
  ! An example of better way may be to set this based on the thickness of the other layers in the column.

  do k=1, nz
    do j=js-shift-1, je+shift+1 ; do i=is-shift-1, ie+shift+1 ! We want these to run over same values as places where we compute fluxes. 
      !h_mask(i,j,k) = 1.0
      !if (h(i,j,k) < CS%h_min_mask) h_mask(i,j,k) = 0.0
      h_mask(i,j,k) = 0.5 * (1 + tanh((h(i,j,k) - CS%h_min_mask) / CS%h_mask_width))
    enddo ; enddo
  enddo

end subroutine calculate_h_mask

!> Calculates the velocity gradients at the center points in 3D.
subroutine vel_gradients(u, v, G, GV, dudx, dudy, dvdx, dvdy, CS)
  type(ocean_grid_type),                     intent(in)    :: G   !< Ocean grid structure
  type(verticalGrid_type),                   intent(in)    :: GV  !< The ocean's vertical grid structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)),intent(in)    :: u   !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)),intent(in)    :: v   !< The meridional velocity [L T-1 ~> m s-1].
  ! Center points 
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(out) :: dudx, dvdy, dudy, dvdx   ! components of the velocity gradient tensor on the i,j points (cell-center)
  type(thickness_flux_ann_CS), intent(in) :: CS !< Control structure for thickness_flux_ann
      
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

!> Apply upslope limiter to the ANN fluxes
!> This ensures that the ANN fluxes don't start to increase the APE of the flow near grounded isopycnals.
!> The physicality of this is arguable, but if not used leads to numerical issues.
!> Some energetics based schemes may help remove this strict condition in future. 
!> Tendencies from the resolved part of the flow may still be able to move isopycnals relative to topo (as they should).
subroutine upslope_limiter(h, Sfn_unlim_u, Sfn_unlim_v, G, Gv, US, CS, tv) 
  type(ocean_grid_type), intent(in) :: G
  type(verticalGrid_type), intent(in) :: GV
  type(unit_scale_type), intent(in) :: US
  type(thermo_var_ptrs),                      intent(in)    :: tv     !< Thermodynamics structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(in) :: h !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(inout)   :: Sfn_unlim_u
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(inout)   :: Sfn_unlim_v
  type(thickness_flux_ann_CS), intent(in) :: CS !< Control structure for thickness_flux_ann

  real :: e(SZI_(G),SZJ_(G),SZK_(GV)+1) ! heights of interfaces, relative to mean
                                         ! sea level [Z ~> m], positive up.
  integer :: i, j, k, is, ie, js, je, nz
  real :: dz_neglect

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  dz_neglect = GV%dZ_subroundoff

  ! Calculates interface heights, e, in [Z ~> m].
  call find_eta(h, tv, G, GV, US, e, halo_size=1)

  do k=1, nz
    
    do j=js, je ; do I=is-1, ie
      if (Sfn_unlim_u(I,j,k) > 0.0) then ! if flow to the east
        if (e(i,j,K) < e(i+1,j,nz+1)) then ! if fully blocked by topography to the east
          Sfn_unlim_u(I,j,k) = 0.0
        elseif (e(i,j,K+1) < e(i+1,j,nz+1)) then! if partially blocked by topography to the east
          Sfn_unlim_u(I,j,k) = Sfn_unlim_u(I,j,k) * ((e(i,j,K) - e(i+1,j,nz+1)) / (h(i,j,k) + dz_neglect))
        endif 
      else ! if flow to the west
        if (e(i,j,nz+1) > e(i+1,j,K)) then ! if fully blocked by topography to the west
          Sfn_unlim_u(I,j,k) = 0.0
        elseif (e(i,j,nz+1) > e(i+1,j,K+1) ) then ! if partially blocked by topography to the west
          Sfn_unlim_u(I,j,k) = Sfn_unlim_u(I,j,k) * ((e(i+1,j,K) - e(i,j,nz+1)) / (h(i+1,j,k) + dz_neglect))
        endif
      endif
    enddo ; enddo

    do J=js-1, je ; do i=is, ie
      if (Sfn_unlim_v(i,J,k) > 0.0) then ! if flow to the north
        if (e(i,j,K) < e(i,j+1,nz+1)) then ! if fully blocked by topography to the north
          Sfn_unlim_v(i,J,k) = 0.0
        elseif (e(i,j,K+1) < e(i,j+1,nz+1)) then ! if partially blocked by topography to the north
          Sfn_unlim_v(i,J,k) = Sfn_unlim_v(i,J,k) * ((e(i,j,K) - e(i,j+1,nz+1)) / (h(i,j,k) + dz_neglect))
        endif
      else ! if flow to the south
        if (e(i,j,nz+1) > e(i,j+1,K)) then ! if fully blocked by topography to the south
          Sfn_unlim_v(i,J,k) = 0.0
        elseif (e(i,j,nz+1) > e(i,j+1,K+1)) then ! if partially blocked by topography to the south
          Sfn_unlim_v(i,J,k) = Sfn_unlim_v(i,J,k) * ((e(i,j+1,K) - e(i,j,nz+1)) / (h(i,j+1,k) + dz_neglect))
        endif
      endif
    enddo ; enddo

  enddo
    
end subroutine upslope_limiter


subroutine upslope_limiter_old(h, uhTrANN, vhTrANN, G, Gv, US, CS, tv) 
  type(ocean_grid_type), intent(in) :: G
  type(verticalGrid_type), intent(in) :: GV
  type(unit_scale_type), intent(in) :: US
  type(thermo_var_ptrs),                      intent(in)    :: tv     !< Thermodynamics structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(in) :: h !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(inout)   :: uhTrANN   !< Zonal ANN h transport u*h*dy [L2 H T-1 ~> m3 s-1 or kg s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(inout)   :: vhTrANN   !< Meridional ANN h transport v*h*dx [L2 H T-1 ~> m3 s-1 or kg s-1]
  type(thickness_flux_ann_CS), intent(in) :: CS !< Control structure for thickness_flux_ann

  real :: e(SZI_(G),SZJ_(G),SZK_(GV)+1) ! heights of interfaces, relative to mean
                                         ! sea level [Z ~> m], positive up.
  integer :: i, j, k, is, ie, js, je, nz
  real :: dz_neglect

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  dz_neglect = GV%dZ_subroundoff

  ! Calculates interface heights, e, in [Z ~> m].
  call find_eta(h, tv, G, GV, US, e, halo_size=1)

  do k=1, nz
    
    do j=js, je ; do I=is-1, ie
      if (uhTrANN(I,j,k) > 0.0) then ! if flow to the east
        if (e(i,j,K) < e(i+1,j,nz+1)) then ! if fully blocked by topography to the east
          uhTrANN(I,j,k) = 0.0
        elseif (e(i,j,K+1) < e(i+1,j,nz+1)) then! if partially blocked by topography to the east
          uhTrANN(I,j,k) = uhTrANN(I,j,k) * ((e(i,j,K) - e(i+1,j,nz+1)) / (h(i,j,k) + dz_neglect))
        endif 
      else ! if flow to the west
        if (e(i,j,nz+1) > e(i+1,j,K)) then ! if fully blocked by topography to the west
          uhTrANN(I,j,k) = 0.0
        elseif (e(i,j,nz+1) > e(i+1,j,K+1) ) then ! if partially blocked by topography to the west
          uhTrANN(I,j,k) = uhTrANN(I,j,k) * ((e(i+1,j,K) - e(i,j,nz+1)) / (h(i+1,j,k) + dz_neglect))
        endif
      endif
    enddo ; enddo

    do J=js-1, je ; do i=is, ie
      if (vhTrANN(i,J,k) > 0.0) then ! if flow to the north
        if (e(i,j,K) < e(i,j+1,nz+1)) then ! if fully blocked by topography to the north
          vhTrANN(i,J,k) = 0.0
        elseif (e(i,j,K+1) < e(i,j+1,nz+1)) then ! if partially blocked by topography to the north
          vhTrANN(i,J,k) = vhTrANN(i,J,k) * ((e(i,j,K) - e(i,j+1,nz+1)) / (h(i,j,k) + dz_neglect))
        endif
      else ! if flow to the south
        if (e(i,j,nz+1) > e(i,j+1,K)) then ! if fully blocked by topography to the south
          vhTrANN(i,J,k) = 0.0
        elseif (e(i,j,nz+1) > e(i,j+1,K+1)) then ! if partially blocked by topography to the south
          vhTrANN(i,J,k) = vhTrANN(i,J,k) * ((e(i,j+1,K) - e(i,j,nz+1)) / (h(i,j+1,k) + dz_neglect))
        endif
      endif
    enddo ; enddo

  enddo
    

end subroutine upslope_limiter_old

!> Apply a flux limiter that prevents ANN fluxes from making isopycnal thickness negative
!> 
subroutine squeeze_limiter(h, uhTrANN, vhTrANN, Sfn_unlim_u, Sfn_unlim_v, slope_x, slope_y, dt, G, Gv, CS, tv)
  type(ocean_grid_type), intent(in) :: G
  type(verticalGrid_type), intent(in) :: GV
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(in) :: h !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(inout)   :: uhTrANN   !< Zonal ANN h transport u*h*dy [L2 H T-1 ~> m3 s-1 or kg s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(inout)   :: vhTrANN   !< Meridional ANN h transport v*h*dx [L2 H T-1 ~> m3 s-1 or kg s-1]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(in)   :: Sfn_unlim_u, slope_x
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(in)   :: Sfn_unlim_v, slope_y
  type(thickness_flux_ann_CS), intent(in) :: CS !< Control structure for thickness_flux_ann
  type(thermo_var_ptrs),                      intent(in)    :: tv
  real, intent(in) :: dt !< Time step [T ~> s]

  !local variables
  real :: slope2_Ratio_u(SZIB_(G),SZJ_(G), SZK_(GV)+1) ! The ratio of the slope squared to slope_max squared [nondim]
  real :: slope2_Ratio_v(SZI_(G),SZJB_(G),SZK_(GV)+1)  ! The ratio of the slope squared to slope_max squared [nondim]
  real :: uhtot(SZIB_(G),SZJ_(G))  ! The vertical sum of uhD [H L2 T-1 ~> m3 s-1 or kg s-1].
  real :: vhtot(SZI_(G),SZJB_(G))  ! The vertical sum of vhD [H L2 T-1 ~> m3 s-1 or kg s-1].
  integer :: i, j, k, is, ie, js, je, nz, nk_linear
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) ::  &
    h_avail, &    ! The mass available for transport out of each face, divided
                                                              ! by dt [H L2 T-1 ~> m3 s-1 or kg s-1].
    h_frac        ! The fraction of the mass in the column above the bottom
                                                              ! interface of a layer that is within a layer [nondim]. 0<h_frac<=1
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1) :: &
    h_avail_rsum  ! The running sum of h_avail above an interface [H L2 T-1 ~> m3 s-1 or kg s-1].
  real :: I_slope_max2  ! The inverse of slope_max squared [L2 Z-2 ~> nondim].
  real :: I4dt          ! 1 / 4 dt [T-1 ~> s-1].
  real :: Sfn_safe      ! The streamfunction that goes linearly back to 0 at the surface
                        ! [H L2 T-1 ~> m3 s-1 or kg s-1].  This is a good value to use when the
                        ! slope is so large as to be meaningless, usually due to weak stratification.
  real :: Sfn_est       ! A preliminary estimate (before limiting) of the overturning
                        ! streamfunction [H L2 T-1 ~> m3 s-1 or kg s-1].
  real :: Sfn_in_h      ! The overturning streamfunction [H L2 T-1 ~> m3 s-1 or kg s-1] (note that
                        ! the units are different from other Sfn vars).
  real :: Rho_avg
  real :: Z_to_H        ! A conversion factor from heights to thicknesses, perhaps based on
                        ! a spatially variable local density [H Z-1 ~> nondim or kg m-3]
  real :: h_neglect ! A thickness that is so small it is usually lost
                     ! in roundoff and can be neglected [H ~> m or kg m-2].
  real :: hn_2          ! Half of h_neglect [H ~> m or kg m-2].
  logical :: use_EOS    ! If true, density is calculated from T & S using an equation of state.

  ! Initialize some useful variables
  use_EOS = associated(tv%eqn_of_state)
  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  nk_linear = max(GV%nkml, 1)
  h_neglect = GV%H_subroundoff
  hn_2 = 0.5*h_neglect
  I4dt = 0.25 / dt
  I_slope_max2 = 1.0 / (CS%slope_max**2)
  uhTrANN(:,:,:) = 0.0
  vhTrANN(:,:,:) = 0.0

  uhtot(:,:) = 0.0
  vhtot(:,:) = 0.0
  ! do j=js,je ; do I=is-1,ie
  !   uhtot(I,j) = 0.0 
  ! enddo ; enddo
  ! do J=js-1,je ; do i=is,ie
  !   vhtot(i,J) = 0.0 
  ! enddo ; enddo

  !
  do j=js-1,je+1 ; do i=is-1,ie+1
    h_avail_rsum(i,j,1) = 0.0
    h_avail(i,j,1) = max(I4dt*G%areaT(i,j)*(h(i,j,1)-GV%Angstrom_H),0.0)
    h_avail_rsum(i,j,2) = h_avail(i,j,1)
    h_frac(i,j,1) = 1.0
  enddo ; enddo
  do k=2,nz
    do j=js-1,je+1; do i=is-1,ie+1
      h_avail(i,j,k) = max(I4dt*G%areaT(i,j)*(h(i,j,k)-GV%Angstrom_H),0.0)
      h_avail_rsum(i,j,k+1) = h_avail_rsum(i,j,k) + h_avail(i,j,k)
      h_frac(i,j,k) = 0.0 
      if (h_avail(i,j,k) > 0.0) h_frac(i,j,k) = h_avail(i,j,k) / h_avail_rsum(i,j,k+1)
    enddo ; enddo
  enddo
  
  ! For the u points. 
  do j=js,je
    do K= nz, 2, -1
      do I=is-1,ie
        if (allocated(tv%SpV_avg) .and. (k > nk_linear) ) then
          Rho_avg = ( ((h(i,j,k) + h(i,j,k-1)) + (h(i+1,j,k) + h(i+1,j,k-1))) + 4.0*hn_2 ) / &
                ( (((h(i,j,k)+hn_2) * tv%SpV_avg(i,j,k))   + ((h(i,j,k-1)+hn_2) * tv%SpV_avg(i,j,k-1))) + &
                  (((h(i+1,j,k)+hn_2)*tv%SpV_avg(i+1,j,k)) + ((h(i+1,j,k-1)+hn_2)*tv%SpV_avg(i+1,j,k-1))) )
          ! Use an average density to convert the volume streamfunction estimate into a mass streamfunction.
          Z_to_H = GV%RZ_to_H*Rho_avg
        else
          Z_to_H = GV%Z_to_H
        endif

        slope2_Ratio_u(I,j,K) = slope_x(I,j,K)**2  * I_slope_max2 !! this probably does not need to be 3D field
        if (slope2_Ratio_u(I,j,K) > 1.0e20) then ! Unlikely to happen, specially since default value of slope_max is 0.01 and slope is limited to be b/w -1 and 1
          slope2_Ratio_u(I,j,K) = 1.0e20
        end if

        if (k > nk_linear) then
          if (use_EOS) then
            if (uhtot(I,j) <= 0.0) then
              ! The transport that must balance the transport below is positive.
              Sfn_safe = uhtot(I,j) * (1.0 - h_frac(i,j,k))
            else !  (uhtot(I,j) > 0.0)
              Sfn_safe = uhtot(I,j) * (1.0 - h_frac(i+1,j,k))
            endif
            ! Determine the actual streamfunction at each interface.
            Sfn_est = (Z_to_H*Sfn_unlim_u(I,j,K) + slope2_Ratio_u(I,j,K)*Sfn_safe) / (1.0 + slope2_Ratio_u(I,j,K))
          else  ! When use_EOS is false, the layers are constant density.
            Sfn_est = Z_to_H*Sfn_unlim_u(I,j,K)
          endif

          ! Make sure that there is enough mass above to allow the streamfunction
          ! to satisfy the boundary condition of 0 at the surface.
          Sfn_in_H = min(max(Sfn_est, -h_avail_rsum(i,j,K)), h_avail_rsum(i+1,j,K))

          ! The actual transport is limited by the mass available in the two
          ! neighboring grid cells.
          uhTrANN(I,j,k) = max(min((Sfn_in_H - uhtot(I,j)), h_avail(i,j,k)), &
                           -h_avail(i+1,j,k))

        else
          if (uhtot(I,j) <= 0.0) then
            uhTrANN(I,j,k) = -uhtot(I,j) * h_frac(i,j,k)
          else !  (uhtot(I,j) > 0.0)
            uhTrANN(I,j,k) = -uhtot(I,j) * h_frac(i+1,j,k)
          endif
        
        endif
      
        uhtot(I,j) = uhtot(I,j) + uhTrANN(I,j,k)
      
      enddo 
    enddo
  enddo

  do J=js-1,je 
    do K= nz, 2, -1
      do i=is,ie
        if (allocated(tv%SpV_avg) .and.  (k > nk_linear) ) then
          Rho_avg = ( ((h(i,j,k) + h(i,j,k-1)) + (h(i,j+1,k) + h(i,j+1,k-1))) + 4.0*hn_2 ) / &
              ( (((h(i,j,k)+hn_2) * tv%SpV_avg(i,j,k))   + ((h(i,j,k-1)+hn_2) * tv%SpV_avg(i,j,k-1))) + &
                (((h(i,j+1,k)+hn_2)*tv%SpV_avg(i,j+1,k)) + ((h(i,j+1,k-1)+hn_2)*tv%SpV_avg(i,j+1,k-1))) )
          ! Use an average density to convert the volume streamfunction estimate into a mass streamfunction.
          Z_to_H = GV%RZ_to_H*Rho_avg
        else
          Z_to_H = GV%Z_to_H
        endif

        slope2_Ratio_v(i,J,K) = slope_y(i,J,K)**2 * I_slope_max2
        if (slope2_Ratio_v(i,J,K) > 1.0e20) then
          slope2_Ratio_v(i,J,K) = 1.0e20
        end if

        if (k > nk_linear) then
          if (use_EOS) then
            if (vhtot(i,J) <= 0.0) then
              ! The transport that must balance the transport below is positive.
              Sfn_safe = vhtot(i,J) * (1.0 - h_frac(i,j,k))
            else !  (vhtot(I,j) > 0.0)
              Sfn_safe = vhtot(i,J) * (1.0 - h_frac(i,j+1,k))
            endif

            ! Find the actual streamfunction at each interface.
            Sfn_est = (Z_to_H*Sfn_unlim_v(i,J,K) + slope2_Ratio_v(i,J,K)*Sfn_safe) / (1.0 + slope2_Ratio_v(i,J,K))
          else  ! When use_EOS is false, the layers are constant density.
            Sfn_est = Z_to_H*Sfn_unlim_v(i,J,K)
          endif

          ! Make sure that there is enough mass above to allow the streamfunction
          ! to satisfy the boundary condition of 0 at the surface.
          Sfn_in_H = min(max(Sfn_est, -h_avail_rsum(i,j,K)), h_avail_rsum(i,j+1,K))

          ! The actual transport is limited by the mass available in the two
          ! neighboring grid cells.
          vhTrANN(i,J,k) = max(min((Sfn_in_H - vhtot(i,J)), h_avail(i,j,k)), -h_avail(i,j+1,k))
        else
          if (vhtot(i,J) <= 0.0) then
            vhTrANN(i,J,k) = -vhtot(i,J) * h_frac(i,j,k)
          else !  (vhtot(i,J) > 0.0)
            vhTrANN(i,J,k) = -vhtot(i,J) * h_frac(i,j+1,k)
          endif
        endif
        vhtot(i,J) = vhtot(i,J) + vhTrANN(i,J,k)

      enddo
    enddo
  enddo

  do j=js,je ; do I=is-1,ie ; uhTrANN(I,j,1) = -uhtot(I,j) ; enddo ; enddo
  do J=js-1,je ; do i=is,ie ; vhTrANN(i,J,1) = -vhtot(i,J) ; enddo ; enddo
  

end subroutine squeeze_limiter

subroutine squeeze_limiter_old(h, uhTrANN, vhTrANN, dt, G, Gv, CS)
  type(ocean_grid_type), intent(in) :: G
  type(verticalGrid_type), intent(in) :: GV
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(in) :: h !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(inout)   :: uhTrANN   !< Zonal ANN h transport u*h*dy [L2 H T-1 ~> m3 s-1 or kg s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(inout)   :: vhTrANN   !< Meridional ANN h transport v*h*dx [L2 H T-1 ~> m3 s-1 or kg s-1]
  type(thickness_flux_ann_CS), intent(in) :: CS !< Control structure for thickness_flux_ann
  real, intent(in) :: dt !< Time step [T ~> s]

  !local variables
  integer :: i, j, k, is, ie, js, je, nz
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) ::  h_avail    ! The mass available for transport out of each face, divided
                                                              ! by dt [H L2 T-1 ~> m3 s-1 or kg s-1].
  real :: I4dt          ! 1 / 4 dt [T-1 ~> s-1].

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  I4dt = 0.25 / dt

  do k = 1, nz
    do j=js-1,je+1 ; do i=is-1,ie+1
      ! Maximum volume that may be transported out of any face (1/4 of the volume of the cell)
      h_avail(i,j,k) = max(I4dt*G%areaT(i,j)*(h(i,j,k) - GV%Angstrom_H), 0.0)
    enddo ; enddo
  enddo

  do k = 1, nz
    do j=js,je ; do i=is-1,ie
      ! Limit the fluxes to the available mass
      uhTrANN(i,j,k) = max( min(uhTrANN(i,j,k), h_avail(i,j,k)), -h_avail(i+1,j,k))
    enddo ; enddo

    do j=js-1,je ; do i=is,ie
      vhTrANN(i,j,k) = max( min(vhTrANN(i,j,k), h_avail(i,j,k)), -h_avail(i,j+1,k))
    enddo ; enddo
  enddo
      

end subroutine squeeze_limiter_old

!> Init function
! Read parameters and register output fields.
subroutine thickness_flux_ann_init(Time, G, GV, US, param_file, diag, CS)
  type(time_type),         intent(in) :: Time    !< Current model time
  type(ocean_grid_type),   intent(in) :: G       !< Ocean grid structure
  type(verticalGrid_type), intent(in) :: GV      !< Vertical grid structure
  type(unit_scale_type),   intent(in) :: US      !< A dimensional unit scaling type
  type(param_file_type),   intent(in) :: param_file !< Parameter file handles
  type(diag_ctrl), target, intent(inout) :: diag !< Diagnostics control structure
  type(thickness_flux_ann_CS), intent(inout) :: CS !< Control structure for thickness_flux_ann

  ! Local variables
  character(len=40) :: mdl = "MOM_thickness_flux_ann"
# include "version_variable.h"  
  
  CS%initialized = .true.
  CS%diag => diag

  ! Read all relevant parameters and write them to the model log.
  call log_version(param_file, mdl, version, "")
  
  ! Gather all parameters that are needed. 
  call get_param(param_file, mdl, "thickness_flux_ann_coeff", CS%ann_coeff, &
                      "Coefficient to multiply the thickness flux ANN output by", default=1.0, units="nondim")
  call get_param(param_file, mdl, "thickness_flux_ann_window", CS%ann_window, &
                        "Number of horizontal grid points to use in the thickness flux ANN window", default=3)
  call get_param(param_file, mdl, "thickness_flux_ann_num_layers", CS%thickness_ann_num_layers, &
                      "Number of ANN layers for thickness flux (including input and output as layer)", default=4)
  call get_param(param_file, mdl, "thickness_flux_ann_params_file", CS%thickness_ann_NNfile, &
                      "Thickness_flux ANN parameters netcdf input", default="thickness_flux_ann_params.nc")
  call get_param(param_file, mdl, "thickness_flux_model_type", CS%thickness_ann_model_type, &
                      "Type of ANN model (e.g. options GM_ann, GM_rotated_ann, nondim_ann).", default="GM_ann")     
  call get_param(param_file, mdl, "decompose_h_gradients", CS%decompose_h_gradients, &
                     "If true, decomposes h gradients into steady and transient parts", default=.false.)
  call get_param(param_file, mdl, "h_mask_flux", CS%h_mask_flux, &
                     "If true, masks flux in regions with very small layer thickness", default=.false.)                     
  call get_param(param_file, mdl, "remove_barotropic_flux", CS%remove_barotropic_flux, &
                     "If true, remove barotropic component from flux", default=.false.) 
  call get_param(param_file, mdl, "limit_upslope_flow", CS%limit_upslope_flow, &
                     "If true, limit flux that pushes isopycnals upslope.", default=.false.)                                          
  call get_param(param_file, mdl, "h_min_mask", CS%h_min_mask, &
                      "Mask thickness below this thereshold in the ANN.", default=1.0, units="nondim")
  call get_param(param_file, mdl, "mask_boundary_layers", CS%mask_boundary_layers, &
                     "If true, masks flux in regions near boundaries", default=.false.)                     
  call get_param(param_file, mdl, "delta_BL_top", CS%delta_BL_top, &
                      "Upper boundary layer to be ignored", default=100.0, units="nondim")   
  call get_param(param_file, mdl, "delta_BL_bottom", CS%delta_BL_bottom, &
                      "Lower boundary layer to be ignored", default=500.0, units="nondim")
  call get_param(param_file, mdl, "h_mask_width", CS%h_mask_width, &
                      "Width of the mask transition in the ANN.", default=0.1, units="nondim")
  call get_param(param_file, mdl, "FGR_ANN", CS%FGR_ANN, &
                      "Filter to grid ratio for the ANN", default=1.0, units="nondim")
  call get_param(param_file, mdl, "KD_SMOOTH", CS%kappa_smooth, &
                 "A diapycnal diffusivity that is used to interpolate "//&
                 "more sensible values of T & S into thin layers.", &
                 units="m2 s-1", default=1.0e-6, scale=GV%m2_s_to_HZ_T)
  call get_param(param_file, mdl, "KHTH_SLOPE_MAX", CS%slope_max, &
                 "A slope beyond which the calculated isopycnal slope is "//&
                 "not reliable and is scaled away.", units="nondim", default=0.01, scale=US%L_to_Z)

  ! Initialize the ANN
  call ann_init(CS%ann_cs, CS%thickness_ann_num_layers, CS%thickness_ann_NNfile)

  ! Register diagnostics
  CS%id_dhdx = register_diag_field('ocean_model', 'dhdx', diag%axesTL, Time, &
              'Horizontal h gradient in x direction', 'm/m', conversion=US%Z_to_L/US%L_to_m )
  CS%id_dhdy = register_diag_field('ocean_model', 'dhdy', diag%axesTL, Time, &
              'Horizontal h gradient in y direction', 'm/m', conversion=US%Z_to_L/US%L_to_m )
  
  CS%id_slope_x = register_diag_field('ocean_model', 'ANN_slope_x', diag%axesCui, Time, &
           'Zonal slope of neutral surface', 'nondim', conversion=US%Z_to_L)
  CS%id_slope_y =  register_diag_field('ocean_model', 'ANN_slope_y', diag%axesCvi, Time, &
           'Meridional slope of neutral surface', 'nondim', conversion=US%Z_to_L)

  CS%id_dhbardx = register_diag_field('ocean_model', 'dhbardx', diag%axesTL, Time, &
              'Horizontal h gradient due to bottom in x direction', 'm/m', conversion=US%Z_to_L/US%L_to_m )
  CS%id_dhbardy = register_diag_field('ocean_model', 'dhbardy', diag%axesTL, Time, &
              'Horizontal h gradient due to bottom in y direction', 'm/m', conversion=US%Z_to_L/US%L_to_m )
  CS%id_dudx = register_diag_field('ocean_model', 'dudx', diag%axesTL, Time, &
              'Zonal velocity gradient in x direction', 's-1', conversion=US%s_to_T )
  CS%id_dudy = register_diag_field('ocean_model', 'dudy', diag%axesTL, Time, &
              'Zonal velocity gradient in y direction', 's-1', conversion=US%s_to_T )
  CS%id_dvdx = register_diag_field('ocean_model', 'dvdx', diag%axesTL, Time, &
              'Meridional velocity gradient in x direction', 's-1', conversion=US%s_to_T )
  CS%id_dvdy = register_diag_field('ocean_model', 'dvdy', diag%axesTL, Time, &
              'Meridional velocity gradient in y direction', 's-1', conversion=US%s_to_T )
  CS%id_Fx = register_diag_field('ocean_model', 'Fx', diag%axesTL, Time, &
              'ANN output in x direction', 'm2 s-1', conversion=US%L_to_m**2*US%s_to_T)
  CS%id_Fy = register_diag_field('ocean_model', 'Fy', diag%axesTL, Time, &
              'ANN output in y direction', 'm2 s-1', conversion=US%L_to_m**2*US%s_to_T)
  CS%id_uhTrANN = register_diag_field('ocean_model', 'uhTrANN', diag%axesCuL, Time, &
              'Zonal ANN h transport ~ u*h*dy', 'm3 s-1', conversion=US%L_to_m**3*US%s_to_T)
  CS%id_vhTrANN = register_diag_field('ocean_model', 'vhTrANN', diag%axesCvL, Time, &
              'Meridional ANN h transport ~ v*h*dx', 'm3 s-1', conversion=US%L_to_m**3*US%s_to_T)
  CS%id_h_mask = register_diag_field('ocean_model', 'h_mask', diag%axesTL, Time, &
              'Mask based on the thickness', 'nondim', conversion=1.0)
  CS%id_boundary_mask = register_diag_field('ocean_model', 'boundary_mask', diag%axesTL, Time, &
              'Mask the boundary regions with steep slopes', 'nondim', conversion=1.0)            

  CS%id_sfn_unlim_x =  register_diag_field('ocean_model', 'ANN_sfn_unlim_x', diag%axesCui, Time, &
           'Parameterized Zonal Overturning Streamfunction before limiting/smoothing', &
           'm3 s-1', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_sfn_unlim_y =  register_diag_field('ocean_model', 'ANN_sfn_unlim_y', diag%axesCvi, Time, &
           'Parameterized Meridional Overturning Streamfunction before limiting/smoothing', &
           'm3 s-1', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)


end subroutine thickness_flux_ann_init

!> End function
subroutine thickness_flux_ann_end(CS)
  type(thickness_flux_ann_CS), intent(inout) :: CS !< Control structure for thickness_flux_ann

end subroutine thickness_flux_ann_end

! ! ! Old routine that may be used if we want to make this modules standalone and apart from the thickness diffuse module. 
!> Applies the thickness transport calculated in each layer using an ANN,
!! and updates the thicknesses, h. 
! subroutine thickness_flux_ann(h, u, v, uhtr, vhtr, dt, G, GV, US, CS)
!   type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
!   type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
!   type(unit_scale_type),                      intent(in)    :: US     !< A dimensional unit scaling type
!   real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(inout) :: h      !< Layer thickness [H ~> m or kg m-2]
!   real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(in)    :: u !< Zonal velocity
!   real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(in)    :: v !< Meridional velocity
!   real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(inout) :: uhtr   !< Accumulated zonal mass flux
!                                                                       !! [L2 H ~> m3 or kg]
!   real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(inout) :: vhtr   !< Accumulated meridional mass flux
!                                                                     !! [L2 H ~> m3 or kg]
!   real,                                       intent(in)    :: dt !< Time step [T ~> s]
!   type(thickness_flux_ann_CS),                intent(inout) :: CS !< Control structure for thickness_flux_ann

!   real :: uhTrANN(SZIB_(G),SZJ_(G),SZK_(GV)) ! Zonal ANN h transport ~ u*h*dy [L2 H T-1 ~> m3 s-1 or kg s-1]
!   real :: vhTrANN(SZI_(G),SZJB_(G),SZK_(GV)) ! Meridional ANN h transport ~ v*h*dx [L2 H T-1 ~> m3 s-1 or kg s-1]
!   integer :: i, j, k, is, ie, js, je, nz
!   real :: h_neglect ! A thickness that is so small it is usually lost
!                     ! in roundoff and can be neglected [H ~> m or kg m-2].


!   is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
!   h_neglect = GV%H_subroundoff

!   uhTrANN = 0.0
!   vhTrANN = 0.0

!   !> Calculate the thickness fluxes using the ANN
!   call thickness_flux_ann_full(h, u, v, uhTrANN, vhTrANN, G, GV, US, CS)

!   ! Update the layer thickness 
!   !$OMP parallel do default(shared)
!   do k=1,nz
!     do j=js,je ; do I=is-1,ie
!       uhtr(I,j,k) = uhtr(I,j,k) + uhTrANN(I,j,k) * dt
!       !if (associated(CDp%uhGM)) CDp%uhGM(I,j,k) = uhD(I,j,k)
!     enddo ; enddo
!     do J=js-1,je ; do i=is,ie
!       vhtr(i,J,k) = vhtr(i,J,k) + vhTrANN(i,J,k) * dt
!       !if (associated(CDp%vhGM)) CDp%vhGM(i,J,k) = vhD(i,J,k)
!     enddo ; enddo
!     do j=js,je ; do i=is,ie
!       h(i,j,k) = h(i,j,k) - dt * G%IareaT(i,j) * &
!           ((uhTrANN(I,j,k) - uhTrANN(I-1,j,k)) + (vhTrANN(i,J,k) - vhTrANN(i,J-1,k)))
!       if (h(i,j,k) < GV%Angstrom_H) h(i,j,k) = GV%Angstrom_H
!     enddo ; enddo
!   enddo

!end subroutine thickness_flux_ann


end module MOM_thickness_flux_ann