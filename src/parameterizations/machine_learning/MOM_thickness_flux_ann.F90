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
  integer :: ann_window  !< Number of horizontal grid points to use in the ANN window.
  logical :: decompose_h_gradients !< If true, decompose the h gradients into steady and transient parts.
  real    :: h_min_mask !< Minimum thickness for the mask
  real    :: h_mask_width !< Width of the mask for the h gradients

  type(diag_ctrl), pointer :: diag => NULL() !< structure used to regulate timing of diagnostics

  !! Diagnostic identifier
  integer :: id_dhdx, id_dhdy, id_dhbardx, id_dhbardy
  integer :: id_h_mask
  integer :: id_Fx, id_Fy, id_uhTrANN, id_vhTrANN
  integer :: id_dudx, id_dudy, id_dvdx, id_dvdy

end type THICKNESS_FLUX_ANN_CS

contains 

!> Calculates the parameterized thickness fluxes for use in the continuity equation.
!> Returns the fluxes at the u,v points.
subroutine thickness_flux_ann_full(h, u, v, uhTrANN, vhTrANN, G, GV, US, CS)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(unit_scale_type),                      intent(in)    :: US     !< A dimensional unit scaling type
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)    :: h      !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(in)    :: u      !< Zonal velocity [L T-1 ~> m s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(in)    :: v      !< Meridional velocity [L T-1 ~> m s-1]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(out)   :: uhTrANN   !< Zonal ANN h transport u*h*dy [L2 H T-1 ~> m3 s-1 or kg s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(out)   :: vhTrANN   !< Meridional ANN h transport v*h*dx [L2 H T-1 ~> m3 s-1 or kg s-1]
  type(thickness_flux_ann_CS),                intent(inout) :: CS !< Control structure for thickness_flux_ann
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
  
  ! Variables for the ANN output 
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: FxC, FyC

  ! Variables for the local stencil 
  real, allocatable :: dhdx_local(:,:), dhdy_local(:,:), dudx_local(:,:), dudy_local(:,:), dvdx_local(:,:), dvdy_local(:,:)
  real, allocatable :: x(:) !To-Do: make this adjustable based on window size.
  real, allocatable :: y(:), y_rot(:)
  real :: vel_grad_mag, h_grad_mag
  real :: tol_vel_grad, tol_h_grad
  real :: NGM_C

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  !Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! Calculate the extra points that grid needs to be extended to, using the ANN window
  ! done as (ann_window-1)/2 as integer
  shift = (CS%ann_window-1)/2
  stencil_points = CS%ann_window * CS%ann_window

  uhTrANN(:,:,:) = 0.0
  vhTrANN(:,:,:) = 0.0

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
  call h_gradients(h, G, GV, dhdx, dhdy, CS)

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
        
        y(:) = y(:) * h_grad_mag * vel_grad_mag * G%areaT(i,j) 
        ! End : Code to work with specific ANN
      endif
      
      FxC(i,j,k) = y(1) * G%mask2dT(i,j) 
      FyC(i,j,k) = y(2) * G%mask2dT(i,j) 
    enddo ; enddo
  
  !> Interpolate fluxes to u, v points
    ! > We multiply the ann coeff at this later stage, so the unchanged F can be used for diagnostics
    do j=js,je ; do i=is-1,ie
      uhTrANN(I,j,k) = 0.5 * (FxC(i,j,k) + FxC(i+1,j,k)) * G%dyCu(I,j) * G%mask2dCu(I,j) * CS%ann_coeff
    enddo ; enddo
    do j=js-1,je ; do i=is,ie
      vhTrANN(i,J,k) = 0.5 * (FyC(i,j,k) + FyC(i,j+1,k)) * G%dxCv(i,J) * G%mask2dCv(i,J) * CS%ann_coeff
    enddo ; enddo

  !> Put any limiters that may be needed. 

  enddo

  !> Apply the no- BT flow condition (for 2 layers)
  !uhTrANN(:,:,1) = - uhTrANN(:,:,2) 
  !vhTrANN(:,:,1) = - vhTrANN(:,:,2) 


  if (CS%id_dhdx > 0) call post_data(CS%id_dhdx, dhdx, CS%diag)
  if (CS%id_dhdy > 0) call post_data(CS%id_dhdy, dhdy, CS%diag)

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

  do j=js-shift-2, je+shift+1 ; do i=is-shift-2, ie+shift+1
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

  do j=js-shift-1, je+shift+1 ; do i=is-shift-1, ie+shift+1
    dhbardx(i,j,nz) =  - de_bottomdx(i,j) * h_mask(i,j,nz)
    dhbardy(i,j,nz) =  - de_bottomdy(i,j) * h_mask(i,j,nz)
  enddo ; enddo

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
  call get_param(param_file, mdl, "h_min_mask", CS%h_min_mask, &
                      "Mask thickness below this thereshold in the ANN.", default=1.0, units="nondim")
  call get_param(param_file, mdl, "h_mask_width", CS%h_mask_width, &
                      "Width of the mask transition in the ANN.", default=0.1, units="nondim")

  ! Initialize the ANN
  call ann_init(CS%ann_cs, CS%thickness_ann_num_layers, CS%thickness_ann_NNfile)

  ! Register diagnostics
  CS%id_dhdx = register_diag_field('ocean_model', 'dhdx', diag%axesTL, Time, &
              'Horizontal h gradient in x direction', 'm/m', conversion=US%Z_to_L/US%L_to_m )
  CS%id_dhdy = register_diag_field('ocean_model', 'dhdy', diag%axesTL, Time, &
              'Horizontal h gradient in y direction', 'm/m', conversion=US%Z_to_L/US%L_to_m )
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


end subroutine thickness_flux_ann_init

!> End function
subroutine thickness_flux_ann_end(CS)
  type(thickness_flux_ann_CS), intent(inout) :: CS !< Control structure for thickness_flux_ann

end subroutine thickness_flux_ann_end

! ! ! Old routine that may be used if we want to make this modules standalone and apart from the thickness diffuse module. 
!> Applies the thickness transport calculated in each layer using an ANN,
!! and updates the thicknesses, h. 
subroutine thickness_flux_ann(h, u, v, uhtr, vhtr, dt, G, GV, US, CS)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(unit_scale_type),                      intent(in)    :: US     !< A dimensional unit scaling type
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(inout) :: h      !< Layer thickness [H ~> m or kg m-2]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(in)    :: u !< Zonal velocity
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(in)    :: v !< Meridional velocity
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(inout) :: uhtr   !< Accumulated zonal mass flux
                                                                      !! [L2 H ~> m3 or kg]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(inout) :: vhtr   !< Accumulated meridional mass flux
                                                                    !! [L2 H ~> m3 or kg]
  real,                                       intent(in)    :: dt !< Time step [T ~> s]
  type(thickness_flux_ann_CS),                intent(inout) :: CS !< Control structure for thickness_flux_ann

  real :: uhTrANN(SZIB_(G),SZJ_(G),SZK_(GV)) ! Zonal ANN h transport ~ u*h*dy [L2 H T-1 ~> m3 s-1 or kg s-1]
  real :: vhTrANN(SZI_(G),SZJB_(G),SZK_(GV)) ! Meridional ANN h transport ~ v*h*dx [L2 H T-1 ~> m3 s-1 or kg s-1]
  integer :: i, j, k, is, ie, js, je, nz
  real :: h_neglect ! A thickness that is so small it is usually lost
                    ! in roundoff and can be neglected [H ~> m or kg m-2].


  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
  h_neglect = GV%H_subroundoff

  uhTrANN = 0.0
  vhTrANN = 0.0

  !> Calculate the thickness fluxes using the ANN
  call thickness_flux_ann_full(h, u, v, uhTrANN, vhTrANN, G, GV, US, CS)

  ! Update the layer thickness 
  !$OMP parallel do default(shared)
  do k=1,nz
    do j=js,je ; do I=is-1,ie
      uhtr(I,j,k) = uhtr(I,j,k) + uhTrANN(I,j,k) * dt
      !if (associated(CDp%uhGM)) CDp%uhGM(I,j,k) = uhD(I,j,k)
    enddo ; enddo
    do J=js-1,je ; do i=is,ie
      vhtr(i,J,k) = vhtr(i,J,k) + vhTrANN(i,J,k) * dt
      !if (associated(CDp%vhGM)) CDp%vhGM(i,J,k) = vhD(i,J,k)
    enddo ; enddo
    do j=js,je ; do i=is,ie
      h(i,j,k) = h(i,j,k) - dt * G%IareaT(i,j) * &
          ((uhTrANN(I,j,k) - uhTrANN(I-1,j,k)) + (vhTrANN(i,J,k) - vhTrANN(i,J-1,k)))
      if (h(i,j,k) < GV%Angstrom_H) h(i,j,k) = GV%Angstrom_H
    enddo ; enddo
  enddo

end subroutine thickness_flux_ann


end module MOM_thickness_flux_ann