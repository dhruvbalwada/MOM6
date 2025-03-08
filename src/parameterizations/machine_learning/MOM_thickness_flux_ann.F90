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

public thickness_flux_ann, thickness_flux_ann_init, thickness_flux_ann_end

!> Control structure/type for thickness flux ANN
type, public :: THICKNESS_FLUX_ANN_CS ; private
  logical :: initialized = .false. !< If true, the module has been initialized.
  logical :: thickness_flux_ann  !< If true, use thickness fluxes are computed using ANN.
  logical :: debug !< if true, write verbose checksums for debugging purposes. 

  type(ann_cs) :: ann_cs !< ANN control structure.
  integer :: thickness_ann_num_layers ! number of layers
  character(len=200) :: thickness_ann_NNfile   ! The name of netcdf file having neural network shape function
  
  real :: ann_coeff  !< Coefficient to multiply the ANN output by.
  integer :: ann_window  !< Number of horizontal grid points to use in the ANN window.

  type(diag_ctrl), pointer :: diag => NULL() !< structure used to regulate timing of diagnostics

  !! Diagnostic identifier
  integer :: id_dhdx, id_dhdy, id_Fx, id_Fy
end type THICKNESS_FLUX_ANN_CS

contains 

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
  write (*,*) "here at last", uhTrANN(4,4,1), uhTrANN(4,4,2)

end subroutine thickness_flux_ann

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
  integer :: i, j, k, is, ie, js, je, nz, shift
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dhdx, dhdy
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dudx, dudy
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dvdx, dvdy
  
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: FxC, FyC

  real, dimension(2) :: y
  real, dimension(2) :: x !To-Do: make this adjustable based on window size.

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  !Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! Calculate the extra points that grid needs to be extended to, using the ANN window
  ! done as (ann_window-1)/2 as integer
  shift = (CS%ann_window-1)/2

  !> Calculates the h and u gradients in full 3D domain
  call h_gradients(h, G, GV, dhdx, dhdy, CS)
  call vel_gradients(u, v, G, GV, dudx, dudy, dvdx, dvdy, CS)
  
  do k=2, nz
  !> Rotation, local normalize etc

  !> Calculate the fluxes at center points 
    do j=js,je ; do i=is,ie
      x(1) = dhdx(i,j,k)
      x(2) = dhdy(i,j,k)
      
      call ann(x, y, CS%ann_cs)    

      !write (*,*) "i,j,k:", i, j,k, ", x,y", x, y
      
      FxC(i,j,k) = y(1) * CS%ann_coeff
      FyC(i,j,k) = y(2) * CS%ann_coeff
    enddo ; enddo
  
  !> Interpolate fluxes to u, v points
    do j=js,je ; do i=is-1,ie
      uhTrANN(I,j,k) = 0.5 * (FxC(i,j,k) + FxC(i+1,j,k)) * G%dyCu(I,j) * G%OBCmaskCu(I,j)
    enddo ; enddo
    do j=js-1,je ; do i=is,ie
      vhTrANN(i,J,k) = 0.5 * (FyC(i,j,k) + FyC(i,j+1,k)) * G%dxCv(i,J) * G%OBCmaskCv(i,J)
    enddo ; enddo

  !> Put any limiters that may be needed. 

  enddo

  if (CS%id_dhdx > 0) call post_data(CS%id_dhdx, dhdx, CS%diag)
  if (CS%id_dhdy > 0) call post_data(CS%id_dhdy, dhdy, CS%diag)

end subroutine thickness_flux_ann_full


!> Calculates the thickness gradients in each layer at the center points in 3D. 
!> TODO: the code below does not have the proper handling of the case with variable bottom.
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
    do j=js-shift, je+shift ; do i=is-shift-1, ie+shift ! extra points needed in the x direction since we interpolate to center
      dhdx_u(I,j,k) = G%IdxCu(i,j) * (h(i+1,j,k) - h(i,j,k)) 
    enddo ; enddo
    ! Calculate the y-gradients at v points
    do j=js-shift-1, je+shift ; do i=is-shift, ie+shift ! extra points needed in the y direction since we interpolate to center
      dhdy_v(i,J,k) = G%IdyCv(i,J) * (h(i,j+1,k) - h(i,j,k))
    enddo ; enddo
    ! Interpolate the gradients to the center points
    ! We need these at +/- shift points because that is the local domain that the ANN will use.
    do j=js-shift, je+shift ; do i=is-shift, ie+shift      
      dhdx(i,j,k) = 0.5 * (dhdx_u(I,j,k) + dhdx_u(I-1,j,k))
      dhdy(i,j,k) = 0.5 * (dhdy_v(i,J,k) + dhdy_v(i,J-1,k))
    enddo ; enddo
  enddo ! end k loop

end subroutine h_gradients

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
    do j=js-shift,je+shift ; do i=is-shift,ie+shift ! has halo 2 ! loops over c points
      dudx(i,j,k) = G%IdxT(i,j)* (u(I,j,k)   - u(I-1,j,k)) !* G%mask2dT(i,j)
      dvdy(i,j,k) = G%IdyT(i,j)* (v(i,J,k)   - v(i,J-1,k)) !* G%mask2dT(i,j)
    enddo ; enddo

    ! Calculate velocity gradients at corner points 
    ! loops over q points (we don't use the soft convention of MOM6 for do loop indices here)
    do j=js-shift-1,je+shift ; do i=is-shift-1,ie+shift 
      dvdx_q(I,J,k) = G%IdxBu(I,J)*(v(i+1,J,k)  - v(i,J,k) ) !* G%mask2dBu(I,J)
      dudy_q(I,J,k) = G%IdyBu(I,J)*(u(I,j+1,k)  - u(I,j,k) ) !* G%mask2dBu(I,J)
    enddo ; enddo

    ! interpolate corner grads to center points 
    do j = js-shift, je+shift; do i = is-shift, ie+shift
      dvdx(i,j,k) =  0.25 * (dvdx_q(I,J,k) + dvdx_q(I-1,J,k) + dvdx_q(I,J-1,k) + dvdx_q(I-1,J-1,k)) !* G%mask2dT(i,j) 
      dudy(i,j,k) =  0.25 * (dudy_q(I,J,k) + dudy_q(I-1,J,k) + dudy_q(I,J-1,k) + dudy_q(I-1,J-1,k)) !* G%mask2dT(i,j) 
    enddo; enddo 
  enddo
end subroutine vel_gradients

!> Init function
! Read parameters and register output fields.
subroutine thickness_flux_ann_init(Time, G, GV, US, param_file, diag,CS)
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
  ! Read parameters
  CS%initialized = .true.
  CS%diag => diag

  ! Read all relevant parameters and write them to the model log.
  call log_version(param_file, mdl, version, "")
  call get_param(param_file, mdl, "THICKNESS_FLUX_ANN", CS%thickness_flux_ann, &
                      "If true, turns on the thickness flux ANN scheme", default=.false.)

  call get_param(param_file, mdl, "thickness_flux_ann_coeff", CS%ann_coeff, &
                      "Coefficient to multiply the thickness flux ANN output by", default=1.0, units="nondim")
  call get_param(param_file, mdl, "thickness_flux_ann_window", CS%ann_window, &
                        "Number of horizontal grid points to use in the thickness flux ANN window", default=3)

  ! Setup ann for thickness fluxes
  call get_param(param_file, mdl, "thickness_flux_ann_num_layers", CS%thickness_ann_num_layers, &
                      "Number of ANN layers for thickness flux", default=4)
  call get_param(param_file, mdl, "thickness_flux_ann_params_file", CS%thickness_ann_NNfile, &
                      "Thickness_flux ANN parameters netcdf input", default="thickness_flux_ann_params.nc")
  call ann_init(CS%ann_cs, CS%thickness_ann_num_layers, CS%thickness_ann_NNfile)

  ! Register diagnostics
  CS%id_dhdx = register_diag_field('ocean_model', 'dhdx', diag%axesTL, Time, &
              'Horizontal h gradient in x direction', 'm/m', &
              conversion=US%Z_to_L)
  CS%id_dhdy = register_diag_field('ocean_model', 'dhdy', diag%axesTL, Time, &
              'Horizontal h gradient in y direction', 'm/m', &
              conversion=US%Z_to_L)

end subroutine thickness_flux_ann_init

!> End function
subroutine thickness_flux_ann_end(CS)
  type(thickness_flux_ann_CS), intent(inout) :: CS !< Control structure for thickness_flux_ann

end subroutine thickness_flux_ann_end

end module MOM_thickness_flux_ann