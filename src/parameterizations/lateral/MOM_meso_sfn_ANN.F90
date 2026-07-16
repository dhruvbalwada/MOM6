! This file is part of MOM6, the Modular Ocean Model version 6.
! See the LICENSE file for licensing information.
! SPDX-License-Identifier: Apache-2.0

!> Implements an ANN-based mesoscale streamfunction parameterization for use
!! with isopycnal height diffusion in MOM_thickness_diffuse.
!!
!! The network reads a nondimensionalized stencil of density gradients,
!! strain rate components, and relative vorticity, and returns two density
!! flux components at the cell center. The dimensionalization in
!! meso_sfn_ANN_compute (multiplication by rho_grad_mag * vel_grad_mag *
!! areaT * ann_coeff) must match the nondimensionalization used when the
!! network was trained -- changing one without the other will produce
!! garbage fluxes. The training procedure is the implicit contract.
!!
!! Density fluxes are converted to a velocity-scale streamfunction
!! Upsilon (Ferrari et al. 2010) by dividing by the local 3-D density
!! gradient magnitude; a configurable clamp acts on Upsilon so the cap is
!! grid-independent. The volume-transport streamfunction passed back to
!! thickness_diffuse is Upsilon * dy_Cu (or dx_Cv), matching MOM6's
!! Sfn_unlim convention.
!!
!! The division by the local gradient is unbounded where the local gradient
!! is small but the stencil-scale gradient scaling the ANN output is not
!! (e.g. along coastlines), and there the answer is set by the clamps.
!! MESO_SFN_UPSILON_FORM selects structurally bounded alternatives:
!! STENCIL_GRAD replaces the horizontal part of the divisor with the same
!! stencil norm that scales the ANN output, and SLOPE_STENCIL feeds the
!! network stencils of bounded (sine-form) isopycnal slopes and reads the
!! rescaled output directly as Upsilon, eliminating the division entirely.
!! SLOPE_CLAMP is like SLOPE_STENCIL but applies the slope limit to the
!! INPUTS: the slope-vector magnitude the network sees is clamped at
!! MESO_SFN_SLOPE_MAX (Perezhogin's input-space bound), rather than limiting
!! only the output rescaling. The two dissociate whether the regularization
!! acts on what the network knows or on what it is allowed to do.
!! All bounded forms limit the effective slope at MESO_SFN_SLOPE_MAX,
!! mirroring the KHTH_SLOPE_MAX limit in GM.
module MOM_meso_sfn_ANN

use MOM_ANN,              only : ANN_init, ANN_apply_array_sio, ANN_end, ANN_CS
use MOM_diag_mediator,    only : post_data, register_diag_field, diag_ctrl, time_type
use MOM_error_handler,    only : MOM_error, FATAL
use MOM_file_parser,      only : get_param, log_version, param_file_type
use MOM_grid,             only : ocean_grid_type
use MOM_isopycnal_slopes, only : calc_isoneutral_slopes
use MOM_string_functions, only : uppercase
use MOM_unit_scaling,     only : unit_scale_type
use MOM_variables,        only : thermo_var_ptrs
use MOM_verticalGrid,     only : verticalGrid_type
use MOM_domains,          only : pass_vector

implicit none ; private

#include <MOM_memory.h>

public :: meso_sfn_ANN_init, meso_sfn_ANN_compute, meso_sfn_ANN_end

!>@{ Integer encodings of MESO_SFN_UPSILON_FORM, the conversion from ANN output to Upsilon
integer, parameter :: UPSILON_LOCAL_GRAD    = 0 !< Divide the flux by the local in-plane density gradient
integer, parameter :: UPSILON_STENCIL_GRAD  = 1 !< Divide the flux by a stencil-aggregate density gradient
integer, parameter :: UPSILON_SLOPE_STENCIL = 2 !< Predict Upsilon directly from slope-stencil inputs
integer, parameter :: UPSILON_SLOPE_CLAMP   = 3 !< As SLOPE_STENCIL, but the slope-vector magnitude fed
                                                !! to the network is clamped at MESO_SFN_SLOPE_MAX
!>@}

!> Control structure for meso-scale streamfunction ANN parameterization
type, public :: MESO_SFN_ANN_CS; private
  logical :: initialized = .false. !< If true, the module has been initialized.
  logical :: debug !< if true, write verbose checksums for debugging purposes.

  real :: ann_coeff  !< Coefficient to multiply the ANN output by.
  real    :: kappa_smooth        !< Vertical diffusivity used to interpolate more sensible values
                                 !! of T & S into thin layers [H Z T-1 ~> m2 s-1 or kg m-1 s-1]
  integer :: ann_window !< Size of the window used in the ANN model.

  type(ANN_CS) :: ann_rho_flux !< ANN instance for off-diagonal and diagonal stress
  character(len=200) :: ann_file_rho_flux !< Path to netcdf file with ANN
  real :: min_dist_from_boundary  !< Minimum distance from bottom for valid interface [Z ~> m]
  real :: mag_grad_floor  !< Floor for density gradient magnitude [R Z-1 ~> kg m-4]
  real :: flux_clamp  !< Maximum magnitude of ANN output density flux [R L T-1 ~> kg m-2 s-1]
  real :: Upsilon_clamp !< Maximum magnitude of the velocity-scale streamfunction
                        !! Upsilon (Ferrari et al. 2010) [L Z T-1 ~> m2 s-1]
  integer :: upsilon_form !< The form of the conversion from ANN output to Upsilon; one of
                          !! UPSILON_LOCAL_GRAD (original), UPSILON_STENCIL_GRAD or
                          !! UPSILON_SLOPE_STENCIL (bounded forms).
  real :: slope_max !< The pointwise-equivalent slope magnitude at which the bounded conversion
                    !! forms limit Upsilon, comparable to KHTH_SLOPE_MAX [Z L-1 ~> nondim]
  type(diag_ctrl), pointer :: diag => NULL() !< structure used to regulate timing of diagnostics
  ! Diagnostic identifiers
  integer :: id_drdx_u !< Diagnostic id for zonal density gradient at u-points.
  integer :: id_drdy_v !< Diagnostic id for meridional density gradient at v-points.
  integer :: id_drdz_u !< Diagnostic id for vertical density gradient at u-points.
  integer :: id_drdz_v !< Diagnostic id for vertical density gradient at v-points.
  integer :: id_drdx_c !< Diagnostic id for zonal density gradient at center points.
  integer :: id_drdy_c !< Diagnostic id for meridional density gradient at center points.
  integer :: id_Fx_c   !< Diagnostic id for zonal density flux at center points.
  integer :: id_Fy_c   !< Diagnostic id for meridional density flux at center points.
  integer :: id_Fx_u   !< Diagnostic id for zonal density flux at u-points.
  integer :: id_Fy_v   !< Diagnostic id for meridional density flux at v-points.
  integer :: id_sfn_u  !< Diagnostic id for volume streamfunction at u-points.
  integer :: id_sfn_v  !< Diagnostic id for volume streamfunction at v-points.
  integer :: id_Ups_u = -1 !< Diagnostic id for Upsilon at u-points before clamping.
  integer :: id_Ups_v = -1 !< Diagnostic id for Upsilon at v-points before clamping.
  integer :: id_Ups_clamp_u = -1 !< Diagnostic id for the mask where MESO_UPSILON_CLAMP binds at u-points.
  integer :: id_Ups_clamp_v = -1 !< Diagnostic id for the mask where MESO_UPSILON_CLAMP binds at v-points.
  integer :: id_s_hat_u = -1 !< Diagnostic id for the effective stencil-aggregate slope at u-points.
  integer :: id_s_hat_v = -1 !< Diagnostic id for the effective stencil-aggregate slope at v-points.
end type MESO_SFN_ANN_CS

contains

!> Compute the ANN-based mesoscale streamfunction on u- and v-points.
!!
!! Computes density gradients and velocity gradients, feeds them through the ANN
!! to get density fluxes at cell centers, then converts those fluxes into a
!! streamfunction on u- and v-points for use in thickness_diffuse.
subroutine meso_sfn_ANN_compute(h, e, sfn_u, sfn_v, G, GV, US, tv, CS, dt, u, v)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(unit_scale_type),                      intent(in)    :: US     !< A dimensional unit scaling type
  type(thermo_var_ptrs),                      intent(in)    :: tv     !< Thermodynamics structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)    :: h      !< Layer thickness [Z ~> m or kg m-2]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1),  intent(in)    :: e      !< Layer thickness [Z ~> m or kg m-2]
  type(MESO_SFN_ANN_CS),                intent(inout) :: CS !< Control structure for thickness_flux_ann
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(out) :: sfn_u  !< Mesoscale volume streamfunction
                                                                     !! on u-points [Z L2 T-1 ~> m3 s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(out) :: sfn_v  !< Mesoscale volume streamfunction
                                                                     !! on v-points [Z L2 T-1 ~> m3 s-1]
  real,                                      intent(in)    :: dt     !< Model time step [T ~> s]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), intent(in)    :: u      !< Zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), intent(in)    :: v      !< Meridional velocity [L T-1 ~> m s-1].

  ! Local variables
  integer :: i, j, k, is, ie, js, je, nz, shift, stencil_points, ii, jj
  integer :: nij, m

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: drdx_u !< Zonal density gradient at u [R L-1 ~> kg m-4]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: drdz_u !< Vertical density gradient at u [R Z-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: drdy_v !< Meridional density gradient at v [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: drdz_v !< Vertical density gradient at v [R Z-1 ~> kg m-4]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: slope_x !< Isopycnal slope in x at u [Z L-1 ~> nondim]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: slope_y !< Isopycnal slope in y at v [Z L-1 ~> nondim]

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: Fx_u !< Zonal density flux at u-points [R L T-1 ~> kg m-2 s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: Fy_v !< Meridional density flux at v-points [R L T-1 ~> kg m-2 s-1]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: drdx_c !< Zonal density gradient at center points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1) :: drdy_c !< Meridional density gradient
                                                          !! at center points [R L-1 ~> kg m-4]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: Fx_c !< Zonal density flux at center points [R L T-1 ~> kg m-2 s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: Fy_c !< Meridional density flux at
                                                       !! center points [R L T-1 ~> kg m-2 s-1]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: rho_norm_c !< Stencil norm of the horizontal density
                                                             !! gradient at center points [R L-1 ~> kg m-4]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: rho_norm_u !< rho_norm_c at u-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: rho_norm_v !< rho_norm_c at v-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: drdz_c !< Vertical density gradient at
                                                         !! center points [R Z-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: s_x_c !< Sine-form (bounded) zonal isopycnal slope
                                                        !! at center points [Z L-1 ~> nondim]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: s_y_c !< Sine-form (bounded) meridional isopycnal slope
                                                        !! at center points [Z L-1 ~> nondim]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: Upsx_c !< Zonal Upsilon at center points in
                                                         !! SLOPE_STENCIL mode [L Z T-1 ~> m2 s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1)  :: Upsy_c !< Meridional Upsilon at center points in
                                                         !! SLOPE_STENCIL mode [L Z T-1 ~> m2 s-1]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: Ups_u !< Upsilon at u-points before clamping [L Z T-1 ~> m2 s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: Ups_v !< Upsilon at v-points before clamping [L Z T-1 ~> m2 s-1]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: clamp_u !< 1 where MESO_UPSILON_CLAMP binds at u-points,
                                                          !! 0 elsewhere [nondim]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: clamp_v !< 1 where MESO_UPSILON_CLAMP binds at v-points,
                                                          !! 0 elsewhere [nondim]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1) :: s_hat_u !< Effective stencil-aggregate slope at
                                                          !! u-points [Z L-1 ~> nondim]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1) :: s_hat_v !< Effective stencil-aggregate slope at
                                                          !! v-points [Z L-1 ~> nondim]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dudx !< du/dx at cell center [T-1 ~> s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dvdy !< dv/dy at cell center [T-1 ~> s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dudy !< du/dy at cell center [T-1 ~> s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: dvdx !< dv/dx at cell center [T-1 ~> s-1]

  real, dimension(SZI_(G),SZJ_(G)) :: norm_y !< Scaling coefficient for ANN outputs [R L-1 T-1 ~> kg m-4 s-1]

  real, allocatable :: drdx_local(:,:) !< Local stencil of drdx [R L-1 ~> kg m-4], or of the
                                       !! sine-form slope s_x in SLOPE_STENCIL mode [Z L-1 ~> nondim]
  real, allocatable :: drdy_local(:,:) !< Local stencil of drdy [R L-1 ~> kg m-4], or of the
                                       !! sine-form slope s_y in SLOPE_STENCIL mode [Z L-1 ~> nondim]
  real, allocatable :: dudx_local(:,:) !< Local stencil of du/dx [T-1 ~> s-1]
  real, allocatable :: dudy_local(:,:) !< Local stencil of du/dy [T-1 ~> s-1]
  real, allocatable :: dvdx_local(:,:) !< Local stencil of dv/dx [T-1 ~> s-1]
  real, allocatable :: dvdy_local(:,:) !< Local stencil of dv/dy [T-1 ~> s-1]
  real, allocatable :: sh_xx_local(:,:) !< Local stencil of normal strain [T-1 ~> s-1]
  real, allocatable :: sh_xy_local(:,:) !< Local stencil of shear strain [T-1 ~> s-1]
  real, allocatable :: vort_local(:,:)  !< Local stencil of relative vorticity [T-1 ~> s-1]
  real :: vel_grad_mag !< Magnitude of velocity gradient tensor over stencil [T-1 ~> s-1]
  real :: rho_grad_mag !< Magnitude of density gradient over stencil [R L-1 ~> kg m-4], or of
                       !! the slope stencil in SLOPE_STENCIL mode [Z L-1 ~> nondim]
  real :: mag_grad3 !< Magnitude of the full 3-D density gradient at a center point [R Z-1 ~> kg m-4]
  real :: s_norm    !< Stencil norm of the sine-form slopes, after limiting [Z L-1 ~> nondim]
  real, allocatable :: x(:,:) !< Input vector to the ANN
  real, allocatable :: y(:,:) !< Output vector from the ANN
  real, allocatable :: yy(:)   !< Local output vector from the ANN
  real :: mag_grad !< Magnitude of 3-D density gradient [R Z-1 ~> kg m-4]
  logical :: use_stanley
  logical :: use_EOS
  real :: dist_from_bot_a, dist_from_bot_b  ! Distance from interface to bottom [Z ~> m]
  real :: dist_from_sfc_a, dist_from_sfc_b  ! Distance from interface to surface [Z ~> m]
  real :: Upsilon_u  ! Velocity-scale streamfunction at u-point (Ferrari et al. 2010) [L Z T-1 ~> m2 s-1]
  real :: Upsilon_v  ! Velocity-scale streamfunction at v-point (Ferrari et al. 2010) [L Z T-1 ~> m2 s-1]
  real :: rho_grad_neglect  ! A density gradient magnitude so small it is lost in
                            ! roundoff; used to prevent division by zero [R L-1 ~> kg m-4]
  real :: vel_grad_neglect  ! A velocity gradient magnitude so small it is lost in
                            ! roundoff; used to prevent division by zero [T-1 ~> s-1]
  real :: slope_neglect     ! A slope magnitude so small it is lost in roundoff;
                            ! used to prevent division by zero [Z L-1 ~> nondim]
  real :: s_mag             ! Magnitude of the slope vector at a center point [Z L-1 ~> nondim]
  logical :: slope_inputs   ! True for the forms that feed the network slope stencils
                            ! (SLOPE_STENCIL and SLOPE_CLAMP) and read its output as Upsilon

  if (.not. CS%initialized) call MOM_error(FATAL, &
      "meso_sfn_ANN_compute: Module MOM_meso_sfn_ANN must be initialized before use.")

  slope_inputs = (CS%upsilon_form == UPSILON_SLOPE_STENCIL) .or. &
                 (CS%upsilon_form == UPSILON_SLOPE_CLAMP)
  use_stanley = .false. ! Not using Stanley smoothing here.
  use_EOS = associated(tv%eqn_of_state)
  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  rho_grad_neglect = 1.0e-30 * US%kg_m3_to_R * US%L_to_m
  vel_grad_neglect = 1.0e-30 * US%T_to_s
  slope_neglect = 1.0e-30 * US%L_to_Z

  ! Allocate the local stencil variables
  allocate(drdx_local(CS%ann_window, CS%ann_window), drdy_local(CS%ann_window, CS%ann_window), &
           dudx_local(CS%ann_window, CS%ann_window), dudy_local(CS%ann_window, CS%ann_window), &
           dvdx_local(CS%ann_window, CS%ann_window), dvdy_local(CS%ann_window, CS%ann_window), &
           sh_xx_local(CS%ann_window, CS%ann_window), sh_xy_local(CS%ann_window, CS%ann_window), &
           vort_local(CS%ann_window, CS%ann_window))

  shift = (CS%ann_window-1)/2
  stencil_points = CS%ann_window * CS%ann_window

  ! Number of horizontal grid points in ANN inference loop below
  nij = (ie - is + 3) * (je - js + 3)
  allocate(x(nij, stencil_points*5), y(nij, 2))
  allocate(yy(2))

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

  rho_norm_c(:,:,:) = 0.0 ; rho_norm_u(:,:,:) = 0.0 ; rho_norm_v(:,:,:) = 0.0
  drdz_c(:,:,:) = 0.0 ; s_x_c(:,:,:) = 0.0 ; s_y_c(:,:,:) = 0.0
  Upsx_c(:,:,:) = 0.0 ; Upsy_c(:,:,:) = 0.0
  Ups_u(:,:,:) = 0.0 ; Ups_v(:,:,:) = 0.0
  clamp_u(:,:,:) = 0.0 ; clamp_v(:,:,:) = 0.0
  s_hat_u(:,:,:) = 0.0 ; s_hat_v(:,:,:) = 0.0
  ! Compute rho gradients
  if (use_EOS) then
    call calc_isoneutral_slopes(G, GV, US, h, e, tv, dt*CS%kappa_smooth, use_stanley, slope_x, slope_y, &
                              drdx_u=drdx_u, drdy_v=drdy_v, drdz_u=drdz_u, drdz_v=drdz_v, halo=3)
  else
    call calc_layered_density_gradients(G, GV, US, h, e, drdx_u, drdy_v, drdz_u, drdz_v, halo=3, &
                                        min_dist_from_boundary=CS%min_dist_from_boundary)
  endif

  ! Interpolate the rho gradients to the center point
  call center_grad_rho(drdx_u, drdy_v, drdx_c, drdy_c, G, GV, CS)

  if (slope_inputs) then
    ! Interpolate the vertical density gradient to center points and build sine-form
    ! slope components, s = grad(rho) / |grad3(rho)|, following the bounded slope
    ! definition in calc_isoneutral_slopes; |s| <= 1 pointwise by construction.
    ! This construction works with any vertical coordinate: the gradients come from
    ! calc_isoneutral_slopes (EOS mode) or calc_layered_density_gradients (layered mode).
    call center_drdz(drdz_u, drdz_v, drdz_c, G, GV, CS)
    do K=1, nz+1
      do j=js-shift-1,je+shift+1 ; do i=is-shift-1,ie+shift+1
        mag_grad3 = sqrt( US%Z_to_L**2*(drdx_c(i,j,K)**2 + drdy_c(i,j,K)**2) + drdz_c(i,j,K)**2 )
        if (mag_grad3 > 0.0) then
          s_x_c(i,j,K) = drdx_c(i,j,K) / mag_grad3
          s_y_c(i,j,K) = drdy_c(i,j,K) / mag_grad3
        endif
      enddo ; enddo
    enddo
    if (CS%upsilon_form == UPSILON_SLOPE_CLAMP) then
      ! Input-space bound (Perezhogin): clamp the slope-VECTOR magnitude the network
      ! sees at slope_max, so the regularization acts on the inputs rather than on the
      ! output rescaling. Clamping the vector magnitude (not each component) keeps the
      ! pointwise limit identical to KHTH_SLOPE_MAX and makes the stencil norm bounded
      ! by 3*slope_max, consistent with the factor-3 convention used elsewhere.
      do K=1, nz+1
        do j=js-shift-1,je+shift+1 ; do i=is-shift-1,ie+shift+1
          s_mag = sqrt( s_x_c(i,j,K)**2 + s_y_c(i,j,K)**2 )
          if (s_mag > CS%slope_max) then
            s_x_c(i,j,K) = s_x_c(i,j,K) * (CS%slope_max / s_mag)
            s_y_c(i,j,K) = s_y_c(i,j,K) * (CS%slope_max / s_mag)
          endif
        enddo ; enddo
      enddo
    endif
  endif

  ! Compute velocity gradients at center points
  call vel_gradients(u, v, G, GV, dudx, dudy, dvdx, dvdy, CS)

  ! Post diagnostics
  if (CS%id_drdx_u > 0) call post_data(CS%id_drdx_u, drdx_u, CS%diag)
  if (CS%id_drdy_v > 0) call post_data(CS%id_drdy_v, drdy_v, CS%diag)

  if (CS%id_drdz_u > 0) call post_data(CS%id_drdz_u, drdz_u, CS%diag)
  if (CS%id_drdz_v > 0) call post_data(CS%id_drdz_v, drdz_v, CS%diag)

  if (CS%id_drdx_c > 0) call post_data(CS%id_drdx_c, drdx_c, CS%diag)
  if (CS%id_drdy_c > 0) call post_data(CS%id_drdy_c, drdy_c, CS%diag)

  ! Compute the density fluxes at center points using the ANN.
  do K = 2, nz
    m = 0
    do j = js-1, je+1 ; do i = is-1, ie+1
      m = m + 1
      if (slope_inputs) then
        ! The density-gradient stencils are replaced by stencils of the bounded slopes.
        ! Their normalized pattern is approximately unchanged (exactly, where the
        ! stratification is uniform across the stencil), so the network trained on
        ! normalized density gradients applies without retraining.
        drdx_local(:,:) = s_x_c(i-shift:i+shift,j-shift:j+shift,K)
        drdy_local(:,:) = s_y_c(i-shift:i+shift,j-shift:j+shift,K)
      else
        drdx_local(:,:) = drdx_c(i-shift:i+shift,j-shift:j+shift,K)
        drdy_local(:,:) = drdy_c(i-shift:i+shift,j-shift:j+shift,K)
      endif
      ! Take the velocity gradients below the interface K
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
          rho_grad_mag = (rho_grad_mag + drdx_local(ii,jj)*drdx_local(ii,jj)) + &
                         drdy_local(ii,jj)*drdy_local(ii,jj)
          vel_grad_mag = ((vel_grad_mag + sh_xx_local(ii,jj)*sh_xx_local(ii,jj)) + &
                          sh_xy_local(ii,jj)*sh_xy_local(ii,jj)) + &
                         vort_local(ii,jj)*vort_local(ii,jj)
        enddo
      enddo
      vel_grad_mag = sqrt(vel_grad_mag) + vel_grad_neglect
      if (slope_inputs) then
        rho_grad_mag = sqrt(rho_grad_mag) + slope_neglect
        ! Limit the stencil-aggregate slope, as GM limits the slope with KHTH_SLOPE_MAX.
        ! The factor of 3 converts the pointwise limit to the 9-point stencil norm.
        s_norm = min(rho_grad_mag, 3.0*CS%slope_max)
        ! Here norm_y rescales the ANN output directly to Upsilon [Z L-1 T-1 ~> s-1].
        norm_y(i,j) = s_norm * vel_grad_mag
      else
        rho_grad_mag = sqrt(rho_grad_mag) + rho_grad_neglect
        norm_y(i,j) = rho_grad_mag * vel_grad_mag
        if (CS%upsilon_form == UPSILON_STENCIL_GRAD) rho_norm_c(i,j,K) = rho_grad_mag
      endif

      ! Normalize inputs
      drdx_local(:,:) = drdx_local(:,:) / rho_grad_mag
      drdy_local(:,:) = drdy_local(:,:) / rho_grad_mag

      sh_xx_local(:,:) = sh_xx_local(:,:)/ vel_grad_mag
      sh_xy_local(:,:) = sh_xy_local(:,:)/ vel_grad_mag
      vort_local(:,:) = vort_local(:,:)/ vel_grad_mag

      ! Prepare input vector for ANN
      x(m,1:stencil_points) = RESHAPE(drdx_local, (/stencil_points/))
      x(m,stencil_points+1:2*stencil_points) = RESHAPE(drdy_local, (/stencil_points/))
      x(m,2*stencil_points+1:3*stencil_points) = RESHAPE(sh_xx_local, (/stencil_points/))
      x(m,3*stencil_points+1:4*stencil_points) = RESHAPE(sh_xy_local, (/stencil_points/))
      x(m,4*stencil_points+1:5*stencil_points) = RESHAPE(vort_local, (/stencil_points/))

    enddo ; enddo

    ! Call the ANN
    call ANN_apply_array_sio(nij, x,y, CS%ann_rho_flux)

    m=0
    do j = js-1, je+1 ; do i = is-1, ie+1
      m=m+1
      ! Dimensionalize the output. The factors applied here must match the
      ! nondimensionalization used when the network was trained; this is
      ! an implicit contract with the training procedure.
      yy(:) = ((y(m,:) * norm_y(i,j)) * G%areaT(i,j)) * CS%ann_coeff

      if (slope_inputs) then
        ! Here yy is already the bounded velocity-scale streamfunction Upsilon, with
        ! |Upsilon| <= |ANN| * 3*slope_max * vel_grad_mag * areaT * ann_coeff, so no
        ! flux clamp is needed. The ANN outputs -u'rho', so we negate as below.
        Upsx_c(i,j,K) = -yy(1)
        Upsy_c(i,j,K) = -yy(2)
      else
        ! Clamp ANN output to prevent extreme values
        yy(1) = max(-CS%flux_clamp, min(CS%flux_clamp, yy(1)))
        yy(2) = max(-CS%flux_clamp, min(CS%flux_clamp, yy(2)))

        ! The sign convention is that ANN outputs -u'rho', so we negate.
        Fx_c(i,j,K) = -yy(1)
        Fy_c(i,j,K) = -yy(2)
      endif

    enddo ; enddo
  enddo

  ! Interpolate the density fluxes (or Upsilon in SLOPE_STENCIL mode) to u and v points.
  if (slope_inputs) then
    call center2uv(Upsx_c, Upsy_c, Ups_u, Ups_v, G, GV)
  else
    call center2uv(Fx_c, Fy_c, Fx_u, Fy_v, G, GV)
  endif
  ! The stencil norms are interpolated with the same weights as the fluxes they scale,
  ! so the flux-to-norm ratio in the conversion below remains bounded.
  if (CS%upsilon_form == UPSILON_STENCIL_GRAD) &
    call center2uv(rho_norm_c, rho_norm_c, rho_norm_u, rho_norm_v, G, GV)

  do K=2, nz
    do j=js,je ; do I=is-1,ie
      ! In layered mode, skip interfaces at the bottom or surface
      if (.not. use_EOS) then
        dist_from_bot_a = e(i,j,K) - e(i,j,nz+1)
        dist_from_bot_b = e(i+1,j,K) - e(i+1,j,nz+1)
        dist_from_sfc_a = e(i,j,1) - e(i,j,K)
        dist_from_sfc_b = e(i+1,j,1) - e(i+1,j,K)
        if (dist_from_bot_a < CS%min_dist_from_boundary .or. &
            dist_from_bot_b < CS%min_dist_from_boundary .or. &
            dist_from_sfc_a < CS%min_dist_from_boundary .or. &
            dist_from_sfc_b < CS%min_dist_from_boundary) then
          sfn_u(I,j,K) = 0.0
          cycle
        endif
      endif
      select case (CS%upsilon_form)
      case (UPSILON_LOCAL_GRAD)
        ! Skip if density gradient is too small (prevents division by ~zero)
        mag_grad = sqrt( US%Z_to_L**2*drdx_u(I,j,K)**2 + drdz_u(I,j,K)**2 )
        if (mag_grad < CS%mag_grad_floor) then
          sfn_u(I,j,K) = 0.0
          cycle
        endif
        ! Velocity-scale (grid-independent) streamfunction, Upsilon in Ferrari et al. 2010.
        Upsilon_u = (Fx_u(I,j,K)/mag_grad) * G%OBCmaskCu(I,j)
      case (UPSILON_STENCIL_GRAD)
        ! As LOCAL_GRAD, but the horizontal part of the divisor is the stencil norm
        ! that scales the ANN numerator, so Upsilon stays bounded even where the
        ! local gradient vanishes.
        mag_grad = sqrt( US%Z_to_L**2*rho_norm_u(I,j,K)**2 + drdz_u(I,j,K)**2 )
        ! Limit the effective stencil-aggregate slope rho_norm_u/mag_grad, as GM limits
        ! the slope with KHTH_SLOPE_MAX; the factor of 3 converts the pointwise limit
        ! to the 9-point stencil norm.
        mag_grad = max(mag_grad, rho_norm_u(I,j,K) / (3.0*CS%slope_max))
        if (mag_grad < CS%mag_grad_floor) then
          sfn_u(I,j,K) = 0.0
          cycle
        endif
        s_hat_u(I,j,K) = rho_norm_u(I,j,K) / mag_grad
        Upsilon_u = (Fx_u(I,j,K)/mag_grad) * G%OBCmaskCu(I,j)
      case (UPSILON_SLOPE_STENCIL, UPSILON_SLOPE_CLAMP)
        ! Upsilon was predicted directly from slope stencils; no division is needed.
        Upsilon_u = Ups_u(I,j,K) * G%OBCmaskCu(I,j)
      end select
      Ups_u(I,j,K) = Upsilon_u
      if (abs(Upsilon_u) > CS%Upsilon_clamp) clamp_u(I,j,K) = 1.0
      Upsilon_u = max(-CS%Upsilon_clamp, min(CS%Upsilon_clamp, Upsilon_u))
      sfn_u(I,j,K) = Upsilon_u * G%dy_Cu(I,j)

    enddo ; enddo
    do J=js-1,je ; do i=is,ie
      if (.not. use_EOS) then
        dist_from_bot_a = e(i,j,K) - e(i,j,nz+1)
        dist_from_bot_b = e(i,j+1,K) - e(i,j+1,nz+1)
        dist_from_sfc_a = e(i,j,1) - e(i,j,K)
        dist_from_sfc_b = e(i,j+1,1) - e(i,j+1,K)
        if (dist_from_bot_a < CS%min_dist_from_boundary .or. &
            dist_from_bot_b < CS%min_dist_from_boundary .or. &
            dist_from_sfc_a < CS%min_dist_from_boundary .or. &
            dist_from_sfc_b < CS%min_dist_from_boundary) then
          sfn_v(i,J,K) = 0.0
          cycle
        endif
      endif
      select case (CS%upsilon_form)
      case (UPSILON_LOCAL_GRAD)
        ! Skip if density gradient is too small (prevents division by ~zero)
        mag_grad = sqrt( US%Z_to_L**2*drdy_v(i,J,K)**2 + drdz_v(i,J,K)**2 )
        if (mag_grad < CS%mag_grad_floor) then
          sfn_v(i,J,K) = 0.0
          cycle
        endif
        ! Velocity-scale (grid-independent) streamfunction, Upsilon in Ferrari et al. 2010.
        Upsilon_v = (Fy_v(i,J,K)/mag_grad) * G%OBCmaskCv(i,J)
      case (UPSILON_STENCIL_GRAD)
        ! As LOCAL_GRAD, but the horizontal part of the divisor is the stencil norm
        ! that scales the ANN numerator, so Upsilon stays bounded even where the
        ! local gradient vanishes.
        mag_grad = sqrt( US%Z_to_L**2*rho_norm_v(i,J,K)**2 + drdz_v(i,J,K)**2 )
        ! Limit the effective stencil-aggregate slope rho_norm_v/mag_grad, as GM limits
        ! the slope with KHTH_SLOPE_MAX; the factor of 3 converts the pointwise limit
        ! to the 9-point stencil norm.
        mag_grad = max(mag_grad, rho_norm_v(i,J,K) / (3.0*CS%slope_max))
        if (mag_grad < CS%mag_grad_floor) then
          sfn_v(i,J,K) = 0.0
          cycle
        endif
        s_hat_v(i,J,K) = rho_norm_v(i,J,K) / mag_grad
        Upsilon_v = (Fy_v(i,J,K)/mag_grad) * G%OBCmaskCv(i,J)
      case (UPSILON_SLOPE_STENCIL, UPSILON_SLOPE_CLAMP)
        ! Upsilon was predicted directly from slope stencils; no division is needed.
        Upsilon_v = Ups_v(i,J,K) * G%OBCmaskCv(i,J)
      end select
      Ups_v(i,J,K) = Upsilon_v
      if (abs(Upsilon_v) > CS%Upsilon_clamp) clamp_v(i,J,K) = 1.0
      Upsilon_v = max(-CS%Upsilon_clamp, min(CS%Upsilon_clamp, Upsilon_v))
      sfn_v(i,J,K) = Upsilon_v * G%dx_Cv(i,J)

    enddo ; enddo
  enddo

  call pass_vector(sfn_u, sfn_v, G%Domain)

  if (.not. slope_inputs) then
    ! Density fluxes are not part of the slope-input prediction paths.
    if (CS%id_Fx_c > 0) call post_data(CS%id_Fx_c, Fx_c, CS%diag)
    if (CS%id_Fy_c > 0) call post_data(CS%id_Fy_c, Fy_c, CS%diag)

    if (CS%id_Fx_u > 0) call post_data(CS%id_Fx_u, Fx_u, CS%diag)
    if (CS%id_Fy_v > 0) call post_data(CS%id_Fy_v, Fy_v, CS%diag)
  endif

  if (CS%id_sfn_u > 0) call post_data(CS%id_sfn_u, sfn_u, CS%diag)
  if (CS%id_sfn_v > 0) call post_data(CS%id_sfn_v, sfn_v, CS%diag)

  if (CS%id_Ups_u > 0) call post_data(CS%id_Ups_u, Ups_u, CS%diag)
  if (CS%id_Ups_v > 0) call post_data(CS%id_Ups_v, Ups_v, CS%diag)
  if (CS%id_Ups_clamp_u > 0) call post_data(CS%id_Ups_clamp_u, clamp_u, CS%diag)
  if (CS%id_Ups_clamp_v > 0) call post_data(CS%id_Ups_clamp_v, clamp_v, CS%diag)
  if (CS%id_s_hat_u > 0) call post_data(CS%id_s_hat_u, s_hat_u, CS%diag)
  if (CS%id_s_hat_v > 0) call post_data(CS%id_s_hat_v, s_hat_v, CS%diag)

  deallocate(drdx_local, drdy_local, dudx_local, dudy_local, dvdx_local, dvdy_local, &
             sh_xx_local, sh_xy_local, vort_local)
  deallocate(x, y, yy)
end subroutine meso_sfn_ANN_compute

!> Interpolate density gradients from u- and v-points to cell centers.
subroutine center_grad_rho(drdx_u, drdy_v, drdx_c, drdy_c, G, GV, CS)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(MESO_SFN_ANN_CS),                intent(inout) :: CS !< Control structure for thickness_flux_ann
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(in) :: drdx_u !< Zonal density gradient
                                                                    !! at u-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(in) :: drdy_v !< Meridional density gradient
                                                                    !! at v-points [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1), intent(inout) :: drdx_c !< Zonal density gradient
                                                                       !! at center [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1), intent(inout) :: drdy_c !< Meridional density gradient
                                                                       !! at center [R L-1 ~> kg m-4]

  integer :: i, j, k, is, ie, js, je, nz, shift

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  shift = (CS%ann_window-1)/2

  do K=1, nz+1
    do j=js-shift-1,je+shift+1 ; do i=is-shift-1,ie+shift+1
      drdx_c(i,j,K) = 0.5 * (drdx_u(i-1,j,K) * G%mask2dCu(i-1,j) + drdx_u(i,j,K) * G%mask2dCu(i,j)) * G%mask2dT(i,j)
      drdy_c(i,j,K) = 0.5 * (drdy_v(i,j-1,K) * G%mask2dCv(i,j-1) + drdy_v(i,j,K) * G%mask2dCv(i,j)) * G%mask2dT(i,j)
    enddo ; enddo
  enddo

end subroutine center_grad_rho

!> Interpolate vertical density gradients from u- and v-points to cell centers.
!! The average is renormalized by the number of wet contributing points so that
!! land neighbors do not artificially weaken the stratification.
subroutine center_drdz(drdz_u, drdz_v, drdz_c, G, GV, CS)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  type(MESO_SFN_ANN_CS),                intent(in) :: CS !< Control structure for thickness_flux_ann
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(in) :: drdz_u !< Vertical density gradient
                                                                     !! at u-points [R Z-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(in) :: drdz_v !< Vertical density gradient
                                                                     !! at v-points [R Z-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1), intent(inout) :: drdz_c !< Vertical density gradient
                                                                       !! at center [R Z-1 ~> kg m-4]

  real :: wsum  ! Number of wet velocity points contributing to the average [nondim]
  integer :: i, j, k, is, ie, js, je, nz, shift

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke

  shift = (CS%ann_window-1)/2

  do K=1, nz+1
    do j=js-shift-1,je+shift+1 ; do i=is-shift-1,ie+shift+1
      wsum = ((G%mask2dCu(I-1,j) + G%mask2dCu(I,j)) + (G%mask2dCv(i,J-1) + G%mask2dCv(i,J)))
      drdz_c(i,j,K) = ((drdz_u(I-1,j,K)*G%mask2dCu(I-1,j) + drdz_u(I,j,K)*G%mask2dCu(I,j)) + &
                       (drdz_v(i,J-1,K)*G%mask2dCv(i,J-1) + drdz_v(i,J,K)*G%mask2dCv(i,J))) &
                      / max(wsum, 1.0) * G%mask2dT(i,j)
    enddo ; enddo
  enddo

end subroutine center_drdz

!> Interpolate two fields from cell centers to u- and v-points.
subroutine center2uv(var1_c, var2_c, var1_u, var2_v, G, GV)
  type(ocean_grid_type),                      intent(in)    :: G      !< Ocean grid structure
  type(verticalGrid_type),                    intent(in)    :: GV     !< Vertical grid structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1),  intent(in)    :: var1_c !< Variable at center points [arbitrary]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1),  intent(in)    :: var2_c !< Variable at center points [arbitrary]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(inout)   :: var1_u !< Variable at u points [arbitrary]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(inout)   :: var2_v !< Variable at v points [arbitrary]

  integer :: i, j, k, is, ie, js, je, nz

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  do K=1, nz+1
    do j=js,je ; do I=is-1,ie
      var1_u(I,j,K) = 0.5 * (var1_c(i,j,K) * G%mask2dT(i,j) + var1_c(i+1,j,K) * G%mask2dT(i+1,j)) * G%mask2dCu(I,j)
    enddo ; enddo
    do J=js-1,je ; do i=is,ie
      var2_v(i,J,K) = 0.5 * (var2_c(i,j,K) * G%mask2dT(i,j) + var2_c(i,j+1,K) * G%mask2dT(i,j+1)) * G%mask2dCv(i,J)
    enddo ; enddo
  enddo

end subroutine center2uv
!> Calculates the velocity gradients at the center points in 3D.
subroutine vel_gradients(u, v, G, GV, dudx, dudy, dvdx, dvdy, CS)
  type(ocean_grid_type),                     intent(in)    :: G   !< Ocean grid structure
  type(verticalGrid_type),                   intent(in)    :: GV  !< The ocean's vertical grid structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)),intent(in)    :: u   !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)),intent(in)    :: v   !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(out) :: dudx !< du/dx [T-1 ~> s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(out) :: dvdy !< dv/dy [T-1 ~> s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(out) :: dudy !< du/dy [T-1 ~> s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(out) :: dvdx !< dv/dx [T-1 ~> s-1]
  type(MESO_SFN_ANN_CS), intent(in) :: CS !< Control structure for thickness_flux_ann

  ! Corner points
  real, dimension(SZIB_(G), SZJB_(G),SZK_(GV)) :: dudy_q !< du/dy at corner points [T-1 ~> s-1]
  real, dimension(SZIB_(G), SZJB_(G),SZK_(GV)) :: dvdx_q !< dv/dx at corner points [T-1 ~> s-1]
  integer :: is, ie, js, je
  integer :: nz
  integer :: i, j, k
  integer :: shift

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke

  shift = (CS%ann_window-1)/2

  do k=1, nz
    ! Calculate velocity gradients at center points directly.
    do j=js-shift-1,je+shift+1 ; do i=is-shift-1,ie+shift+1
      dudx(i,j,k) = G%IdxT(i,j)* (u(I,j,k) * G%mask2dCu(I,j)   - u(I-1,j,k) * G%mask2dCu(I-1,j)) * G%mask2dT(i,j)
      dvdy(i,j,k) = G%IdyT(i,j)* (v(i,J,k) * G%mask2dCv(i,J)   - v(i,J-1,k) * G%mask2dCv(i,J-1)) * G%mask2dT(i,j)
    enddo ; enddo

    ! Calculate velocity gradients at corner points.
    ! Bounds extend one further on the lower side than the center-point loop above
    ! because the 4-point corner-to-center interpolation below reads indices (I-1,J-1).
    do j=js-shift-2,je+shift+1 ; do i=is-shift-2,ie+shift+1
      dvdx_q(I,J,k) = G%IdxBu(I,J)*(v(i+1,J,k)  - v(i,J,k) ) * G%mask2dBu(I,J)
      dudy_q(I,J,k) = G%IdyBu(I,J)*(u(I,j+1,k)  - u(I,j,k) ) * G%mask2dBu(I,J)
      !
    enddo ; enddo

    ! interpolate corner grads to center points
    do j = js-shift-1, je+shift+1; do i = is-shift-1, ie+shift+1
      dvdx(i,j,k) =  0.25 * (((dvdx_q(I,J,k) + dvdx_q(I-1,J,k)) + dvdx_q(I,J-1,k)) + &
                              dvdx_q(I-1,J-1,k)) * G%mask2dT(i,j)
      dudy(i,j,k) =  0.25 * (((dudy_q(I,J,k) + dudy_q(I-1,J,k)) + dudy_q(I,J-1,k)) + &
                              dudy_q(I-1,J-1,k)) * G%mask2dT(i,j)
    enddo; enddo
  enddo
end subroutine vel_gradients
!> Compute density gradients from fixed layer densities and interface heights
!! This is a workaround for running the ANN parameterization in pure layered
!! mode (USE_EOS=False) where calc_isoneutral_slopes won't work.
subroutine calc_layered_density_gradients(G, GV, US, h, e, &
                                          drdx_u, drdy_v, drdz_u, drdz_v, halo, min_dist_from_boundary)
  type(ocean_grid_type),                       intent(in)  :: G
  type(verticalGrid_type),                     intent(in)  :: GV
  type(unit_scale_type),                       intent(in)  :: US
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),   intent(in)  :: h   ! Layer thickness [Z ~> m]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1), intent(in)  :: e   ! Interface heights [Z ~> m]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(out) :: drdx_u ! [R L-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(out) :: drdy_v ! [R L-1 ~> kg m-4]
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)+1), intent(out) :: drdz_u ! [R Z-1 ~> kg m-4]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)+1), intent(out) :: drdz_v ! [R Z-1 ~> kg m-4]
  integer,                                      intent(in)  :: halo
  real,                                         intent(in)  :: min_dist_from_boundary  ! Threshold for boundaries [Z]

  ! Local variables
  real :: drho_k       ! Density difference across interface K [R]
  real :: dz_u, dz_v   ! Vertical length scale at u,v points [Z]
  real :: dedx, dedy   ! Interface slope [Z L-1]
  real :: h_neglect    ! Small thickness [H]
  real :: dist_from_bot_a, dist_from_bot_b  ! Distance from interface to bottom [Z]
  real :: dist_from_sfc_a, dist_from_sfc_b  ! Distance from interface to surface [Z]

  integer :: i, j, k, is, ie, js, je, nz

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
  h_neglect = GV%H_subroundoff
  ! Initialize to zero
  drdx_u(:,:,:) = 0.0
  drdy_v(:,:,:) = 0.0
  drdz_u(:,:,:) = 0.0
  drdz_v(:,:,:) = 0.0

  ! Loop over interfaces (K=1 is surface, K=nz+1 is bottom)
  do K = 2, nz
    ! Density jump across this interface (from GV%Rlay - the target layer densities)
    drho_k = GV%Rlay(k) - GV%Rlay(k-1)  ! [R ~> kg m-3]

    ! --- U-points (zonal gradients) ---
    do j = js-halo, je+halo
      do I = is-1-halo, ie+halo

        ! Check if interface is above bottom on both sides
        ! e is negative (depth), e(nz+1) is the bottom, e(1) is the surface
        dist_from_bot_a = e(i,j,K)   - e(i,j,nz+1)
        dist_from_bot_b = e(i+1,j,K) - e(i+1,j,nz+1)
        dist_from_sfc_a = e(i,j,1)   - e(i,j,K)
        dist_from_sfc_b = e(i+1,j,1) - e(i+1,j,K)

        if (dist_from_bot_a > min_dist_from_boundary .and. &
            dist_from_bot_b > min_dist_from_boundary .and. &
            dist_from_sfc_a > min_dist_from_boundary .and. &
            dist_from_sfc_b > min_dist_from_boundary .and. &
            G%mask2dCu(I,j) > 0.5) then

          ! Average thickness of layers above and below interface at u-point
          dz_u = 0.25 * GV%H_to_Z * ( &
                (h(i,j,k-1) + h(i,j,k)) + (h(i+1,j,k-1) + h(i+1,j,k)) )
          dz_u = max(dz_u, GV%H_to_Z * h_neglect)

          ! Interface height gradient (slope of isopycnal)
          dedx = (e(i+1,j,K) - e(i,j,K)) * G%IdxCu(I,j)  ! [Z L-1]

          ! In a layered model with tilted interfaces:
          !   dρ/dx comes from the interface tilt: (Δρ across interface) * (∂η/∂x) / Δz
          !   dρ/dz is simply Δρ / Δz
          !
          ! Physical interpretation: if interface tilts up to the east,
          ! denser water (layer k) is lifted, creating ∂ρ/∂x < 0

          drdx_u(I,j,K) = drho_k * dedx / dz_u   ! [R L-1]
          drdz_u(I,j,K) = - drho_k / dz_u          ! [R Z-1]

          ! Apply land mask
          drdx_u(I,j,K) = drdx_u(I,j,K) * (G%mask2dCu(I,j) * G%mask2dT(i,j) * G%mask2dT(i+1,j))
          drdz_u(I,j,K) = drdz_u(I,j,K) * (G%mask2dCu(I,j) * G%mask2dT(i,j) * G%mask2dT(i+1,j))
        else
          ! Interface is at/near bottom or surface on at least one side - set gradients to zero
          drdx_u(I,j,K) = 0.0
          drdz_u(I,j,K) = 0.0
        endif
      enddo
    enddo

    ! --- V-points (meridional gradients) ---
    do J = js-1-halo, je+halo
      do i = is-halo, ie+halo
        ! Check if interface is above bottom and below surface on both sides
        dist_from_bot_a = e(i,j,K)   - e(i,j,nz+1)
        dist_from_bot_b = e(i,j+1,K) - e(i,j+1,nz+1)
        dist_from_sfc_a = e(i,j,1)   - e(i,j,K)
        dist_from_sfc_b = e(i,j+1,1) - e(i,j+1,K)

        if (dist_from_bot_a > min_dist_from_boundary .and. &
            dist_from_bot_b > min_dist_from_boundary .and. &
            dist_from_sfc_a > min_dist_from_boundary .and. &
            dist_from_sfc_b > min_dist_from_boundary .and. &
            G%mask2dCv(i,J) > 0.5) then

          ! Interface is a real isopycnal on both sides - compute gradients
          dz_v = 0.25 * GV%H_to_Z * ( &
                 (h(i,j,k-1) + h(i,j,k)) + (h(i,j+1,k-1) + h(i,j+1,k)) )
          dz_v = max(dz_v, GV%H_to_Z * h_neglect)

          ! Interface height gradient
          dedy = (e(i,j+1,K) - e(i,j,K)) * G%IdyCv(i,J)  ! [Z L-1]

          drdy_v(i,J,K) = drho_k * dedy / dz_v   ! [R L-1]
          drdz_v(i,J,K) = -drho_k / dz_v         ! [R Z-1]
        else
          ! Interface is at/near bottom or surface on at least one side, or masked
          drdy_v(i,J,K) = 0.0
          drdz_v(i,J,K) = 0.0
        endif
      enddo
    enddo
  enddo

end subroutine calc_layered_density_gradients

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
  character(len=40) :: upsilon_form_str ! The string value of MESO_SFN_UPSILON_FORM
# include "version_variable.h"

  CS%diag => diag

  call log_version(param_file, mdl, version, &
       "ANN-based mesoscale streamfunction parameterization.")

  ! We don't need to check if use is true, because this is only called if it is.
  call get_param(param_file, mdl, "MESO_SFN_ANN_COEFF", CS%ann_coeff, &
                      "Coefficient to multiply the mesoscale streamfunction ANN output by", default=1.0, units="nondim")

  call get_param(param_file, mdl, "KD_SMOOTH", CS%kappa_smooth, &
                 "A diapycnal diffusivity that is used to interpolate "//&
                 "more sensible values of T & S into thin layers.", &
                 units="m2 s-1", default=1.0e-6, scale=GV%m2_s_to_HZ_T)
  call get_param(param_file, mdl, "MESO_SFN_MIN_DIST_BOUNDARY", CS%min_dist_from_boundary, &
             "Minimum distance from surface or bottom for interface to be considered valid "//&
             "for density gradient calculations in layered mode.", &
             units="m", default=50.0, scale=US%m_to_Z)
  call get_param(param_file, mdl, "MESO_SFN_MAG_GRAD_FLOOR", CS%mag_grad_floor, &
             "Minimum density gradient magnitude below which the streamfunction "//&
             "is set to zero to avoid division by near-zero values.", &
             units="kg m-4", default=1.0e-10, scale=US%kg_m3_to_R*US%Z_to_m)
  call get_param(param_file, mdl, "MESO_SFN_FLUX_CLAMP", CS%flux_clamp, &
             "Maximum magnitude of ANN output density flux before conversion "//&
             "to streamfunction.", &
             units="kg m-2 s-1", default=1.0e2, scale=US%kg_m3_to_R*US%m_to_L*US%T_to_s)
  call get_param(param_file, mdl, "MESO_UPSILON_CLAMP", CS%Upsilon_clamp, &
             "Maximum magnitude of the velocity-scale mesoscale streamfunction "//&
             "(Upsilon in Ferrari et al. 2010).", &
             units="m2 s-1", default=15., scale=US%m_to_L*US%m_to_Z*US%T_to_s)
  call get_param(param_file, mdl, "MESO_SFN_UPSILON_FORM", upsilon_form_str, &
             "The form of the conversion from the ANN prediction to the velocity-scale "//&
             "streamfunction Upsilon: \n"//&
             "\t LOCAL_GRAD - divide the predicted density flux by the local in-plane \n"//&
             "\t\t density gradient magnitude (original, unbounded form). \n"//&
             "\t STENCIL_GRAD - divide the predicted density flux by a gradient magnitude \n"//&
             "\t\t whose horizontal part is the stencil norm that scales the ANN output, \n"//&
             "\t\t so that Upsilon is bounded; the effective slope is limited by \n"//&
             "\t\t MESO_SFN_SLOPE_MAX. \n"//&
             "\t SLOPE_STENCIL - feed the ANN stencils of bounded (sine-form) isopycnal \n"//&
             "\t\t slopes and interpret the rescaled output directly as Upsilon, with \n"//&
             "\t\t no division. \n"//&
             "\t SLOPE_CLAMP - as SLOPE_STENCIL, but the slope-vector magnitude fed to \n"//&
             "\t\t the network is clamped at MESO_SFN_SLOPE_MAX, so the bound acts on \n"//&
             "\t\t the network inputs rather than on the output rescaling.", &
             default="LOCAL_GRAD")
  select case (uppercase(trim(upsilon_form_str)))
    case ("LOCAL_GRAD")    ; CS%upsilon_form = UPSILON_LOCAL_GRAD
    case ("STENCIL_GRAD")  ; CS%upsilon_form = UPSILON_STENCIL_GRAD
    case ("SLOPE_STENCIL") ; CS%upsilon_form = UPSILON_SLOPE_STENCIL
    case ("SLOPE_CLAMP")   ; CS%upsilon_form = UPSILON_SLOPE_CLAMP
    case default
      call MOM_error(FATAL, "meso_sfn_ANN_init: Unrecognized value of "//&
                     "MESO_SFN_UPSILON_FORM = "//trim(upsilon_form_str))
  end select
  call get_param(param_file, mdl, "MESO_SFN_SLOPE_MAX", CS%slope_max, &
             "The pointwise-equivalent isopycnal slope at which the bounded forms of "//&
             "the flux-to-streamfunction conversion limit Upsilon; the limit on the "//&
             "stencil-aggregate slope is 3 times this value. Values of 1 or larger "//&
             "effectively disable the limiter. Comparable to KHTH_SLOPE_MAX.", &
             units="nondim", default=0.01, scale=US%L_to_Z, &
             do_not_log=(CS%upsilon_form==UPSILON_LOCAL_GRAD))
  call get_param(param_file, mdl, "MESO_SFN_ANN_WINDOW", CS%ann_window, &
                      "Number of horizontal grid points to use in the thickness flux ANN window", default=1)
  ! The stencil reads drdx_c(i-shift:i+shift,...) with shift=(ann_window-1)/2.
  ! halo=3 is requested in meso_sfn_ANN_compute, so shift must be <= 3.
  if (CS%ann_window < 1 .or. CS%ann_window > 3 .or. mod(CS%ann_window, 2) == 0) &
    call MOM_error(FATAL, "meso_sfn_ANN_init: MESO_SFN_ANN_WINDOW must be an odd integer in [1,3].")
  call get_param(param_file, mdl, "MESO_SFN_ANN_FILE", CS%ann_file_rho_flux, &
               "ANN parameters for prediction of density fluxes (netcdf)", &
               default="INPUT/rho_flux.nc")
  call ANN_init(CS%ann_rho_flux, CS%ann_file_rho_flux)

  ! Register diagnostic fields
  CS%id_drdx_u = register_diag_field('ocean_model', 'meso_sfn_drdx_u', diag%axesCui, Time, &
           'Zonal density gradient used in meso sfn', &
           'kg m-4', conversion=US%R_to_kg_m3*US%m_to_L)
  CS%id_drdy_v = register_diag_field('ocean_model', 'meso_sfn_drdy_v', diag%axesCvi, Time, &
           'Meridional density gradient used in meso sfn', &
           'kg m-4', conversion=US%R_to_kg_m3*US%m_to_L)
  CS%id_drdz_u = register_diag_field('ocean_model', 'meso_sfn_drdz_u', diag%axesCui, Time, &
           'Vertical density gradient at u points used in meso sfn', &
           'kg m-4', conversion=US%R_to_kg_m3*US%m_to_Z)
  CS%id_drdz_v = register_diag_field('ocean_model', 'meso_sfn_drdz_v', diag%axesCvi, Time, &
           'Vertical density gradient at v points used in meso sfn', &
           'kg m-4', conversion=US%R_to_kg_m3*US%m_to_Z)
  CS%id_drdx_c = register_diag_field('ocean_model', 'meso_sfn_drdx_c', diag%axesTi, Time, &
           'Zonal density gradient at center points used in meso sfn', &
           'kg m-4', conversion=US%R_to_kg_m3*US%m_to_L)
  CS%id_drdy_c = register_diag_field('ocean_model', 'meso_sfn_drdy_c', diag%axesTi, Time, &
           'Meridional density gradient at center points used in meso sfn', &
           'kg m-4', conversion=US%R_to_kg_m3*US%m_to_L)
  CS%id_Fx_c = register_diag_field('ocean_model', 'meso_sfn_flux_x_c', diag%axesTi, Time, &
           'Zonal density flux at center points used in meso sfn', &
           'kg m-2 s-1', conversion=US%R_to_kg_m3*US%L_to_m*US%s_to_T)
  CS%id_Fy_c = register_diag_field('ocean_model', 'meso_sfn_flux_y_c', diag%axesTi, Time, &
           'Meridional density flux at center points used in meso sfn', &
           'kg m-2 s-1', conversion=US%R_to_kg_m3*US%L_to_m*US%s_to_T)
  CS%id_Fx_u = register_diag_field('ocean_model', 'meso_sfn_flux_x_u', diag%axesCui, Time, &
           'Zonal density flux at u points used in meso sfn', &
           'kg m-2 s-1', conversion=US%R_to_kg_m3*US%L_to_m*US%s_to_T)
  CS%id_Fy_v = register_diag_field('ocean_model', 'meso_sfn_flux_y_v', diag%axesCvi, Time, &
           'Meridional density flux at v points used in meso sfn', &
           'kg m-2 s-1', conversion=US%R_to_kg_m3*US%L_to_m*US%s_to_T)
  CS%id_sfn_u = register_diag_field('ocean_model', 'meso_sfn_unlim_u', diag%axesCui, Time, &
           'Meso-scale volume streamfunction at u points', &
           'm3 s-1', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_sfn_v = register_diag_field('ocean_model', 'meso_sfn_unlim_v', diag%axesCvi, Time, &
           'Meso-scale volume streamfunction at v points', &
           'm3 s-1', conversion=US%Z_to_m*US%L_to_m**2*US%s_to_T)
  CS%id_Ups_u = register_diag_field('ocean_model', 'meso_sfn_Upsilon_u', diag%axesCui, Time, &
           'Velocity-scale mesoscale streamfunction at u points, before clamping', &
           'm2 s-1', conversion=US%L_to_m*US%Z_to_m*US%s_to_T)
  CS%id_Ups_v = register_diag_field('ocean_model', 'meso_sfn_Upsilon_v', diag%axesCvi, Time, &
           'Velocity-scale mesoscale streamfunction at v points, before clamping', &
           'm2 s-1', conversion=US%L_to_m*US%Z_to_m*US%s_to_T)
  CS%id_Ups_clamp_u = register_diag_field('ocean_model', 'meso_sfn_Upsilon_clamp_u', diag%axesCui, Time, &
           'Mask of points where MESO_UPSILON_CLAMP limits the streamfunction at u points', 'nondim')
  CS%id_Ups_clamp_v = register_diag_field('ocean_model', 'meso_sfn_Upsilon_clamp_v', diag%axesCvi, Time, &
           'Mask of points where MESO_UPSILON_CLAMP limits the streamfunction at v points', 'nondim')
  if (CS%upsilon_form == UPSILON_STENCIL_GRAD) then
    CS%id_s_hat_u = register_diag_field('ocean_model', 'meso_sfn_s_hat_u', diag%axesCui, Time, &
             'Effective stencil-aggregate slope in the bounded Upsilon conversion at u points', &
             'nondim', conversion=US%Z_to_L)
    CS%id_s_hat_v = register_diag_field('ocean_model', 'meso_sfn_s_hat_v', diag%axesCvi, Time, &
             'Effective stencil-aggregate slope in the bounded Upsilon conversion at v points', &
             'nondim', conversion=US%Z_to_L)
  endif

  CS%initialized = .true.
end subroutine meso_sfn_ANN_init
!> Finalizes the meso-scale streamfunction ANN parameterization
!!
subroutine meso_sfn_ANN_end(CS)
  type(MESO_SFN_ANN_CS), intent(inout) :: CS !< Control structure

  ! Deallocate anything that needs to be.
  call ANN_end(CS%ann_rho_flux)

end subroutine meso_sfn_ANN_end

end module MOM_meso_sfn_ANN
