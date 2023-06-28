!> General use ANN in MOM6
module MOM_ann
    
! This file is part of MOM6. See LICENSE.md for the license

! Here add use to the bits that will be needed
! use MOM_* only : *
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_file_parser,   only : get_param, log_version, param_file_type

!
implicit none ; private

#include <MOM_memory.h>

public ann_init, ann !, ann_end

!> Type for a layer
type, private :: layer_type; private 
    integer :: output_width 
    integer :: input_width 

    real, allocatable :: A(:,:)
    real, allocatable :: b(:)

end type layer_type


!> Control structure/type for ANN
type, public :: ANN_CS ; private 
    ! Parameters
    integer :: num_layers ! number of layers 
    integer, allocatable :: layer_sizes(:) ! size of each layer, including input

    ! How do have arbitrary number of weight matrices ? 
    ! for arbitrary number of layers.
    ! for now choose 3, as it works for me. 
    
    type(layer_type), allocatable :: layers(:)
    !real, allocatable :: A0(:,:), A1(:,:), A2(:,:) 
    !real, allocatable :: b0(:), b1(:), b2(:) 

end type ANN_CS

contains 

!> Init function 
! Read parameters and register output fields.
subroutine ann_init(CS, use_ANN, param_file)
    type(ANN_CS), intent(inout) :: CS !< ANN control structure.
    logical, intent(out) :: use_ANN !< If true, turns on ANN module.
    type(param_file_type),   intent(in)    :: param_file !< Parameter file parser structure.

    integer :: i
#include "version_variable.h"
    character(len=40) :: mdl = "MOM_ann"

    call log_version(param_file, mdl, version, "")

    call get_param(param_file, mdl, "USE_ANN", use_ANN, &
                   "If true, turns on the ANN", default=.false.)
    if (.not. use_ANN) return

    ! Read in number of layers and their sizes
    CS%num_layers = 4
    
    allocate(CS%layer_sizes(CS%num_layers))
    CS%layer_sizes = [2, 24, 24, 2]
    
    ! Allocate the layers
    allocate(CS%layers(CS%num_layers-1)) ! since this contains the matrices that move info from one layer to another, it is one size smaller. 
    
    ! Allocate the A, b matrices for each layers
    do i = 1,CS%num_layers-1
        CS%layers(i)%output_width = CS%layer_sizes(i+1)
        CS%layers(i)%input_width = CS%layer_sizes(i)

        allocate(CS%layers(i)%A(CS%layers(i)%input_width, CS%layers(i)%output_width), source=0.)
        allocate(CS%layers(i)%b(CS%layers(i)%output_width), source=0.)
    enddo

end subroutine ann_init

! Main function to be called 
! This function should take a vector x and output a vector y.
! The sizes of these vector would have to be adjusted dynamically. 
! The role of ann will be to successively apply the dense and relu layers
! as the architecture dictates. 
subroutine ann(x, y, CS)
    type(ANN_CS), intent(in) :: CS ! ANN control structure

    real, dimension(CS%layer_sizes(1)), intent(in) :: x ! input 
    real, dimension(CS%layer_sizes(CS%num_layers)), intent(out) :: y ! output 
    
    real, allocatable :: x_1(:), x_2(:) ! intermediate states. 
    integer :: i

    ! start by allocating and assigning the input
    allocate(x_1(CS%layer_sizes(1)), source=0.)
    x_1 = x

    do i = 1, CS%num_layers -1 
        ! allocate and assign states
        allocate(x_2(CS%layer_sizes(i+1)), source=0.)

        ! Call the dense operations (matmul)
        call dense(CS%layers(i)%A, CS%layers(i)%b, x_1, x_2, CS%layer_sizes(i+1), CS%layer_sizes(i) )

        ! Call the activation functions if needed

        ! swap allocations and move forward 
        deallocate(x_1)
        allocate(x_1(CS%layer_sizes(i+1)), source=0.)
        x_1 = x_2
        deallocate(x_2)
    enddo
    ! Finally assign output, which goes out. 
    y = x_1


end subroutine ann


! A dense layer
subroutine dense(A, b, x, y, m, n)
    integer, intent(in) :: m, n ! sizes, where m is output and n is input from this operation.

    real, dimension(n), intent(in) :: x
    real, dimension(m), intent(out) :: y
    
    real, dimension(n, m), intent(in) :: A
    real, dimension(m), intent(in) :: b

    integer :: i, j
    real :: temp_y

    ! Do a y = matmul(x, A)
    ! Following the row-vector convention from JAX.
    do j=1,m
        y(j) = 0.
        do i=1,n
            ! Multiply by kernel
            y(j) = y(j) + x(i) * A(i, j)
        enddo
        ! Add bias
        y(j) = y(j) + b(j)
    enddo


end subroutine dense

! A relu activation layer
subroutine relu(x, y, m)
    integer, intent(in) :: m ! size of input and output vectors
    real, dimension(m), intent(in) ::x
    real, dimension(m), intent(out) ::y

    integer :: i 

    do i=1,m
        y(i) = max(x(i), 0.0)
    enddo

end subroutine relu

! A end function to deallocate arrays that were allocated here can be added.
! subroutine ann_end()

! end subroutine ann()


end module MOM_ann

