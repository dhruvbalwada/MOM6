!> General use ANN in MOM6
module MOM_ann
    
! This file is part of MOM6. See LICENSE.md for the license

! Here add use to the bits that will be needed
! use MOM_* only : *
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_file_parser,   only : get_param, log_version, param_file_type
use MOM_io, only : MOM_read_data
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
    character(len=200) :: NNfile   ! The name of netcdf file having neural network shape function

    ! How do have arbitrary number of weight matrices ? 
    ! for arbitrary number of layers.
    ! for now choose 3, as it works for me. 
    
    type(layer_type), allocatable :: layers(:)
    real, allocatable :: input_norms(:), output_norms(:)
    
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
    character(len=1) :: A = 'A'
    character(len=1) :: b = 'b'
    character(len=1) :: layer_num_str
    character(len=3) :: matrix_name

#include "version_variable.h"
    character(len=40) :: mdl = "MOM_ann"

    call log_version(param_file, mdl, version, "")

    call get_param(param_file, mdl, "USE_ANN", use_ANN, &
                   "If true, turns on the ANN", default=.false.)
    if (.not. use_ANN) return

    ! Read in number of layers and their sizes
    call get_param(param_file, mdl, "ANN_num_layers", CS%num_layers, &
                   "Number of ANN layers", default=4)
    call get_param(param_file, mdl, "ANN_PARAMS_FILE", CS%NNfile, &
                   "ANN parameters netcdf input", default="not_specified")

    ! Read size of layers
    allocate(CS%layer_sizes(CS%num_layers))

    call MOM_read_data(CS%NNfile,"layer_sizes",CS%layer_sizes)
    !write (*,*) "layer sizes", CS%layer_sizes
    !CS%layer_sizes = [2, 24, 24, 2]

    ! Read norms
    allocate(CS%input_norms(CS%layer_sizes(1)))
    allocate(CS%output_norms(CS%layer_sizes(CS%num_layers)))

    call MOM_read_data(CS%NNfile, 'input_norms', CS%input_norms)
    call MOM_read_data(CS%NNfile, 'output_norms', CS%output_norms)
    
    ! Allocate the layers
    allocate(CS%layers(CS%num_layers-1)) ! since this contains the matrices that move info from one layer to another, it is one size smaller. 
    
    ! Allocate the A, b matrices for each layers
    do i = 1,CS%num_layers-1
        CS%layers(i)%output_width = CS%layer_sizes(i+1)
        CS%layers(i)%input_width = CS%layer_sizes(i)

        !write(layer_num_str, '(I0)') i-1
        ! note that the order of dimensions is reversed
        ! https://stackoverflow.com/questions/47085101/netcdf-startcount-exceeds-dimension-bound
        allocate(CS%layers(i)%A(CS%layers(i)%output_width, CS%layers(i)%input_width), source=0.)
        matrix_name = trim(A) // trim(layer_num_str)
        !write (*,*) "Reading", matrix_name, "Size", CS%layers(i)%input_width, CS%layers(i)%output_width
        ! How to read in 2D?
        ! What is the weird format? 
        call MOM_read_data(CS%NNfile, matrix_name, CS%layers(i)%A, &
                            (/1,1,1,1/),(/CS%layers(i)%output_width,CS%layers(i)%input_width,1,1/))

        !write (*,*) "Reading", matrix_name, CS%layers(i)%A


        allocate(CS%layers(i)%b(CS%layers(i)%output_width), source=0.)
        matrix_name = trim(b) // trim(layer_num_str)
        !write (*,*) "Reading", matrix_name
        call MOM_read_data(CS%NNfile, matrix_name, CS%layers(i)%b)
        !write (*,*) "Reading", matrix_name, CS%layers(i)%b
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

    !write(*,*) x
    ! start by allocating and assigning the input
    allocate(x_1(CS%layer_sizes(1)), source=0.)
    x_1 = x
    ! Normalize input
    do i = 1,CS%layer_sizes(1)
        x_1(i) = x_1(i) / CS%input_norms(i)
    enddo
    
    !write(*,*) "input", x_1, "size", CS%layer_sizes(1)

    do i = 1, CS%num_layers -1 
        ! allocate and assign states
        allocate(x_2(CS%layer_sizes(i+1)), source=0.)
        !write(*,*) "input", x_1, "output", x_2

        ! Call the dense operations (matmul)
        call dense(CS%layers(i)%A, CS%layers(i)%b, x_1, x_2, CS%layer_sizes(i+1), CS%layer_sizes(i) )
        !write(*,*) "input", x_1, "output", x_2

        ! Call the activation functions (if needed)
        if (i < CS%num_layers-1) then
            call relu(x_2, CS%layer_sizes(i+1))
        endif

        ! swap allocations and move forward 
        deallocate(x_1)
        allocate(x_1(CS%layer_sizes(i+1)), source=0.)
        x_1 = x_2
        deallocate(x_2)
    enddo
    ! Finally assign output, which goes out. 
    y = x_1

    ! un-normalize output
    do i = 1, CS%layer_sizes(CS%num_layers)
        y(i) = y(i) * CS%output_norms(i)
    enddo




end subroutine ann


! A dense layer
subroutine dense(A, b, x, y, m, n)
    integer, intent(in) :: m, n ! sizes, where m is output and n is input from this operation.

    real, dimension(n), intent(in) :: x
    real, dimension(m), intent(out) :: y
    
    real, dimension(m, n), intent(in) :: A
    real, dimension(m), intent(in) :: b

    integer :: i, j

    ! Do a y = matmul(x, A)
    ! JAX follows row vector convention
    ! in FORTRAN the matrices are transposed.
    do j=1,m ! ouput 
        y(j) = 0.
        do i=1,n ! input
            ! Multiply by kernel
            y(j) = y(j) + ( x(i) * A(j, i) )
        enddo
        ! Add bias
        y(j) = y(j) + b(j)
    enddo


end subroutine dense

! A relu activation layer
subroutine relu(x, n)
    integer, intent(in) :: n ! size of input and output vectors
    real, dimension(n), intent(inout) :: x
    !real, dimension(m), intent(out) ::y

    integer :: i 

    do i=1,n
        x(i) = max(x(i), 0.0)
    enddo

end subroutine relu

! A end function to deallocate arrays that were allocated here can be added.
! subroutine ann_end()

! end subroutine ann()


end module MOM_ann