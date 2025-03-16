# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
from cuquantum.densitymat import (
    full_matrix_product,
    LocalDenseMatrixOperator,
    DenseMixedState,
    WorkStream,
    Operator,
    GPUCallback,
)

dev = cp.cuda.Device()  # get current device
props = cp.cuda.runtime.getDeviceProperties(dev.id)
print("===== device info ======")
print("GPU-local-id:", dev.id)
print("GPU-name:", props["name"].decode())
print("GPU-clock:", props["clockRate"])
print("GPU-memoryClock:", props["memoryClockRate"])
print("GPU-nSM:", props["multiProcessorCount"])
print("GPU-major:", props["major"])
print("GPU-minor:", props["minor"])

np.random.seed(42)
cp.random.seed(42)

# Parameters
dtype = "complex128"
hilbert_space_dims = (4, 3, 5, 7)
num_modes = len(hilbert_space_dims)
modes_string = "abcdefghijkl"[: len(hilbert_space_dims)]
dissipation_strength = 0.1
batch_size = 1


def take_complex_conjugate_transpose(arr):
    return arr.transpose(tuple(range(num_modes, 2 * num_modes)) + tuple(range(0, num_modes)) + (2 * num_modes,)).conj()


# create a WorkStream, holding workspace and stream (cp.cuda.get_current_stream() at the time of construction by default)
ctx = WorkStream()

#############
# Hamiltonian
#############

### create a time-dependent full matrix operator which updates only the diagonal entries of the matrix operator

## create the time-independent component which will also serve as GPU array containing the MatrixOperator data
# this needs to be an F-ordered GPU cp.ndarray, the shape of which reflects the local hilbert space dimensions and the batch size
static_matrix_op_arr = cp.empty((*hilbert_space_dims, *hilbert_space_dims, batch_size), dtype=dtype, order="F")
static_matrix_op_arr[:] = cp.random.normal(size=static_matrix_op_arr.shape)
if "complex" in dtype:
    static_matrix_op_arr[:] += 1j * (cp.random.normal(size=static_matrix_op_arr.shape))
# make it hermitian
static_matrix_op_arr += take_complex_conjugate_transpose(static_matrix_op_arr)

## create the time-dependent component as a callback function
# first prepare some data that will be used inside the callback
# create an array holding the diagonal elements of the kronecker sum of number operators for each site
number_operator_diag = cp.zeros(hilbert_space_dims, dtype=dtype, order="F")
identity_diag = cp.ones(hilbert_space_dims, dtype=dtype)
for i, dim in enumerate(hilbert_space_dims):
    number_operator_diag += cp.einsum("ijkl," + "ijkl"[i] + "->ijkl", identity_diag, cp.arange(dim))

# flatten and add dummy batch dimension
number_operator_diag = number_operator_diag.reshape((np.prod(hilbert_space_dims), 1))
# indices for indexing into diagonal elements of the
diag_inds = cp.diag_indices(np.prod(hilbert_space_dims))


# since this is a time-dependent matrix-operator, we need to define a callback function to update its elements
# `DenseLocalMatrixOperator` only supports callbacks which expect cp.ndarray as second (callback parameters) and third positional argument (the buffer containing the operators elements) and modify the third positional argument in place
def matrix_op_callback(t: float, args: cp.ndarray, arr: cp.ndarray) -> None:
    """
    This function updates the GPU array `arr` for a time-dependent matrix operator in place.
    In particular the diagonal is updated with an elementwise trigonometric function of the diagonal elements of the sum each sites number operator
    `args` is a cp.ndarray with shape (number of parameters, batch size)
    `arr` is a cp.ndarray with shape (*hilbert_space_dims, *hilbert_space_dims, batch size)
    """
    # when this callback function is invoked through the library it executes within a cupy.cuda.Stream context
    # since cp.ndarray operations are implicitly stream ordered, we don't need to use the stream explicitly here though

    stream = cp.cuda.get_current_stream()
    # the device on which `arr` is located will be the current cp.cuda.Device
    assert cp.cuda.Device() == arr.device

    # extract the coefficients, portable between batched and unbatched MatrixOperator
    omegas = args[0, :]

    # hilbert_space_dims available from enclosing scope, could also be infered from `arr` shape
    matrix_dim = np.prod(hilbert_space_dims)
    batch_size = arr.shape[-1]
    # the matrix operator is passed a tensor, in order to update the diagonal entries, we need to reshape into a matrix (no copy)
    matricized_arr_view = arr.reshape(matrix_dim, matrix_dim, batch_size)
    # use diagonal indices from enclosing scope instead of recomputing them here
    # if batch dimension > 1, broadcasting over the batch index is implicit in th
    matricized_arr_view[diag_inds] = cp.sin(cp.pi * t * omegas * number_operator_diag)
    print(f"User-defined callback function updated matrix operator in place.")
    return


## create a LocalDenseMatrixOperator instance with the static and dynamic components
# the callback needs to be wrapped as `GPUCallback` and we need to specify that it is an inplace callback
hamiltonian_matrix_op = LocalDenseMatrixOperator(
    static_matrix_op_arr, callback=GPUCallback(matrix_op_callback, is_inplace=True)
)
print("Created time-dependent dense matrix Hamiltonian defining the closed quantum system.")

## compose an `OperatorTerm` describing the closed system evolution under the time-dependent Hamiltonian defined by hamiltonian_matrix_op
# `full_matrix_product` accepts a tuples of `LocalDenseMatrixOperators` and booleans as optional positional arguments
# the first boolean specifies whether the first element of the tuple (a `LocalDenseMatrixOperator`) acts on the bra (`True`) or ket (`False``, default) modes of the quantum state
# the second boolean specifies whether to apply the complex conjugate transpose of the `LocalDenseMatrixOperator` (defaults to `False`)
hamiltonian_term = full_matrix_product((hamiltonian_matrix_op, False, False), coeff=1)
print("Created `OperatorTerm` consisting of the time-dependent Hamiltonian.")
# express the commutator [H, rho] where rho is the quantum state and H the Hamiltonian:

#############
# Dissipators
#############

## create a non-hermitian dense matrix operator as a dissipative term
matrix_op_arr = cp.empty((*hilbert_space_dims, *hilbert_space_dims, batch_size), dtype=dtype, order="F")
matrix_op_arr[:] = cp.random.normal(size=static_matrix_op_arr.shape)
if "complex" in dtype:
    matrix_op_arr[:] = 1j * (cp.random.normal(size=static_matrix_op_arr.shape))
dissipative_matrix_op = LocalDenseMatrixOperator(matrix_op_arr)
print("Created a dense matrix operator defining a single Lindblad dissipator.")
## compose the dissipative part of the Lindblad superoperator
# `LocalDenseMatrixOperators` are applied to the quantum state in the order they are passed (left to right below)
dissipative_ket_terms = full_matrix_product(
    (dissipative_matrix_op, False, True), (dissipative_matrix_op, False, False), coeff=-dissipation_strength / 2
)
# dual method flips the order and whether the components of the product are applied to bra or ket modes of the quantum state
dissipative_bra_terms = dissipative_ket_terms.dual()
dissipative_two_sided_terms = full_matrix_product(
    (dissipative_matrix_op, False, False), (dissipative_matrix_op, True, True), coeff=dissipation_strength
)
dissipative_terms = dissipative_two_sided_terms + dissipative_bra_terms + dissipative_ket_terms
print("Created `OperatorTerm`s defining the dissipative terms in the Lindblad equation.")

#############
# Liouvillian
#############

closed_system_op = Operator(hilbert_space_dims, (hamiltonian_term, -1j, False), (hamiltonian_term, +1j, True))
print("Created closed system Liouvillian Super-Operator.")

## compose the full Lindbladian as an `Operator`
liouvillian = closed_system_op + Operator(hilbert_space_dims, (dissipative_terms,))
print("Created Lindbladian superoperator from closed system liouvillian and dissipative terms.")


##########################
# Initial and final states
##########################

# get random normalized initial mixed state
rho = DenseMixedState(ctx, hilbert_space_dims, batch_size, dtype)
rho.attach_storage(cp.empty(rho.storage_size, dtype=dtype))
rho_arr = rho.view()
rho_arr[:] = cp.random.normal(size=rho_arr.shape)
if "complex" in dtype:
    rho_arr[:] += 1j * (cp.random.normal(size=rho_arr.shape))
rho_arr += take_complex_conjugate_transpose(rho_arr)
rho_arr /= rho.trace()
print("Created a Haar random normalized mixed quantum state.")
# get empty output state
rho_out = rho.clone(cp.zeros_like(rho.storage, order="F"))

print("Created zero initialized output state.")

#################
# Operator action
#################

# callback parameters
t = 0.3
num_params = 1
callback_args = cp.empty((num_params, batch_size), dtype=dtype, order="F")
callback_args[:] = cp.random.rand(num_params, batch_size)

# compute action of lindbladian on quantum state
liouvillian.prepare_action(ctx, rho)
print("Prepared the action of a  time-dependent, dissipative superoperator based on full matrices on a mixed quantum state.")
liouvillian.compute_action(t, callback_args, rho, rho_out)
print("Computed the action of a time-dependent, dissipative superoperator based on full matrices on a mixed quantum state.")

print("Finished computation and exit.")
