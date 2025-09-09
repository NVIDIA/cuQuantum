# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
from cuquantum.densitymat import tensor_product, DensePureState, DenseMixedState, WorkStream, Operator, OperatorAction, CPUCallback, OperatorSpectrumSolver, OperatorSpectrumConfig

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
print("========================")


def make_operator_hermitian(tensor: np.ndarray) -> np.ndarray:

    original_shape = tensor.shape
    rank = tensor.ndim

    if rank % 2 != 0:
        raise ValueError(f"Operator tensor must have an even rank, but got rank {rank}.")

    half_rank = rank // 2
    dim_out = np.prod(original_shape[:half_rank])
    dim_in = np.prod(original_shape[half_rank:])

    if dim_out != dim_in:
        raise ValueError(
            f"For an operator to be Hermitian, its input and output space "
            f"dimensions must match. Got input dim {dim_in} and output dim {dim_out}."
        )

    matrix = tensor.reshape(dim_out, dim_in)
    hermitian_matrix = 0.5 * (matrix + matrix.conj().T)
    hermitian_tensor = hermitian_matrix.reshape(original_shape)
    return hermitian_tensor

## TODO: Switch callback args format to 2D ndarray, although tuple is still supported for portability with batch_size > 1
# define the shape of the composite tensor product space
hilbert_space_dims = (4, 5, 2, 6, 3, 7)  # six quantum degrees of freedom

# define some elementary tensor operators
A = np.random.random((hilbert_space_dims[2],) * 2)  # one-body elementary tensor operator

B = np.random.random(  # two-body elementary tensor operator
    (
        hilbert_space_dims[3],
        hilbert_space_dims[5],
    )
    * 2
)

C = np.random.random((hilbert_space_dims[1],) * 2)  # one-body elementary tensor operator

print("Defined elementary operators A, B, C.")

A = make_operator_hermitian(A)
B = make_operator_hermitian(B)
C = make_operator_hermitian(C)


# define a scalar callback function (time-dependent coefficient)
def my_callback(t, args):  # args is an arbitrary list of real user-defined parameters
    _omega = args[0]
    return np.sin(np.pi * _omega * t)  # return the scalar parameterized coefficient at time t


# construct tensor products of elementary tensor operators
ab = tensor_product(
    (
        A,  # elementary tensor operator
        (2,),  # quantum degrees of freedom it acts on
    ),
    (
        B,  # elementary tensor operator
        (3, 5),  # quantum degrees of freedom it acts on
    ),
    coeff=1.0,  # constant (static) coefficient
)

bc = tensor_product(
    (
        B,  # elementary tensor operator
        (3, 5),  # quantum degrees of freedom it acts on
    ),
    (
        C,  # elementary tensor operator
        (1,),  # quantum degrees of freedom it acts on
    ),
    coeff=CPUCallback(my_callback),  # time-dependent parameterized coefficient represented by a user-defined callback function
)

# construct different operator terms
term1 = ab + bc  # an operator term composed of a sum of two tensor operator products

term2 = tensor_product(  # an operator term composed of a single elementary tensor operator
    (
        C,  # elementary tensor operator
        (1,),  # quantum degrees of freedom it acts on
    ),
)

print("Created OperatorTerms term1 and term2.")

# construct the Hamiltonian operator from two operator terms
hamiltonian = Operator(
    hilbert_space_dims,  # shape of the composite tensor space
    (term1,),  # first operator term with a default coefficient 1.0
    (
        term2,
        CPUCallback(my_callback),
    ),  # second operator term modulated by a parameterized time-dependent coefficient (callback function)
)

print("Created Hamiltonian Operator from term1 and term2.")


num_modes = len(hilbert_space_dims)
def take_complex_conjugate_transpose(arr):
    return arr.transpose(tuple(range(num_modes, 2 * num_modes)) + tuple(range(0, num_modes)) + (2 * num_modes,)).conj()


# Initialize computation context and prepare for eigenvalue spectrum calculation
ctx = WorkStream() 

# Define batch size for the eigenvalue spectrum calculation
batch_size = 1
hilbert_vol = np.prod(hilbert_space_dims)

# Define configuration options for the eigenvalue spectrum calculation
max_num_eigvals = 5
min_block_size = 4
max_buffer_ratio = 25
max_restarts = 10


# Create a sequence of pure states |ψ_i⟩
states = []
for _ in range(1,max_num_eigvals+1):
    state = DensePureState(ctx, hilbert_space_dims, batch_size, "float64")
    state.allocate_storage()
    state.storage[:] = cp.random.randn(hilbert_vol * batch_size)
    norm = state.norm()
    state.inplace_scale(1.0 / cp.sqrt(norm))
    states.append(state)
states = tuple(states)


# Configure the eigenvalue spectrum calculation using OperatorSpectrumConfig data class
config = OperatorSpectrumConfig(
    min_krylov_block_size=min_block_size,
    max_buffer_ratio=max_buffer_ratio,
    max_restarts=max_restarts
)


# Create an OperatorSpectrumSolver instance with the specified configuration
spectrum = OperatorSpectrumSolver(hamiltonian, "SA", True, config)

# Prepare the eigenvalue spectrum calculation
spectrum.prepare(ctx, states[0], max_num_eigvals=max_num_eigvals)

# Compute the eigenvalue spectrum
result = spectrum.compute(t=0.0, params=[0.0], states=states, tol=1e-10)

print("Spectrum computation completed!")
print(f"Computed eigenvalues: {result.evals.flatten()}")



