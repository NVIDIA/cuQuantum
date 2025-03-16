# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
from cuquantum.densitymat import (
    tensor_product,
    DenseOperator,
    DenseMixedState,
    WorkStream,
    Operator,
    OperatorAction,
    CPUCallback,
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
print("========================")

###
# This example shows how batching can be applied to the components from which we define `Operator`s and their coefficients.
# Batching can be applied to `DenseOperator`, `MultidiagonalOperator`, `DenseLocalMatrixOperator` in the same manner, here we focus on `DenseOperator` as well as coefficients.
###

## define the shape of the composite tensor product space
hilbert_space_dims = (4, 3, 2, 5, 2, 3)  # six quantum degrees of freedom

##define the batch size
# note that there can only be a single batch size > 1 for all interacting objects in the API, i.e
# if the quantum state is batched, any components of the Operator / OperatorAction which is applied to the state need to either have the batch size or be unbatched (batch size of 1)
batch_size = 3

## define some elementary tensor operators
A_batched = DenseOperator(
    cp.random.random((hilbert_space_dims[2],) * 2 + (batch_size,))  # one-body batched elementary tensor operator
)

B_batched = DenseOperator(
    np.random.random(  # two-body batched elementary tensor operator
        (
            hilbert_space_dims[3],
            hilbert_space_dims[5],
        )
        * 2
        + (batch_size,)
    )
)

C = DenseOperator(np.random.random((hilbert_space_dims[1],) * 2))  # one-body non-batched elementary tensor operator

# Creating a densed operator is optional, but recommended if the operator appears in multiple terms
D = np.random.random((hilbert_space_dims[4],) * 2)  # one-body non-batched elementary tensor operator


print("Defined elementary operators A, B, C, D.")


## define a callback function for time-dependent batched coefficients
num_params = 1  # number of parameters, not counting in batching passed to all callback functions


def a_cpu_batched_coefficient_callback(t: float, args: np.ndarray):
    # `args`` is an array of shape (number of parameters x batch size)
    # this function is expected to return an ndarray of size batch size
    omegas = args[0, :]  # this callback function is written to handle arbitrary batch sizes
    return np.sin(np.pi * omegas * t)  # return the scalar parameterized coefficient at time t


## construct tensor products of elementary tensor operators
ab = tensor_product(
    (
        A_batched,  # batched elementary tensor operator
        (2,),  # quantum degrees of freedom it acts on
    ),
    (
        B_batched,  # batched elementary tensor operator
        (3, 5),  # quantum degrees of freedom it acts on
    ),
    coeff=1.3,  # constant non-batched coefficient
)

bc_batched = tensor_product(
    (
        B_batched,  # elementary tensor operator
        (3, 5),  # quantum degrees of freedom it acts on
    ),
    (
        C,  # non-batched elementary tensor operator
        (1,),  # quantum degrees of freedom it acts on
    ),
    coeff=CPUCallback(
        a_cpu_batched_coefficient_callback
    ),  # time-dependent parameterized coefficient represented by a user-defined callback function
    batch_size=batch_size,  # we need to explicitly specify batch size of the coefficients returned by the callback function here since it cannot be inferred from the callback
)

cd_batched = tensor_product(
    (
        C,  # non-batched elementary tensor operator
        (1,),  # quantum degrees of freedom it acts on
    ),
    (
        D,  # non-batched elementary tensor operator
        (4,),  # quantum degrees of freedom it acts on
    ),
    coeff=np.linspace(1.4, 2.4, batch_size),  # batched constant coefficients
)

c = tensor_product(  # an operator term composed of a single elementary tensor operator without a batched coefficient
    (
        C,  # elementary tensor operator
        (1,),  # quantum degrees of freedom it acts on
    ),
)

# construct different operator terms
term1 = ab + bc_batched  # an operator term composed of a sum of two tensor operator products

term2 = c

term3 = cd_batched

print("Created OperatorTerms term1 through term3.")

# construct the Hamiltonian operator from two operator terms
hamiltonian = Operator(
    hilbert_space_dims,  # shape of the composite tensor space
    (term1, 1.2),  # first operator term with a static coefficient
    (
        term2,
        CPUCallback(a_cpu_batched_coefficient_callback),
    ),  # second operator term modulated by a batched parameterized time-dependent coefficient (callback function)
)

# `append` method allows to pass `coeff` and `duality` keyword arguments when adding an `OperatorTerm` to an `Operator` in place
# the += overload will use the default values for these keyword ar
hamiltonian.append(term3, coeff=np.arange(1, batch_size + 1))

print("Created Hamiltonian Operator from term1 through term3.")

# construct the Liouvillian for the von Neumann equation
liouvillian = hamiltonian - hamiltonian.dual()  # Hamiltonian action on the left minus Hamiltonian action on the right: [H, *]

print("Created Liouvillian Operator from Hamiltonian.")

# open a work stream
ctx = WorkStream()

# construct the Liouvillian action for a single quantum state
liouvillian_action = OperatorAction(ctx, (liouvillian,))

print("Created Liouvillian OperatorAction from Liouvillian.")

# create a batched mixed quantum state (density matrix) with zero initialized data buffer
rho0 = DenseMixedState(ctx, hilbert_space_dims, batch_size, "complex128")
slice_shape, slice_offsets = rho0.local_info
rho0.attach_storage(cp.zeros(rho0.storage_size, dtype=rho0.dtype))
# set storage to a Haar random unnormalized state
# for MGMN execution, the data buffer may be larger than the locally stored slice of the state
# the view method returns a tensor shaped view on the local slice (the full state for single-GPU execution)
rho0.view()[:] = cp.random.normal(size=slice_shape) + (1j * cp.random.normal(size=slice_shape))
# for non-random initialization and MGMN execution, we would use slice_offsets to determine how to set the elements
norm = rho0.norm().get()
rho0.inplace_scale(np.sqrt(1 / norm))
assert np.allclose(rho0.norm().get(), 1)

print("Created a Haar random normalized mixed quantum state (not physical due to lack of hermitianity).")

# two ways of creating another mixed quantum state of the same shape and init it to zero
rho1 = rho0.clone(cp.zeros_like(rho0.storage))
rho2 = DenseMixedState(ctx, hilbert_space_dims, batch_size, "complex128")
rho2.allocate_storage()

print("Created a zero-initialized output mixed quantum state.")

# prepare operator action on a mixed quantum state
liouvillian_action.prepare(ctx, (rho0,))

print("Prepared Liouvillian action through OperatorAction.prepare.")

# set the parameters for the callback function to some value
# to avoid data movement to GPU in every compute call, we do it once here (relevant for larger batch sizes)
callback_args = cp.arange(1, 1 + (batch_size) * num_params).reshape(num_params, batch_size)
# to avoid reallocation of Fortran-ordered cp.ndarray for every compute call, we do it once here (relevant for larger batch sizes)
callback_args = cp.asfortranarray(callback_args)

# compute the operator action on a given quantum state
liouvillian_action.compute(
    0.0,  # time value
    callback_args,  # user-defined parameters, preferable passed as 2D ndarray of shape num_params x batch_size
    (rho0,),  # input quantum state
    rho1,  # output quantum state
)

print("Computed Liouvillian action through OperatorAction.compute.")

# alternatively, prepare the operator action directly via the operator
liouvillian.prepare_action(ctx, rho0)

print("Prepared Liouvillian action through Operator.prepare_action.")

## compute the operator action directly via the operator

liouvillian.compute_action(
    0.0,  # time value
    callback_args,  # user-defined parameters, with batchin these need to be passed as 2D np.ndarray / cp.ndarray of shape num_params x batch_size
    rho0,  # input quantum state
    rho2,  # output quantum state
)
print("Computed Liouvillian action through Operator.compute_action.")
assert cp.allclose(rho1.view(), rho2.view())
print("Finished computation and exit.")
