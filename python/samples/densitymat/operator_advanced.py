# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
from cuquantum.densitymat import (
    tensor_product,
    DenseMixedState,
    DenseOperator,
    WorkStream,
    Operator,
    OperatorAction,
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


# define the shape of the composite tensor product space
hilbert_space_dims = (4, 5, 2, 6, 3, 7)  # six quantum degrees of freedom

# define some elementary tensor operators
A = np.random.random((hilbert_space_dims[2],) * 2)  # one-body elementary tensor operator


b = np.random.random(  # two-body elementary tensor operator, arbitrary strides also supported
    (
        hilbert_space_dims[3],
        hilbert_space_dims[5],
    )
    * 2
)


# We can wrap the NDArray in a TensorOperator, which is useful if we want to use the same tensor multiple times.
B = DenseOperator(b)


# one-body elementary tensor callback operator
def c_callback(t, args):
    return np.random.random((hilbert_space_dims[1],) * 2)


c_representative_tensor = c_callback(0.0, ())
# making an instance of TensorOperator is optional, we can also pass the tuple of (c_representative_tensor, c_callback) directly below in palce of `C`
C = DenseOperator(c_representative_tensor, c_callback)

print("Defined elementary operators A, B, C.")


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
    coeff=my_callback,  # time-dependent parameterized coefficient represented by a user-defined callback function
)

# construct different operator terms
term1 = ab + bc  # an OperatorTerm composed of a sum of two tensor operator products
term1 += bc  # `OperatorTerm` also supports in-place addition

term2 = tensor_product(  # an operator term composed of a single elementary tensor operator
    (
        C,  # elementary tensor operator
        (1,),  # quantum degrees of freedom it acts on
        (False,),  # operator action duality (side: left/right) for each quantum degree of freedom
    ),
    coeff=1.0,  # constant (static) coefficient
)

print("Created OperatorTerms term1 and term2.")

# construct the Hamiltonian operator from two operator terms
hamiltonian = Operator(
    hilbert_space_dims,  # shape of the composite tensor space
    (term1,),  # first operator term with a default coefficient 1.0
    (
        term2,
        my_callback,
    ),  # second operator term modulated by a parameterized time-dependent coefficient (callback function)
)

print("Created Hamiltonian Operator from term1 and term2.")

# construct the Liouvillian for the von Neumann equation
liouvillian = (
    hamiltonian - hamiltonian.dual()
)  # Hamiltonian action on the left minus Hamiltonian action on the right: [H, *]

print("Created Liouvillian Operator from Hamiltonian.")

# open a work stream over a CUDA stream
my_stream = cp.cuda.Stream()
ctx = WorkStream(stream=my_stream)

# construct the Liouvillian action for a single quantum state
liouvillian_action = OperatorAction(ctx, (liouvillian,))

print("Created Liouvillian OperatorAction from Liouvillian.")

# create a mixed quantum state (density matrix) with zero initialized data buffer
batch_size = 1
rho0 = DenseMixedState(ctx, hilbert_space_dims, batch_size, "complex128")
slice_shape, slice_offsets = rho0.local_info
rho0.attach_storage(cp.zeros(rho0.storage_size, dtype=rho0.dtype))
# set storage to a Haar random unnormalized state
# for MGMN execution, the data buffer may be larger than the locally stored slice of the state
# the view method returns a tensor shaped view on the local slice (the full state for single-GPU execution)
rho0.view()[:] = cp.random.normal(size=slice_shape) + (
    1j * cp.random.normal(size=slice_shape)
)
# for non-random initialization and MGMN execution, we would use slice_offsets to determine how to set the elements
norm = rho0.norm().get()[()]
rho0.inplace_scale(np.sqrt(1 / norm))
assert np.isclose(rho0.norm().get()[()], 1)

print(
    "Created a Haar random normalized mixed quantum state (not physical due to lack of hermitianity)."
)

# two ways of creating another mixed quantum state of the same shape and init it to zero
rho1 = rho0.clone(cp.zeros_like(rho0.storage))
rho2 = DenseMixedState(ctx, hilbert_space_dims, batch_size, "complex128")
rho2.allocate_storage()

print("Created a zero-initialized output mixed quantum state.")


# prepare operator action on a mixed quantum state
liouvillian_action.prepare(ctx, (rho0,))

print("Prepared Liouvillian action through OperatorAction.prepare.")

# set a parameter for the callback function to some value
omega = 2.4

# compute the operator action on a given quantum state
liouvillian_action.compute(
    0.0,  # time value
    (omega,),  # user-defined parameters
    (rho0,),  # input quantum state
    rho1,  # output quantum state
)

print("Computed Liouvillian action through OperatorAction.compute.")

# alternatively, prepare the operator action directly via the operator
liouvillian.prepare_action(ctx, rho0)

print("Prepared Liouvillian action through Operator.prepare_action.")

# compute the operator action directly via the operator
liouvillian.compute_action(
    0.0,  # time value
    (omega,),  # user-defined parameters
    rho0,  # input quantum state
    rho2,  # output quantum state
)
# OperatorAction.compute and Operator.compute_action are accumulative, so rho0p should now be equivalent to twice the action of the Liouvillian.

print("Computed Liouvillian action through Operator.compute_action.")

# prepare the operator expectation value computation
liouvillian.prepare_expectation(ctx, rho0)

print("Prepared expectation through Operator.prepare_expectation.")

# compute the operator expectation value
expval = liouvillian.compute_expectation(
    0.0,  # time value
    (omega,),  # user-defined parameters
    rho1,  # input quantum state
)

print("Computed expectation through Operator.compute_expectation.")

# we can compute the operator action again without another prepare call
liouvillian.compute_action(
    0.0,  # time value
    (omega,),  # user-defined parameters
    rho0,  # input quantum state
    rho1,  # output quantum state
)

print("Computed Liouvillian action through Operator.compute_action.")

# assuming we want to some other task in between it may be useful to release the workspace
ctx.release_workspace()

# releasing the workspace has no user-facing side effects
# for example, we do not need to reissue prepare calls
liouvillian.compute_action(
    0.0,  # time value
    (omega,),  # user-defined parameters
    rho0,  # input quantum state
    rho2,  # output quantum state
)

print("Computed Liouvillian action through Operator.compute_action after releasing workspace.")

# synchronize the work stream
my_stream.synchronize()

print("Finished computation and exit.")
