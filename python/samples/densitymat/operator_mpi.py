# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
from mpi4py import MPI

from cuquantum.densitymat import (
    tensor_product,
    DensePureState,
    DenseOperator,
    WorkStream,
    OperatorTerm,
    Operator,
    OperatorAction,
)

NUM_DEVICES = cp.cuda.runtime.getDeviceCount()
rank = MPI.COMM_WORLD.Get_rank()
dev = cp.cuda.Device(rank % NUM_DEVICES)
dev.use()
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


# create Workstream on the current device
ctx = WorkStream(device_id=dev.id)

# setup MPI communicator
ctx.set_communicator(comm=MPI.COMM_WORLD.Dup(), provider="MPI")

# define the shape of the composite tensor product space
hilbert_space_dims = (4, 4, 4, 4, 4)  # six quantum degrees of freedom
batch_size = 2

# define some elementary tensor operators
identity = DenseOperator(np.eye(hilbert_space_dims[0], dtype="complex128"))
op_term = OperatorTerm(dtype="complex128")
for i in range(len(hilbert_space_dims)):
    op_term += tensor_product(
        (
            identity,
            (1,),
        )
    )
# This operator will just be proportional to the identity
op = Operator(hilbert_space_dims, (op_term,))
op_action = OperatorAction(ctx, (op,))


def set_ditstring(state, batch_index, ditstring: list):
    """
    Set's the state's coefficient at for the `batch_index`'th quantum state to the product state in the computational basis encoded by `ditstring`.
    """
    slice_shape, slice_offsets = state.local_info
    ditstring = np.asarray(
        ditstring
        + [
            batch_index,
        ],
        dtype="int",
    )
    ditstring_is_local = True
    state_inds = []
    for slice_dim, slice_offset, state_dit in zip(
        slice_shape, slice_offsets, ditstring
    ):
        ditstring_is_local = state_dit in range(slice_offset, slice_offset + slice_dim)
        if not ditstring_is_local:
            break
        else:
            state_inds.append(
                range(slice_offset, slice_offset + slice_dim).index(state_dit)
            )
    if ditstring_is_local:
        strides = (1,) + tuple(np.cumprod(np.array(slice_shape)[:-1]))
        ind = np.sum(strides * np.array(state_inds))
        state.storage[ind] = 1.0


# product states to be set for each batch state
global_ditstrings = [[0, 1, 3, 2, 0], [1, 0, 3, 2, 1]]

# make initial state
state = DensePureState(ctx, hilbert_space_dims, batch_size, "complex128")
required_buffer_size = state.storage_size
state.attach_storage(cp.zeros((required_buffer_size,), dtype="complex128"))
# set product states for each batch input state
for batch_ind in range(batch_size):
    set_ditstring(state, batch_ind, global_ditstrings[batch_ind])
# more ways to make a State instance
state_out = state.clone(cp.zeros(required_buffer_size, dtype="complex128"))
another_state = DensePureState(ctx, hilbert_space_dims, batch_size, "complex128")
another_state.allocate_storage()
# prepare and compute Operator action
op.prepare_action(ctx, state)
op.compute_action(0.0, [], state, state_out)
state_out_slice = state_out.view()
# compute Operator action
op_action.compute(
    0.0,
    [],
    [
        state,
    ],
    another_state,
)
# OperatorAction and Operator for this specific example have the same effect
assert cp.allclose(another_state.view(), state_out_slice)
