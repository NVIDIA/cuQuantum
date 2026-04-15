# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example demonstrating distributed density matrix operations using NCCL backend.

This example uses MPI to initialize an NCCL communicator for GPU-to-GPU communication.

Run with:
    mpirun -n <num_gpus> python operator_mpi_nccl_example.py

Requirements:
    - mpi4py
    - nvmath.distributed (for simple NCCL communicator initialization and cleanup, although users can also manage communicators manually)
    - CUDENSITYMAT_COMM_LIB environment variable pointing to the NCCL interface library
"""

import cupy as cp
import numpy as np
from mpi4py import MPI
import nvmath.distributed

from cuquantum.densitymat import (
    tensor_product,
    DensePureState,
    DenseOperator,
    MultidiagonalOperator,
    WorkStream,
    OperatorTerm,
    Operator,
    OperatorAction,
)


def ordered_print(msg):
    """Print messages in rank order for clean output."""
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    for i in range(size):
        if i == rank:
            print(f"Rank {i}: {msg}")
        MPI.COMM_WORLD.Barrier()


# Get device info and set up GPU
NUM_DEVICES = cp.cuda.runtime.getDeviceCount()
# Create a duplicate of COMM_WORLD for this application
comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
dev = cp.cuda.Device(rank % NUM_DEVICES)
dev.use()
props = cp.cuda.runtime.getDeviceProperties(dev.id)
ordered_print("===== device info ======")
ordered_print(f"GPU-local-id: {dev.id}")
ordered_print(f"GPU-name: {props['name'].decode()}")
ordered_print(f"GPU-clock: {props['clockRate']}")
ordered_print(f"GPU-memoryClock: {props['memoryClockRate']}")
ordered_print(f"GPU-nSM: {props['multiProcessorCount']}")
ordered_print(f"GPU-major: {props['major']}")
ordered_print(f"GPU-minor: {props['minor']}")
ordered_print("========================")

# Create WorkStream on the current device
ctx = WorkStream(device_id=dev.id)
ordered_print("Created WorkStream (execution context) on current device.")

# Setup NCCL communicator (initialized via nvmath.distributed)
nvmath.distributed.initialize(dev.id, comm, backends=["nccl"])
nccl_comm_ptr = nvmath.distributed.get_context().nccl_comm
ctx.set_communicator(nccl_comm_ptr, provider="NCCL")
ordered_print("Set NCCL communicator on execution context, enabling distributed computation.")

# Define the shape of the composite tensor product space
hilbert_space_dims = (4, 4, 4, 4, 4, 4)  # six quantum degrees of freedom
batch_size = 2

# Define some elementary tensor operators
identity = DenseOperator(np.eye(hilbert_space_dims[0], dtype="complex128"))
identity_sparse = MultidiagonalOperator(
    np.ones((hilbert_space_dims[0], 1), dtype="complex128"),
    offsets=[
        0,
    ],
)
ordered_print("Defined dense and sparse identity elementary operator.")

op_term = OperatorTerm(dtype="complex128")
for i in range(len(hilbert_space_dims) - 1):
    op_term += tensor_product(
        (
            identity,
            (i,),
        ),
        (
            identity_sparse,
            (i + 1,),
        ),
    )
# This operator will just be proportional to the identity
op = Operator(hilbert_space_dims, (op_term,))
ordered_print(
    "Created Operator corresponding to the action of products of nearest neighbor identity operators for a one-dimensional lattice."
)

op_action = OperatorAction(ctx, (op,))
ordered_print("Created OperatorAction from previously defined Operator.")


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
    for slice_dim, slice_offset, state_dit in zip(slice_shape, slice_offsets, ditstring):
        ditstring_is_local = state_dit in range(slice_offset, slice_offset + slice_dim)
        if not ditstring_is_local:
            break
        else:
            state_inds.append(range(slice_offset, slice_offset + slice_dim).index(state_dit))
    if ditstring_is_local:
        strides = (1,) + tuple(np.cumprod(np.array(slice_shape)[:-1]))
        ind = np.sum(strides * np.array(state_inds))
        state.storage[ind] = 1.0


# Product states to be set for each batch state
global_ditstrings = [[0, 1, 3, 2, 0, 2], [1, 0, 3, 2, 1, 0]]

# Make initial state
state = DensePureState(ctx, hilbert_space_dims, batch_size, "complex128")
ordered_print("Created dense quantum state instance with distributed storage.")
required_buffer_size = state.storage_size
state.attach_storage(cp.zeros((required_buffer_size,), dtype="complex128"))
ordered_print("Attach local storage slice to dense quantum state instance.")

# Set product states for each batch input state
for batch_ind in range(batch_size):
    set_ditstring(state, batch_ind, global_ditstrings[batch_ind])
ordered_print("Set distributed quantum state to desired product state.")

# More ways to make a State instance
state_out = state.clone(cp.zeros(required_buffer_size, dtype="complex128"))
another_state = DensePureState(ctx, hilbert_space_dims, batch_size, "complex128")
another_state.allocate_storage()
ordered_print("Created two zero initialized distributed quantum states to accumulate into.")

# Prepare and compute action of Operator
op.prepare_action(ctx, state)
ordered_print("Prepared Operator for action on a distributed quantum state.")

op.compute_action(0.0, None, state, state_out)
ordered_print("Accumulated action of Operator on a distributed quantum state into a zero initialized output quantum state.")

# Compute OperatorAction
op_action.compute(
    0.0,
    None,
    [
        state,
    ],
    another_state,
)
ordered_print("Accumulated the result of OperatorAction on a distributed quantum state into a zero initialized quantum state.")

assert cp.allclose(another_state.view(), state_out.view())
ordered_print("Verified that the result of OperatorAction and Operator on the same distributed input quantum state match.")

assert cp.allclose(state_out.view(), state.view() * (len(hilbert_space_dims) - 1))
ordered_print(
    "Verified that an Operator corresponding to sum of products of identity matrix scales the input quantum state by the number of terms in the sum."
)

# Clean up the duplicated communicator
comm.Free()
