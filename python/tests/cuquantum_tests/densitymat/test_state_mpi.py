# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum.densitymat import (
    WorkStream,
    DensePureState,
    DenseMixedState,
)

import cupy as cp
import pytest
from mpi4py import MPI

NUM_DEVICES = cp.cuda.runtime.getDeviceCount()
# TODO: mostly redundant with tests in test_context.py, consolidate in the future


@pytest.mark.parametrize(
    "hilbert_space_dims",
    [
        (
            2,
            2,
            2,
        )
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("mixed", [True, False])
@pytest.mark.parametrize("dtype", ["complex128", "complex64", "float64", "float32"])
def test_creation(hilbert_space_dims, batch_size, mixed, dtype):
    comm = MPI.COMM_WORLD
    my_comm = comm.Dup()
    rank = comm.Get_rank()
    size = comm.Get_size()
    cp.cuda.Device(rank % NUM_DEVICES).use()
    ctx = WorkStream(device_id=rank % NUM_DEVICES)
    ctx.set_communicator(my_comm, provider="MPI")
    State = DensePureState if not mixed else DenseMixedState
    state = State(ctx, hilbert_space_dims, batch_size, dtype)
    slice_shape, offsets = state.local_info
    assert len(slice_shape) == len(offsets)
    assert len(slice_shape) == 1 + len(hilbert_space_dims) * (2 if mixed else 1)
    storage_size = state.storage_size
    state_storage_buf = cp.zeros((storage_size,), dtype=state.dtype)
    state.attach_storage(state_storage_buf)
    state_view = state.view()
    state_view[:] = cp.random.rand(*state_view.shape)
    assert state_view.shape == slice_shape
    state = None
    state = State(ctx, hilbert_space_dims, batch_size, dtype)
    state.allocate_storage()
    assert state.storage.size == storage_size
    assert state.storage.size == state.storage_size
