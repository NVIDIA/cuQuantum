# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum.densitymat import (
    WorkStream,
    DensePureState,
    DenseMixedState,
)

import cupy as cp
import pytest

from .distributed_utils import (
    mpi_comm as comm,
    CURRENT_DEVICE_ID,
    skip_if_provider_unavailable,
    set_communicator_for_provider,
)

# TODO: mostly redundant with tests in test_context.py, consolidate in the future

# mark all tests in this file as mpi tests
pytestmark = pytest.mark.mpi


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
@pytest.mark.parametrize("provider", ["MPI", "NCCL"])
def test_creation(hilbert_space_dims, batch_size, mixed, dtype, provider):
    skip_if_provider_unavailable(provider)
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    device_id = CURRENT_DEVICE_ID
    with cp.cuda.Device(device_id):
        ctx = WorkStream(device_id=device_id)
        set_communicator_for_provider(ctx, provider)
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
