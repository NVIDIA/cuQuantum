# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum.densitymat import WorkStream

import cupy as cp
import numpy as np
from mpi4py import MPI
import pytest

# nvmath-python >= 0.9 no longer ships its own NCCL bindings; NCCL is now
# provided by the standalone ``nccl4py`` package (imported as ``nccl.core``).
import nccl.core as nccl

from .distributed_utils import (
    mpi_comm as comm,
    CURRENT_DEVICE_ID,
    skip_if_provider_unavailable,
    set_communicator_for_provider,
)

# mark all tests in this file as mpi tests
pytestmark = pytest.mark.mpi


@pytest.mark.parametrize("provider", ["MPI", "NCCL"])
def test_work_stream_communicator_from_mpi_comm(provider):
    """Test setting communicator from mpi4py.MPI.Comm object."""
    skip_if_provider_unavailable(provider)
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    device_id = CURRENT_DEVICE_ID

    with cp.cuda.Device(device_id):
        ctx = WorkStream(device_id=device_id)
        set_communicator_for_provider(ctx, provider)
        assert size == ctx.get_num_ranks()
        assert rank == ctx.get_proc_rank()


@pytest.mark.parametrize("sequence_type", [tuple, list])
def test_work_stream_mpi_communicator_from_pointer(sequence_type):
    """Test setting MPI communicator from (pointer, size) sequence."""
    skip_if_provider_unavailable("MPI")
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    device_id = CURRENT_DEVICE_ID

    with cp.cuda.Device(device_id):
        ctx = WorkStream(device_id=device_id)
        comm_ptr = MPI._addressof(comm)
        mpi_comm_size = MPI._sizeof(MPI.Comm)
        ctx.set_communicator(sequence_type([comm_ptr, mpi_comm_size]), provider="MPI")
        assert size == ctx.get_num_ranks()
        assert rank == ctx.get_proc_rank()


def test_work_stream_mpi_communicator_from_int_pointer():
    """Test setting MPI communicator from integer pointer."""
    skip_if_provider_unavailable("MPI")

    rank = comm.Get_rank()
    size = comm.Get_size()
    device_id = CURRENT_DEVICE_ID

    with cp.cuda.Device(device_id):
        ctx = WorkStream(device_id=device_id)
        comm_ptr = MPI._addressof(comm)
        ctx.set_communicator(comm_ptr, provider="MPI")
        assert size == ctx.get_num_ranks()
        assert rank == ctx.get_proc_rank()


def _bootstrap_nccl_communicator(rank, size):
    """Create an externally-managed NCCL communicator using nccl4py.

    All ranks must call this collectively. A unique id is generated on every
    rank to obtain a same-sized buffer, then rank 0's bytes are broadcast via
    MPI so all ranks join the same communicator.
    """
    unique_id = nccl.get_unique_id()
    comm.Bcast(unique_id.as_ndarray.view(np.int8), root=0)
    return nccl.Communicator.init(nranks=size, rank=rank, unique_id=unique_id)


@pytest.mark.parametrize("sequence_type", [tuple, list])
def test_work_stream_nccl_communicator_from_pointer(sequence_type):
    """Test setting NCCL communicator from (pointer, size) sequence with externally managed ncclComm_t."""
    skip_if_provider_unavailable("NCCL")

    rank = comm.Get_rank()
    size = comm.Get_size()
    device_id = CURRENT_DEVICE_ID

    with cp.cuda.Device(device_id):
        nccl_comm = _bootstrap_nccl_communicator(rank, size)
        try:
            ctx = WorkStream(device_id=device_id)
            # Pass (ncclComm_t value, size) - library_handle wraps it in numpy array internally
            # The size value is not actually used (library uses itemsize of internal holder)
            ctx.set_communicator(
                sequence_type([nccl_comm.ptr, np.dtype(np.intp).itemsize]),
                provider="NCCL"
            )
            assert size == ctx.get_num_ranks()
            assert rank == ctx.get_proc_rank()
        finally:
            # Clean up externally managed NCCL communicator
            nccl_comm.destroy()


def test_work_stream_nccl_communicator_from_int_pointer():
    """Test setting NCCL communicator from integer pointer with externally managed ncclComm_t."""
    skip_if_provider_unavailable("NCCL")

    rank = comm.Get_rank()
    size = comm.Get_size()
    device_id = CURRENT_DEVICE_ID

    with cp.cuda.Device(device_id):
        nccl_comm = _bootstrap_nccl_communicator(rank, size)
        try:
            ctx = WorkStream(device_id=device_id)
            ctx.set_communicator(int(nccl_comm.ptr), provider="NCCL")
            assert size == ctx.get_num_ranks()
            assert rank == ctx.get_proc_rank()
        finally:
            ctx = None
            cp.cuda.Device().synchronize()
            nccl_comm.finalize()
            nccl_comm.destroy()
