# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum.densitymat import WorkStream

import cupy as cp
from mpi4py import MPI
import pytest

NUM_DEVICES = cp.cuda.runtime.getDeviceCount()

# mark all tests in this file as mpi tests
pytestmark = pytest.mark.mpi

def test_work_stream_mpi():
    comm = MPI.COMM_WORLD
    my_comm = comm.Dup()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # test setting with instance of mpi4py.Comm
    cp.cuda.Device(rank % NUM_DEVICES).use()
    ctx = WorkStream(device_id=rank % NUM_DEVICES)
    ctx.set_communicator(comm, provider="MPI")
    assert size == ctx.get_num_ranks()
    assert rank == ctx.get_proc_rank()

    # test setting with pointer to communicator and its size
    ctx = WorkStream(device_id=rank % NUM_DEVICES)
    comm_ptr = MPI._addressof(comm)
    mpi_comm_size = MPI._sizeof(MPI.Comm)
    ctx.set_communicator((comm_ptr, mpi_comm_size), provider="MPI")
    assert size == ctx.get_num_ranks()
    assert rank == ctx.get_proc_rank()
    