# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from cuquantum.densitymat import WorkStream

import cupy as cp
import weakref
import pytest
from mpi4py import MPI

NUM_DEVICES = cp.cuda.runtime.getDeviceCount()


def test_work_stream_mpi():
    comm = MPI.COMM_WORLD
    my_comm = comm.Dup()
    rank = comm.Get_rank()
    size = comm.Get_size()

    cp.cuda.Device(rank % NUM_DEVICES).use()
    ctx = WorkStream(device_id=rank % NUM_DEVICES)
    ctx.set_communicator(comm, provider="MPI")
    assert size == ctx.get_num_ranks()
    assert rank == ctx.get_proc_rank()
