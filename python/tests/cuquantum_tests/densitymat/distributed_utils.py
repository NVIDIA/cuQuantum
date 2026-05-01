# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shared utilities for cuDensityMat distributed (MPI/NCCL) tests.
"""

import os
import pytest

from mpi4py import MPI
import numpy as np
import cupy as cp

import nvmath.distributed
from cuquantum.densitymat import WorkStream

# Global MPI communicator for distributed tests
mpi_comm = MPI.COMM_WORLD.Dup()

# Number of available CUDA devices
NUM_DEVICES = cp.cuda.runtime.getDeviceCount()
device_id = mpi_comm.Get_rank() % NUM_DEVICES
CURRENT_DEVICE_ID = device_id

def get_available_provider():
    """
    Detect which distributed provider is available based on CUDENSITYMAT_COMM_LIB.
    
    Returns "MPI", "NCCL", or None if the environment variable is not set.
    """
    comm_lib = os.environ.get("CUDENSITYMAT_COMM_LIB", "")
    if "nccl" in comm_lib.lower():
        return "NCCL"
    elif "mpi" in comm_lib.lower():
        return "MPI"
    return None

AVAILABLE_PROVIDER = get_available_provider()
NCCL_COMM_PTR = None
if AVAILABLE_PROVIDER == "NCCL":
    with cp.cuda.Device(device_id):
        nvmath.distributed.initialize(device_id, mpi_comm, backends=["nccl"])
        nccl_comm = nvmath.distributed.get_context().nccl_comm
        if isinstance(nccl_comm, int):
            NCCL_COMM_PTR = nccl_comm
        else:
            NCCL_COMM_PTR = nccl_comm.ptr

def skip_if_provider_unavailable(provider: str):
    """Skip test if the requested provider doesn't match the loaded interface."""
    if AVAILABLE_PROVIDER is None:
        pytest.skip("CUDENSITYMAT_COMM_LIB not set")
    if provider != AVAILABLE_PROVIDER:
        pytest.skip(f"Test requires {provider} provider but {AVAILABLE_PROVIDER} is loaded")

def set_communicator_for_provider(ctx: WorkStream, provider: str):
    """Set the communicator on the WorkStream for the given provider.
    
    Args:
        ctx: The WorkStream to configure.
        provider: "MPI" or "NCCL".
        comm: Optional MPI communicator to use. If None, uses the global mpi_comm.
    """
    if provider == "NCCL":
        ctx.set_communicator(NCCL_COMM_PTR, provider="NCCL")
    elif provider == "MPI":
        ctx.set_communicator(mpi_comm, provider="MPI")
    else:
        raise ValueError(f"Unknown provider: {provider}")
