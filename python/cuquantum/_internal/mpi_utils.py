# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
MPI helper functions (internal use).
"""


def get_mpi_comm_pointer(comm):
    """Simple helper to get the address to and size of a ``MPI_Comm`` handle.

    Args:
        comm (mpi4py.MPI.Comm): An MPI communicator.

    Returns:
        tuple: A pair of int values representing the address and the size.
    """
    try:
        from mpi4py import MPI  # init!
    except ImportError as e:
        raise RuntimeError("please install mpi4py") from e

    if not isinstance(comm, MPI.Comm):
        raise ValueError("invalid MPI communicator")
    comm_ptr = MPI._addressof(comm)  # = MPI_Comm*
    mpi_comm_size = MPI._sizeof(MPI.Comm)
    return comm_ptr, mpi_comm_size
