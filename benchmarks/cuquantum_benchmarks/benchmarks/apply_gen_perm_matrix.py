# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

from cuquantum import custatevec as cusv

from .._utils import (check_sequence, check_targets_controls, dtype_to_cuda_type,
                      precision_str_to_dtype, wrap_with_nvtx)


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


def test_apply_generalized_permutation_matrix(
        n_qubits, dtype_sv,
        targets, controls, adjoint,
        diag, dtype_diag, location_diag,  # for D
        perm_table, location_perm,        # for P
        n_warmup, n_repeat, *,
        benchmark_data=None):
    # TODO: allow controlling seed?
    if diag is False and not perm_table:
        raise ValueError("need to specify at least --has-diag or --has-perm/--perm-table")

    logger.debug(f"{n_qubits=}")
    logger.debug(f"{dtype_sv=}")
    logger.debug(f"{targets=}")
    logger.debug(f"{controls=}")
    logger.debug(f"{adjoint=}")
    logger.debug(f"{diag=}")
    logger.debug(f"{dtype_diag=}")
    logger.debug(f"{location_diag=}")
    if isinstance(perm_table, bool) or len(perm_table) <= 16:
        logger.debug(f"{perm_table=}")
    else:
        logger.debug("perm_table = (omitted due to length)")
    logger.debug(f"{location_perm=}")
    logger.debug(f"{n_warmup=}")
    logger.debug(f"{n_repeat=}")

    check_targets_controls(targets, controls, n_qubits)
    n_targets = len(targets)
    n_controls = len(controls)

    # cuStateVec handle initialization
    handle = cusv.create()
    stream = cp.cuda.Stream()
    cusv.set_stream(handle, stream.ptr)

    size_sv = (2 ** n_qubits)
    dtype_sv = precision_str_to_dtype(dtype_sv)
    sv = cp.ones((size_sv,), dtype=dtype_sv)
    data_type_sv = dtype_to_cuda_type(dtype_sv)

    # the diagonal matrix can live on either host (np) or device (cp)
    matrix_dim = (2 ** n_targets)
    dtype_diag = precision_str_to_dtype(dtype_diag)
    xp_diag = cp if location_diag == 'device' else np
    if diag:
        # it's better to just call rng.uniform(), but it's not there until CuPy v12.0.0
        # rng_diag = xp_diag.random.default_rng(seed=1234)
        # diag = rng_diag.uniform(0.7, 1.3, size=matrix_dim).astype(dtype_diag)
        diag = 0.6 * xp_diag.random.random(size=matrix_dim).astype(dtype_diag) + 0.7
        if isinstance(diag, cp.ndarray):
            diag_ptr = diag.data.ptr
        elif isinstance(diag, np.ndarray):
            diag_ptr = diag.ctypes.data
        else:
            raise ValueError
    else:
        diag_ptr = 0
    data_type_diag = dtype_to_cuda_type(dtype_diag)

    # the permutation table can live on either host (np) or device (cp)
    xp_perm = cp if location_perm == 'device' else np
    if perm_table:
        if perm_table is True:
            original_perm_table = xp_perm.arange(0, matrix_dim, dtype=xp_perm.int64)
            perm_table = xp_perm.copy(original_perm_table)
            # it'd have been nice to seed an rng and call rng.shuffle(), but CuPy does
            # not support it yet...
            while True:
                xp_perm.random.shuffle(perm_table)
                # check if the matrix is not diagonal
                if not (original_perm_table == perm_table).all():
                    break
        else:  # a user-provided list
            check_sequence(perm_table, expected_size=matrix_dim, name="perm_table")
            perm_table = xp_perm.asarray(perm_table, dtype=xp_perm.int64)

        if isinstance(perm_table, cp.ndarray):
            perm_table_ptr = perm_table.data.ptr
        elif isinstance(perm_table, np.ndarray):
            perm_table_ptr = perm_table.ctypes.data
        else:
            raise ValueError
    else:
        perm_table_ptr = 0

    cp.cuda.Device().synchronize()  # ensure data prep is done before switching stream

    ####################################################################################

    # manage the workspace
    workspace_size = cusv.apply_generalized_permutation_matrix_get_workspace_size(
        handle, data_type_sv, n_qubits, perm_table_ptr, diag_ptr,
        data_type_diag, targets, n_targets, n_controls)

    with stream:
        if workspace_size > 0:
            workspace = cp.cuda.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        # apply diagonal/permutation gate
        apply_generalized_permutation_matrix = wrap_with_nvtx(
            cusv.apply_generalized_permutation_matrix,
            "apply_generalized_permutation_matrix")
        args = (
            handle, sv.data.ptr, data_type_sv, n_qubits, perm_table_ptr,
            diag_ptr, data_type_diag, adjoint, targets, n_targets,
            controls, 0,  # TODO: support control bit values
            n_controls, workspace_ptr, workspace_size)
        result = benchmark(
            apply_generalized_permutation_matrix,
            args,
            n_warmup=n_warmup, n_repeat=n_repeat)

    # destroy handle
    cusv.destroy(handle)

    logger.debug(str(result))
    cpu_time = np.average(result.cpu_times)
    gpu_time = np.average(result.gpu_times[0])
    memory_footprint = (2. ** (n_qubits - n_controls)) * 2. * np.dtype(dtype_sv).itemsize
    logger.debug(f"effective bandwidth = {memory_footprint / gpu_time * 1e-9} (GB/s)")

    return cpu_time, gpu_time
