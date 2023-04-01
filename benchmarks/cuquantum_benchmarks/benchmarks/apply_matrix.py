# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np
import cupy as cp
from cupyx.profiler import benchmark

from cuquantum import custatevec as cusv

from .._utils import (dtype_to_cuda_type, dtype_to_compute_type, precision_str_to_dtype,
                      random_unitary, L2flush, benchmark_with_prerun)


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


def test_apply_matrix(
        n_qubits, targets, controls, dtype_sv, dtype_mat, layout, adjoint,
        n_warmup, n_repeat, location, *, flush_l2=False):
    logger.debug(f"{n_qubits = }")
    logger.debug(f"{targets = }")
    logger.debug(f"{controls = }")
    logger.debug(f"{dtype_sv = }")
    logger.debug(f"{dtype_mat = }")
    logger.debug(f"{layout = }")
    logger.debug(f"{adjoint = }")
    logger.debug(f"{location = }")
    logger.debug(f"{n_warmup = }")
    logger.debug(f"{n_repeat = }")
    logger.debug(f"{flush_l2 = }")
    
    dtype_sv = precision_str_to_dtype(dtype_sv)
    dtype_mat = precision_str_to_dtype(dtype_mat)
    xp = cp if location == 'device' else np
    layout = cusv.MatrixLayout.ROW if layout == "row" else cusv.MatrixLayout.COL

    # simple sanity checks
    assert len(targets) >= 1, "must have at least 1 target qubit"
    _targets = set(targets)
    assert len(_targets) == len(targets), "target qubit IDs cannot overlap"
    _controls = set(controls)
    assert len(_controls) == len(controls), "control qubits IDs cannot overlap"
    assert len(_targets & _controls) == 0, "targets and controls cannot overlap"
    _involved = targets + controls
    assert 0 <= min(_involved) and max(_involved) < n_qubits, f"involved qubit IDs must be in range [0, {n_qubits})"

    size_sv = 2**n_qubits
    n_targets = len(targets)
    n_controls = len(controls)

    # passing data ptr is slightly faster
    targets_data = np.asarray(targets, dtype=np.int32)
    targets = targets_data.ctypes.data
    controls_data = np.asarray(controls, dtype=np.int32)
    controls = controls_data.ctypes.data

    # the statevector must reside on device
    sv = cp.ones((size_sv,), dtype=dtype_sv)
    sv /= np.sqrt(size_sv)
    # assert cp.allclose(cp.sum(cp.abs(sv)**2), 1)
    data_type_sv = dtype_to_cuda_type(dtype_sv)

    # the gate matrix can live on either host (np) or device (cp)
    matrix_dim = 2**n_targets
    matrix = xp.asarray(random_unitary(matrix_dim), dtype=dtype_mat)
    data_type_mat = dtype_to_cuda_type(dtype_mat)
    if isinstance(matrix, cp.ndarray):
        matrix_ptr = matrix.data.ptr
    elif isinstance(matrix, np.ndarray):
        matrix_ptr = matrix.ctypes.data
    else:
        raise ValueError

    compute_type = dtype_to_compute_type(dtype_mat)  # TODO: make this independent?
    cp.cuda.Device().synchronize()  # ensure data prep is done before switching stream

    ####################################################################################
    
    # cuStateVec handle initialization
    handle = cusv.create()
    stream = cp.cuda.Stream()
    cusv.set_stream(handle, stream.ptr)

    # get the workspace size
    workspace_size = cusv.apply_matrix_get_workspace_size(
        handle,
        data_type_sv, n_qubits,
        matrix_ptr, data_type_mat, layout, adjoint, n_targets, n_controls,
        compute_type)
    
    # apply gate
    with stream:
        # manage workspace
        if workspace_size > 0:
            workspace = cp.cuda.memory.alloc(workspace_size)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        args = (handle,
                sv.data.ptr, data_type_sv, n_qubits,
                matrix_ptr, data_type_mat, layout, adjoint,
                targets, n_targets,
                controls, 0, n_controls,  # TODO: support control bit values
                compute_type, workspace_ptr, workspace_size)

        if flush_l2:
            l2flusher = L2flush()
            def f(*args, **kwargs):
                l2flusher.flush()  # clear L2 cache

            result = benchmark_with_prerun(
                cusv.apply_matrix,
                args,
                n_warmup=n_warmup, n_repeat=n_repeat,
                pre_run=f)
        else:
            result = benchmark(
                cusv.apply_matrix,
                args,
                n_warmup=n_warmup, n_repeat=n_repeat)
    
    # destroy handle
    cusv.destroy(handle)

    logger.debug(str(result))  # this is nice-looking, if _PerfCaseResult.__repr__ is there
    #logger.debug(f"(CPU times: {result.cpu_times}")
    #logger.debug(f"(GPU times: {result.gpu_times[0]}")
    cpu_time = np.average(result.cpu_times)
    gpu_time = np.average(result.gpu_times[0])
    mem_access = (2. ** (n_qubits - n_controls)) * 2. * np.dtype(dtype_sv).itemsize
    logger.debug(f"effective bandwidth = {mem_access/gpu_time*1e-9} (GB/s)")

    return cpu_time, gpu_time
