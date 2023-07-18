# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import sys

import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

import cuquantum.cutensornet as cutn
from cuquantum.cutensornet import tensor

from .._utils import precision_str_to_dtype, wrap_with_nvtx
try:
    path = os.environ.get('CUTENSORNET_APPROX_TN_UTILS_PATH', '')
    if path and os.path.isfile(path):
        sys.path.insert(1, os.path.dirname(path))
    from approxTN_utils import tensor_decompose
except ImportError:
    tensor_decompose = None


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


def benchmark_tensor_decompose(
        expr, shape, precision, is_complex, method, algorithm, n_warmup, n_repeats, check_ref, *,
        benchmark_data=None):
    logger.debug(f"{expr=}")
    logger.debug(f"{shape=}")
    logger.debug(f"{precision=}")
    logger.debug(f"{is_complex=}")
    logger.debug(f"{method=}")
    logger.debug(f"{algorithm=}")
    logger.debug(f"{n_warmup=}")
    logger.debug(f"{n_repeats=}")
    logger.debug(f"{check_ref=}")

    cp.random.seed(5678)  # TODO: set me
    handle = cutn.create()
    options = {'handle': handle}
    decomp_subscripts = expr

    # sanity checks
    expr_in = expr.split('->')[0]
    assert len(expr_in) == len(shape), \
           f"the input shape {shape} mismatches with the input modes {expr_in}"
    if check_ref and tensor_decompose is None:
        raise RuntimeError("--check-reference is not supported") 

    dtype_r = precision_str_to_dtype(precision, False)
    t_in = cp.random.random(shape, dtype=dtype_r)
    if is_complex:
        dtype = precision_str_to_dtype(precision)
        t_in = t_in.astype(dtype)
        t_in += 1j*cp.random.random(shape, dtype=dtype_r)
        assert t_in.dtype == dtype
    
    t_numpy = t_in.get()

    if method == "QR":
        kwargs = {'options': options}
        if check_ref:
            options_ref = {'method':'qr'}
    elif method == "SVD":
        try:
            kwargs = {'options': options, 'method': tensor.SVDMethod(algorithm=algorithm)}
        except TypeError as e:
            if algorithm != "gesvd":
                raise ValueError(f"{algorithm} requires cuQuantum v23.06+") from e
            else:
                kwargs = {'options': options, 'method': tensor.SVDMethod()}
        if check_ref:
            options_ref = {'method':'svd'}
    else:
        assert False
    cp.cuda.Device().synchronize()  # ensure data prep is done

    decompose = wrap_with_nvtx(tensor.decompose, "decompose")

    results = benchmark(decompose,
                        (decomp_subscripts, t_in), kwargs=kwargs,
                        n_repeat=n_repeats, n_warmup=n_warmup)

    if check_ref:
        decompose_ref = wrap_with_nvtx(tensor_decompose, "tensor_decompose")

        results_cupy = benchmark(decompose_ref,
                                 (decomp_subscripts, t_in), kwargs=options_ref,
                                 n_repeat=n_repeats, n_warmup=n_warmup)
    
        results_numpy = benchmark(decompose_ref,
                                  (decomp_subscripts, t_numpy), kwargs=options_ref,
                                  n_repeat=n_repeats, n_warmup=n_warmup)

    cutn.destroy(handle)

    logger.debug(str(results))
    if check_ref:
        logger.debug("ref (CuPy):")
        logger.debug(str(results_cupy))
        benchmark_data['cupy_time'] = max(
            np.average(results_cupy.cpu_times), np.average(results_cupy.gpu_times[0]))
        logger.debug("ref (NumPy):")
        logger.debug(str(results_numpy))
        benchmark_data['numpy_time'] = np.average(results_numpy.cpu_times)

    cpu_time = np.average(results.cpu_times)
    gpu_time = np.average(results.gpu_times[0])

    return cpu_time, gpu_time
