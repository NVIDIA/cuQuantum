# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import sys

import cupy
import numpy
import opt_einsum
import pytest

import cuquantum
from cuquantum import cutensornet as cutn
from cuquantum.cutensornet._internal.utils import infer_object_package

from .data import backend_names, dtype_names, einsum_expressions
from .test_utils import atol_mapper, EinsumFactory, rtol_mapper
from .test_utils import compute_and_normalize_numpy_path
from .test_utils import deselect_contract_tests
from .test_utils import get_stream_for_backend
from .test_utils import set_path_to_optimizer_options


# TODO: parametrize compute type?
@pytest.mark.uncollect_if(func=deselect_contract_tests)
@pytest.mark.parametrize(
    "use_numpy_path", (False, True)
)
@pytest.mark.parametrize(
    "stream", (None, True)
)
@pytest.mark.parametrize(
    "order", ("C", "F")
)
@pytest.mark.parametrize(
    "dtype", dtype_names
)
@pytest.mark.parametrize(
    "xp", backend_names
)
@pytest.mark.parametrize(
    "einsum_expr_pack", einsum_expressions
)
class TestContract:

    def _test_runner(
            self, func, einsum_expr_pack, xp, dtype, order,
            stream, use_numpy_path, **kwargs):
        einsum_expr = copy.deepcopy(einsum_expr_pack)
        if isinstance(einsum_expr, list):
            einsum_expr, network_opts, optimizer_opts, _ = einsum_expr
        else:
            network_opts = optimizer_opts = None
        assert isinstance(einsum_expr, (str, tuple))

        factory = EinsumFactory(einsum_expr)
        operands = factory.generate_operands(
            factory.input_shapes, xp, dtype, order)
        backend = sys.modules[infer_object_package(operands[0])]
        if stream:
            stream = get_stream_for_backend(backend)

        path = None
        if use_numpy_path:
            try:
                path = compute_and_normalize_numpy_path(
                    factory.convert_by_format(operands, dummy=True),
                    len(operands))
            except NotImplementedError:
                # we can't support the returned NumPy path, just skip
                pytest.skip("NumPy path is either not found or invalid")

        data = factory.convert_by_format(operands)
        if func is cuquantum.contract:
            return_info = kwargs.pop('return_info')
            if path is not None:
                optimizer_opts = set_path_to_optimizer_options(
                    optimizer_opts, path)
            out = func(
                *data, options=network_opts, optimize=optimizer_opts,
                stream=stream, return_info=return_info)
            if return_info:
                out, (path, info) = out
                assert isinstance(path, list)
                assert isinstance(info, cuquantum.OptimizerInfo)
        else:  # cuquantum.einsum()
            optimize = kwargs.pop('optimize')
            if optimize == 'path':
                optimize = path if path is not None else False
            try:
                out = func(*data, optimize=optimize)
            except cutn.cuTensorNetError as e:
                if (optimize is not True
                        and "CUTENSORNET_STATUS_NOT_SUPPORTED" in str(e)):
                    pytest.skip("cuquantum.einsum() fail -- TN too large?")
                else:
                    raise

        if stream:
            stream.synchronize()
        backend_out = sys.modules[infer_object_package(out)]
        assert backend_out is backend
        assert out.dtype == operands[0].dtype

        out_ref = opt_einsum.contract(
            *data, backend="torch" if "torch" in xp else xp)
        assert backend.allclose(
            out, out_ref, atol=atol_mapper[dtype], rtol=rtol_mapper[dtype])

    @pytest.mark.parametrize(
        "return_info", (False, True)
    )
    def test_contract(
            self, einsum_expr_pack, xp, dtype, order,
            stream, use_numpy_path, return_info):
        self._test_runner(
            cuquantum.contract, einsum_expr_pack, xp, dtype, order,
            stream, use_numpy_path, return_info=return_info)

    @pytest.mark.parametrize(
        "optimize", (False, True, "path")
    )
    def test_einsum(
            self, einsum_expr_pack, xp, dtype, order,
            stream, use_numpy_path, optimize):
        self._test_runner(
            cuquantum.einsum, einsum_expr_pack, xp, dtype, order,
            stream, use_numpy_path, optimize=optimize)
