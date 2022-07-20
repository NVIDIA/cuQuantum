# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import re
import sys

import cupy
import numpy
import opt_einsum
import pytest

from cuquantum import Network
from cuquantum.cutensornet._internal.utils import infer_object_package

from .data import backend_names, dtype_names, einsum_expressions
from .testutils import atol_mapper, EinsumFactory, rtol_mapper
from .testutils import compute_and_normalize_numpy_path
from .testutils import set_path_to_optimizer_options


# TODO: parametrize compute type?
@pytest.mark.parametrize(
    "use_numpy_path", (False, True)
)
@pytest.mark.parametrize(
    "stream", (None, True)
)
@pytest.mark.parametrize(
    "autotune", (False, 5)
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
class TestNetwork:

    def test_network(
            self, einsum_expr_pack, xp, dtype, order, autotune,
            stream, use_numpy_path):
        einsum_expr = copy.deepcopy(einsum_expr_pack)
        if isinstance(einsum_expr, list):
            einsum_expr, network_opts, optimizer_opts, overwrite_dtype = einsum_expr
            if dtype != overwrite_dtype:
                pytest.skip(f"skipping {dtype} is requested")
        else:
            network_opts = optimizer_opts = None
        assert isinstance(einsum_expr, (str, tuple))

        factory = EinsumFactory(einsum_expr)
        operands = factory.generate_operands(
            factory.input_shapes, xp, dtype, order)
        backend = sys.modules[infer_object_package(operands[0])]
        if stream:
            if backend is numpy:
                stream = cupy.cuda.Stream()  # implementation detail
            else:
                stream = backend.cuda.Stream()
        data = factory.convert_by_format(operands)
        tn = Network(*data, options=network_opts)

        # We already test tn as a context manager in the samples, so let's test
        # explicitly calling tn.free() here.
        try:
            if not use_numpy_path:
                _, info = tn.contract_path(optimize=optimizer_opts)
                uninit_f_str = re.compile("{.*}")
                assert uninit_f_str.search(str(info)) is None
            else:
                try:
                    path_ref = compute_and_normalize_numpy_path(
                        factory.convert_by_format(operands, dummy=True),
                        len(operands))
                except NotImplementedError:
                    # we can't support the returned NumPy path, just skip
                    pytest.skip("NumPy path is either not found or invalid")
                else:
                    optimizer_opts = set_path_to_optimizer_options(
                        optimizer_opts, path_ref)
                    path, _ = tn.contract_path(optimizer_opts)
                    assert path == path_ref  # round-trip test

            if autotune:
                tn.autotune(iterations=autotune, stream=stream)
            out = tn.contract(stream=stream)
            if stream:
                stream.synchronize()
            backend_out = sys.modules[infer_object_package(out)]
            assert backend_out is backend
            assert out.dtype == operands[0].dtype

            out_ref = opt_einsum.contract(
                *data, backend="torch" if "torch" in xp else xp)
            assert backend.allclose(
                out, out_ref, atol=atol_mapper[dtype], rtol=rtol_mapper[dtype])

            # TODO: test tn.reset_operands()
        finally:
            tn.free()
