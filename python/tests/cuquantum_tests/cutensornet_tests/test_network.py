# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import copy
import re
import sys

import cupy as cp
import numpy as np
import opt_einsum as oe
import pytest

from cuquantum import cutensornet as cutn
from cuquantum import Network
from cuquantum.cutensornet._internal.utils import infer_object_package

from .data import backend_names, dtype_names, einsum_expressions
from .test_utils import atol_mapper, EinsumFactory, rtol_mapper
from .test_utils import check_intermediate_modes
from .test_utils import compute_and_normalize_numpy_path
from .test_utils import deselect_contract_tests
from .test_utils import get_stream_for_backend
from .test_utils import set_path_to_optimizer_options


# TODO: parametrize compute type?
@pytest.mark.uncollect_if(func=deselect_contract_tests)
@pytest.mark.parametrize(
    "release_workspace", (True, False)
)
@pytest.mark.parametrize(
    "reset_none", (True, )
)
@pytest.mark.parametrize(
    "gradient", (False, "random", "all")
)
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
            stream, use_numpy_path, gradient, reset_none, release_workspace):
        einsum_expr = copy.deepcopy(einsum_expr_pack)
        if isinstance(einsum_expr, list):
            einsum_expr, network_opts, optimizer_opts, _ = einsum_expr
        else:
            network_opts = optimizer_opts = None
        assert isinstance(einsum_expr, (str, tuple))

        # prepare operands and other needed test config
        factory = EinsumFactory(einsum_expr)
        operands = factory.generate_operands(
            factory.input_shapes, xp, dtype, order)
        qualifiers, picks = factory.generate_qualifiers(xp, gradient)
        factory.setup_torch_grads(xp, picks, operands)
        backend = sys.modules[infer_object_package(operands[0])]
        data = factory.convert_by_format(operands)
        if stream:
            stream = get_stream_for_backend(backend)
        tn = Network(*data, options=network_opts, qualifiers=qualifiers)

        # We already test tn as a context manager in the samples, so let's test
        # explicitly calling tn.free() here.
        try:
            if not use_numpy_path:
                path, info = self._setup_path(tn, optimizer_opts)
                uninit_f_str = re.compile("{.*}")
                assert uninit_f_str.search(str(info)) is None
                check_intermediate_modes(
                    info.intermediate_modes, factory.input_modes,
                    factory.output_modes[0], path)
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
                    path, _ = self._setup_path(tn, optimizer_opts)
                    # round-trip test
                    # note that within each pair it could have different order
                    assert all(map(lambda x, y: sorted(x) == sorted(y), path, path_ref))

            if autotune:
                tn.autotune(iterations=autotune, stream=stream, release_workspace=release_workspace)
            # check the result
            out, out_ref = self._verify_contract(
                tn, operands, backend, data, xp, dtype, stream, release_workspace)
            self._verify_gradient(
                tn, operands, backend, data, xp, dtype,
                gradient, out, out_ref, picks, stream, release_workspace)

            if reset_none:
                tn.reset_operands(None, stream=stream)

            # generate new data (by picking a nonzero seed) and bind them
            # to the TN
            operands = factory.generate_operands(
                factory.input_shapes, xp, dtype, order, seed=100)
            factory.setup_torch_grads(xp, picks, operands)
            data = factory.convert_by_format(operands)
            tn.reset_operands(*operands, stream=stream)

            # check the result
            out, out_ref = self._verify_contract(
                tn, operands, backend, data, xp, dtype, stream, release_workspace)
            self._verify_gradient(
                tn, operands, backend, data, xp, dtype,
                gradient, out, out_ref, picks, stream, release_workspace)
        finally:
            tn.free()

    def _setup_path(self, tn, optimizer_opts):
        try:
            path, info = tn.contract_path(optimize=optimizer_opts)
        except cutn.cuTensorNetError as e:
            # differentiating some edge TNs is not yet supported
            if "NOT_SUPPORTED" in str(e):
                pytest.skip("this TN is currently not supported")
            else:
                raise
        return path, info

    def _setup_gradients(self, tn, output_grad, stream, release_workspace):
        try:
            input_grads = tn.gradients(output_grad, stream=stream, release_workspace=release_workspace)
        except cutn.cuTensorNetError as e:
            # differentiating some edge TNs is not yet supported
            if "NOT_SUPPORTED" in str(e):
                pytest.skip("this TN is currently not supported")
            else:
                raise
        return input_grads

    def _verify_contract(
            self, tn, operands, backend, data, xp, dtype, stream, release_workspace):
        out = tn.contract(stream=stream, release_workspace=release_workspace)
        if stream:
            stream.synchronize()

        # check contraction result types
        assert sys.modules[infer_object_package(out)] is backend
        assert out.dtype == operands[0].dtype

        # check contraction
        out_ref = oe.contract(*data, backend=("torch" if "torch" in xp else xp))
        assert backend.allclose(
            out, out_ref, atol=atol_mapper[dtype], rtol=rtol_mapper[dtype])

        return out, out_ref

    def _verify_gradient(
            self, tn, operands, backend, data, xp, dtype,
            gradient, out, out_ref, picks, stream, release_workspace):
        if gradient is False:
            return

        # compute gradients
        output_grad = backend.ones_like(out)
        input_grads = self._setup_gradients(tn, output_grad, stream, release_workspace)
        if stream:
            stream.synchronize()

        # check gradient result types
        assert all((sys.modules[infer_object_package(grad)] is backend)
                   if grad is not None else True
                   for grad in input_grads)
        assert all((grad.dtype == operands[0].dtype)
                   if grad is not None else True
                   for grad in input_grads)

        # given simplicity & CI time constraints we only do grad
        # verification with torch tensors
        if "torch" in xp:
            output_grad = backend.ones_like(out_ref)
            out_ref.backward(output_grad)

            # check gradients
            try:
                is_close = backend.tensor(tuple(
                    backend.allclose(
                        cutn_grad, op.grad,
                        atol=atol_mapper[dtype], rtol=rtol_mapper[dtype])
                    if cutn_grad is not None else cutn_grad is op.grad
                    for cutn_grad, op in zip(input_grads, operands)
                ))
                assert all(is_close)
            except AssertionError as e:
                # for easier debugging
                print(tuple(op.shape for op in operands))
                print(input_grads)
                print(tuple(op.grad for op in operands))
                raise
