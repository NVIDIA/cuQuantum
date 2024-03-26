# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
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
from .test_utils import deselect_gradient_tests
from .test_utils import get_stream_for_backend
from .test_utils import set_path_to_optimizer_options


# TODO: parametrize compute type?
@pytest.mark.parametrize(
    "use_numpy_path", (False, True)
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
class _TestContractBase:

    def _test_runner(
            self, func, einsum_expr_pack, xp, dtype, order,
            use_numpy_path, gradient, **kwargs):
        einsum_expr = copy.deepcopy(einsum_expr_pack)
        if isinstance(einsum_expr, list):
            einsum_expr, network_opts, optimizer_opts, _ = einsum_expr
        else:
            network_opts = optimizer_opts = None
        assert isinstance(einsum_expr, (str, tuple))

        factory = EinsumFactory(einsum_expr)
        operands = factory.generate_operands(
            factory.input_shapes, xp, dtype, order)
        qualifiers, picks = factory.generate_qualifiers(xp, gradient)
        factory.setup_torch_grads(xp, picks, operands)
        backend = sys.modules[infer_object_package(operands[0])]
        stream = kwargs.get('stream')
        if stream:
            stream_obj = get_stream_for_backend(backend)
            if stream == "as_int":
                if backend is numpy or backend is cupy:
                    stream = stream_obj.ptr
                else:
                    pytest.skip("we do not support torch operands + "
                                "a raw stream pointer")
            else:
                stream = stream_obj

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
            try:
                out = func(
                    *data, options=network_opts, optimize=optimizer_opts,
                    stream=stream, return_info=return_info)
                if stream:
                    stream_obj.synchronize()
            except cutn.cuTensorNetError as e:
                # differentiating some edge TNs is not yet supported
                if "NOT_SUPPORTED" in str(e):
                    pytest.skip("this TN is currently not supported")
                else:
                    raise
            except MemoryError as e:
                if "Insufficient memory" in str(e):
                    # not enough memory available to process, just skip
                    pytest.skip("Insufficient workspace memory available.")
                else:
                    raise

            if return_info:
                out, (path, info) = out
                assert isinstance(path, list)
                assert isinstance(info, cuquantum.OptimizerInfo)

            if gradient:
                # compute gradients
                output_grad = backend.ones_like(out)
                try:
                    out.backward(output_grad)
                except cutn.cuTensorNetError as e:
                    # differentiating some edge TNs is not yet supported;
                    if "NOT_SUPPORTED" in str(e):
                        # we don't wanna skip because we can still verify
                        # contraction ouput
                        gradient = None
                    else:
                        raise

            if gradient:
                input_grads = tuple(op.grad for op in operands)

                # check gradient result types
                assert all((sys.modules[infer_object_package(grad)] is backend)
                           if grad is not None else True
                           for grad in input_grads)
                assert all((grad.dtype == operands[0].dtype)
                           if grad is not None else True
                           for grad in input_grads)

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
            except MemoryError as e:
                if "Insufficient memory" in str(e):
                    # not enough memory available to process, just skip
                    pytest.skip("Insufficient workspace memory available.")
                else:
                    raise

        backend_out = sys.modules[infer_object_package(out)]
        assert backend_out is backend
        assert out.dtype == operands[0].dtype

        # check contraction
        factory.setup_torch_grads(xp, picks, operands)
        out_ref = opt_einsum.contract(
            *data, backend="torch" if "torch" in xp else xp)
        assert backend.allclose(
            out, out_ref, atol=atol_mapper[dtype], rtol=rtol_mapper[dtype])

        # check gradients
        if gradient and func is cuquantum.contract:
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
 

@pytest.mark.uncollect_if(func=(deselect_contract_tests,
                                deselect_gradient_tests))
@pytest.mark.parametrize(
    "gradient", (False, "random", "all")
)
@pytest.mark.parametrize(
    "stream", (None, True, "as_int")
)
@pytest.mark.parametrize(
    "return_info", (False, True)
)
class TestContract(_TestContractBase):

    def test_contract(
            self, einsum_expr_pack, xp, dtype, order,
            use_numpy_path, gradient, stream, return_info):
        self._test_runner(
            cuquantum.contract, einsum_expr_pack, xp, dtype, order,
            use_numpy_path, gradient, stream=stream, return_info=return_info)


# einsum does not support gradient (at some point we should deprecate it...)
@pytest.mark.uncollect_if(func=deselect_contract_tests)
@pytest.mark.parametrize(
    "optimize", (False, True, "path")
)
class TestEinsum(_TestContractBase):

    def test_einsum(
            self, einsum_expr_pack, xp, dtype, order,
            use_numpy_path, optimize):
        self._test_runner(
            cuquantum.einsum, einsum_expr_pack, xp, dtype, order,
            use_numpy_path, None, optimize=optimize)
