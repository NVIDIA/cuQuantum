# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import copy
import re
import sys
import itertools

import opt_einsum as oe
import pytest

try:
    import cupy as cp
except ImportError:
    cp = None
import numpy as np

from nvmath.internal.utils import infer_object_package

from cuquantum.bindings import cutensornet as cutn
from cuquantum.tensornet import Network

from .utils.data import BACKEND_MEMSPACE, dtype_names, einsum_expressions
from .utils.helpers import EinsumFactory, get_contraction_tolerance
from .utils.helpers import check_intermediate_modes
from .utils.helpers import compute_and_normalize_numpy_path
from .utils.helpers import _BaseTester
from .utils.helpers import get_stream_for_backend
from .utils.helpers import set_path_to_optimizer_options
from .utils.helpers import Deselector

class NetworkBaseTester(_BaseTester):

    def _test_runner(
            self, einsum_expr_pack, xp, dtype, order, autotune,
            stream, use_numpy_path, gradient, reset_none, release_workspace, rng):
        einsum_expr = copy.deepcopy(einsum_expr_pack)
        if isinstance(einsum_expr, list):
            einsum_expr, network_opts, optimizer_opts, _ = einsum_expr
        else:
            network_opts = optimizer_opts = None
        assert isinstance(einsum_expr, (str, tuple))

        # prepare operands and other needed test config
        factory = EinsumFactory(einsum_expr, rng)
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

            # generate new data and bind them to the TN, 
            # this is now done automatically by the factory to continously evolve with its rng
            operands = factory.generate_operands(
                factory.input_shapes, xp, dtype, order)
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
            out, out_ref, **get_contraction_tolerance(dtype))

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
            assert dtype is not None
            try:
                for cutn_grad, op in zip(input_grads, operands):
                    if cutn_grad is None or op.grad is None:
                        if cutn_grad is None and op.grad is None:
                            continue
                        raise RuntimeError(f"Inconsistent types between cutn and torch reference for input tensor gradients: cutn: {type(cutn_grad)} mismatching torch: {type(op.grad)}")
                    backend.testing.assert_close(
                            cutn_grad, op.grad,
                            **get_contraction_tolerance(dtype))
            except AssertionError as e:
                # for easier debugging
                print(tuple(op.shape for op in operands))
                print(input_grads)
                print(tuple(op.grad for op in operands))
                raise


@pytest.mark.parametrize(
    "gradient", (False, "random", "all")
)
@pytest.mark.parametrize(
    "dtype", dtype_names
)
@pytest.mark.parametrize(
    "xp", BACKEND_MEMSPACE
)
class TestNetworkFunctionality(NetworkBaseTester):

    @pytest.mark.parametrize(
        "release_workspace", (True, False)
    )
    def test_release_workspace(self, xp, dtype, gradient, release_workspace):
        rng = self._get_rng(xp, gradient, dtype, release_workspace, "release_workspace")
        self._test_runner(
            "ea,fb,abcd,gc,hd->efgh", # einsum_expr_pack
            xp, 
            dtype, 
            "C", #order,
            3, # autotune
            None, # stream
            False, # use_numpy_path
            gradient, # gradient
            False, # reset_none
            release_workspace, # release_workspace
            rng,
        )
    
    @pytest.mark.parametrize(
        "order", ("C", "F")
    )
    def test_order(self, xp, dtype, gradient, order):
        rng = self._get_rng(xp, gradient, dtype, order, "order")
        self._test_runner(
            ((2, 3, 4), (3, 4, 5), (2, 1), (1, 5), None), # einsum_expr_pack
            xp, 
            dtype, 
            order, # order
            False, # autotune
            None, # stream
            True, # use_numpy_path
            gradient, # gradient
            True, # reset_none
            False, # release_workspace
            rng,
        )

    @pytest.mark.parametrize(
        "reset_none", (True, False)
    )
    def test_reset_operands(self, xp, dtype, gradient, reset_none):
        rng = self._get_rng(xp, gradient, dtype, reset_none, "reset_operands")
        self._test_runner(
            (('a', 'b'), ('b', 'c', 'd'), ('a',)), # einsum_expr_pack
            xp,
            dtype, 
            "F", # order
            False, # autotune
            True, # stream
            False, # use_numpy_path
            gradient, # gradient
            reset_none, # reset_none
            True, # release_workspace
            rng,
        )
    
    @pytest.mark.parametrize(
        "stream", (None, True)
    )
    def test_stream(self, xp, dtype, gradient, stream):
        rng = self._get_rng(xp, gradient, dtype, stream, "stream")
        self._test_runner(
            ["abc,bcd,ade", {}, {"slicing": {"min_slices": 4}}, None], # einsum_expr_pack
            xp, 
            dtype, 
            "C", # order
            False, # autotune
            stream, # stream
            False, # use_numpy_path
            gradient, # gradient
            True, # reset_none
            False, # release_workspace
            rng,
        )
    
    @pytest.mark.parametrize(
        "autotune", (False, 3)
    )
    def test_autotune(self, xp, dtype, gradient, autotune):
        rng = self._get_rng(xp, gradient, dtype, autotune, "autotune")
        self._test_runner(
            [((5, 4, 3), (3, 4, 6), (6, 5), None), {}, {}, None], # einsum_expr_pack
            xp, 
            dtype, 
            "F", # order
            autotune, # autotune
            None, # stream
            False, # use_numpy_path
            gradient, # gradient
            False, # reset_none
            False, # release_workspace
            rng,
        )
    
    @pytest.mark.parametrize(
        "use_numpy_path", (False, True)
    )
    def test_use_numpy_path(self, xp, dtype, gradient, use_numpy_path):
        rng = self._get_rng(xp, gradient, dtype, use_numpy_path, "use_numpy_path")
        self._test_runner(
            "abc,ace,abd->de", # einsum_expr_pack
            xp, 
            dtype, 
            "C", # order
            False, # autotune
            None, # stream
            use_numpy_path, # use_numpy_path
            gradient, # gradient
            True, # reset_none
            True, # release_workspace
            rng,
        )


GRADIENT_OPTIONS = (False, "random", "all")

NUM_NETWORK_TESTS = len(GRADIENT_OPTIONS) * len(dtype_names) * len(BACKEND_MEMSPACE) * len(einsum_expressions)

NUM_TESTS_PER_CASE = 5

@pytest.mark.uncollect_if(func=Deselector.deselect_contract_tests)
@pytest.mark.parametrize(
    "gradient", GRADIENT_OPTIONS
)
@pytest.mark.parametrize(
    "dtype", dtype_names
)
@pytest.mark.parametrize(
    "xp", BACKEND_MEMSPACE
)
@pytest.mark.parametrize(
    "einsum_expr_pack", einsum_expressions
)
class TestNetworkCorrectness(NetworkBaseTester):

    def _test_config_iterator(self, einsum_expr_pack, xp, dtype, gradient):

        rng = self._get_rng(einsum_expr_pack, xp, dtype, gradient)
        TEST_CONFIGS = list(itertools.product(
            [None, True], # stream
            ['C', 'F'], # order
            [False, 3], # autotune
            [True, False], # use_numpy_path
            [True, False], # release_workspace,
            [True, False], # reset_none
        ))
        
        rng.shuffle(TEST_CONFIGS)
        yield from TEST_CONFIGS

    def test_network(self, einsum_expr_pack, xp, dtype, gradient):
        rng = self._get_rng(einsum_expr_pack, xp, dtype, gradient, "network")
        config_iter = self._test_config_iterator(einsum_expr_pack, xp, dtype, gradient)
        for _ in range(NUM_TESTS_PER_CASE):
            stream, order, autotune, use_numpy_path, release_workspace, reset_none = next(config_iter)
            self._test_runner(
                einsum_expr_pack,
                xp,
                dtype,
                order,
                autotune,
                stream,
                use_numpy_path,
                gradient,
                reset_none,
                release_workspace,
                rng,
            )


@contextlib.contextmanager
def disable_cupy_memory_pool():
    old_allocator = cp.cuda.get_allocator()
    cp.cuda.set_allocator(None)
    try:
        yield
    finally:
        cp.cuda.set_allocator(old_allocator)


@pytest.mark.skipif(cp is None, reason="cupy is not installed")
def test_none_memory_pool():
    entry_allocator = cp.cuda.get_allocator()
    with disable_cupy_memory_pool():
        a = cp.ones((2,2), dtype="float")
        b = cp.ones((2,2), dtype="float")

        qualifiers = np.zeros(2, dtype=cutn.tensor_qualifiers_dtype)
        qualifiers[0]["requires_gradient"] = True
        with Network("ab,bc->ac", a, b, qualifiers=qualifiers) as tn:
            tn.contract_path()
            out = tn.contract()
            act = cp.ones_like(out)
            grad = tn.gradients(act)[0]
            assert cp.allclose(grad, 2)
    exit_allocator = cp.cuda.get_allocator()
    assert entry_allocator is exit_allocator