# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import gc
import itertools
import sys
from typing import Any

import opt_einsum
import pytest

from nvmath.internal.utils import infer_object_package

from cuquantum import MemoryLimitExceeded
from cuquantum.tensornet import contract, einsum, OptimizerInfo
from cuquantum.tensornet import tensor_network
from cuquantum.tensornet._internal import grad_torch
from cuquantum.bindings import cutensornet as cutn

from .utils.data import BACKEND_MEMSPACE, dtype_names, einsum_expressions
from .utils.helpers import EinsumFactory, get_contraction_tolerance
from .utils.helpers import compute_and_normalize_numpy_path
from .utils.helpers import get_stream_for_backend
from .utils.helpers import set_path_to_optimizer_options
from .utils.helpers import _BaseTester
from .utils.helpers import Deselector
from .utils.helpers import cleanup_between_tests # fixture to clean up between tests


class _TestContractBase(_BaseTester):

    def _get_config_iterator(self, xp, dtype, gradient, einsum_expr_pack):
        rng = self._get_rng(einsum_expr_pack, xp, dtype, gradient)
        TEST_CONFIGS = list(itertools.product(
            [True, False], # use_numpy_path
            ["C", "F"], # order
            [None, True], # stream
            [True, False], # return_info
        ))
        rng.shuffle(TEST_CONFIGS)
        yield from TEST_CONFIGS

    def _test_runner(
            self, func, einsum_expr_pack, xp, dtype, order,
            use_numpy_path, gradient, rng, **kwargs):
        einsum_expr = copy.deepcopy(einsum_expr_pack)
        if isinstance(einsum_expr, list):
            einsum_expr, network_opts, optimizer_opts, _ = einsum_expr
        else:
            network_opts = optimizer_opts = None
        assert isinstance(einsum_expr, (str, tuple))

        factory = EinsumFactory(einsum_expr, rng)
        operands = factory.generate_operands(
            factory.input_shapes, xp, dtype, order)
        qualifiers, picks = factory.generate_qualifiers(xp, gradient)
        factory.setup_torch_grads(xp, picks, operands)
        backend_name = infer_object_package(operands[0])
        backend = sys.modules[backend_name]
        stream = kwargs.get('stream')
        if stream:
            stream_obj = get_stream_for_backend(backend)
            if stream == "as_int":
                if backend_name == "numpy" and stream_obj is None:
                    # CuPy is not installed, skip this test
                    pytest.skip("CuPy is not installed, skipping this test for numpy with stream as int")
                elif backend_name in {"numpy", "cupy"}:
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
        if func is contract:
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
            except MemoryLimitExceeded as e:
                pytest.skip("Insufficient workspace memory available.")

            if return_info:
                out, (path, info) = out
                assert isinstance(path, list)
                assert isinstance(info, OptimizerInfo)

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

        else:  # einsum()
            optimize = kwargs.pop('optimize')
            if optimize == 'path':
                optimize = path if path is not None else False
            try:
                out = func(*data, optimize=optimize)
            except cutn.cuTensorNetError as e:
                if (optimize is not True
                        and "CUTENSORNET_STATUS_NOT_SUPPORTED" in str(e)):
                    pytest.skip("einsum() fail -- TN too large?")
                else:
                    raise
            except MemoryLimitExceeded as e:
                pytest.skip("Insufficient workspace memory available.")

        backend_out = sys.modules[infer_object_package(out)]
        assert backend_out is backend
        assert out.dtype == operands[0].dtype

        # check contraction
        factory.setup_torch_grads(xp, picks, operands)
        out_ref = opt_einsum.contract(
            *data, backend="torch" if "torch" in xp else xp)
        tolerance = get_contraction_tolerance(dtype)
        assert backend.allclose(
            out, out_ref, **tolerance)

        # check gradients
        if gradient and func is contract:
            out_ref.backward(output_grad)

            # check gradients
            try:
                is_close = backend.tensor(tuple(
                    backend.allclose(
                        cutn_grad, op.grad,
                        **tolerance)
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


@pytest.mark.uncollect_if(func=Deselector.deselect_gradient_tests)
@pytest.mark.parametrize(
    "dtype", dtype_names
)
@pytest.mark.parametrize(
    "gradient", (False, "random", "all")
)
@pytest.mark.parametrize(
    "xp", BACKEND_MEMSPACE
)
class TestContractFunctionality(_TestContractBase):

    @pytest.mark.parametrize(
        "stream", (None, True, "as_int")
    )
    def test_stream(self, xp, gradient, dtype, stream):
        rng = self._get_rng(xp, gradient, dtype, stream, "stream")
        self._test_runner(
            contract,
            ["abc,bcd,ade", {}, {"slicing": {"min_slices": 4}}, None], # einsum_expr_pack
            xp,
            dtype,
            "C", # order
            False, # use_numpy_path
            gradient,
            rng,
            stream=stream,
            return_info=False,
        )

    @pytest.mark.parametrize(
        "return_info", (False, True)
    )
    def test_return_info(self, xp, gradient, dtype, return_info):
        rng = self._get_rng(xp, gradient, dtype, return_info, "return_info")
        self._test_runner(
            contract,
            [((5, 4, 3), (3, 4, 6), (6, 5), None), {}, {}, None], # einsum_expr_pack
            xp,
            dtype,
            "F", # order
            True, # use_numpy_path
            gradient,
            rng,
            stream=True,
            return_info=return_info,
        )

    @pytest.mark.parametrize(
        "order", ("C", "F")
    )
    def test_order(self, xp, gradient, dtype, order):
        rng = self._get_rng(xp, gradient, dtype, order, "order")
        self._test_runner(
            contract,
            "abc,ace,abd->de", # einsum_expr_pack
            xp,
            dtype,
            order,
            False, # use_numpy_path
            gradient,
            rng,
            stream=None,
            return_info=False,
        )

    @pytest.mark.parametrize(
        "use_numpy_path", (False, True)
    )
    def test_use_numpy_path(self, xp, gradient, dtype, use_numpy_path):
        rng = self._get_rng(xp, gradient, dtype, use_numpy_path, "use_numpy_path")
        self._test_runner(
            contract,
            (('a', 'b'), ('b', 'c', 'd'), ('a',)), # einsum_expr_pack
            xp,
            dtype,
            "C", # order
            use_numpy_path,
            gradient,
            rng,
            stream=True,
            return_info=True,
        )

NUM_TESTS_PER_CASE = 5


@pytest.mark.uncollect_if(func=Deselector.deselect_contract_tests)
@pytest.mark.parametrize(
    "gradient", (False, "random", "all")
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
class TestContract(_TestContractBase):

    def test_contract(self, einsum_expr_pack, xp, dtype, gradient):
        rng = self._get_rng(einsum_expr_pack, xp, dtype, gradient, "contract")
        config_iter = self._get_config_iterator(
            xp, dtype, gradient, einsum_expr_pack)
        for _ in range(NUM_TESTS_PER_CASE):
            use_numpy_path, order, stream, return_info = next(config_iter)
            self._test_runner(
                contract, einsum_expr_pack, xp, dtype, order,
                use_numpy_path, gradient, rng, stream=stream, return_info=return_info)


# einsum does not support gradient (at some point we should deprecate it...)
@pytest.mark.uncollect_if(func=Deselector.deselect_einsum_tests)
@pytest.mark.parametrize(
    "optimize", (False, True, "path")
)
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
    "xp", BACKEND_MEMSPACE
)
@pytest.mark.parametrize(
    "einsum_expr_pack", einsum_expressions
)
class TestEinsum(_TestContractBase):

    def test_einsum(
            self, einsum_expr_pack, xp, dtype, order,
            use_numpy_path, optimize):
        rng = self._get_rng(einsum_expr_pack, xp, dtype, optimize, "einsum")
        self._test_runner(
            einsum, einsum_expr_pack, xp, dtype, order,
            use_numpy_path, None, rng, optimize=optimize)


def _make_fake_torch_network(
        torch: Any, instances: list[Any], *, fail_forward: bool = False) -> type[Any]:
    """Create a fake Torch-backed Network that tracks free() calls."""

    class FakeNetwork:
        def __init__(self, *operands: Any, **kwargs: Any) -> None:
            self.package = "torch"
            self.operand = next(operand for operand in operands if torch.is_tensor(operand))
            self.device_id = 0
            self.is_interleaved = False
            self.valid_state = True
            self.free_calls = 0
            instances.append(self)

        def contract_path(self, optimize: Any = None) -> tuple[list[Any], None]:
            if fail_forward:
                raise RuntimeError("forward failed")
            return [], None

        def contract(self, stream: Any = None) -> Any:
            return self.operand * 2

        def gradients(self, output_grad: Any, stream: Any = None) -> tuple[Any]:
            return (output_grad * 2,)

        def free(self) -> None:
            if self.valid_state:
                self.valid_state = False
                self.free_calls += 1

    return FakeNetwork


@pytest.mark.parametrize(
    "operand_requires_grad, grad_enabled",
    ((False, True), (True, False)),
)
def test_torch_contract_frees_network_without_autograd(
        monkeypatch: pytest.MonkeyPatch,
        operand_requires_grad: bool,
        grad_enabled: bool) -> None:
    """Verify immediate Network cleanup when no autograd graph is created."""

    torch = pytest.importorskip("torch")
    instances: list[Any] = []
    monkeypatch.setattr(
        tensor_network, "Network", _make_fake_torch_network(torch, instances))
    operand = torch.ones(2, requires_grad=operand_requires_grad)

    with torch.set_grad_enabled(grad_enabled):
        output = contract("i->i", operand)

    # Exactly one free proves both no leak and idempotent ownership.
    assert instances[0].free_calls == 1
    # Neither disabled grad mode nor a constant input can produce an autograd graph.
    assert not output.requires_grad


@pytest.mark.parametrize("completion", ("backward", "discard"))
def test_torch_contract_frees_network_after_graph_completion(
        monkeypatch: pytest.MonkeyPatch, completion: str) -> None:
    """Verify Network cleanup after backward or graph disposal."""

    torch = pytest.importorskip("torch")
    instances: list[Any] = []
    monkeypatch.setattr(
        tensor_network, "Network", _make_fake_torch_network(torch, instances))

    class FakeDevice:
        def __init__(self, *args: Any) -> None:
            pass

        def set_current(self) -> None:
            pass

    monkeypatch.setattr(grad_torch, "Device", FakeDevice)
    operand = torch.ones(2, requires_grad=True)
    output = contract("i->i", operand)

    # A live graph must retain the Network until backward or graph disposal.
    assert instances[0].free_calls == 0
    if completion == "backward":
        output.sum().backward()
        # The fake gradient confirms that backward still maps gradients correctly.
        assert torch.equal(operand.grad, torch.full_like(operand, 2))
    else:
        del output
        gc.collect()

    # Either graph-completion route must release the Network exactly once.
    assert instances[0].free_calls == 1


def test_torch_contract_frees_network_on_forward_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Network cleanup when the autograd forward path fails."""

    torch = pytest.importorskip("torch")
    instances: list[Any] = []
    monkeypatch.setattr(
        tensor_network, "Network",
        _make_fake_torch_network(torch, instances, fail_forward=True))
    operand = torch.ones(2, requires_grad=True)

    with pytest.raises(RuntimeError, match="forward failed"):
        contract("i->i", operand)

    # Cleanup on the error path must destroy the internally owned Network once.
    assert instances[0].free_calls == 1


def test_torch_contract_frees_network_on_grad_torch_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Network cleanup when grad_torch fails to import on first use."""
    torch = pytest.importorskip("torch")
    instances: list[Any] = []
    monkeypatch.setattr(tensor_network, "Network", _make_fake_torch_network(torch, instances))
    # This module already imported grad_torch; undo that so contract() imports it again.
    monkeypatch.delattr(sys.modules["cuquantum.tensornet._internal"], "grad_torch")
    monkeypatch.setitem(sys.modules, "cuquantum.tensornet._internal.grad_torch", None)

    with pytest.raises(ImportError, match="grad_torch"):
        contract("i->i", torch.ones(2))

    assert instances[0].free_calls == 1
