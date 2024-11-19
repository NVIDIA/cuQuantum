# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Sequence
from itertools import product

import cupy as cp
import numpy as np
import pytest
from cuquantum.densitymat import WorkStream, DensePureState, DenseMixedState
from cuquantum.cutensornet._internal import utils


def get_state(ctx, hilbert_space_dims, batch_size, package, dtype, init="random", mixed=False):
    ctor_args = (ctx, hilbert_space_dims, batch_size, dtype)
    state = DenseMixedState(*ctor_args) if mixed else DensePureState(*ctor_args)
    shape, offsets = state.local_info

    _state = package.empty(shape, dtype=dtype, order="F")
    if init == "random":
        _state[:] = package.random.rand(*_state.shape) - 0.5
        if "complex" in dtype:
            _state[:] += 1j * (package.random.rand(*_state.shape) - 0.5)
    elif init == "zeros":
        _state[:] = 0.0
    state.attach_storage(_state)
    return state


@pytest.fixture
def work_stream():
    # NOTE: If random seeds are set at module or class level, some single-precision tests fail
    np.random.seed(42)
    cp.random.seed(42)
    return WorkStream()


@pytest.fixture(params=list(product([(2,), (2, 3)], [cp], ["random"], [True, False])))
def state(request, work_stream):
    hilbert_space_dims, package, init, mixed = request.param

    def _state(batch_size, dtype):
        return get_state(work_stream, hilbert_space_dims, batch_size, package, dtype, init, mixed)

    return _state


class TestState:

    @pytest.mark.parametrize("batch_size,factors", [(1, 2.0)])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
    def test_inplace_scale_different_dtypes(self, state, batch_size, dtype, factors):
        psi = state(batch_size, dtype)
        psi_arr = psi.storage.get()

        psi.inplace_scale(factors)
        scaled_psi_arr = psi.storage.get()

        factors_np = self._return_numpy_factors(factors, dtype)
        ref = psi_arr * factors_np
        np.testing.assert_allclose(scaled_psi_arr, ref)

    @pytest.mark.parametrize("batch_size,dtype", [(2, "float64")])
    @pytest.mark.parametrize(
        "factors", [2.0, (2.0, 3.0), np.array([2.0, 3.0]), cp.array([2.0, 3.0])]
    )
    def test_inplace_scale_different_factors(self, state, batch_size, dtype, factors):
        psi = state(batch_size, dtype)
        psi_arr = psi.storage.get()

        psi.inplace_scale(factors)
        scaled_psi_arr = psi.storage.get()

        factors_np = self._return_numpy_factors(factors, dtype)
        ref = psi_arr * factors_np
        np.testing.assert_allclose(scaled_psi_arr, ref)

    @pytest.mark.parametrize("dtype", ["float64"])
    @pytest.mark.parametrize(
        "batch_size,factors",
        [
            (2, (1.0, 2.0, 3.0)),
            (2, np.array([[1.0, 2.0], [3.0, 4.0]])),
        ],
    )
    def test_inplace_scale_improper_factors_shape(self, state, batch_size, dtype, factors):
        psi = state(batch_size, dtype)
        with pytest.raises(ValueError):
            psi.inplace_scale(factors)

    @pytest.mark.parametrize("dtype", ["float64"])
    @pytest.mark.parametrize(
        "batch_size,factors",
        [
            (2, {1.0, 2.0}),
            (2, (1, 2 + 3j)),
        ],
    )
    def test_inplace_scale_improper_factors_type(self, state, batch_size, dtype, factors):
        psi = state(batch_size, dtype)
        with pytest.raises(TypeError):
            psi.inplace_scale(factors)

    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize(
        "dtype,factors",
        [
            ("complex128", np.array([1.0, 2.0 + 3.0j])),
            ("float32", cp.array([2.0, 3.0], dtype="float64")),
            pytest.param(
                "float64", cp.array([1 + 2j, 3 + 4j]), marks=pytest.mark.xfail(raises=TypeError)
            ),
        ],
    )
    def test_inplace_scale_dtype_factors_compatibility(self, state, batch_size, dtype, factors):
        psi = state(batch_size, dtype)
        psi.inplace_scale(factors)

    @pytest.mark.parametrize("batch_size,factors", [(1, 2.0)])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
    def test_inplace_accumulate_different_dtypes(self, state, batch_size, dtype, factors):
        psi1 = state(batch_size, dtype)
        psi2 = psi1.clone(cp.zeros_like(psi1.storage))
        psi2.storage[:] = cp.random.rand(*psi2.storage.shape)
        psi1_arr = psi1.storage.get()
        psi2_arr = psi2.storage.get()

        psi1.inplace_accumulate(psi2, factors)
        accumulated_psi_arr = psi1.storage.get()

        factors_np = self._return_numpy_factors(factors, dtype)
        ref = factors_np * psi2_arr + psi1_arr
        np.testing.assert_allclose(accumulated_psi_arr, ref)

    @pytest.mark.parametrize("batch_size,dtype", [(2, "float64")])
    @pytest.mark.parametrize(
        "factors", [2.0, (2.0, 3.0), np.array([2.0, 3.0]), cp.array([2.0, 3.0])]
    )
    def test_inplace_accumulate_different_factors(self, state, batch_size, dtype, factors):
        psi1 = state(batch_size, dtype)
        psi2 = psi1.clone(cp.zeros_like(psi1.storage))
        psi2.storage[:] = cp.random.rand(*psi2.storage.shape)
        psi1_arr = psi1.storage.get()
        psi2_arr = psi2.storage.get()

        psi1.inplace_accumulate(psi2, factors)
        accumulated_psi_arr = psi1.storage.get()

        factors_np = self._return_numpy_factors(factors, dtype)
        ref = factors_np * psi2_arr + psi1_arr
        np.testing.assert_allclose(accumulated_psi_arr, ref)

    @pytest.mark.parametrize("dtype", ["float64"])
    @pytest.mark.parametrize(
        "batch_size,factors",
        [
            (2, (1.0, 2.0, 3.0)),
            (2, np.array([[1.0, 2.0], [3.0, 4.0]])),
        ],
    )
    def test_inplace_accumulate_improper_factors_shape(self, state, batch_size, dtype, factors):
        psi1 = state(batch_size, dtype)
        psi2 = psi1.clone(cp.zeros_like(psi1.storage))
        with pytest.raises(ValueError):
            psi1.inplace_accumulate(psi2, factors)

    @pytest.mark.parametrize("dtype", ["float64"])
    @pytest.mark.parametrize(
        "batch_size,factors",
        [
            (1, 1 + 2j),
            (2, {1.0, 2.0}),
            (2, (1, 2 + 3j)),
        ],
    )
    def test_inplace_accumulate_improper_factors_type(self, state, batch_size, dtype, factors):
        psi1 = state(batch_size, dtype)
        psi2 = psi1.clone(cp.zeros_like(psi1.storage))
        with pytest.raises(TypeError):
            psi1.inplace_accumulate(psi2, factors)

    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize(
        "dtype,factors",
        [
            ("complex128", np.array([1.0, 2.0 + 3.0j])),
            ("float32", cp.array([2.0, 3.0], dtype="float64")),
            pytest.param(
                "float64", cp.array([1 + 2j, 3 + 4j]), marks=pytest.mark.xfail(raises=TypeError)
            ),
        ],
    )
    def test_inplace_accumulate_dtype_factors_compatibility(
        self, state, batch_size, dtype, factors
    ):
        psi1 = state(batch_size, dtype)
        psi2 = psi1.clone(cp.zeros_like(psi1.storage))
        psi1.inplace_accumulate(psi2, factors)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
    def test_state_inner_product(self, state, batch_size, dtype):
        psi1 = state(batch_size, dtype)
        psi2 = psi1.clone(cp.zeros_like(psi1.storage))
        psi2.storage[:] = cp.random.rand(*psi2.storage.shape)
        psi1_arr = psi1.storage.get()
        psi2_arr = psi2.storage.get()

        inner_prod = psi1.inner_product(psi2)
        inner_prod_arr = inner_prod.get()

        psi1_arr = psi1_arr.reshape((-1, psi1_arr.shape[-1]), order="F")
        psi2_arr = psi2_arr.reshape((-1, psi2_arr.shape[-1]), order="F")
        ref = np.zeros((batch_size,), dtype=inner_prod.dtype)
        for i in range(batch_size):
            ref[i] = np.vdot(psi1_arr[:, i], psi2_arr[:, i])

        np.testing.assert_allclose(inner_prod_arr, ref, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
    def test_norm(self, state, batch_size, dtype):
        psi = state(batch_size, dtype)
        psi_arr = psi.storage.get()
        norm = psi.norm().get()

        psi_arr = psi_arr.reshape((-1, psi_arr.shape[-1]), order="F")
        ref = np.empty((batch_size,), dtype=psi.storage.real.dtype)
        for i in range(batch_size):
            ref[i] = np.vdot(psi_arr[:, i], psi_arr[:, i]).real

        np.testing.assert_allclose(norm, ref, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
    def test_trace(self, state, batch_size, dtype):
        psi = state(batch_size, dtype)
        psi_arr = psi.storage.get()

        trace = psi.trace()

        ref = cp.empty((batch_size,), dtype=dtype)
        if isinstance(psi, DensePureState):
            psi_arr = psi_arr.reshape((-1, psi_arr.shape[-1]), order="F")
            for i in range(batch_size):
                ref[i] = np.vdot(psi_arr[:, i], psi_arr[:, i])
        else:
            psi_arr = psi_arr.reshape(
                (np.prod(psi.hilbert_space_dims), np.prod(psi.hilbert_space_dims), batch_size),
                order="F",
            )
            for i in range(batch_size):
                ref[i] = cp.trace(psi_arr[:, :, i])

        cp.testing.assert_allclose(trace, ref, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
    def test_attach_storage(self, state, batch_size, dtype):
        psi = state(batch_size, dtype)
        shape, _ = psi.local_info
        with utils.device_ctx(psi._ctx.device_id):
            psi_arr = cp.zeros(shape, dtype=dtype, order="F")
            psi_arr_wrong_shape = cp.zeros([x + 1 for x in shape], dtype=dtype, order="F")
            psi_arr_c_order = cp.zeros(shape, dtype=dtype, order="C")

        psi.attach_storage(psi_arr)
        with pytest.raises(ValueError):
            psi.clone(psi_arr_wrong_shape)
        if len(psi.hilbert_space_dims) > 1:
            with pytest.raises(ValueError):
                psi.clone(psi_arr_c_order)

    @staticmethod
    def _return_numpy_factors(factors, dtype):
        if isinstance(factors, Sequence):
            factors_np = np.array(factors, dtype=dtype)
        elif isinstance(factors, cp.ndarray):
            factors_np = factors.get()
        else:  # single number or numpy array
            factors_np = factors
        return factors_np
