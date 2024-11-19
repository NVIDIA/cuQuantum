# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ElementaryOperator."""

from itertools import product

import pytest
import numpy as np
import scipy as sp
import cupy as cp
from scipy.sparse import dia_matrix as sp_dia_matrix
from cupyx.scipy.sparse import dia_matrix as cp_dia_matrix

from cuquantum.densitymat import DenseOperator, MultidiagonalOperator


np.random.seed(42)
cp.random.seed(42)


@pytest.fixture(scope="class")
def callback_args():
    t = 1.0
    args = [1.0, 2.0, 3.0]
    return t, args


@pytest.fixture
def dense_operator(request):
    hilbert_space_dims, order, package, has_callback = request.param
    shape = (*hilbert_space_dims, *hilbert_space_dims)
    if has_callback:
        data = package.empty(shape, order=order)

        def callback(t, args):
            _data = np.empty(shape, order=order)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    _data[i, j] = np.sin((i + 2 * j) * t * np.sum(args))
            return _data

    else:
        data = package.asarray(np.random.rand(*shape), order=order)
        callback = None
    return DenseOperator(data, callback)


@pytest.fixture
def dense_operator_(request):
    hilbert_space_dims, order, package, has_callback = request.param
    shape = (*hilbert_space_dims, *hilbert_space_dims)
    if has_callback:
        data = package.empty(shape, order=order)

        def callback(t, args):
            _data = np.empty(shape, order=order)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    _data[i, j] = np.cos((i + 2 * j) * t * np.sum(args))
            return _data

    else:
        data = package.asarray(np.random.rand(*shape), order=order)
        callback = None
    return DenseOperator(data, callback)


@pytest.fixture
def multidiagonal_operator(request):
    dim, num_diags, order, package, has_callback = request.param
    if has_callback:
        data = package.empty((dim, num_diags), order=order)
        shape = data.shape

        def callback(t, args):
            _data = np.empty(shape, order=order)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    _data[i, j] = np.sin((i + 2 * j) * t * np.sum(args))
            return _data

    else:
        data = package.asarray(np.random.random((dim, num_diags)), order=order)
        callback = None
    offsets = list(np.random.choice(range(-dim + 1, dim + 1), size=num_diags, replace=False))
    return MultidiagonalOperator(data, offsets, callback=callback)


@pytest.fixture
def multidiagonal_operator_(request):
    dim, num_diags, order, package, has_callback = request.param
    if has_callback:
        data = package.empty((dim, num_diags), order=order)
        shape = data.shape

        def callback(t, args):
            _data = np.empty(shape, order=order)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    _data[i, j] = np.cos((i + 2 * j) * t * np.sum(args))
            return _data

    else:
        data = package.random.random((dim, num_diags))
        callback = None
    offsets = list(np.random.choice(range(-dim + 1, dim + 1), size=num_diags, replace=False))
    return MultidiagonalOperator(data, offsets, callback=callback)


@pytest.fixture
def dia_matrix(request):
    dim, num_diags, dia_matrix_func, package = request.param
    data = package.random.random((num_diags, dim))
    offsets = list(np.random.choice(range(-dim + 1, dim + 1), size=num_diags, replace=False))
    return dia_matrix_func((data, offsets), shape=(dim, dim))


@pytest.mark.usefixtures("callback_args")
@pytest.mark.parametrize(
    "dense_operator",
    list(
        product(
            [(3,), (3, 4)],
            ["C", "F"],
            [np, cp],
            [False, True],
        )
    ),
    indirect=True,
)
class TestDenseOperatorUnaryOperations:

    @pytest.mark.parametrize("scalar", [2.3])
    def test_left_scalar_multiplication(self, dense_operator, callback_args, scalar):
        dense_op = dense_operator
        t, args = callback_args

        scaled_dense_op = scalar * dense_op

        dense_op_arr = dense_op.to_array(t, args)
        ref = scalar * dense_op_arr
        np.testing.assert_allclose(scaled_dense_op.to_array(t, args), ref)

    @pytest.mark.parametrize("scalar", [2.3])
    def test_right_scalar_multiplication(self, dense_operator, callback_args, scalar):
        dense_op = dense_operator
        t, args = callback_args

        scaled_dense_op = dense_op * scalar

        dense_op_arr = dense_op.to_array(t, args)
        ref = dense_op_arr * scalar
        np.testing.assert_allclose(scaled_dense_op.to_array(t, args), ref)

    def test_conjugate_transpose(self, dense_operator, callback_args):
        dense_op = dense_operator
        t, args = callback_args
        dense_op_arr = dense_operator.to_array(t, args)

        dense_op_dag = dense_op.dag()

        n = dense_op_arr.ndim
        indices = list(range(n // 2, n)) + list(range(n // 2))
        ref = dense_op_arr.transpose(*indices).conj()
        np.testing.assert_allclose(dense_op_dag.to_array(t, args), ref)


@pytest.mark.parametrize(
    "dense_operator,dense_operator_",
    list(
        product(
            list(product([(3,)], ["C", "F"], [np], [False, True])),
            list(product([(3,)], ["C", "F"], [np], [False, True])),
        )
    )
    + list(
        product(
            list(product([(3,)], ["C", "F"], [cp], [False, True])),
            list(product([(3,)], ["C", "F"], [cp], [False, True])),
        )
    ),
    indirect=True,
)
class TestDenseOperatorBinaryOperations:

    def test_addition(self, dense_operator, dense_operator_, callback_args):
        dense_op1 = dense_operator
        dense_op2 = dense_operator_
        t, args = callback_args

        dense_op_sum = dense_op1 + dense_op2

        dense_op1_arr = dense_op1.to_array(t, args)
        dense_op2_arr = dense_op2.to_array(t, args)
        ref = dense_op1_arr + dense_op2_arr
        np.testing.assert_allclose(dense_op_sum.to_array(t, args), ref)

    def test_subtraction(self, dense_operator, dense_operator_, callback_args):
        dense_op1 = dense_operator
        dense_op2 = dense_operator_
        t, args = callback_args

        dense_op_diff = dense_op1 - dense_op2

        dense_op1_arr = dense_op1.to_array(t, args)
        dense_op2_arr = dense_op2.to_array(t, args)
        ref = dense_op1_arr - dense_op2_arr
        np.testing.assert_allclose(dense_op_diff.to_array(t, args), ref)

    def test_matrix_multiplication(self, dense_operator, dense_operator_, callback_args):
        dense_op1 = dense_operator
        dense_op2 = dense_operator_
        t, args = callback_args

        dense_op_prod = dense_op1 @ dense_op2

        if len(dense_op1.shape) == 2:
            subscripts = "ab,bc->ac"
        elif len(dense_op1.shape) == 4:
            subscripts = "abcd,cdef->abef"

        dense_op1_arr = dense_op1.to_array(t, args)
        dense_op2_arr = dense_op2.to_array(t, args)
        ref = np.einsum(subscripts, dense_op1_arr, dense_op2_arr)
        np.testing.assert_allclose(dense_op_prod.to_array(t, args), ref)


@pytest.mark.parametrize(
    "multidiagonal_operator",
    list(product([4], [3], ["C", "F"], [np, cp], [False, True])),
    indirect=True,
)
class TestMultidiagonalOperatorUnaryOperations:

    @pytest.mark.parametrize("scalar", [2.3])
    def test_left_scalar_multiplication(self, multidiagonal_operator, callback_args, scalar):
        dia_op = multidiagonal_operator
        t, args = callback_args
        dia_op_arr = dia_op.to_array(t, args)

        scaled_dia_op = scalar * dia_op

        ref = scalar * dia_op_arr
        np.testing.assert_allclose(scaled_dia_op.to_array(t, args), ref)

    @pytest.mark.parametrize("scalar", [2.3])
    def test_right_scalar_multiplication(self, multidiagonal_operator, callback_args, scalar):
        dia_op = multidiagonal_operator
        t, args = callback_args
        dia_op_arr = multidiagonal_operator.to_array(t, args)

        scaled_dia_op = dia_op * scalar

        ref = scalar * dia_op_arr
        np.testing.assert_allclose(scaled_dia_op.to_array(t, args), ref)

    def test_conjugate_transpose(self, multidiagonal_operator, callback_args):
        dia_op = multidiagonal_operator
        t, args = callback_args
        dia_op_arr = dia_op.to_array(t, args)

        dia_op_dag = dia_op.dag()

        ref = dia_op_arr.conj().T
        np.testing.assert_allclose(dia_op_dag.to_array(t, args), ref)


@pytest.mark.parametrize(
    "multidiagonal_operator,multidiagonal_operator_",
    list(
        product(
            list(product([4], [3], ["C", "F"], [np], [False, True])),
            list(product([4], [2], ["C", "F"], [np], [False, True])),
        )
    )
    + list(
        product(
            list(product([4], [3], ["C", "F"], [cp], [False, True])),
            list(product([4], [2], ["C", "F"], [cp], [False, True])),
        )
    ),
    indirect=True,
)
class TestMultidiagonalOperatorBinaryOperations:

    def test_addition(self, multidiagonal_operator, multidiagonal_operator_, callback_args):
        dia_op1 = multidiagonal_operator
        dia_op2 = multidiagonal_operator_
        t, args = callback_args
        dia_op1_arr = dia_op1.to_array(t, args)
        dia_op2_arr = dia_op2.to_array(t, args)

        dia_op_sum = dia_op1 + dia_op2

        ref = dia_op1_arr + dia_op2_arr
        np.testing.assert_allclose(dia_op_sum.to_array(t, args), ref)

    def test_subtraction(self, multidiagonal_operator, multidiagonal_operator_, callback_args):
        dia_op1 = multidiagonal_operator
        dia_op2 = multidiagonal_operator_
        t, args = callback_args
        dia_op1_arr = dia_op1.to_array(t, args)
        dia_op2_arr = dia_op2.to_array(t, args)

        dia_op_diff = dia_op1 - dia_op2

        ref = dia_op1_arr - dia_op2_arr
        np.testing.assert_allclose(dia_op_diff.to_array(t, args), ref)

    def test_matrix_multiplication(
        self, multidiagonal_operator, multidiagonal_operator_, callback_args
    ):
        dia_op1 = multidiagonal_operator
        dia_op2 = multidiagonal_operator_
        t, args = callback_args
        dia_op1_arr = dia_op1.to_array(t, args)
        dia_op2_arr = dia_op2.to_array(t, args)

        dia_op_prod = dia_op1 @ dia_op2

        ref = dia_op1_arr @ dia_op2_arr
        np.testing.assert_allclose(dia_op_prod.to_array(t, args), ref)


@pytest.mark.parametrize(
    "dense_operator,multidiagonal_operator",
    list(
        product(
            list(product([(4,)], ["C", "F"], [np], [False, True])),
            list(product([4], [2], ["C", "F"], [np], [False, True])),
        )
    )
    + list(
        product(
            list(product([(4,)], ["C", "F"], [cp], [False, True])),
            list(product([4], [2], ["C", "F"], [cp], [False, True])),
        )
    ),
    indirect=True,
)
class TestMixedOperations:

    def test_dense_multidiagonal_addition(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        t, args = callback_args
        dense_op_arr = dense_op.to_array(t, args)
        dia_op_arr = dia_op.to_array(t, args)

        dense_dia_op_sum = dense_op + dia_op

        ref = dense_op_arr + dia_op_arr
        np.testing.assert_allclose(dense_dia_op_sum.to_array(t, args), ref)

    def test_multidiagonal_dense_addition(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        t, args = callback_args
        dense_op_arr = dense_op.to_array(t, args)
        dia_op_arr = dia_op.to_array(t, args)

        dense_dia_op_sum = dia_op + dense_op

        ref = dense_op_arr + dia_op_arr
        np.testing.assert_allclose(dense_dia_op_sum.to_array(t, args), ref)

    def test_dense_multidiagonal_subtraction(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        t, args = callback_args
        dense_op_arr = dense_op.to_array(t, args)
        dia_op_arr = dia_op.to_array(t, args)

        dense_dia_op_diff = dense_op - dia_op

        ref = dense_op_arr - dia_op_arr
        np.testing.assert_allclose(dense_dia_op_diff.to_array(t, args), ref)

    def test_multidiagonal_dense_subtraction(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        t, args = callback_args
        dense_op_arr = dense_op.to_array(t, args)
        dia_op_arr = dia_op.to_array(t, args)

        dense_dia_op_diff = dia_op - dense_op

        ref = dia_op_arr - dense_op_arr
        np.testing.assert_allclose(dense_dia_op_diff.to_array(t, args), ref)

    def test_dense_multidiagonal_matrix_multiplication(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        t, args = callback_args
        dense_op_arr = dense_op.to_array(t, args)
        dia_op_arr = dia_op.to_array(t, args)

        dense_dia_op_prod = dense_op @ dia_op

        ref = dense_op_arr @ dia_op_arr
        np.testing.assert_allclose(dense_dia_op_prod.to_array(t, args), ref)

    def test_multidiagonal_dense_matrix_multiplication(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        t, args = callback_args
        dense_op_arr = dense_op.to_array(t, args)
        dia_op_arr = dia_op.to_array(t, args)

        dense_dia_op_prod = dia_op @ dense_op

        ref = dia_op_arr @ dense_op_arr
        np.testing.assert_allclose(dense_dia_op_prod.to_array(t, args), ref)
