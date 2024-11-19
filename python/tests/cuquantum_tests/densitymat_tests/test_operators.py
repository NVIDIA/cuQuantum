# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp
import pytest

from cuquantum.densitymat import (
    tensor_product,
    DenseOperator,
    MultidiagonalOperator,
    Operator,
    OperatorAction,
    WorkStream,
)


@pytest.fixture
def work_stream():
    np.random.seed(42)
    cp.random.seed(42)
    return WorkStream()


def get_dia_example(hilbert_space_dims):
    A = DenseOperator(np.random.rand(*((hilbert_space_dims[2],) * 2)))
    B = MultidiagonalOperator(np.random.rand(hilbert_space_dims[3], 3), [-1, 0, 1])

    ab = tensor_product((A, (2,)), (B, (3,)), coeff=lambda t, args: np.sin(args[0] * t))
    ab2 = tensor_product((A, (2,)), (B, (3,)), coeff=2)
    return ab, ab2


def get_dense_example(hilbert_space_dims):
    A = DenseOperator(np.random.rand(*((hilbert_space_dims[2],) * 2)))
    B = DenseOperator(np.random.rand(*((hilbert_space_dims[3], hilbert_space_dims[5]) * 2)))

    ab = tensor_product((A, (2,)), (B, (3, 5)), coeff=lambda t, args: np.sin(args[0] * t))
    ab2 = tensor_product((A, (2,)), (B, (3, 5)), coeff=2)
    return ab, ab2


class TestOperators:

    @pytest.mark.parametrize("hilbert_space_dims", [(4, 5, 2, 6, 3, 7)])
    @pytest.mark.parametrize("example_getter", [get_dense_example, get_dia_example])
    def test_operator_term(self, hilbert_space_dims, example_getter):
        a, b = example_getter(hilbert_space_dims)
        c, d = example_getter(hilbert_space_dims)
        a_times_b = self._test_multiplication_operatorterms(a, b)
        c_times_d = self._test_multiplication_operatorterms(c, d)
        a_plus_b = self._test_addition_operatorterms(a, b)
        c_plus_d = self._test_addition_operatorterms(c, d)
        a_plus_b_times_c_plus_d = self._test_multiplication_operatorterms(a_plus_b, c_plus_d)
        a_times_b_plus_c_times_d = self._test_addition_operatorterms(a_times_b, c_times_d)
        assert len(a_plus_b_times_c_plus_d.terms) == 4
        assert len(a_times_b_plus_c_times_d.terms) == 2

        # test dag method
        ops = []
        for op in a.terms[0][::-1]:
            ops.append(op)
        adag = a.dag()
        ops_dag = []
        for op in adag.terms[0]:
            ops_dag.append(op)
        for op, op_dag in zip(ops, ops_dag):
            n = len(op.shape)
            indices = list(range(n // 2, n)) + list(range(n // 2))
            np.testing.assert_allclose(op.to_array().transpose(*indices).conjugate(), op_dag.to_array())

        # test dual method
        ops = []
        for op in a.terms[0][::-1]:
            ops.append(op)
        adual = a.dual()
        ops_dual = []
        for op in adual.terms[0]:
            ops_dual.append(op)
        assert ops == ops_dual

    @pytest.mark.parametrize("hilbert_space_dims", [(4, 5, 2, 6, 3, 7)])
    @pytest.mark.parametrize("example_getter", [get_dense_example, get_dia_example])
    def test_tensor_product(self, work_stream, hilbert_space_dims, example_getter):
        ctx = work_stream

        a, b = example_getter(hilbert_space_dims)
        # test addition
        self._test_addition_operatorterms(a, b)

        # test out-of-place multiplication
        a_scaled = a * 2
        a_scaled = 2 * a_scaled
        assert a._coefficients[0].scalar * 4 == a_scaled._coefficients[0].scalar

        # test scalar operators
        iden = tensor_product(dtype="float64")
        scaled_iden = tensor_product(coeff=2.0, dtype="float64")
        two_ids = iden + scaled_iden
        general = iden + example_getter(hilbert_space_dims)[0]
        iden._maybe_instantiate(ctx, hilbert_space_dims)
        two_ids._maybe_instantiate(ctx, hilbert_space_dims)
        general._maybe_instantiate(ctx, hilbert_space_dims)

    def _test_addition_operatorterms(self, a, b):
        ab = a + b
        assert len(ab.terms) == len(a.terms) + len(b.terms)
        return ab

    def _test_multiplication_operatorterms(self, a, b):
        ab = a * b
        assert len(ab.terms) == len(a.terms) * len(b.terms)
        counter = 0
        for term_a in a.terms:
            for term_b in b.terms:
                ab.terms[counter][: len(term_a)] == term_a
                ab.terms[counter][len(term_a) :] == term_b
                counter += 1
        return ab

    @pytest.mark.parametrize("hilbert_space_dims", [(4, 5, 2, 6, 3, 7)])
    @pytest.mark.parametrize("n_ops", [1, 2])
    def test_operator_action(self, work_stream, hilbert_space_dims, n_ops):
        ctx = work_stream
        ops = []
        for _ in range(n_ops):
            a, b = get_dense_example(hilbert_space_dims)
            ops.append(Operator(hilbert_space_dims, (a, np.random.rand()), (b, np.random.rand())))
        OperatorAction(ctx, ops)
