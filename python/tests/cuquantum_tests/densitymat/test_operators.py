# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import numpy as np
import cupy as cp
import pytest

from cuquantum.densitymat import (
    tensor_product,
    DenseOperator,
    MultidiagonalOperator,
    Operator,
    OperatorTerm,
    OperatorAction,
    DensePureState,
    WorkStream,
    CPUCallback,
    GPUCallback
)


@pytest.fixture
def work_stream():
    np.random.seed(42)
    cp.random.seed(42)
    return WorkStream()


def get_dia_example(hilbert_space_dims, batch_size_ops = (1,1), batch_size_term = 1):
    A = DenseOperator(np.random.rand(*((hilbert_space_dims[2],) * 2), batch_size_ops[0]))
    B = MultidiagonalOperator(np.random.rand(hilbert_space_dims[3], 3, batch_size_ops[1]), [-1, 0, 1])

    ab = tensor_product((A, (2,)), (B, (3,)), coeff=CPUCallback(lambda t, args: np.sin(args[0] * t) * np.arange(1,batch_size_term+1), is_inplace=False), batch_size = batch_size_term)
    ab2 = tensor_product((A, (2,)), (B, (3,)), coeff=np.arange(1,batch_size_term+1))
    return ab, ab2


def get_dense_example(hilbert_space_dims, batch_size_ops = (1,1), batch_size_term = 1):
    A =  DenseOperator(np.random.rand(*((hilbert_space_dims[2],) * 2), batch_size_ops[0]))
    B = DenseOperator(np.random.rand(*((hilbert_space_dims[3], hilbert_space_dims[5]) * 2), batch_size_ops[1]))

    ab = tensor_product((A, (2,)), (B, (3, 5)), coeff=CPUCallback(lambda t, args: np.sin(args[0] * t) * np.arange(1,batch_size_term+1), is_inplace=False), batch_size = batch_size_term)
    ab2 = tensor_product((A, (2,)), (B, (3, 5)), coeff=np.arange(1,batch_size_term+1) if batch_size_term > 1 else 2.0)
    return ab, ab2


class TestOperators:

    @pytest.fixture
    def callback_operator(self, work_stream):
        def create_operator(batch_size):
            hilbert_space_dims = (2,)
            elem_op = DenseOperator(np.eye(2, dtype=np.complex128))
            term = tensor_product(
                (elem_op, (0,)),
                coeff=CPUCallback(lambda t, args: args[0], is_inplace=False),
                batch_size=batch_size,
            )
            return work_stream, hilbert_space_dims, Operator(hilbert_space_dims, (term,))

        return create_operator

    def test_operator_term_append_empty_scalar_product(self):
        matrix_term = OperatorTerm(dtype="complex128")
        matrix_term.append_matrix_product([], [], [], coeff=2.0)
        assert matrix_term.terms == [[]]
        assert matrix_term.modes == [[]]
        assert matrix_term.duals == [[]]
        assert matrix_term._conjugations == [[]]

        elementary_term = OperatorTerm(dtype="complex128")
        elementary_term.append_elementary_product([], None, [], coeff=3.0)
        assert elementary_term.terms == [[]]
        assert elementary_term.modes == [[]]
        assert elementary_term.duals == [[]]
        assert elementary_term._conjugations == [[]]

    def test_operator_term_rejects_mismatched_append_dtype(self):
        term = OperatorTerm(dtype="complex128")
        complex_op = DenseOperator(np.eye(2, dtype=np.complex128))
        float_op = DenseOperator(np.eye(2, dtype=np.float64))

        term.append_elementary_product([complex_op], [(0,)], [(False,)])
        with pytest.raises(TypeError, match="same data type"):
            term.append_elementary_product([float_op], [(1,)], [(False,)])

    def test_operator_term_iadd_infers_dtype(self) -> None:
        """Infer the dtype from the appended term when the destination is empty."""
        op = DenseOperator(np.eye(2, dtype=np.complex128))
        other = OperatorTerm(dtype="complex128")
        other.append_elementary_product([op], [(0,)], [(False,)])

        term = OperatorTerm()
        term += other

        assert term.dtype == other.dtype
        assert term.terms == other.terms

    def test_operator_term_iadd_keeps_two_untyped_empty_terms_empty(self) -> None:
        """Treat adding two empty, untyped terms as a no-op."""
        term = OperatorTerm()

        term += OperatorTerm()

        assert term.dtype is None
        assert term.terms == []
        assert term.modes == []
        assert term.duals == []
        assert term._conjugations == []

    def test_operator_term_iadd_rejects_mismatched_dtype(self) -> None:
        """Reject in-place addition of terms with different dtypes."""
        complex_op = DenseOperator(np.eye(2, dtype=np.complex128))
        float_op = DenseOperator(np.eye(2, dtype=np.float64))
        term = OperatorTerm(dtype="complex128")
        term.append_elementary_product([complex_op], [(0,)], [(False,)])
        other = OperatorTerm(dtype="float64")
        other.append_elementary_product([float_op], [(0,)], [(False,)])

        with pytest.raises(TypeError, match="dtype"):
            term += other

        assert term.dtype == "complex128"
        assert len(term.terms) == 1

    def test_operator_action_dtype_check_requires_matching_dtype(self) -> None:
        """Exercise the existing-dtype validation branch directly."""
        # The public constructor starts with no dtype and cannot reach this branch.
        action = OperatorAction.__new__(OperatorAction)
        action._dtype = "complex128"

        action._set_or_check_dtype([SimpleNamespace(dtype="complex128")])
        with pytest.raises(TypeError, match="same dtype"):
            action._set_or_check_dtype([SimpleNamespace(dtype="float64")])

    def test_operator_action_rejects_mixed_dtype_operators(self, work_stream) -> None:
        """Reject mixed-dtype operators through the public constructor."""
        hilbert_space_dims = (2,)
        complex_term = tensor_product(
            (DenseOperator(np.eye(2, dtype=np.complex128)), (0,))
        )
        float_term = tensor_product(
            (DenseOperator(np.eye(2, dtype=np.float64)), (0,))
        )
        operators = [
            Operator(hilbert_space_dims, (complex_term,)),
            Operator(hilbert_space_dims, (float_term,)),
        ]

        with pytest.raises(ValueError, match="more than one dtype"):
            OperatorAction(work_stream, operators)

    @pytest.mark.parametrize(
        "params, expected_batch_values",
        [
            ([0.25], np.array([0.25])),
            (np.array([[0.25, 0.75]]), np.array([0.25, 0.75])),
        ],
    )
    def test_operator_compute_action_accepts_callback_params(
        self, callback_operator, params, expected_batch_values
    ) -> None:
        """Pass a single-batch sequence and batched 2-D array through the public API."""
        batch_size = expected_batch_values.size
        ctx, hilbert_space_dims, operator = callback_operator(batch_size)
        state_in = DensePureState(ctx, hilbert_space_dims, batch_size, "complex128")
        state_out = DensePureState(ctx, hilbert_space_dims, batch_size, "complex128")
        state_in.attach_storage(cp.ones((2, batch_size), dtype=cp.complex128, order="F"))
        state_out.attach_storage(cp.zeros((2, batch_size), dtype=cp.complex128, order="F"))

        operator.prepare_action(ctx, state_in)
        operator.compute_action(0.0, params, state_in, state_out)

        expected = np.broadcast_to(expected_batch_values, (2, batch_size))
        np.testing.assert_allclose(state_out.storage.get(), expected)

    @pytest.mark.parametrize(
        "params",
        [np.array(0.1), np.array([0.1]), np.zeros((2, 3, 4))],
    )
    def test_operator_compute_action_rejects_invalid_callback_param_dimensions(
        self, callback_operator, params
    ) -> None:
        """Reject invalid parameter ranks through the public operator action API."""
        ctx, hilbert_space_dims, operator = callback_operator(1)
        state_in = DensePureState(ctx, hilbert_space_dims, 1, "complex128")
        state_out = DensePureState(ctx, hilbert_space_dims, 1, "complex128")
        state_in.attach_storage(cp.ones(2, dtype=cp.complex128))
        state_out.attach_storage(cp.zeros(2, dtype=cp.complex128))
        operator.prepare_action(ctx, state_in)

        with pytest.raises(ValueError, match="2-D array"):
            operator.compute_action(0.0, params, state_in, state_out)

    @pytest.mark.parametrize("hilbert_space_dims", [(4, 5, 2, 6, 3, 7)])
    @pytest.mark.parametrize("batch_sizes", ([(1,1),(1,1),1,1],
                                            [(1,1),(1,1),2,1],
                                            [(2,2),(2,2),1,1],
                                            [(2,2),(2,2),2,1],
                                            [(1,1),(2,2),1,1],
                                            [(1,1),(2,2),2,1],
                                            [(1,1),(2,2),2,2],
                                            [(1,1),(1,1),2,2],                                            
                                            pytest.param([(3,3),(3,2),1,1], marks=pytest.mark.xfail(raises=ValueError)),
                                            pytest.param([(2,2),(3,3),1,1], marks=pytest.mark.xfail(raises=ValueError)),
                                            pytest.param([(2,2),(1,1),3,3], marks=pytest.mark.xfail(raises=ValueError)),
                                            pytest.param([(2,1),(1,2),3,3], marks=pytest.mark.xfail(raises=ValueError)),
                                            pytest.param([(2,2),(1,1),1,3], marks=pytest.mark.xfail(raises=ValueError)),
                                            pytest.param([(1,1),(2,2),3,3], marks=pytest.mark.xfail(raises=ValueError)),
                                            )
                            )
    @pytest.mark.parametrize("example_getter", [get_dense_example, get_dia_example])
    def test_operator_term(self, hilbert_space_dims, batch_sizes, example_getter):
        a, b = example_getter(hilbert_space_dims, batch_sizes[0], batch_sizes[2])
        c, d = example_getter(hilbert_space_dims, batch_sizes[1], batch_sizes[3])
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
            if isinstance(op, MultidiagonalOperator):
                _op=op.to_dense()
                _op_dag = op_dag.to_dense()
            else:
                _op = op
                _op_dag = op_dag
            _op_dag_data = np.empty_like(_op.data)
            for i in range(op.batch_size):
                _op_dag_data[...,i]=_op.data[...,i].transpose(*indices).conjugate()
            np.testing.assert_allclose(_op_dag_data, _op_dag.data)

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
        assert a._coefficients[0].static_coeff * 4 == a_scaled._coefficients[0].static_coeff

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

    @pytest.mark.parametrize("hilbert_space_dims", [(4, 5, 2, 6, 3, 7)])
    @pytest.mark.parametrize("n_ops", [1, 2])
    def test_operator(self, work_stream, hilbert_space_dims, n_ops):
        ctx = work_stream
        ops = []
        op=Operator(hilbert_space_dims)
        # in place addition Operator.append(OperatorTerm)
        for _ in range(n_ops):
            a, b = get_dense_example(hilbert_space_dims)
            op.append(a, coeff = np.random.rand())
            op.append(b, coeff = np.random.rand())
        op_action = OperatorAction(ctx, [op])
        op_action = None
        # in place addition Operator += OperatorTerm
        op=Operator(hilbert_space_dims)
        for _ in range(n_ops):
            a, b = get_dense_example(hilbert_space_dims)
            op += a
            op += b
        op_action = OperatorAction(ctx, [op])
        op_action = None
        # in  place addition Operator += Operator
        op=Operator(hilbert_space_dims)
        for _ in range(n_ops):
            a, b = get_dense_example(hilbert_space_dims)
            op += Operator(hilbert_space_dims, (a, np.random.rand()), (b, np.random.rand()))
        op_action = OperatorAction(ctx, [op])
        op_action = None

        


    
