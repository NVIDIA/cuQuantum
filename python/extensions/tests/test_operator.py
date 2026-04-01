# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the Operator class.
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import ElementaryOperator, MatrixOperator, OperatorTerm, Operator


key = jax.random.key(0)


@pytest.fixture(scope="class")
def handle():
    """
    Fixture to create a cuDensityMat handle.
    """
    handle_ = cudm.create()
    yield handle_
    cudm.destroy(handle_)


@pytest.fixture(scope="class")
def data(request):
    """
    Fixture to create a random tensor data buffer.
    """
    shape, dtype = request.param
    return jax.random.normal(key, shape, dtype=dtype)


data_ = data


class TestOperator:
    """
    Test the cuDensityMat Operator class.
    """

    dims = (3, 4, 5)

    @pytest.mark.parametrize(
        "data",
        [
            ((3, 3), jnp.complex128),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "data_",
        [
            ((4, 4), jnp.complex64),
        ],
        indirect=True,
    )
    def test_append_fail_mixed_dtypes(self, data, data_):
        """
        Test appending operator products of different dtypes fails.
        """
        base_op1 = ElementaryOperator(data)
        base_op2 = ElementaryOperator(data_)
        
        op_term1 = OperatorTerm(self.dims)
        op_term2 = OperatorTerm(self.dims)

        op_term1.append([base_op1], modes=(0,))
        op_term2.append([base_op2], modes=(1,))

        op = Operator(self.dims)
        with pytest.raises(ValueError):
            op.append(op_term1)
            op.append(op_term2)

    @pytest.mark.parametrize(
        "data",
        [
            ((3, 3), jnp.complex128),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "data_",
        [
            ((4, 4), jnp.complex128),
        ],
        indirect=True,
    )
    def test_create_elementary(self, data, data_, handle):
        """
        Test operator opaque handle creation.
        """
        base_op1 = ElementaryOperator(data)
        base_op2 = ElementaryOperator(data_)

        op_term = OperatorTerm(self.dims)
        op_term.append([base_op1, base_op2], modes=(0, 1))

        op = Operator(self.dims)
        op.append(op_term)

        op._create(handle)
        assert op._ptr is not None

        for op_term in op.op_terms:
            assert op_term._ptr is not None
            for base_op in op_term.op_prods[0]:
                assert base_op._ptr is not None

        op._destroy()
        op_term._destroy()
        base_op1._destroy()
        base_op2._destroy()

    @pytest.mark.parametrize(
        "data",
        [
            ((3, 4, 5, 3, 4, 5), jnp.complex128),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "data_",
        [
            ((3, 4, 5, 3, 4, 5), jnp.complex128),
        ],
        indirect=True,
    )
    def test_create_matrix(self, data, data_, handle):
        """
        Test operator opaque handle creation.
        """
        base_op1 = MatrixOperator(data)
        base_op2 = MatrixOperator(data_)

        op_term = OperatorTerm(self.dims)
        op_term.append([base_op1, base_op2])

        op = Operator(self.dims)
        op.append(op_term)

        op._create(handle)
        assert op._ptr is not None

        for op_term in op.op_terms:
            assert op_term._ptr is not None
            for base_op in op_term.op_prods[0]:
                assert base_op._ptr is not None

        op._destroy()
        op_term._destroy()
        base_op1._destroy()
        base_op2._destroy()

    @pytest.mark.parametrize(
        "data",
        [
            ((3, 3), jnp.complex128),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "data_",
        [
            ((4, 4), jnp.complex128),
        ],
        indirect=True,
    )
    def test_destroy_elementary(self, data, data_, handle):
        """
        Test operator opaque handle destruction.
        """
        base_op1 = ElementaryOperator(data)
        base_op2 = ElementaryOperator(data_)

        op_term = OperatorTerm(self.dims)
        op_term.append([base_op1, base_op2], modes=(0, 1))
        op_term.append([base_op1,], modes=(0,))
        op_term2 = OperatorTerm(self.dims)
        op_term2.append([base_op2,], modes=(1,))

        op = Operator(self.dims)
        op2 = Operator(self.dims)
        op.append(op_term)
        op2.append(op_term)
        op2.append(op_term2)

        op._create(handle)
        op2._create(handle)

        op._destroy()
        assert op._ptr is None
        for op_term in op.op_terms:
            assert op_term._ptr is not None
            for base_op in op_term.op_prods[0]:
                assert base_op._ptr is not None
        op2._destroy()
        op_term._destroy()
        op_term2._destroy()
        base_op1._destroy()
        base_op2._destroy()

    @pytest.mark.parametrize(
        "data",
        [
            ((3, 4, 5, 3, 4, 5), jnp.complex128),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "data_",
        [
            ((3, 4, 5, 3, 4, 5), jnp.complex128),
        ],
        indirect=True,
    )
    def test_destroy_matrix(self, data, data_, handle):
        """
        Test operator opaque handle destruction.
        """
        base_op1 = MatrixOperator(data)
        base_op2 = MatrixOperator(data_)

        op_term = OperatorTerm(self.dims)
        op_term.append([base_op1, base_op2])

        op_term2 = OperatorTerm(self.dims)
        op_term2.append([base_op1,],)
        op_term2.append([base_op2,],)
        

        op = Operator(self.dims)
        op2 = Operator(self.dims)
        op.append(op_term)
        op2.append(op_term2)
        op2.append(op_term)

        op._create(handle)
        op2._create(handle)

        op._destroy()
        assert op._ptr is None
        for op_term in op.op_terms:
            assert op_term._ptr is not None
            for base_op in op_term.op_prods[0]:
                assert base_op._ptr is not None
        op2._destroy()
        op_term._destroy()
        op_term2._destroy()
        base_op1._destroy()
        base_op2._destroy()
