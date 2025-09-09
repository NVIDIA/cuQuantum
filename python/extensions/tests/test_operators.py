# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test the cuDensityMat API for JAX.
"""

import pytest

from itertools import product

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import ElementaryOperator, MatrixOperator, OperatorTerm, Operator


jax.config.update("jax_enable_x64", True)


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


class TestElementaryOperator:
    """
    Test the cuDensityMat ElementaryOperator class.
    """

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    def test_init_dense(self, data):
        """
        Test initializing dense elementary operator.
        """
        elem_op = ElementaryOperator(data)

        assert elem_op.num_modes == len(data.shape) // 2
        assert elem_op.mode_extents == data.shape[:len(data.shape) // 2]
        assert elem_op.data.shape == data.shape
        assert elem_op.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE

    @pytest.mark.parametrize(
        "data",
        [
            ((3, 5, 3), jnp.complex128),
            ((3, 5, 3, 4), jnp.complex128),
        ],
        indirect=True,
    )
    def test_init_dense_fail(self, data):
        """
        Test initializing dense elementary operator with invalid data.
        """
        with pytest.raises(ValueError):
            ElementaryOperator(data)

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 2)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    @pytest.mark.parametrize("offsets", [(0, 1)])
    def test_init_multidiagonal(self, data, offsets):
        """ 
        Test initializing multidiagonal elementary operator.
        """
        elem_op = ElementaryOperator(data, offsets=offsets)
        assert elem_op.num_modes == len(data.shape) // 2
        assert elem_op.mode_extents == data.shape[:len(data.shape) // 2]
        assert elem_op.data.shape == data.shape
        assert elem_op.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL
        assert elem_op.offsets == (0, 1)

    @pytest.mark.parametrize(
        "data,offsets",
        [
            (((3, 5, 3, 5), jnp.complex128), (0, 1)),
            (((4, 3), jnp.complex128), (0, 1)),
            (((4, 3), jnp.complex128), (0, 1, 0)),
        ],
        indirect=["data"],
    )
    def test_init_multidiagonal_fail(self, data, offsets):
        """
        Test initializing multidiagonal elementary operator with invalid offsets.
        """
        with pytest.raises(ValueError):
            ElementaryOperator(data, offsets=offsets)

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    def test_create(self, data, handle):
        """
        Test elementary operator opaque handle creation.
        """
        elem_op = ElementaryOperator(data)

        # Test that _ptr is None before _create.
        assert elem_op._ptr is None

        # Test that _create sets the pointer.
        elem_op._create(handle)
        assert elem_op._ptr is not None

        # Test that calling _create again won't overwrite the pointer.
        ptr = elem_op._ptr
        elem_op._create(handle)
        assert elem_op._ptr == ptr

        elem_op._destroy()

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    def test_destroy(self, data, handle):
        """
        Test elementary operator opaque handle destruction.
        """
        elem_op = ElementaryOperator(data)

        # Test that _create sets the pointer.
        elem_op._create(handle)
        assert elem_op._ptr is not None

        # Test that _destroy sets the pointer to None.
        elem_op._destroy()
        assert elem_op._ptr is None

        # Test that calling _destroy again has no effect.
        elem_op._destroy()
        assert elem_op._ptr is None


class TestMatrixOperator:
    """
    Test the cuDensityMat MatrixOperator class.
    """

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    def test_init_dense(self, data):
        """
        Test initializing dense matrix operator.
        """
        elem_op = MatrixOperator(data)

        assert elem_op.num_modes == len(data.shape) // 2
        assert elem_op.mode_extents == data.shape[:len(data.shape) // 2]
        assert elem_op.data.shape == data.shape

    @pytest.mark.parametrize(
        "data",
        [
            ((3, 5, 3), jnp.complex128),
            ((3, 5, 3, 4), jnp.complex128),
        ],
        indirect=True,
    )
    def test_init_dense_fail(self, data):
        """
        Test initializing dense matrix operator with invalid data.
        """
        with pytest.raises(ValueError):
            MatrixOperator(data)

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    def test_create(self, data, handle):
        """
        Test matrix operator opaque handle creation.
        """
        elem_op = MatrixOperator(data)

        # Test that _ptr is None before _create.
        assert elem_op._ptr is None

        # Test that _create sets the pointer.
        elem_op._create(handle)
        assert elem_op._ptr is not None

        # Test that calling _create again won't overwrite the pointer.
        ptr = elem_op._ptr
        elem_op._create(handle)
        assert elem_op._ptr == ptr

        elem_op._destroy()

    @pytest.mark.parametrize(
        "data",
        list(product(
            [(4, 4), (3, 5, 3, 5)],
            [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
        )),
        indirect=True,
    )
    def test_destroy(self, data, handle):
        """
        Test matrix operator opaque handle creation.
        """
        elem_op = MatrixOperator(data)

        # Test that _create sets the pointer.
        elem_op._create(handle)
        assert elem_op._ptr is not None

        # Test that _destroy sets the pointer to None.
        elem_op._destroy()
        assert elem_op._ptr is None

        # Test that calling _destroy again has no effect.
        elem_op._destroy()
        assert elem_op._ptr is None


class TestOperatorTerm:
    """
    Test the cuDensityMat OperatorTerm class.
    """

    dims = (3, 4, 5)

    @pytest.mark.parametrize(
        "data",
        [
            ((4, 4), jnp.complex128),
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
    @pytest.mark.parametrize(
        "operator_class",
        [
            ElementaryOperator,
            MatrixOperator,
        ],
    )
    def test_append_single(self, data, data_, operator_class):
        """
        Test appending a single operator product.
        """
        base_op1 = operator_class(data)
        base_op2 = operator_class(data_)

        op_term = OperatorTerm(self.dims)
        assert op_term.dims == self.dims

        if operator_class is ElementaryOperator:
            op_term.append([base_op1, base_op2], modes=(0, 1), duals=(False, True))
        else:
            op_term.append([base_op1, base_op2], conjs=(False, True), duals=(False, True))

        assert len(op_term.op_prods) == 1
        if operator_class is ElementaryOperator:
            assert len(op_term.modes) == 1
        else:
            assert len(op_term.conjs) == 1
        assert len(op_term.duals) == 1
        assert len(op_term.coeffs) == 1
        assert len(op_term.coeff_callbacks) == 1
        assert len(op_term.coeff_grad_callbacks) == 1

        assert op_term.op_prods[0] == (base_op1, base_op2)
        if operator_class is ElementaryOperator:
            assert op_term.modes[0] == (0, 1)
        else:
            assert op_term.conjs[0] == (False, True)
        assert op_term.duals[0] == (False, True)
        assert op_term.coeffs[0] == 1.0
        assert op_term.coeff_callbacks[0] is None
        assert op_term.coeff_grad_callbacks[0] is None

    @pytest.mark.parametrize(
        "data",
        [
            ((4, 4), jnp.complex128),
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
    @pytest.mark.parametrize(
        "operator_class",
        [
            ElementaryOperator,
            MatrixOperator,
        ],
    )
    def test_append_multiple(self, data, data_, operator_class):
        """
        Test appending multiple operator products.
        """
        base_op1 = operator_class(data)
        base_op2 = operator_class(data_)

        op_term = OperatorTerm(self.dims)
        assert op_term.dims == self.dims

        if operator_class is ElementaryOperator:
            op_term.append([base_op1], modes=(0,), duals=(False,))
            op_term.append([base_op2], modes=(1,), duals=(True,))
        else:
            op_term.append([base_op1], conjs=(False,), duals=(False,))
            op_term.append([base_op2], conjs=(True,), duals=(True,))

        assert len(op_term.op_prods) == 2
        if operator_class is ElementaryOperator:
            assert len(op_term.modes) == 2
        else:
            assert len(op_term.conjs) == 2
        assert len(op_term.duals) == 2
        assert len(op_term.coeffs) == 2
        assert len(op_term.coeff_callbacks) == 2
        assert len(op_term.coeff_grad_callbacks) == 2

        assert op_term.op_prods[0] == (base_op1,)
        if operator_class is ElementaryOperator:
            assert op_term.modes[0] == (0,)
        else:
            assert op_term.conjs[0] == (False,)
        assert op_term.duals[0] == (False,)
        assert op_term.coeffs[0] == 1.0
        assert op_term.coeff_callbacks[0] is None
        assert op_term.coeff_grad_callbacks[0] is None

        assert op_term.op_prods[1] == (base_op2,)
        if operator_class is ElementaryOperator:
            assert op_term.modes[1] == (1,)
        else:
            assert op_term.conjs[1] == (True,)
        assert op_term.duals[1] == (True,)
        assert op_term.coeffs[1] == 1.0
        assert op_term.coeff_callbacks[1] is None
        assert op_term.coeff_grad_callbacks[1] is None

    @pytest.mark.parametrize(
        "data",
        [
            ((4, 4), jnp.complex128),
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
    @pytest.mark.parametrize(
        "operator_class",
        [
            ElementaryOperator,
            MatrixOperator,
        ],
    )
    def test_append_fail_mixed_dtypes(self, data, data_, operator_class):
        """
        Test appending operator products of different dtypes fails.
        """
        base_op1 = operator_class(data)
        base_op2 = operator_class(data_)

        op_term = OperatorTerm(self.dims)

        with pytest.raises(ValueError):
            op_term.append([base_op1, base_op2])

    @pytest.mark.parametrize(
        "data",
        [
            ((4, 4), jnp.complex128),
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
    def test_append_fail_mixed_operator_types(self, data, data_):
        """
        Test appending operator products of different operator types fails.
        """
        elem_op = ElementaryOperator(data)
        matrix_op = MatrixOperator(data_)

        op_term = OperatorTerm(self.dims)

        with pytest.raises(ValueError):
            op_term.append([elem_op, matrix_op])

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
        Test operator term opaque handle creation.
        """
        base_op1 = ElementaryOperator(data)
        base_op2 = ElementaryOperator(data_)

        op_term = OperatorTerm(self.dims)

        op_term.append([base_op1, base_op2], modes=(0, 1))

        assert op_term._ptr is None

        op_term._create(handle)
        assert op_term._ptr is not None
        for base_op in op_term.op_prods[0]:
            assert base_op._ptr is not None

        # Test that calling _create again won't overwrite the pointer.
        ptr = op_term._ptr
        base_op_ptrs = [base_op._ptr for base_op in op_term.op_prods[0]]
        op_term._create(handle)
        assert op_term._ptr == ptr
        for base_op, base_op_ptr in zip(op_term.op_prods[0], base_op_ptrs):
            assert base_op._ptr == base_op_ptr

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
        Test operator term opaque handle creation.
        """
        base_op1 = MatrixOperator(data)
        base_op2 = MatrixOperator(data_)

        op_term = OperatorTerm(self.dims)

        op_term.append([base_op1, base_op2])

        assert op_term._ptr is None

        op_term._create(handle)
        assert op_term._ptr is not None
        for base_op in op_term.op_prods[0]:
            assert base_op._ptr is not None

        # Test that calling _create again won't overwrite the pointer.
        ptr = op_term._ptr
        base_op_ptrs = [base_op._ptr for base_op in op_term.op_prods[0]]
        op_term._create(handle)
        assert op_term._ptr == ptr
        for base_op, base_op_ptr in zip(op_term.op_prods[0], base_op_ptrs):
            assert base_op._ptr == base_op_ptr

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
        Test operator term opaque handle destruction.
        """
        base_op1 = ElementaryOperator(data)
        base_op2 = ElementaryOperator(data_)

        op_term = OperatorTerm(self.dims)
        op_term2 = OperatorTerm(self.dims)
        op_term.append([base_op1, base_op2], modes=(0, 1))
        op_term2.append([base_op1,], modes=(0,))


        # Test that _create sets the pointer.
        op_term._create(handle)
        op_term2._create(handle)
        assert op_term._ptr is not None

        # Test that _destroy sets the pointer to None.
        op_term._destroy()
        assert op_term._ptr is None
        for base_op in op_term.op_prods[0]:
            assert base_op._ptr is not None
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
        Test operator term opaque handle destruction.
        """
        base_op1 = MatrixOperator(data)
        base_op2 = MatrixOperator(data_)

        op_term = OperatorTerm(self.dims)
        op_term2 = OperatorTerm(self.dims)

        op_term.append([base_op1, base_op2])
        op_term2.append([base_op1, base_op2])

        # Test that _create sets the pointer.
        op_term._create(handle)
        op_term2._create(handle)
        assert op_term._ptr is not None

        # Test that _destroy sets the pointer to None.
        op_term._destroy()
        assert op_term._ptr is None
        for base_op in op_term.op_prods[0]:
            assert base_op._ptr is not None
        op_term2._destroy()
        base_op1._destroy()
        base_op2._destroy()


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
