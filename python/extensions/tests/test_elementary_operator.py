# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the ElementaryOperator class.
"""

from itertools import product

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import ElementaryOperator


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
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_init_dense(self, data, batch_size):
        """
        Test initializing dense elementary operator.
        """
        # TODO: This should be done in a better way, by creating random buffer rather than broadcasting.
        data_batched = jnp.broadcast_to(data, (batch_size, *data.shape))
        elem_op = ElementaryOperator(data_batched)

        assert elem_op.num_modes == len(data.shape) // 2
        assert elem_op.mode_extents == data.shape[:len(data.shape) // 2]
        assert elem_op.data.shape == (batch_size, *data.shape)
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
    @pytest.mark.parametrize("diag_offsets", [(0, 1)])
    def test_init_multidiagonal(self, data, diag_offsets):
        """ 
        Test initializing multidiagonal elementary operator.
        """
        elem_op = ElementaryOperator(data, diag_offsets=diag_offsets)

        assert elem_op.num_modes == len(data.shape) // 2
        assert elem_op.mode_extents == data.shape[:len(data.shape) // 2]
        assert elem_op.data.shape == (1, *data.shape)
        assert elem_op.sparsity == cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL
        assert elem_op.diag_offsets == (0, 1)

    @pytest.mark.parametrize(
        "data,diag_offsets",
        [
            (((3, 5, 3, 5), jnp.complex128), (0, 1)),
            (((4, 3), jnp.complex128), (0, 1)),
            (((4, 3), jnp.complex128), (0, 1, 0)),
        ],
        indirect=["data"],
    )
    def test_init_multidiagonal_fail(self, data, diag_offsets):
        """
        Test initializing multidiagonal elementary operator with invalid diag_offsets.
        """
        with pytest.raises(ValueError):
            ElementaryOperator(data, diag_offsets=diag_offsets)

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
