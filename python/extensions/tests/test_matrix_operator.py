# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the MatrixOperator class.
"""

from itertools import product

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import MatrixOperator


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
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_init_dense(self, data, batch_size):
        """
        Test initializing dense matrix operator.
        """
        data_batched = jnp.broadcast_to(data, (batch_size, *data.shape))
        elem_op = MatrixOperator(data_batched)

        assert elem_op.num_modes == len(data.shape) // 2
        assert elem_op.mode_extents == data.shape[:len(data.shape) // 2]
        assert elem_op.data.shape == (batch_size, *data.shape)

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
