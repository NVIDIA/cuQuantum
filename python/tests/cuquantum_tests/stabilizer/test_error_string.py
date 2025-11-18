# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Error string functionality test of cuStabilizer python bindings."""

import pytest
import cupy as cp
from cuquantum.bindings import custabilizer as cust


def test_get_error_string_success():
    """Test get_error_string with SUCCESS status."""
    error_msg = cust.get_error_string(cust.Status.SUCCESS)
    assert error_msg is not None
    assert len(error_msg) > 0
    assert "success" in error_msg.lower()


def test_get_error_string_invalid_value():
    """Test get_error_string with INVALID_VALUE status."""
    error_msg = cust.get_error_string(cust.Status.INVALID_VALUE)
    assert error_msg is not None
    assert "invalid" in error_msg.lower()


def test_get_error_string_alloc_failed():
    """Test get_error_string with ALLOC_FAILED status."""
    error_msg = cust.get_error_string(cust.Status.ALLOC_FAILED)
    assert error_msg is not None
    assert "alloc" in error_msg.lower() or "memory" in error_msg.lower()


def test_get_error_string_internal_error():
    """Test get_error_string with INTERNAL_ERROR status."""
    error_msg = cust.get_error_string(cust.Status.INTERNAL_ERROR)
    assert error_msg is not None
    assert "internal" in error_msg.lower() or "error" in error_msg.lower()


def test_get_error_string_insufficient_workspace():
    """Test get_error_string with INSUFFICIENT_WORKSPACE status."""
    error_msg = cust.get_error_string(cust.Status.INSUFFICIENT_WORKSPACE)
    assert error_msg is not None
    assert "workspace" in error_msg.lower() or "insufficient" in error_msg.lower()


def test_get_error_string_not_supported():
    """Test get_error_string with NOT_SUPPORTED status."""
    error_msg = cust.get_error_string(cust.Status.NOT_SUPPORTED)
    assert error_msg is not None
    assert "support" in error_msg.lower()


def test_get_error_string_cuda_error():
    """Test get_error_string with CUDA_ERROR status."""
    error_msg = cust.get_error_string(cust.Status.CUDA_ERROR)
    assert error_msg is not None
    assert "cuda" in error_msg.lower()


def test_error_in_exception():
    """Test that error strings appear in exceptions raised by cuStabilizer."""
    handle = cust.create()
    
    try:
        # create circuit with invalid buffer size
        circuit_string = "H 0"
        buffer_size = cust.circuit_size_from_string(handle, circuit_string)
        
        # allocate buffer that's too small (should trigger INSUFFICIENT_WORKSPACE)
        device_buffer = cp.cuda.alloc(1)
        
        with pytest.raises(Exception) as exc_info:
            circuit = cust.create_circuit_from_string(handle, circuit_string, device_buffer.ptr, 1)
        
        # check that the exception message contains meaningful error information
        assert "workspace" in str(exc_info.value).lower() or "insufficient" in str(exc_info.value).lower()

    finally:
        cust.destroy(handle)


if __name__ == "__main__":
    test_get_error_string_success()
    test_get_error_string_invalid_value()
    test_get_error_string_alloc_failed()
    test_get_error_string_internal_error()
    test_get_error_string_insufficient_workspace()
    test_get_error_string_not_supported()
    test_get_error_string_cuda_error()
    test_error_in_exception()
    print("All error string tests passed!")

