# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated with version 25.09.0. Do not modify it directly.

from cpython.memoryview cimport PyMemoryView_FromMemory
from cpython.buffer cimport PyBUF_READ, PyBUF_WRITE

import math

import numpy as np
import cupy as cp

from .cycudensitymat cimport CUDENSITYMAT_CALLBACK_DEVICE_CPU, CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD


cdef inline object _reconstruct_cpu_params(int64_t batch_size, int32_t num_params, const double * _params_, int flag):
    """
    Reconstruct NumPy array for params/params_grad from pointer and size.
    """
    params_size = sizeof(double) * num_params * batch_size
    params_buffer = PyMemoryView_FromMemory(<char *>_params_, params_size, flag)
    params = np.ndarray((num_params, batch_size), dtype=np.float64, buffer=params_buffer, order='F')
    return params


cdef inline object _reconstruct_cpu_storage(mode_extents, int64_t batch_size, data_type, void * _storage_):
    """
    Reconstruct NumPy array for storage from pointer and size.
    """
    storage_size = data_type.itemsize * math.prod(mode_extents) * batch_size
    storage_buffer = PyMemoryView_FromMemory(<char *>_storage_, storage_size, PyBUF_WRITE)
    storage = np.ndarray((*mode_extents, batch_size), dtype=data_type, buffer=storage_buffer, order='F')
    return storage


cdef inline object _reconstruct_gpu_params(int64_t batch_size, int32_t num_params, const double * _params_):
    """
    Reconstruct CuPy array for params/params_grad from pointer and size.
    """
    params_size = sizeof(double) * num_params * batch_size        
    if _params_ == NULL:
        device_id = cp.cuda.Device().id
    else:
        device_id = -1
    params_memory = cp.cuda.UnownedMemory(<intptr_t>_params_, params_size, None, device_id = device_id)
    params_memory_ptr = cp.cuda.MemoryPointer(params_memory, 0)
    params = cp.ndarray((num_params, batch_size), dtype=cp.float64, memptr=params_memory_ptr, order='F')
    return params


cdef inline object _reconstruct_gpu_storage(mode_extents, int64_t batch_size, data_type, void * _storage_):
    """
    Reconstruct CuPy array for storage from pointer and size.
    """
    storage_size = data_type.itemsize * math.prod(mode_extents) * batch_size
    storage_memory = cp.cuda.UnownedMemory(<intptr_t>_storage_, storage_size, None)
    storage_memory_ptr = cp.cuda.MemoryPointer(storage_memory, 0)
    storage = cp.ndarray((*mode_extents, batch_size), dtype=data_type, memptr=storage_memory_ptr, order='F')
    return storage


cdef inline void _hold_scalar_callback_reference(intptr_t obj, scalar_callback):
    """
    Hold reference to a scalar callback.
    """
    if scalar_callback is not None:
        _callback_holders[obj].add((<WrappedScalarCallback>scalar_callback).callback)


cdef inline void _hold_tensor_callback_reference(intptr_t obj, tensor_callback):
    """
    Hold reference to a tensor callback.
    """
    if tensor_callback is not None:
        _callback_holders[obj].add((<WrappedTensorCallback>tensor_callback).callback)


cdef inline void _hold_scalar_gradient_callback_reference(intptr_t obj, scalar_gradient_callback):
    """
    Hold reference to a scalar gradient callback.
    """
    if scalar_gradient_callback is not None:
        _callback_holders[obj].add((<WrappedScalarGradientCallback>scalar_gradient_callback).callback)


cdef inline void _hold_tensor_gradient_callback_reference(intptr_t obj, tensor_gradient_callback):
    """
    Hold reference to a tensor gradient callback.
    """
    if tensor_gradient_callback is not None:
        _callback_holders[obj].add((<WrappedTensorGradientCallback>tensor_gradient_callback).callback)


cdef inline void _hold_tensor_gradient_callback_references(intptr_t obj, tensor_gradient_callbacks):
    """
    Hold references to a list of tensor gradient callbacks.
    """
    for tensor_gradient_callback in tensor_gradient_callbacks:
        _hold_tensor_gradient_callback_reference(obj, tensor_gradient_callback)


cdef inline _WrappedScalarCallback _convert_scalar_callback(scalar_callback):
    """
    Convert a wrapped scalar callback from a cdef class to a C struct.
    """
    cdef _WrappedScalarCallback _scalar_callback_
    if scalar_callback is not None:
        _scalar_callback_ = (<WrappedScalarCallback>scalar_callback)._struct
    else:
        _scalar_callback_.callback = NULL
        _scalar_callback_.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU
        _scalar_callback_.wrapper = NULL
    return _scalar_callback_


cdef inline _WrappedTensorCallback _convert_tensor_callback(tensor_callback):
    """
    Convert a wrapped tensor callback from a cdef class to a C struct.
    """
    cdef _WrappedTensorCallback _tensor_callback_
    if tensor_callback is not None:
        _tensor_callback_ = (<WrappedTensorCallback>tensor_callback)._struct
    else:
        _tensor_callback_.callback = NULL
        _tensor_callback_.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU
        _tensor_callback_.wrapper = NULL
    return _tensor_callback_


cdef inline _WrappedScalarGradientCallback _convert_scalar_gradient_callback(scalar_gradient_callback):
    """
    Convert a wrapped scalar gradient callback from a cdef class to a C struct.
    """
    cdef _WrappedScalarGradientCallback _scalar_gradient_callback_
    if scalar_gradient_callback is not None:
        _scalar_gradient_callback_ = (<WrappedScalarGradientCallback>scalar_gradient_callback)._struct
    else:
        _scalar_gradient_callback_.callback = NULL
        _scalar_gradient_callback_.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU
        _scalar_gradient_callback_.wrapper = NULL
        _scalar_gradient_callback_.direction = CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD
    return _scalar_gradient_callback_


cdef inline _WrappedTensorGradientCallback _convert_tensor_gradient_callback(tensor_gradient_callback):
    """
    Convert a wrapped tensor gradient callback from a cdef class to a C struct.
    """
    cdef _WrappedTensorGradientCallback _tensor_gradient_callback_
    if tensor_gradient_callback is not None:
        _tensor_gradient_callback_ = (<WrappedTensorGradientCallback>tensor_gradient_callback)._struct
    else:
        _tensor_gradient_callback_.callback = NULL
        _tensor_gradient_callback_.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU
        _tensor_gradient_callback_.wrapper = NULL
        _tensor_gradient_callback_.direction = CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD
    return _tensor_gradient_callback_
