# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interface to seamlessly use Cupy ndarray objects.
"""

__all__ = ['CupyTensor']

import cupy
import numpy

from . import utils
from .package_ifc import StreamHolder
from .tensor_ifc import Tensor


class CupyTensor(Tensor):
    """
    Tensor wrapper for cupy ndarrays.
    """
    name = 'cupy'
    module = cupy
    name_to_dtype = Tensor.create_name_dtype_map(conversion_function=lambda name: cupy.dtype(name), exception_type=TypeError)

    def __init__(self, tensor):
        super().__init__(tensor)

    @property
    def data_ptr(self):
        return self.tensor.data.ptr

    @property
    def device(self):
        return 'cuda'

    @property
    def device_id(self):
        return self.tensor.device.id

    @property
    def dtype(self):
        """Name of the data type"""
        return self.tensor.dtype.name

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def strides(self):
        return tuple(stride_in_bytes // self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    def numpy(self, stream_holder=StreamHolder()):
        stream = stream_holder.obj
        # cupy/cupy#7820
        if stream is not None:
            with stream:
                out = self.tensor.get(stream=stream)
            stream.synchronize()
        else:
            out = self.tensor.get(stream=stream)
            # note that synchronization semantics are inconsistent for stream_holder 
            # TODO: add synchronization on current stream here (better to recompose the if/else branch then)
        return out

    @classmethod
    def empty(cls, shape, **context):
        """
        Create an empty tensor of the specified shape and data type.
        """
        name = context.get('dtype', 'float32')
        dtype = CupyTensor.name_to_dtype[name]
        device = context.get('device', None)
        strides = context.get('strides', None)

        if isinstance(device, cupy.cuda.Device):
           device_id = device.id
        elif isinstance(device, int):
           device_id = device
        else:
            raise ValueError(f"The device must be specified as an integer or cupy.cuda.Device instance, not '{device}'.")

        with utils.device_ctx(device_id):
            if strides:
                # need an explicit allocation due to cupy/cupy#7818
                size = dtype.itemsize
                for s in shape:
                    size = size * s
                ptr = cupy.cuda.alloc(size)
                # when strides is not None, it should be of unit counts not bytes
                strides = tuple(s * dtype.itemsize for s in strides)
                tensor = cupy.ndarray(shape, dtype=dtype, strides=strides, memptr=ptr)
            else:
                tensor = cupy.ndarray(shape, dtype=dtype)

        return tensor

    def to(self, device='cpu', stream_holder=StreamHolder()):
        """
        Create a copy of the tensor on the specified device (integer or 
          'cpu'). Copy to  Numpy ndarray if CPU, otherwise return Cupy type.
        """
        if device == 'cpu':
            return self.numpy(stream_holder=stream_holder)

        if not isinstance(device, int):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device}'.")

        with utils.device_ctx(device), stream_holder.ctx:
            tensor_device = cupy.asarray(self.tensor)

        return tensor_device

    def copy_(self, src, stream_holder=StreamHolder()):
        """
        Inplace copy of src (copy the data from src into self).
        """
        with stream_holder.ctx:
            cupy.copyto(self.tensor, src)

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, cupy.ndarray)
    
    def update_extents_strides(self, extents, strides):
        strides = [i * self.tensor.itemsize for i in strides]
        self.tensor = cupy.ndarray(extents, dtype=self.tensor.dtype, memptr=self.tensor.data, strides=strides)