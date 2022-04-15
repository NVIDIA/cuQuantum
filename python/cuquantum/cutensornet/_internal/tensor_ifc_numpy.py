# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interface to seamlessly use Numpy ndarray objects.
"""

__all__ = ['NumpyTensor']

import cupy
import numpy

from .tensor_ifc import Tensor

class NumpyTensor(Tensor):
    """
    Tensor wrapper for numpy ndarrays.
    """
    name = 'numpy'
    module = numpy
    name_to_dtype = Tensor.create_name_dtype_map(conversion_function=lambda name: numpy.dtype(name), exception_type=TypeError)

    def __init__(self, tensor):
        super().__init__(tensor)

    @property
    def data_ptr(self):
        return self.tensor.ctypes.data

    @property
    def device(self):
        return 'cpu'

    @property
    def device_id(self):
        return None

    @property
    def dtype(self):
        """Name of the data type"""
        return self.tensor.dtype.name

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def strides(self):
        return tuple(stride_in_bytes / self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    def numpy(self):
        return self.tensor

    @classmethod
    def empty(cls, shape, **context):
        """
        Create an empty tensor of the specified shape and data type.
        """
        name = context.get('dtype', 'float32')
        dtype = NumpyTensor.name_to_dtype[name]
        return cls(module.empty(shape, dtype=dtype))

    def to(self, device='cpu'):
        """
        Create a copy of the tensor on the specified device (integer or 
          'cpu'). Copy to  Cupy ndarray on the specified device if it 
          is not CPU. Otherwise, return self.
        """
        if device == 'cpu':
            return self

        if not isinstance(device, int):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device}'.")

        with cupy.cuda.Device(device):
            tensor_device = cupy.asarray(self.tensor)

        return tensor_device

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, numpy.ndarray)

