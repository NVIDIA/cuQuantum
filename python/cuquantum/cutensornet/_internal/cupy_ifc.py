# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interface to seamlessly use Cupy ndarray objects.
"""

__all__ = ['CupyTensor']

import cupy
import numpy

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
        return tuple(stride_in_bytes / self.tensor.itemsize for stride_in_bytes in self.tensor.strides)

    def numpy(self):
        return self.tensor.get()

    @classmethod
    def empty(cls, shape, **context):
        """
        Create an empty tensor of the specified shape and data type.
        """
        name = context.get('dtype', 'float32')
        dtype = CupyTensor.name_to_dtype[name]
        device = context.get('device', None)
        with cupy.cuda.Device(device=device):
            tensor = cupy.empty(shape, dtype=dtype)

        return tensor

    def to(self, device='cpu'):
        """
        Create a copy of the tensor on the specified device (integer or 
          'cpu'). Copy to  Numpy ndarray if CPU, otherwise return Cupy type.
        """
        if device == 'cpu':
            return self.numpy()

        if not isinstance(device, int):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device}'.")

        with cupy.cuda.Device(device):
            tensor_device = cupy.asarray(self.tensor)

        return tensor_device

    def copy_(self, src):
        """
        Inplace copy of src (copy the data from src into self).
        """

        cupy.copyto(self.tensor, src)

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, cupy.ndarray)

