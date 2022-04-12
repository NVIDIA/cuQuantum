# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interface class to encapsulate low-level calls to get or set optimizer information.
"""

__all__ = ['OptimizerInfoInterface']

from collections.abc import Sequence
import operator

import numpy as np

from cuquantum import cutensornet as cutn


def _parse_and_map_sliced_modes(sliced_modes, mode_map_user_to_ord, size_dict, dtype_mode=np.int32, dtype_extent=np.int64):
    """
    Parse user-provided sliced modes and create individual, contiguous sliced_modes and sliced extents array.
    """

    num_sliced_modes = len(sliced_modes)
    if num_sliced_modes == 0:
        return num_sliced_modes, np.zeros((num_sliced_modes,), dtype=dtype_mode), np.zeros((num_sliced_modes,), dtype=dtype_extent)

    # The sliced modes have already passed basic checks when creating the OptimizerOptions dataclass.

    pairs =  not isinstance(sliced_modes[0], str) and isinstance(sliced_modes[0], Sequence)
    if pairs:
        sliced_modes, sliced_extents = zip(*sliced_modes)
    else:
        sliced_extents = np.ones((num_sliced_modes,), dtype=dtype_extent)

    sliced_modes = np.asarray([mode_map_user_to_ord[m] for m in sliced_modes], dtype=dtype_mode)
    remainder = tuple(size_dict[m] % e for m, e in zip(sliced_modes, sliced_extents))
    if any(remainder):
        raise ValueError("The sliced extents must evenly divide the original extents of the corresponding mode.")

    return num_sliced_modes, sliced_modes, np.asanyarray(sliced_extents, dtype=dtype_extent)


InfoEnum = cutn.ContractionOptimizerInfoAttribute

class OptimizerInfoInterface(object):
    """
    """
    def __init__(self, network):
        """
        """
        self.network = network

        get_dtype = cutn.contraction_optimizer_info_get_attribute_dtype

        self._flop_count = np.zeros((1,), dtype=get_dtype(InfoEnum.FLOP_COUNT))
        self._largest_tensor = np.zeros((1,), dtype=get_dtype(InfoEnum.LARGEST_TENSOR))
        self._num_slices = np.zeros((1,), dtype=get_dtype(InfoEnum.NUM_SLICES))
        self._num_sliced_modes = np.zeros((1,), dtype=get_dtype(InfoEnum.NUM_SLICED_MODES))
        self._slicing_overhead = np.zeros((1,), dtype=get_dtype(InfoEnum.SLICING_OVERHEAD))

        self.num_contraction = len(self.network.operands) - 1
        self._path = np.zeros((2*self.num_contraction, ), dtype=np.int32)

    @staticmethod
    def _get_scalar_attribute(network, name, attribute):
        """
        name      = cutensornet enum for the attribute
        attribute = numpy ndarray object into which the value is stored by cutensornet
        """
        assert network.optimizer_info_ptr is not None, "Internal error"
        cutn.contraction_optimizer_info_get_attribute(network.handle, network.optimizer_info_ptr, name, attribute.ctypes.data, attribute.dtype.itemsize)

    @staticmethod
    def _set_scalar_attribute(network, name, attribute, value):
        """
        name      = cutensornet enum for the attribute
        attribute = numpy ndarray object into which the value is stored
        value     = the value to set the the attribute to
        """
        assert network.optimizer_info_ptr is not None, "Internal error"
        attribute[0] = value
        cutn.contraction_optimizer_info_set_attribute(network.handle, network.optimizer_info_ptr, name, attribute.ctypes.data, attribute.dtype.itemsize)

    @property
    def num_slices(self):
        """
        The number of slices in the network.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.NUM_SLICES, self._num_slices)

        return int(self._num_slices)

    @num_slices.setter
    def num_slices(self, number):
        """
        Set the number of slices in the network.
        """
        OptimizerInfoInterface._set_scalar_attribute(network, InfoEnum.NUM_SLICES, self._num_slices, number)

    @property
    def flop_count(self):
        """
        The cost of contracting the network.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.FLOP_COUNT, self._flop_count)

        return float(self._flop_count)

    @property
    def largest_intermediate(self):
        """
        The size of the largest intermediate.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.LARGEST_TENSOR, self._largest_tensor)

        return float(self._largest_tensor)


    @property
    def slicing_overhead(self):
        """
        The slicing overhead.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.SLICING_OVERHEAD, self._slicing_overhead)

        return float(self._slicing_overhead)

    @property
    def path(self):
        """
        Return the contraction path in linear format.
        """

        network = self.network

        path_wrapper = cutn.ContractionPath(self.num_contraction, self._path.ctypes.data)
        size = path_wrapper.get_size()
        cutn.contraction_optimizer_info_get_attribute(network.handle, network.optimizer_info_ptr, InfoEnum.PATH, path_wrapper.get_path(), size)

        path = tuple(zip(*[iter(self._path)]*2))

        return path

    @path.setter
    def path(self, path):
        """
        Set the path.
        """
        from functools import reduce

        get_dtype = cutn.contraction_optimizer_info_get_attribute_dtype

        network = self.network

        num_contraction = len(path)
        if num_contraction != len(network.operands) - 1:
            raise ValueError(f"The length of the contraction path ({num_contraction}) must be one less than the number of operands ({len(network.operands)}).")

        path = reduce(operator.concat, path)
        self._path = np.array(path, dtype=np.int32)
        path_wrapper = cutn.ContractionPath(num_contraction, self._path.ctypes.data)
        size = path_wrapper.get_size()
        cutn.contraction_optimizer_info_set_attribute(network.handle, network.optimizer_info_ptr, InfoEnum.PATH, path_wrapper.get_path(), size)

    @property
    def num_sliced_modes(self):
        """
        The number of sliced modes in the network.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.NUM_SLICED_MODES, self._num_sliced_modes)

        return int(self._num_sliced_modes)

    @num_sliced_modes.setter
    def num_sliced_modes(self, number):
        """
        Set the number of sliced_modes in the network.
        """
        OptimizerInfoInterface._set_scalar_attribute(self.network, InfoEnum.NUM_SLICED_MODES, self._num_sliced_modes, number)

    @property
    def sliced_mode_extent(self):
        """
        Return the sliced modes as a sequence of (sliced mode, sliced extent) pairs.
        """

        get_dtype = cutn.contraction_optimizer_info_get_attribute_dtype

        network = self.network

        num_sliced_modes = self.num_sliced_modes

        sliced_modes = np.zeros((num_sliced_modes,), dtype=get_dtype(InfoEnum.SLICED_MODE))
        size = num_sliced_modes * sliced_modes.dtype.itemsize
        cutn.contraction_optimizer_info_get_attribute(network.handle, network.optimizer_info_ptr, InfoEnum.SLICED_MODE, sliced_modes.ctypes.data, size)
        sliced_modes = tuple(network.mode_map_ord_to_user[m] for m in sliced_modes)    # Convert to user mode labels

        sliced_extents = np.zeros((num_sliced_modes,), dtype=get_dtype(InfoEnum.SLICED_EXTENT))
        size = num_sliced_modes * sliced_extents.dtype.itemsize
        cutn.contraction_optimizer_info_get_attribute(network.handle, network.optimizer_info_ptr, InfoEnum.SLICED_EXTENT, sliced_extents.ctypes.data, size)

        return tuple(zip(sliced_modes, sliced_extents))

    @sliced_mode_extent.setter
    def sliced_mode_extent(self, sliced_modes):
        """
        Set the sliced modes (and possibly sliced extent).

        sliced_mode = sequence of sliced modes, or sequence of (sliced mode, sliced extent) pairs
        """

        network = self.network

        num_sliced_modes, sliced_modes, sliced_extents = _parse_and_map_sliced_modes(sliced_modes, network.mode_map_user_to_ord, network.size_dict)

        # Set the number of sliced modes first
        self.num_sliced_modes = num_sliced_modes

        size = num_sliced_modes * sliced_modes.dtype.itemsize
        cutn.contraction_optimizer_info_set_attribute(network.handle, network.optimizer_info_ptr, InfoEnum.SLICED_MODE, sliced_modes.ctypes.data, size)

        size = num_sliced_modes * sliced_extents.dtype.itemsize
        cutn.contraction_optimizer_info_set_attribute(network.handle, network.optimizer_info_ptr, InfoEnum.SLICED_EXTENT, sliced_extents.ctypes.data, size)

