# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interface class to encapsulate low-level calls to get or set optimizer information.
"""

__all__ = ['OptimizerInfoInterface']

from collections.abc import Sequence
import itertools
import operator

import numpy as np

from cuquantum.bindings import cutensornet as cutn


def _parse_and_map_sliced_modes(sliced_modes, mode_map_user_to_ord, size_dict):
    """
    Parse user-provided sliced modes, create and return a contiguous (sliced mode, slide extent) array of
      type `cutn.slice_info_pair_dtype`.
    """

    num_sliced_modes = len(sliced_modes)
    slice_info_array = np.empty((num_sliced_modes,), dtype=cutn.slice_info_pair_dtype)

    if num_sliced_modes == 0:
        return slice_info_array

    # The sliced modes have already passed basic checks when creating the OptimizerOptions dataclass.

    pairs =  not isinstance(sliced_modes[0], str) and isinstance(sliced_modes[0], Sequence)
    if pairs:
        sliced_modes, sliced_extents = zip(*sliced_modes)
    else:
        sliced_extents = (1,)

    # Check for invalid mode labels.
    invalid_modes = tuple(filter(lambda k: k not in mode_map_user_to_ord, sliced_modes))
    if invalid_modes:
       message = f"Invalid sliced mode labels: {invalid_modes}"
       raise ValueError(message)

    slice_info_array["sliced_mode"] = sliced_modes = [mode_map_user_to_ord[m] for m in sliced_modes]
    remainder = any(size_dict[m] % e for m, e in itertools.zip_longest(sliced_modes, sliced_extents, fillvalue=1))
    if remainder:
        raise ValueError("The sliced extents must evenly divide the original extents of the corresponding mode.")
    slice_info_array["sliced_extent"] = sliced_extents

    return slice_info_array


InfoEnum = cutn.ContractionOptimizerInfoAttribute


class OptimizerInfoInterface:

    def __init__(self, network):
        """
        """
        self.network = network

        get_dtype = cutn.contraction_optimizer_info_get_attribute_dtype

        self._flop_count = np.zeros((1,), dtype=get_dtype(InfoEnum.FLOP_COUNT))
        self._largest_tensor = np.zeros((1,), dtype=get_dtype(InfoEnum.LARGEST_TENSOR))
        self._num_slices = np.zeros((1,), dtype=get_dtype(InfoEnum.NUM_SLICES))
        self._num_sliced_modes = np.zeros((1,), dtype=get_dtype(InfoEnum.NUM_SLICED_MODES))
        self._slicing_config = np.zeros((1,), dtype=get_dtype(InfoEnum.SLICING_CONFIG))
        self._slicing_overhead = np.zeros((1,), dtype=get_dtype(InfoEnum.SLICING_OVERHEAD))

        self.num_contraction = len(self.network.operands) - 1
        self._path = np.zeros((1,), dtype=get_dtype(InfoEnum.PATH))

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

        return self._num_slices.item()

    @property
    def flop_count(self):
        """
        The cost of contracting the network.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.FLOP_COUNT, self._flop_count)

        return self._flop_count.item()

    @property
    def largest_intermediate(self):
        """
        The size of the largest intermediate.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.LARGEST_TENSOR, self._largest_tensor)

        return self._largest_tensor.item()


    @property
    def slicing_overhead(self):
        """
        The slicing overhead.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.SLICING_OVERHEAD, self._slicing_overhead)

        return self._slicing_overhead.item()

    @property
    def path(self):
        """
        Return the contraction path in linear format.
        """
        path = np.empty((2*self.num_contraction,), dtype=np.int32)
        self._path["data"] = path.ctypes.data
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.PATH, self._path)

        return list(zip(*[iter(path)]*2))

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

        if num_contraction > 0:
            path = reduce(operator.concat, path)
        path_array = np.asarray(path, dtype=np.int32)

        # Construct the path type.
        path = np.array((num_contraction, path_array.ctypes.data), dtype=get_dtype(InfoEnum.PATH))

        # Set the attribute.
        OptimizerInfoInterface._set_scalar_attribute(self.network, InfoEnum.PATH, self._path, path)

    @property
    def num_sliced_modes(self):
        """
        The number of sliced modes in the network.
        """
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.NUM_SLICED_MODES, self._num_sliced_modes)

        return self._num_sliced_modes.item()

    @property
    def sliced_mode_extent(self):
        """
        Return the sliced modes as a sequence of (sliced mode, sliced extent) pairs.
        """

        get_dtype = cutn.contraction_optimizer_info_get_attribute_dtype

        network = self.network

        num_sliced_modes = self.num_sliced_modes

        slice_info_array = np.empty((num_sliced_modes,), dtype=cutn.slice_info_pair_dtype)

        slicing_config = self._slicing_config
        slicing_config["num_sliced_modes"] = num_sliced_modes
        slicing_config["data"] = slice_info_array.ctypes.data
        OptimizerInfoInterface._get_scalar_attribute(self.network, InfoEnum.SLICING_CONFIG, slicing_config)

        sliced_modes = tuple(network.mode_map_ord_to_user[m] for m in slice_info_array["sliced_mode"])  # Convert to user mode labels
        sliced_extents = slice_info_array["sliced_extent"]

        return tuple(zip(sliced_modes, sliced_extents))

    @sliced_mode_extent.setter
    def sliced_mode_extent(self, sliced_modes):
        """
        Set the sliced modes (and possibly sliced extent).

        sliced_mode = sequence of sliced modes, or sequence of (sliced mode, sliced extent) pairs
        """

        get_dtype = cutn.contraction_optimizer_info_get_attribute_dtype

        network = self.network

        # Construct the slicing config type.
        slice_info_array = _parse_and_map_sliced_modes(sliced_modes, network.mode_map_user_to_ord, network.size_dict)
        slicing_config = np.array((len(slice_info_array), slice_info_array.ctypes.data), dtype=get_dtype(InfoEnum.SLICING_CONFIG))

        # Set the attribute.
        OptimizerInfoInterface._set_scalar_attribute(network, InfoEnum.SLICING_CONFIG, self._slicing_config, slicing_config)

    @property
    def intermediate_modes(self):
        """
        Return a sequence of mode labels for all the intermediate tensors.
        """
        get_dtype = cutn.contraction_optimizer_info_get_attribute_dtype
        network = self.network

        num_intermediate_modes = np.zeros((max(1, self.num_contraction),), dtype=get_dtype(InfoEnum.NUM_INTERMEDIATE_MODES))    # Output modes included
        size = num_intermediate_modes.nbytes
        cutn.contraction_optimizer_info_get_attribute(network.handle, network.optimizer_info_ptr, InfoEnum.NUM_INTERMEDIATE_MODES, num_intermediate_modes.ctypes.data, size)

        intermediate_modes = np.zeros((num_intermediate_modes.sum(),), dtype=get_dtype(InfoEnum.INTERMEDIATE_MODES))
        size = intermediate_modes.nbytes
        cutn.contraction_optimizer_info_get_attribute(network.handle, network.optimizer_info_ptr, InfoEnum.INTERMEDIATE_MODES, intermediate_modes.ctypes.data, size)

        count, out = 0, list()
        mode_type = tuple if (network.is_interleaved or network.has_ellipses) else ''.join
        for n in num_intermediate_modes:
            out.append(mode_type(map(lambda m: network.mode_map_ord_to_user[m], intermediate_modes[count:count+n])))    # Convert to user mode labels
            count += n
        assert count == num_intermediate_modes.sum()
        return tuple(out)
