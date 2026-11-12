# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal utility functions for the pauliprop experimental module."""

from typing import Sequence, Callable, Protocol, Literal
import weakref

import numpy as np

import cuquantum.bindings.cupauliprop as cupp
from cuquantum.bindings.cupauliprop import SortOrder 
from ..truncation import Truncation

# Type alias for sort order return types and accepted input values
SortOrderLiteral = None | Literal["internal", "little_endian_bitwise"]


def sort_order_to_cupp(value: SortOrder | SortOrderLiteral) -> SortOrder:
    """Convert Python sort order to C API SortOrder enum.
    
    Args:
        value: Python sort order. One of:
            - cupp.SortOrder enum value (passed through)
            - None: No sorting
            - "internal": Library-chosen sort order
            - "little_endian_bitwise": Little-endian bitwise sort
            
    Returns:
        The corresponding cupp.SortOrder enum value.
        
    Raises:
        ValueError: If value is not a valid sort order.
    """
    # Pass through if already a SortOrder enum
    if isinstance(value, SortOrder):
        return value
    if value is None:
        return SortOrder.NONE
    elif value == "internal":
        return SortOrder.INTERNAL
    elif value == "little_endian_bitwise":
        return SortOrder.LITTLE_ENDIAN_BITWISE
    else:
        raise ValueError(f"Invalid sort order: {value}.")


def sort_order_from_cupp(c_sort_order: "cupp.SortOrder") -> SortOrderLiteral:
    """Convert C API SortOrder enum to Python sort order.
    
    Args:
        c_sort_order: The C API SortOrder enum value.
        
    Returns:
        Python sort order: None, "internal", or "little_endian_bitwise".
    """
    if c_sort_order == cupp.SortOrder.NONE:
        return None
    elif c_sort_order == cupp.SortOrder.INTERNAL:
        return "internal"
    else:  # LITTLE_ENDIAN_BITWISE
        return "little_endian_bitwise"


def register_finalizer(instance: object, destructor: Callable[[int], None], ptr: int, logger=None, name: str = "") -> weakref.finalize:
    """
    Register a weak reference finalizer for safe C library resource cleanup.
    
    Using weakref.finalize instead of __del__ ensures the cleanup function
    is called before interpreter shutdown begins tearing down modules.
    
    Args:
        instance: The Python object instance to attach the finalizer to.
        destructor: The C library cleanup function (e.g., cupp.destroy).
        ptr: The pointer/handle to pass to the destructor.
        logger: Optional logger for debug logging of destruction.
        name: Optional name for the object type (for logging).
        
    Returns:
        The weakref.finalize object (can be called manually via finalizer()).
    """
    def cleanup(p: int, destroy: Callable[[int], None], log, obj_name: str) -> None:
        if p is not None:
            if log is not None:
                log.debug(f"{obj_name} destroyed (ptr={p})")
            destroy(p)
    
    return weakref.finalize(instance, cleanup, ptr, destructor, logger, name)


def is_all_zeros(seq: Sequence[int]) -> bool:
    """Check if a sequence of integers contains only zeros.
    
    Args:
        seq: A sequence of integers to check.
        
    Returns:
        True if all elements are zero, False otherwise.
    """
    return all(x == 0 for x in seq)


def is_c_contiguous(shape: Sequence[int], strides: Sequence[int] | None) -> bool:
    """Check if a tensor with the given shape and strides is contiguous in row-major (C) order.

    The C library is given only a pointer and a size, and reads the buffer in memory order.
    That matches the tensor's logical order only for a row-major layout: a column-major
    (or otherwise permuted) tensor is dense but would be read with its elements reordered.
    Dimensions of extent 1 may have any stride.

    Args:
        shape: The shape of the tensor (extents of each dimension).
        strides: The strides of the tensor (in number of elements, not bytes), or ``None``
            for a row-major tensor.

    Returns:
        True if the tensor is row-major contiguous, False otherwise.
    """
    if strides is None:
        return True
    if len(shape) != len(strides):
        return False
    if any(extent == 0 for extent in shape):
        return True
    expected = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if extent != 1 and stride != expected:
            return False
        expected *= extent
    return True


class _Truncation:
    which_truncation: cupp.TruncationStrategyKind
    params: cupp.CoefficientTruncationParams | cupp.PauliWeightTruncationParams
    strategy: cupp.TruncationStrategy
    
    def _init_strategy(self):
        self.strategy = cupp.TruncationStrategy()
        self.strategy.strategy = self.which_truncation
        self.strategy.param_struct = self.params.ptr
    
    @property
    def params_ptr(self) -> int:
        return self.params.ptr


class _CoefficientTruncation(_Truncation):
    which_truncation = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
    def __init__(self, cutoff: float):
        self._cutoff_as_array = np.array([cutoff], dtype=np.float64) #FIXME: is this required or does the constructor take its argument by value?
        self.params = cupp.CoefficientTruncationParams()
        self.params.cutoff = cutoff
        self._init_strategy()

    @property
    def cutoff(self) -> float:
        return self._cutoff_as_array[()]
   
    
class _WeightTruncation(_Truncation):
    which_truncation = cupp.TruncationStrategyKind.TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED
    def __init__(self, cutoff: int):
        self._cutoff_as_array = np.array([cutoff], dtype=np.int32) #FIXME: is this required or does the constructor take its argument by value?
        self.params = cupp.PauliWeightTruncationParams()
        self.params.cutoff = cutoff
        self._init_strategy()
    
    @property
    def cutoff(self) -> int:
        return self._cutoff_as_array[()]


def create_truncation_strategies(truncation: Truncation | None) -> list[_Truncation]:
    """
    Convert a sequence of Truncation objects to a list of TruncationStrategy structs.
    
    Args:
        truncation: Truncation object.
        
    Returns:
        List of TruncationStrategy structs (empty list if truncations is None or empty).
    """
    truncation_strategies = []
    if truncation:
        if truncation.pauli_weight_cutoff is not None:
            truncation_strategies.append(_WeightTruncation(truncation.pauli_weight_cutoff))
        if truncation.pauli_coeff_cutoff is not None:
            truncation_strategies.append(_CoefficientTruncation(truncation.pauli_coeff_cutoff))
    return truncation_strategies

def convert_truncation_strategies(truncation_strategies: list[_Truncation] | None) -> list[cupp.TruncationStrategy] | None:
    """Convert internal truncation strategies to C API TruncationStrategy structs."""
    if not truncation_strategies:
        return None
    return [s.strategy for s in truncation_strategies]

