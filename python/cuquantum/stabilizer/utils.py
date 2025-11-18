# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import Tuple, Union, TypeVar
from nvmath.internal import utils as nvmath_utils
from nvmath import memory

try:
    import cupy as cp
except ImportError:
    cp = np

Array = Union[np.ndarray, "cp.ndarray"]
Stream = Union[int, "cp.cuda.Stream", None]

T = TypeVar("T", bound=Array)
def _unpack_arrays(num_bits: int, *args: Tuple[T, ...]) -> Tuple[T]:
    """
    Package-transparent unpacking of arrays from bit-packed format.
    """
    ret: Tuple[T] = tuple()
    for a in args:
        shape = a.shape
        if isinstance(a, np.ndarray):
            unpacked: T = np.unpackbits(a.view("uint8"), bitorder="little")
        else:
            unpacked: T = cp.unpackbits(a.view("uint8"), bitorder="little")
        newshape = (shape[0], shape[1] * 8)
        unpacked = unpacked.reshape(newshape)
        ret += (unpacked[:, :num_bits],)
    return ret


def _pack_arrays(num_bits: int, *args: Tuple[T, ...]) -> Tuple[T]:
    """
    Package-transparent packing of arrays.
    """
    ret: Tuple[T] = ()
    stride = (num_bits + 31) // 32
    inputs = list(args)
    for i in range(len(args)):
        a = args[i].reshape(-1, num_bits)
        need_pad = stride * 32 - a.shape[1]
        assert need_pad >= 0, f"stride={stride} num_bits={num_bits}, shape={a.shape}"
        if need_pad > 0:
            if isinstance(a, np.ndarray):
                a = np.pad(a, ((0, 0), (0, need_pad)))
            else:
                a = cp.pad(a, ((0, 0), (0, need_pad)))
        inputs[i] = a
    for a in inputs:
        if isinstance(a, np.ndarray):
            packed: T = np.packbits(a.view("uint8"), bitorder="little")
        else:
            packed: T = cp.packbits(a.view("uint8"), bitorder="little")
        ret += (packed,)
    return ret


def _ptr_as_cupy(ptr, size, **kwargs) -> "cp.ndarray":
    if isinstance(ptr, cp.ndarray):
        return cp.ndarray(**kwargs, memptr=ptr.data)
    mem = cp.cuda.UnownedMemory(
        nvmath_utils.get_ptr_from_memory_pointer(ptr), size, owner=None
    )
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    return cp.ndarray(**kwargs, memptr=memptr)


def _get_memptr(x: Union[memory.MemoryPointer, "cp.ndarray"]) -> memory.MemoryPointer:
    if x is None:
        return None
    if isinstance(x, cp.ndarray):
        return x.data.ptr
    else:
        return nvmath_utils.get_ptr_from_memory_pointer(x)


__all__ = [
    "Array",
    "Stream",
    "_unpack_arrays",
    "_pack_arrays",
    "_get_memptr",
    "_ptr_as_cupy",
]
