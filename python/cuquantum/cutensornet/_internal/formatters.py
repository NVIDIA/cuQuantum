# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Formatters for printing data.
"""

import numpy as np

class MemoryStr(object):
    """
    A simple type to pretty-print memory-like values.
    """

    def __init__(self, memory, base_unit='B'):
        self.memory = memory
        self.base_unit = base_unit
        self.base = 1024

    def __str__(self):
        """
        Convert large values to powers of 1024 for readability.
        """

        base, base_unit, memory = self.base, self.base_unit, self.memory

        if memory < base:
            value, unit = memory, base_unit
        elif memory < base**2:
            value, unit = memory/base, f'Ki{base_unit}'
        elif memory < base**3:
            value, unit = memory/base**2, f'Mi{base_unit}'
        else:
            value, unit = memory/base**3, f'Gi{base_unit}'

        return f"{value:0.2f} {unit}"


def array2string(array_like):
    """
    String representation of an array-like object with possible truncation of "interior" values to limit string size.

    The NumPy function "set_printoptions" can be used to control the display of the array.
    """

    return np.array2string(np.asanyarray(array_like, dtype='object'), separator=', ', formatter={'object': lambda s: s})

