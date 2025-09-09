# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

""" Interface for pluggable memory handlers.
"""

from typing import Optional, Union

__all__ = ['BaseCUDAMemoryManager', 'MemoryPointer', 'MemoryLimitExceeded']

from nvmath.memory import BaseCUDAMemoryManager, MemoryPointer


class MemoryLimitExceeded(MemoryError):
    """
    This exception is raised when the operation requires more device memory than what was specified in operation options.

    Attributes:
        - limit: int
            The memory limit in bytes.
        - requirement: int
            Memory required to perform the operation.
        - device_id: int
            The device selected to run the operation.

    If the options was set to str, this value is the calculated limit.
    """
    limit: int
    device_id: int
    requirement: int

    def __init__(self,
                 limit:int,
                 requirement:int,
                 device_id:int,
                 specified: Optional[Union[str, int]]=None):
        message = f"""GPU memory limit exceeded. Device id: {device_id}.
The memory limit is {limit}, while the minimum workspace size needed is {requirement}.
"""
        if specified is not None:
            message += f"Memory limit specified by options: {specified}."

        super().__init__(message)
        self.limit = limit
        self.requirement = requirement
        self.device_id = device_id