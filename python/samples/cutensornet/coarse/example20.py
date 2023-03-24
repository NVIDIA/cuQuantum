# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating a simple memory manager plugin using a PyTorch tensor as a memory buffer.
"""
import logging
import torch

from cuquantum import contract, MemoryPointer


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

class TorchMemMgr:
    def __init__(self, device):
        self.device = device
        self.logger = logging.getLogger()

    def memalloc(self, size):
        buffer = torch.empty((size, ), device=self.device, dtype=torch.int8, requires_grad=False)
        device_pointer = buffer.data_ptr()
        self.logger.info(f"The user memory allocator has allocated {size} bytes at pointer {device_pointer}.")

        def create_finalizer():
            def finalizer():
                buffer    # Keep buffer alive for as long as it is needed.
                self.logger.info("The memory allocation has been released.")
            return finalizer

        return MemoryPointer(device_pointer, size, finalizer=create_finalizer())

device_id = 0
a = torch.rand((3,2), device=device_id)
b = torch.rand((2,3), device=device_id)

r = contract("ij,jk", a, b, options={'allocator' : TorchMemMgr(device_id)})
