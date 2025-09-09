# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

try:
    import cupy as cp
except ImportError:
    cp = None

import pytest


# Free up cupy memory between tests
@pytest.fixture(scope="function", autouse=True)
def cleanup_between_tests():
    yield
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
