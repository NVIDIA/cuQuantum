# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

try:
    import cupy as cp
    # This is future proof: In the future when CuPy enables cuQuantum Python
    # as an optional backend, we don't want to create a circular dependency
    # that ultimately tests against ourselves. Here we enable CUB as the only
    # optinaly backend and exclude cuTENSOR/cuQuantum Python/etc, using CuPy's
    # private API (for development/testing).
    cp._core.set_reduction_accelerators(['cub'])
    cp._core.set_routine_accelerators(['cub'])
except ImportError:
    pass

try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except ImportError:
    pass