# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from cuquantum.bindings.custatevec import *

warnings.warn(
    "custatevec module is deprecated and will be removed in a future release, please switch to bindings.custatevec.",
    DeprecationWarning,
    stacklevel=2
)
