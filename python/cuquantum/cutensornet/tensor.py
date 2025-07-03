# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

from ..tensornet.tensor import *
from .._internal.utils import deprecate

warnings.warn(
    "This module is deprecated and will be removed in the next release; please switch to cuquantum.tensornet.tensor instead.",
    DeprecationWarning,
    stacklevel=2
)

deprecation_message = "This API has been deprecated and will be removed in the next release, please switch to cuquantum.tensornet.tensor for the same API"

decompose = deprecate(decompose, deprecation_message, UserWarning)
DecompositionOptions = deprecate(DecompositionOptions, deprecation_message, UserWarning)
QRMethod = deprecate(QRMethod, deprecation_message, UserWarning)
#SVDInfo = deprecate(SVDInfo, deprecation_message)
# NOTE: SVDInfo is a miss here since it's returned by contract/contract_path, 
# we want to use a single class such that cuquantum.tensor.decompose & cuquantum.tensornet.tensor.decompose returns the same output class.
SVDMethod = deprecate(SVDMethod, deprecation_message, UserWarning)

del deprecate, deprecation_message