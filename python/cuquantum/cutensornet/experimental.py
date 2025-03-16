# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

from ..tensornet.experimental import *
from .._internal.utils import deprecate

warnings.warn(
    "This module is deprecated and will be removed in a future release; please switch to cuquantum.tensornet.experimental instead.",
    DeprecationWarning,
    stacklevel=2
)

deprecation_message = "This API has been deprecated and will be removed in a future release, please switch to cuquantum.tensornet.experimental for the same API"

contract_decompose = deprecate(contract_decompose, deprecation_message)
ContractDecomposeAlgorithm = deprecate(ContractDecomposeAlgorithm, deprecation_message)
ContractDecomposeInfo = deprecate(ContractDecomposeInfo, deprecation_message)
MPSConfig = deprecate(MPSConfig, deprecation_message)
TNConfig = deprecate(TNConfig, deprecation_message)
NetworkOperator = deprecate(NetworkOperator, deprecation_message)
NetworkState = deprecate(NetworkState, deprecation_message)

del deprecate, deprecation_message