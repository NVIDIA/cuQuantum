# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

from ..tensornet.experimental import *
from .._internal.utils import deprecate

warnings.warn(
    "This module is deprecated and will be removed in the next release; please switch to cuquantum.tensornet.experimental instead.",
    DeprecationWarning,
    stacklevel=2
)

deprecation_message = "This API has been deprecated and will be removed in the next release, please switch to cuquantum.tensornet.experimental for the same API"

contract_decompose = deprecate(contract_decompose, deprecation_message, UserWarning)
ContractDecomposeAlgorithm = deprecate(ContractDecomposeAlgorithm, deprecation_message, UserWarning)
#ContractDecomposeInfo = deprecate(ContractDecomposeInfo, deprecation_message, UserWarning)
# NOTE: ContractDecomposeInfo is a miss here since it's returned by contract_decompose, 
# we want to use a single class such that cuquantum.cutensornet.experimental.contract_decompose & cuquantum.tensornet.experimental.contract_decompose returns the same output class.
MPSConfig = deprecate(MPSConfig, deprecation_message, UserWarning)
TNConfig = deprecate(TNConfig, deprecation_message, UserWarning)
NetworkOperator = deprecate(NetworkOperator, deprecation_message, UserWarning)
NetworkState = deprecate(NetworkState, deprecation_message, UserWarning)

del deprecate, deprecation_message