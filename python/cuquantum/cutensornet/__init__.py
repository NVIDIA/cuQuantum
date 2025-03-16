# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

from cuquantum.bindings.cutensornet import *
from cuquantum._internal.utils import deprecate
from ..tensornet import *
del tensor, experimental # clean up wild card import
from . import experimental
from . import tensor

warnings.warn(
    "cutensornet module is deprecated; use tensornet module for pythonic APIs and bindings.cutensornet for bindings instead.",
    DeprecationWarning,
    stacklevel=2
)

# add deprecation warnings for APIs that are directly imported under this module path
deprecation_message = "This API has been deprecated from cuquantum module and will be removed in a future release, please switch to cuquantum.tensornet for the same API"
# NOTE: tensor_qualifiers_dtype is a miss here since we can't issue warnings in an object

contract = deprecate(contract, deprecation_message)
contract_path = deprecate(contract_path, deprecation_message)
einsum = deprecate(einsum, deprecation_message)
einsum_path = deprecate(einsum_path, deprecation_message)
CircuitToEinsum = deprecate(CircuitToEinsum, deprecation_message)
Network = deprecate(Network, deprecation_message)
NetworkOptions = deprecate(NetworkOptions, deprecation_message)
# OptimizerInfo = deprecate(OptimizerInfo, deprecation_message)
# NOTE: OptimizerInfo is a miss here since it's returned by contract/contract_path, 
# we want to use a single class such that cuquantum.contract_path & cuquantum.tensornet.contract_path returns the same output class.
 
OptimizerOptions = deprecate(OptimizerOptions, deprecation_message)
PathFinderOptions = deprecate(PathFinderOptions, deprecation_message)
ReconfigOptions = deprecate(ReconfigOptions, deprecation_message)
SlicerOptions = deprecate(SlicerOptions, deprecation_message)

del deprecate, deprecation_message