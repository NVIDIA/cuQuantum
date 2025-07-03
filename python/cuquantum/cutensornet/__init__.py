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
    "cutensornet module is deprecated and will be removed in the next release; use tensornet module for pythonic APIs and bindings.cutensornet for bindings instead.",
    DeprecationWarning,
    stacklevel=2
)

# add deprecation warnings for APIs that are directly imported under this module path
deprecation_message = "This API has been deprecated from cuquantum module and will be removed in the next release, please switch to cuquantum.tensornet for the same API"
# NOTE: tensor_qualifiers_dtype is a miss here since we can't issue warnings in an object

contract = deprecate(contract, deprecation_message, UserWarning)
contract_path = deprecate(contract_path, deprecation_message, UserWarning)
einsum = deprecate(einsum, deprecation_message, UserWarning)
einsum_path = deprecate(einsum_path, deprecation_message, UserWarning)
CircuitToEinsum = deprecate(CircuitToEinsum, deprecation_message, UserWarning)
Network = deprecate(Network, deprecation_message, UserWarning)
NetworkOptions = deprecate(NetworkOptions, deprecation_message, UserWarning)
# OptimizerInfo = deprecate(OptimizerInfo, deprecation_message)
# NOTE: OptimizerInfo is a miss here since it's returned by contract/contract_path, 
# we want to use a single class such that cuquantum.contract_path & cuquantum.tensornet.contract_path returns the same output class.
 
OptimizerOptions = deprecate(OptimizerOptions, deprecation_message, UserWarning)
PathFinderOptions = deprecate(PathFinderOptions, deprecation_message, UserWarning)
ReconfigOptions = deprecate(ReconfigOptions, deprecation_message, UserWarning)
SlicerOptions = deprecate(SlicerOptions, deprecation_message, UserWarning)

del deprecate, deprecation_message
# new APIs should only be accessible from cuquantum.tensornet
del CirqParserOptions, QiskitParserOptions