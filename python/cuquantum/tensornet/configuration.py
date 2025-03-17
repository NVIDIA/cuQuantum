# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
A collection of types for defining options to cutensornet.
"""

__all__ = ['NetworkOptions', 'OptimizerInfo', 'OptimizerOptions', 'PathFinderOptions', 
    'ReconfigOptions', 'SlicerOptions', 'MemoryLimitExceeded']

import collections
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Hashable, Iterable, Literal, Optional, Tuple, Union


import cuquantum
from cuquantum.bindings import cutensornet as cutn
from .._internal import enum_utils
from .._internal import formatters
from .._internal.mem_limit import check_memory_str
from .memory import BaseCUDAMemoryManager

@dataclass
class NetworkOptions(object):
    """A data class for providing options to the :class:`cuquantum.Network` object.

    Attributes:
        compute_type (cuquantum.ComputeType): CUDA compute type. A suitable compute type will be selected if not specified.
        device_id: CUDA device ordinal (used if the tensor network resides on the CPU). Device 0 will be used if not specified.
        handle: cuTensorNet library handle. A handle will be created if one is not provided.
        logger (logging.Logger): Python Logger object. The root logger will be used if a logger object is not provided.
        memory_limit: Maximum memory available to cuTensorNet. It can be specified as a value (with optional suffix like
            K[iB], M[iB], G[iB]) or as a percentage. The default is 80% of the device memory.
        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :meth:`Network.autotune` and :meth:`Network.contract`.
            When ``blocking`` is ``True``, these methods do not return until the operation is complete. When blocking is ``"auto"``,
            the methods return immediately when the input tensors are on the GPU. The execution methods always block when the
            input tensors are on the CPU. The default is ``True``.
        allocator: An object that supports the :class:`BaseCUDAMemoryManager` protocol, used to draw device memory. If an
            allocator is not provided, a memory allocator from the library package will be used
            (:func:`torch.cuda.caching_allocator_alloc` for PyTorch operands, :func:`cupy.cuda.alloc` otherwise).
    """
    compute_type : Optional[int] = None
    device_id : Optional[int] = None
    handle : Optional[int] = None
    logger : Optional[Logger] = None
    memory_limit : Optional[Union[int, str]] = r'80%'
    blocking : Literal[True, "auto"] = True
    allocator : Optional[BaseCUDAMemoryManager] = None

    def __post_init__(self):
        #  Defer creating handle as well as computing the memory limit till we know the device the network is on.

        if self.compute_type is not None:
            self.compute_type = cuquantum.ComputeType(self.compute_type)

        if self.device_id is None:
            self.device_id = 0

        check_memory_str(self.memory_limit, "memory limit")

        if self.blocking != True and self.blocking != "auto":
            raise ValueError("The value specified for blocking must be either True or 'auto'.")

        if self.allocator is not None and not isinstance(self.allocator, BaseCUDAMemoryManager):
            raise TypeError("The allocator must be an object of type that fulfils the BaseCUDAMemoryManager protocol.")

    def _check_compatible_with_state(self):
        if self.compute_type is not None:
            raise ValueError(f"NetworkState and NetworkOperator only supports default compute_type")
        
# Generate the options dataclasses from ContractionOptimizerConfigAttributes.

_create_options = enum_utils.create_options_class_from_enum
_opt_conf_enum = cutn.ContractionOptimizerConfigAttribute
_get_dtype = cutn.contraction_optimizer_config_get_attribute_dtype

PathFinderOptions = _create_options('PathFinderOptions', _opt_conf_enum, _get_dtype, "path finder", 'GRAPH_(?P<option_name>.*)')

SlicerOptions = _create_options('SlicerOptions', _opt_conf_enum, _get_dtype, 'slicer', 'SLICER_(?P<option_name>.*)')

ReconfigOptions = _create_options('ReconfigOptions', _opt_conf_enum, _get_dtype, 'reconfiguration', 'RECONFIG_(?P<option_name>.*)')

del _create_options, _opt_conf_enum, _get_dtype

PathType = Iterable[Tuple[int, int]]
ModeSequenceType = Iterable[Hashable]
ModesSequenceType = Iterable[ModeSequenceType]
ModeExtentSequenceType = Iterable[Tuple[Hashable, int]]
KeywordArgType = Dict


@dataclass
class OptimizerOptions(object):
    """A data class for providing options to the cuTensorNet optimizer.

    Attributes:
        samples: Number of samples for hyperoptimization. See `CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES`.
        threads: Number of threads for the hyperoptimizer. See `CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS`.
        path: Options for the path finder (:class:`~cuquantum.PathFinderOptions` object or dict containing the ``(parameter, value)``
            items for ``PathFinderOptions``). Alternatively, the path can be provided as a sequence of pairs in the
            :func:`numpy.einsum_path` format.
        slicing: Options for the slicer (:class:`~cuquantum.SlicerOptions` object or dict containing the ``(parameter, value)`` items for
            ``SlicerOptions``). Alternatively, a sequence of sliced modes or sequence of ``(sliced mode, sliced extent)`` pairs
            can be directly provided.
        reconfiguration: Options for the reconfiguration algorithm as a :class:`~cuquantum.ReconfigOptions` object or dict containing the
            ``(parameter, value)`` items for ``ReconfigOptions``.
        seed: Optional seed for the random number generator. See `CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED`.
        cost_function: The objective function to use for finding the optimal contraction path.
            See `CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE`.
        smart: Whether the optimizer can execute the pre-defined, intelligent heuristics to accelerate path optimization.
            See `CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION`.
    """
    samples : Optional[int] = None
    threads : Optional[int] = None
    path : Optional[Union[PathFinderOptions, PathType]] = None
    slicing : Optional[Union[SlicerOptions, ModeSequenceType, ModeExtentSequenceType]] = None
    reconfiguration : Optional[ReconfigOptions] = None
    seed : Optional[int] = None
    cost_function: Optional[int] = None
    smart: Optional[bool] = None

    def _check_option(self, option, option_class, checker=None):
        if isinstance(option, option_class):
            return option

        if option is None:
            option = option_class()
        elif isinstance(option, KeywordArgType):
            option = option_class(**option)
        elif checker is not None:
            checker()

        return option

    def _check_specified_path(self):
        if not isinstance(self.path, collections.abc.Sequence):
            raise TypeError("The path must be a sequence of pairs in the linear format accepted by numpy.einsum_path.")

        for pair in self.path:
            if not isinstance(pair, collections.abc.Sequence) or len(pair) != 2:
                raise TypeError("The path must be a sequence of pairs in the linear format accepted by numpy.einsum_path.")

    def _check_specified_slices(self):
        if not isinstance(self.slicing, collections.abc.Sequence):
            raise TypeError("Slicing must be specified as a sequence of modes or as a sequence of (mode, extent) pairs.")

        pair = False
        for slc in self.slicing:
            if isinstance(slc, collections.abc.Sequence) and not isinstance(slc, str):
                pair = True
                break

        for s in self.slicing:
            if pair and (isinstance(s, str) or not isinstance(s, collections.abc.Sequence) or len(s) != 2):
                raise TypeError("Slicing must be specified as a sequence of modes or as a sequence of (mode, extent) pairs.")

    def _check_int(self, attribute, name):
        message = f"Invalid value ({attribute}) for '{name}'. Expect positive integer or None."
        if not isinstance(attribute, (type(None), int)):
            raise ValueError(message)
        if isinstance(attribute, int) and attribute < 0:
            raise ValueError(message)

    def __post_init__(self):
        self._check_int(self.samples, "samples")
        self.path = self._check_option(self.path, PathFinderOptions, self._check_specified_path)
        self.slicing = self._check_option(self.slicing, SlicerOptions, self._check_specified_slices)
        self.reconfiguration = self._check_option(self.reconfiguration, ReconfigOptions, None)
        self._check_int(self.seed, "seed")
        if self.cost_function is not None:
            self.cost_function = cuquantum.cutensornet.OptimizerCost(self.cost_function)
        if self.smart is not None:
            self.smart = cuquantum.cutensornet.SmartOption(self.smart)


@dataclass
class OptimizerInfo(object):
    """A data class for capturing optimizer information.

    Attributes:
        largest_intermediate: The number of elements in the largest intermediate tensor. See `CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR`.
        opt_cost: The FLOP count of the optimized contraction path. See `CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT`.
        path: The contraction path as a sequence of pairs in the :func:`numpy.einsum_path` format.
        slices: A sequence of ``(sliced mode, sliced extent)`` pairs.
        num_slices : The number of slices.
        intermediate_modes: A sequence of mode labels for all intermediate tensors
    """
    largest_intermediate : float
    opt_cost : float
    path : PathType
    slices : ModeExtentSequenceType
    num_slices : int
    intermediate_modes : ModesSequenceType

    def __str__(self):
        path = [str(p) for p in self.path]
        slices = [str(s) for s in self.slices]
        intermediate_modes = [str(m) for m in self.intermediate_modes]
        s = f"""Optimizer Information:
    Largest intermediate = {formatters.MemoryStr(self.largest_intermediate, base_unit='Elements')}
    Optimized cost = {self.opt_cost:.3e} FLOPs
    Path = {formatters.array2string(path)}"""
        if len(slices):
            s += f"""
    Number of slices = {self.num_slices}
    Slices = {formatters.array2string(slices)}"""
        else:
            s += """
    Slicing not needed."""
        s += f"""
    Intermediate tensor mode labels = {formatters.array2string(intermediate_modes)}"""

        return s


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
