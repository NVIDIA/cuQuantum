# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tensor network contraction with the standard einsum interface using cutensornet.
"""

__all__ = ['contract', 'contract_path', 'einsum', 'einsum_path', 'Network']

import collections
import dataclasses
import functools
import logging
import os
import sys

import cupy as cp
import numpy as np

from cuquantum import cutensornet as cutn
from . import configuration
from . import memory
from ._internal import einsum_parser
from ._internal import formatters
from ._internal import optimizer_ifc
from ._internal import tensor_wrapper
from ._internal import typemaps
from ._internal import utils


class InvalidNetworkState(Exception):
    pass


class Network:

    """
    Network(subscripts, *operands, options=None)

    Create a tensor network object specified as an Einstein summation expression.

    The Einstein summation convention provides an elegant way of representing many tensor network operations. This object
    allows the user to invest considerable effort into computing the best contraction path as well as autotuning the contraction
    upfront for repeated contractions over the same network *topology* (different input tensors, or "operands", with the same
    Einstein summation expression). Also see :meth:`~Network.contract_path` and :meth:`autotune`.

    For the Einstein summation expression, both the explicit and implicit forms are supported.

    In the implicit form, the output mode labels are inferred from the summation expression and *reordered lexicographically*.
    An example is the expression ``'ij,jh'``, for which the output mode labels are ``'hi'``. (This corresponds to a matrix
    multiplication followed by a transpose.)

    In the explicit form, output mode labels can be directly stated following the identifier ``'->'`` in the summation expression.
    An example is the expression ``'ij,jh->ih'`` (which corresponds to a matrix multiplication).

    To specify an Einstein summation expression, both the subscript format (as shown above) and the interleaved format
    are supported.

    The interleaved format is an alternative way for specifying the operands and their mode labels as
    ``Network(op0, modes0, op1, modes1, ..., [modes_out])``, where ``opN``
    is the N-th operand and ``modesN`` is a sequence of hashable and comparable objects (strings, integers, etc) representing the
    N-th operand's mode labels.

    Ellipsis broadcasting is supported.

    Additional information on various operations on the network can be obtained by passing in a :class:`logging.Logger` object
    to :class:`NetworkOptions` or by setting the appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    Args:
        subscripts : The mode labels (subscripts) defining the Einstein summation expression as a comma-separated sequence of
            characters. Unicode characters are allowed in the expression thereby expanding the size of the tensor network that
            can be specified using the Einstein summation convention.
        operands : A sequence of tensors (ndarray-like objects). The currently supported types are :class:`numpy.ndarray`,
            :class:`cupy.ndarray`, and :class:`torch.Tensor`.
        options : Specify options for the tensor network as a :class:`~cuquantum.NetworkOptions` object. Alternatively, a `dict`
            containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
            the value will be set to the default-constructed ``NetworkOptions`` object.

    See Also:
        :meth:`~Network.contract_path`, :meth:`autotune`, :meth:`~Network.contract`, :meth:`reset_operands`

    Examples:

        >>> from cuquantum import Network
        >>> import numpy as np

        Define the parameters of the tensor network:

        >>> expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
        >>> shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

        Create the input tensors using NumPy:

        >>> operands = [np.random.rand(*shape) for shape in shapes]

        Create a :class:`Network` object:

        >>> n = Network(expr, *operands)

        Find the best contraction order:

        >>> path, info = n.contract_path({'samples': 500})

        Autotune the network:

        >>> n.autotune(iterations=5)

        Perform the contraction. The result is of the same type and on the same device as the operands:

        >>> r1 = n.contract()

        Reset operands to new values:

        >>> operands = [i*operand for i, operand in enumerate(operands, start=1)]
        >>> n.reset_operands(*operands)

        Get the result of the new contraction:

        >>> r2 = n.contract()
        >>> from math import factorial
        >>> np.allclose(r2, factorial(len(operands))*r1)
        True

        Finally, free network resources. If this call isn't made, it may hinder further operations (especially if the
        network is large) since the memory will be released only when the object goes out of scope. (*To avoid having
        to explicitly make this call, it is recommended to use the* :class:`Network` *object as a context manager*.)

        >>> n.free()

        If the operands are on the GPU, they can also be updated using in-place operations. In this case, the call
        to :meth:`reset_operands` can be skipped -- subsequent :meth:`~Network.contract` calls will use the same
        operands (with updated contents). The following example illustrates this using CuPy operands and also demonstrates
        the usage of a :class:`Network` context (so as to skip calling :meth:`free`):

        >>> import cupy as cp
        >>> expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
        >>> shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]
        >>> operands = [cp.random.rand(*shape) for shape in shapes]
        >>>
        >>> with Network(expr, *operands) as n:
        ...     path, info = n.contract_path({'samples': 500})
        ...     n.autotune(iterations=5)
        ...
        ...     # Perform the contraction
        ...     r1 = n.contract()
        ...
        ...     # Update the operands in place
        ...     for i, operand in enumerate(operands, start=1):
        ...         operand *= i
        ...
        ...     # Perform the contraction with the updated operand values
        ...     r2 = n.contract()
        ...
        ... # The resources used by the network are automatically released when the context ends.
        >>>
        >>> from math import factorial
        >>> cp.allclose(r2, factorial(len(operands))*r1)
        array(True)

        PyTorch CPU and GPU tensors can be passed as input operands in the same fashion.

        See :func:`contract` for more examples on specifying the Einstein summation expression as well
        as specifying options for the tensor network and the optimizer.
    """

    def __init__(self, *operands, options=None):
        """
        __init__(subscripts, *operands, options=None)
        """

        options = utils.check_or_create_options(configuration.NetworkOptions, options, "network options")
        self.options = options

        # Logger.
        self.logger = options.logger if options.logger is not None else logging.getLogger()
        self.logger.info(f"CUDA runtime version = {cutn.get_cudart_version()}")
        self.logger.info(f"cuTensorNet version = {cutn.MAJOR_VER}.{cutn.MINOR_VER}.{cutn.PATCH_VER}")
        self.logger.info("Beginning network creation...")

        # Parse Einsum expression.
        self.operands, self.inputs, self.output, self.size_dict, self.mode_map_user_to_ord, self.mode_map_ord_to_user = einsum_parser.parse_einsum(*operands)

        # Copy operands to device if needed.
        self.network_location = 'cuda'
        self.device_id = utils.get_network_device_id(self.operands)
        if self.device_id is None:
            self.network_location = 'cpu'
            self.device_id = options.device_id
            self.operands = tensor_wrapper.to(self.operands, self.device_id)

        # Infer the library package the operands belong to.
        self.package = utils.get_operands_package(self.operands)

        # The output class is that of the first wrapped device operand.
        self.output_class = self.operands[0].__class__

        self.device = cp.cuda.Device(self.device_id)

        # Set memory allocator.
        self.allocator = options.allocator if options.allocator is not None else memory._MEMORY_MANAGER[self.package](self.device_id, self.logger)

        # Set memory limit.
        self.memory_limit = utils.get_memory_limit(self.options.memory_limit, self.device)
        self.logger.info(f"The memory limit is {formatters.MemoryStr(self.memory_limit)}.")

        # Define data types.
        self.data_type = utils.get_operands_dtype(self.operands)
        if self.data_type not in typemaps.NAME_TO_COMPUTE_TYPE:
            message = f"""Unsupported data type.
The data type '{self.data_type}' is currently not supported.
"""
            raise ValueError(message)
        self.compute_type = options.compute_type if options.compute_type is not None else typemaps.NAME_TO_COMPUTE_TYPE[self.data_type]

        # Prepare data for cutensornet.
        num_inputs = len(self.inputs)
        num_modes_out = len(self.output)

        extents_in = tuple(o.shape for o in self.operands)
        strides_in = tuple(o.strides for o in self.operands)
        self.operands_data, alignments_in = utils.get_operands_data(self.operands)
        modes_in = tuple(tuple(m for m in _input) for _input in self.inputs)
        num_modes_in = tuple(len(m) for m in modes_in)

        self.contraction, modes_out, extents_out, strides_out, alignment_out = utils.create_output_tensor(
                self.output_class, self.package, self.output, self.size_dict, self.device, self.data_type)

        # Create/set handle.
        if options.handle is not None:
            self.own_handle = False
            self.handle = options.handle
        else:
            self.own_handle = True
            with self.device:
                self.handle = cutn.create()

        # Network definition.
        self.network = cutn.create_network_descriptor(self.handle, num_inputs,
                num_modes_in, extents_in, strides_in, modes_in, alignments_in,        # inputs
                num_modes_out, extents_out, strides_out, modes_out, alignment_out,    # output
                typemaps.NAME_TO_DATA_TYPE[self.data_type], self.compute_type)

        # Keep output extents for creating new tensors, if needed.
        self.extents_out = extents_out

        # Path optimization atributes.
        self.optimizer_config_ptr, self.optimizer_info_ptr = None, None
        self.optimized = False

        # Workspace attributes.
        self.workspace_desc = cutn.create_workspace_descriptor(self.handle)
        self.workspace_ptr, self.workspace_size = None, None

        # Contraction plan attributes.
        self.plan = None

        # Autotuning attributes.
        self.autotune_pref_ptr = None
        self.autotuned = False

        self.valid_state = True

        self.logger.info("The network has been created.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()

    def _check_valid_network(self, *args, **kwargs):
        """
        """
        if not self.valid_state:
            raise InvalidNetworkState("The network cannot be used after resources are free'd")

    def _check_optimized(self, *args, **kwargs):
        """
        """
        what = kwargs['what']
        if not self.optimized:
            raise RuntimeError(f"{what} cannot be performed before contract_path() has been called.")

    def _free_plan_resources(self, exception=None):
        """
        Free resources allocated in network contraction planning.
        """

        if self.plan is not None:
            cutn.destroy_contraction_plan(self.plan)
            self.plan = None

        return True

    def _free_workspace_memory(self, exception=None):
        """
        Free workspace by releasing the MemoryPointer object.
        """
        self.workspace_ptr = None

        return True

    def _free_path_resources(self, exception=None):
        """
        Free resources allocated in path computation.
        """

        if self.optimizer_config_ptr is not None:
            cutn.destroy_contraction_optimizer_config(self.optimizer_config_ptr)
            self.optimizer_config_ptr = None

        if self.optimizer_info_ptr is not None:
            cutn.destroy_contraction_optimizer_info(self.optimizer_info_ptr)
            self.optimizer_info_ptr = None

        self._free_workspace_memory()
        self.workspace_size = None

        self._free_plan_resources()

        return True

    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_optimized, "Workspace memory allocation")
    @utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory_perhaps(self, stream_ctx):
        if self.workspace_ptr is not None:
            return

        assert self.workspace_size is not None, "Internal Error."

        self.logger.debug("Allocating memory for contracting the tensor network...")
        with self.device, stream_ctx:
            try:
                self.workspace_ptr = self.allocator.memalloc(self.workspace_size)
            except TypeError as e:
                message = "The method 'memalloc' in the allocator object must conform to the interface in the "\
                          "'BaseCUDAMemoryManager' protocol."
                raise TypeError(message) from e
        self.logger.debug(f"Finished allocating memory of size {formatters.MemoryStr(self.workspace_size)} for contraction.")

        device_ptr = utils.get_ptr_from_memory_pointer(self.workspace_ptr)
        cutn.workspace_set(self.handle, self.workspace_desc, cutn.Memspace.DEVICE, device_ptr, self.workspace_size)
        self.logger.debug(f"The workspace memory (device pointer = {device_ptr}) has been set in the workspace descriptor.")

    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_optimized, "Workspace size calculation")
    def _calculate_workspace_size(self):
        """
        Allocate workspace for cutensornet.
        """

        # Release workspace already allocated, if any, because the new requirements are likely different.
        self.workspace_ptr = None

        cutn.workspace_compute_sizes(self.handle, self.network, self.optimizer_info_ptr, self.workspace_desc)

        min_size = cutn.workspace_get_size(self.handle, self.workspace_desc, cutn.WorksizePref.MIN, cutn.Memspace.DEVICE)
        max_size = cutn.workspace_get_size(self.handle, self.workspace_desc, cutn.WorksizePref.MAX, cutn.Memspace.DEVICE)

        if self.memory_limit < min_size:
            message = f"""Insufficient memory.
The memory limit specified is {self.memory_limit}, while the minimum workspace size needed is {min_size}.
"""
            raise RuntimeError(message)

        self.workspace_size = max_size if max_size < self.memory_limit else self.memory_limit
        self.logger.info(f"The workspace size requirements range from {formatters.MemoryStr(min_size)} to "\
                         f"{formatters.MemoryStr(max_size)}.")
        self.logger.info(f"The workspace size has been set to {formatters.MemoryStr(self.workspace_size)}.")

        # Set workspace size to enable contraction planning. The device pointer will be set later during allocation.
        cutn.workspace_set(self.handle, self.workspace_desc, cutn.Memspace.DEVICE, 0, self.workspace_size)


    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_optimized, "Planning")
    @utils.atomic(_free_plan_resources, method=True)
    def _create_plan(self):
        """
        Create network plan.
        """

        self.logger.debug("Creating contraction plan...")

        if self.plan:
            cutn.destroy_contraction_plan(self.plan)

        self.plan = cutn.create_contraction_plan(self.handle, self.network, self.optimizer_info_ptr, self.workspace_desc)

        self.logger.debug("Finished creating contraction plan.")

    def _set_opt_config_options(self, options):
        """
        Set ContractionOptimizerConfig options if the value is not None.

        Args:
            options: A PathFinderOptions, SlicerOptions, or ReconfigOptions object.
        """
        for field in dataclasses.fields(options):
            name, value = field.name, getattr(options, field.name)
            if value is None:
                continue

            enum = options.option_to_enum[name]
            self._set_opt_config_option(name, enum, value)

    def _set_opt_config_option(self, name, enum, value):
        """
        Set a single ContractionOptimizerConfig option if the value is not None.

        Args:
            name: The name of the attribute.
            enum: A ContractionOptimizerConfigAttribute to set.
            value: The value to which the attribute is set to.
        """
        if value is None:
            return

        dtype = cutn.contraction_optimizer_config_get_attribute_dtype(enum)
        value = np.array((value,), dtype=dtype)
        cutn.contraction_optimizer_config_set_attribute(self.handle, self.optimizer_config_ptr, enum, value.ctypes.data, value.dtype.itemsize)
        self.logger.info(f"The optimizer config attribute '{name}' has been set to {value[0]}.")

    @utils.precondition(_check_valid_network)
    def _set_optimizer_options(self, optimize):
        """
        """
        # Loop over the options and set if not None.

        assert isinstance(optimize.path, configuration.PathFinderOptions), "Internal error."

        # PathFinder options.
        self._set_opt_config_options(optimize.path)

        # Slicer options.
        if isinstance(optimize.slicing, configuration.SlicerOptions):
            self._set_opt_config_options(optimize.slicing)

        # Reconfiguration options.
        self._set_opt_config_options(optimize.reconfiguration)

        # The "global" options.
        ConfEnum = cutn.ContractionOptimizerConfigAttribute

        enum = ConfEnum.HYPER_NUM_SAMPLES
        self._set_opt_config_option('samples', enum, optimize.samples)

        enum = ConfEnum.HYPER_NUM_THREADS
        self._set_opt_config_option('threads', enum, optimize.threads)

        enum = ConfEnum.SEED
        self._set_opt_config_option('seed', enum, optimize.seed)

    @utils.precondition(_check_valid_network)
    @utils.atomic(_free_path_resources, method=True)
    def contract_path(self, optimize=None):
        """Compute the best contraction path together with any slicing that is needed to ensure that the contraction can be
        performed within the specified memory limit.

        Args:
            optimize :  This parameter specifies options for path optimization as an :class:`OptimizerOptions` object. Alternatively, a
                dictionary containing the parameters for the ``OptimizerOptions`` constructor can also be provided. If not
                specified, the value will be set to the default-constructed ``OptimizerOptions`` object.

        Returns:
            tuple: A 2-tuple (``path``, ``opt_info``):

                - ``path`` :  A sequence of pairs of operand ordinals representing the best contraction order in the
                  :func:`numpy.einsum_path` format.
                - ``opt_info`` : An object of type :class:`OptimizerInfo` containing information about the best contraction order.

        Notes:

            - If the path is provided, the user has to set the sliced modes too if slicing is desired.
        """

        optimize = utils.check_or_create_options(configuration.OptimizerOptions, optimize, "path optimizer options")

        if self.optimizer_config_ptr is None:
            self.optimizer_config_ptr = cutn.create_contraction_optimizer_config(self.handle)
        if self.optimizer_info_ptr is None:
            self.optimizer_info_ptr = cutn.create_contraction_optimizer_info(self.handle, self.network)

        opt_info_ifc = optimizer_ifc.OptimizerInfoInterface(self)

        # Compute path (or set provided path).
        if isinstance(optimize.path, configuration.PathFinderOptions):
            # Set optimizer options.
            self._set_optimizer_options(optimize)
            # Find "optimal" path.
            self.logger.info("Finding optimal path as well as sliced modes...")
            cutn.contraction_optimize(self.handle, self.network, self.optimizer_config_ptr, self.memory_limit, self.optimizer_info_ptr)
            self.logger.info("Finished finding optimal path as well as sliced modes.")
        else:
            self.logger.info("Setting user-provided path...")
            opt_info_ifc.path = optimize.path
            self.logger.info("Finished setting user-provided path.")

        # Set slicing if provided.
        if not isinstance(optimize.slicing, configuration.SlicerOptions):
            self.logger.info("Setting user-provided sliced modes...")
            opt_info_ifc.sliced_mode_extent = optimize.slicing
            self.logger.info("Finished setting user-provided sliced modes.")

        self.num_slices = opt_info_ifc.num_slices
        assert self.num_slices > 0

        # Create OptimizerInfo object.
        largest_intermediate = opt_info_ifc.largest_intermediate
        opt_cost = opt_info_ifc.flop_count
        path = opt_info_ifc.path
        slices = opt_info_ifc.sliced_mode_extent

        opt_info = configuration.OptimizerInfo(largest_intermediate, opt_cost, path, slices)
        self.logger.info(f"{opt_info}")

        self.optimized = True

        # Calculate workspace size required.
        self._calculate_workspace_size()

        # Create plan.
        self._create_plan()

        return opt_info.path, opt_info

    def _set_autotune_options(self, options):
        """
        Set ContractionAutotunePreference options if the value is not None.

        Args:
            options: dict of name : (enum, value) AutotunePreference parameters.
        """
        for name in options:
            enum, value = options[name]
            if value is None:
                continue

            self._set_autotune_option(name, enum, value)

    def _set_autotune_option(self, name, enum, value):
        """
        Set a single ContractionAutotunePreference option if the value is not None.

        Args:
            name: The name of the attribute.
            enum: A ContractionAutotunePreferenceAttribute to set.
            value: The value to which the attribute is set to.
        """
        if value is None:
            return

        dtype = cutn.contraction_autotune_preference_get_attribute_dtype(enum)
        value = np.array((value,), dtype=dtype)
        cutn.contraction_autotune_preference_set_attribute(self.handle, self.autotune_pref_ptr, enum, value.ctypes.data, value.dtype.itemsize)
        self.logger.info(f"The autotune preference '{name}' has been set to {value[0]}.")

    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_optimized, "Autotuning")
    def autotune(self, *, iterations=3, stream=None):
        """Autotune the network to reduce the contraction cost.

        This is an optional step that is recommended if the :class:`Network` object is used to perform multiple contractions.

        Args:
            iterations: The number of iterations for autotuning. See `CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS`.
            stream: Provide the CUDA stream to use for the autotuning operation. Acceptable inputs include ``cudaStream_t``
                (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
                the current stream will be used.
        """

        message = utils.check_autotune_params(iterations)
        self.logger.info(message)
        if self.autotune_pref_ptr is None:
            self.autotune_pref_ptr = cutn.create_contraction_autotune_preference(self.handle)

        AutoEnum = cutn.ContractionAutotunePreferenceAttribute
        options = {'iterations': (AutoEnum.MAX_ITERATIONS, iterations)}
        self._set_autotune_options(options)

        # Allocate device memory (in stream context) if needed.
        stream, stream_ctx, stream_ptr = utils.get_or_create_stream(self.device, stream, self.package)
        self._allocate_workspace_memory_perhaps(stream_ctx)

        # Check if we still hold an output tensor; if not, create a new one.
        if self.contraction is None:
            self.contraction = utils.create_empty_tensor(self.output_class, self.extents_out, self.data_type, self.device_id, stream_ctx)

        self.logger.info(f"Starting autotuning...")
        with self.device:
            start = stream.record()
            cutn.contraction_autotune(self.handle, self.plan, self.operands_data, self.contraction.data_ptr,
                 self.workspace_desc, self.autotune_pref_ptr, stream_ptr)
            end = stream.record()
            end.synchronize()
            elapsed = cp.cuda.get_elapsed_time(start, end)

        self.autotuned = True
        self.logger.info(f"The autotuning took {elapsed:.3f} ms to complete.")

    @utils.precondition(_check_valid_network)
    def reset_operands(self, *operands):
        """Reset the operands held by this :class:`Network` instance.

        This method is not needed when the operands
        reside on the GPU and in-place operations are used to update the operand values.

        This method will perform various checks on the new operands to make sure:

            - The shapes, strides, datatypes match those of the old ones.
            - The packages that the operands belong to match those of the old ones.
            - If input tensors are on GPU, the library package, device, and alignments must match.

        Args:
            operands: See :class:`Network`'s documentation.
        """

        if len(operands) != len(self.operands):
            message = f"Mismatch in the number of operands ({len(operands)} provided, need {len(self.operands)})."
            raise ValueError(message)

        self.logger.info("Resetting operands...")
        # First wrap operands.
        operands = tensor_wrapper.wrap_operands(operands)

        utils.check_operands_match(self.operands, operands, 'dtype', "data type")
        utils.check_operands_match(self.operands, operands, 'shape', 'shape')
        utils.check_operands_match(self.operands, operands, 'strides', 'strides')

        device_id = utils.get_network_device_id(operands)
        if device_id is None:
            # Copy to existing device pointers because the new operands are on the CPU.
            tensor_wrapper.copy_(operands, self.operands)
        else:
            package = utils.get_operands_package(operands)
            if self.package != package:
                message = f"Library package mismatch: '{self.package}' => '{package}'"
                raise TypeError(message)

            if self.device_id != device_id:
                raise ValueError(f"The new operands must be on the same device ({device_id}) as the original operands "
                                 f"({self.device_id}).")

            _, orig_alignments = utils.get_operands_data(self.operands)
            new_operands_data, new_alignments = utils.get_operands_data(operands)
            utils.check_alignments_match(orig_alignments, new_alignments)

            # Finally, replace the original data pointers by the new ones.
            self.operands_data = new_operands_data
        self.logger.info("The operands have been reset.")

    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_optimized, "Contraction")
    def contract(self, *, stream=None):
        """Contract the network and return the result.

        Args:
            stream: Provide the CUDA stream to use for the autotuning operation. Acceptable inputs include ``cudaStream_t``
                (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
                the current stream will be used.

        Returns:
            The result is of the same type and on the same device as the operands.
        """

        # Allocate device memory (in stream context) if needed.
        stream, stream_ctx, stream_ptr = utils.get_or_create_stream(self.device, stream, self.package)
        self._allocate_workspace_memory_perhaps(stream_ctx)

        # Check if we still hold an output tensor; if not, create a new one.
        if self.contraction is None:
            self.contraction = utils.create_empty_tensor(self.output_class, self.extents_out, self.data_type, self.device_id, stream_ctx)

        self.logger.info("Starting network contraction...")
        with self.device:
            start = stream.record()
            for s in range(self.num_slices):
                cutn.contraction(self.handle, self.plan, self.operands_data, self.contraction.data_ptr,
                    self.workspace_desc, s, stream_ptr)
            end = stream.record()
            end.synchronize()
            elapsed = cp.cuda.get_elapsed_time(start, end)

        self.logger.info(f"The contraction took {elapsed:.3f} ms to complete.")

        if self.network_location == 'cpu':
            out = self.contraction.to('cpu')
        else:
            out = self.contraction.tensor
        self.contraction = None  # We cannot overwrite what we've already handed to users.
        return out

    def free(self):
        """Free network resources.

        It is recommended that the :class:`Network` object be used within a context, but if it is not possible then this
        method must be called explicitly to ensure that the network resources are properly cleaned up.
        """

        if not self.valid_state:
            return

        try:
            self._free_path_resources()

            if self.autotune_pref_ptr is not None:
                cutn.destroy_contraction_autotune_preference(self.autotune_pref_ptr)
                self.autotune_pref_ptr = None

            if self.workspace_desc is not None:
                cutn.destroy_workspace_descriptor(self.workspace_desc)
                self.workspace_desc = None

            if self.network is not None:
                cutn.destroy_network_descriptor(self.network)
                self.network = None

            if self.handle is not None and self.own_handle:
                cutn.destroy(self.handle)
                self.handle = None
                self.own_handle = False
        except Exception as e:
            self.logger.critical("Internal error: only part of the network resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The network resources have been released.")


def contract(*operands, options=None, optimize=None, stream=None, return_info=False):
    """
    contract(subscripts, *operands, options=None, optimize=None, stream=None, return_info=False)

    Evaluate the Einstein summation convention on the operands.

    Explicit as well as implicit form is supported for the Einstein summation expression. In addition to the subscript format,
    the interleaved format is also supported as a means of specifying the operands and their mode labels. See :class:`Network`
    for more detail on the types of operands as well as for examples.

    Args:
        subscripts : The mode labels (subscripts) defining the Einstein summation expression as a comma-separated sequence of
            characters. Unicode characters are allowed in the expression thereby expanding the size of the tensor network that
            can be specified using the Einstein summation convention.
        operands : A sequence of tensors (ndarray-like objects). The currently supported types are :class:`numpy.ndarray`,
            :class:`cupy.ndarray`, and :class:`torch.Tensor`.
        options : Specify options for the tensor network as a :class:`~cuquantum.NetworkOptions` object. Alternatively, a `dict`
            containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
            the value will be set to the default-constructed ``NetworkOptions`` object.
        optimize :  This parameter specifies options for path optimization as an :class:`OptimizerOptions` object. Alternatively, a
            dictionary containing the parameters for the ``OptimizerOptions`` constructor can also be provided. If not
            specified, the value will be set to the default-constructed ``OptimizerOptions`` object.
        stream: Provide the CUDA stream to use for the autotuning operation. Acceptable inputs include ``cudaStream_t``
            (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
            the current stream will be used.
        return_info : If true, information about the best contraction order will also be returned.

    Returns:
        If ``return_info`` is `False`, the output tensor (ndarray-like object) of the same type and on the same device
        as the operands containing the result of the contraction; otherwise, a 2-tuple consisting of the output tensor and an
        :class:`OptimizerInfo` object that contains information about the best contraction order etc.

    .. note::
        It is encouraged for users to maintain the library handle themselves so as to reduce the context initialization time:

        .. code-block:: python

            from cuquantum import cutensornet, NetworkOptions, contract

            handle = cutensornet.create()
            network_opts = NetworkOptions(handle=handle, ...)
            out = contract(..., options=network_opts, ...)
            # ... the same handle can be reused for further calls ...
            # when it's done, remember to destroy the handle
            cutensornet.destroy(handle)

    Examples:

        Use NumPy operands:

        >>> from cuquantum import contract
        >>> import numpy as np
        >>> a = np.arange(6.).reshape(3, 2)
        >>> b = np.arange(6.).reshape(2, 3)

        Perform matrix multiplication using the explicit form. The result ``r`` is a NumPy ndarray (with the computation
        performed on the GPU):

        >>> r = contract('ij,jk->ik', a, b)

        Implicit form:

        >>> r = contract('ij,jk', a, b)

        Interleaved format using characters for mode labels:

        >>> r = contract(a, ['i', 'j'], b, ['j', 'k'], ['i', 'k'], return_info=True)

        Interleaved format using string labels for mode labels and implicit form:

        >>> r = contract(a, ['first', 'second'], b, ['second', 'third'])

        Interleaved format using integer mode labels and explicit form:

        >>> r = contract(a, [1, 2], b, [2, 3], [1, 3])

        Obtain information ``i`` on the best contraction path along with the result ``r``:

        >>> r, i = contract('ij,jk', a, b, return_info=True)

        Provide options for the tensor network:

        >>> from cuquantum import NetworkOptions
        >>> n = NetworkOptions(device_id=1)
        >>> r = contract('ij,jk->ik', a, b, options=n)

        Alternatively, the options can be provided as a dict instead of a :class:`NetworkOptions` object:

        >>> r = contract('ij,jk->ik', a, b, options={'device_id': 1})

        Specify options for the optimizer:

        >>> from cuquantum import OptimizerOptions, PathFinderOptions
        >>> p = PathFinderOptions(imbalance_factor=230, cutoff_size=8)
        >>> o = OptimizerOptions(path=p, seed=123)
        >>> r = contract('ij,jk,kl', a, b, a, optimize=o)

        Alternatively, the options above can be provided as a dict:

        >>> r = contract('ij,jk,kl', a, b, a, optimize={'path': {'imbalance_factor': 230, 'cutoff_size': 8}, 'seed': 123})

        Specify the path directly:

        >>> o = OptimizerOptions(path = [(0,2), (0,1)])
        >>> r = contract('ij,jk,kl', a, b, a, optimize=o)

        Perform elementwise multiplication :math:`a \odot b^T` using the ellipsis shorthand notation:

        >>> r = contract('...,...', a, b.T)

        Obtain the double inner product :math:`a : b^T` (Frobenius inner product for real-valued tensors) using the
        ellipsis shorthand notation:

        >>> r = contract('...,...->', a, b.T)

        Use CuPy operands. The result ``r`` is a CuPy ndarray on the same device as the operands, and ``dev`` is any valid
        device ID on your system that you wish to use to store the tensors and compute the contraction:

        >>> import cupy
        >>> dev = 0
        >>> with cupy.cuda.Device(dev):
        ...     a = cupy.arange(6.).reshape(3, 2)
        ...     b = cupy.arange(6.).reshape(2, 3)
        >>> r = contract('ij,jk', a, b)

        Use PyTorch operands. The result ``r`` is a PyTorch tensor on the same device (``dev``) as the operands:

    .. doctest::
        :skipif: torch is None

        >>> import torch
        >>> dev = 0
        >>> a = torch.arange(6., device=f'cuda:{dev}').reshape(3, 2)
        >>> b = torch.arange(6., device=f'cuda:{dev}').reshape(2, 3)
        >>> r = contract('ij,jk', a, b)
    """

    options = utils.check_or_create_options(configuration.NetworkOptions, options, "network options")

    optimize = utils.check_or_create_options(configuration.OptimizerOptions, optimize, "path optimizer options")

    # Create network.
    with Network(*operands, options=options) as network:

        # Compute path.
        opt_info = network.contract_path(optimize=optimize)

        # Skip autotuning since the network is contracted only once.

        # Contraction.
        output = network.contract(stream=stream)

    if return_info:
        return output, opt_info

    return output


def contract_path(*operands, options=None, optimize=None):
    """
    contract_path(subscripts, *operands, options=None, optimize=None)

    Evaluate the "best" contraction order by allowing the creation of intermediate tensors.

    Explicit as well as implicit form is supported for the Einstein summation expression. In addition to the subscript format,
    the interleaved format is also supported as a means of specifying the operands and their mode labels. See :class:`Network`
    for more detail on the types of operands as well as for examples.

    Args:
        subscripts : The mode labels (subscripts) defining the Einstein summation expression as a comma-separated sequence of
            characters. Unicode characters are allowed in the expression thereby expanding the size of the tensor network that
            can be specified using the Einstein summation convention.
        operands : A sequence of tensors (ndarray-like objects). The currently supported types are :class:`numpy.ndarray`,
            :class:`cupy.ndarray`, and :class:`torch.Tensor`.
        options : Specify options for the tensor network as a :class:`~cuquantum.NetworkOptions` object. Alternatively, a `dict`
            containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
            the value will be set to the default-constructed ``NetworkOptions`` object.
        optimize :  This parameter specifies options for path optimization as an :class:`OptimizerOptions` object. Alternatively, a
            dictionary containing the parameters for the ``OptimizerOptions`` constructor can also be provided. If not
            specified, the value will be set to the default-constructed ``OptimizerOptions`` object.

    Returns:
        tuple: A 2-tuple (``path``, ``opt_info``):

            - ``path`` :  A sequence of pairs of operand ordinals representing the best contraction order in the
              :func:`numpy.einsum_path` format.
            - ``opt_info`` : An object of type :class:`OptimizerInfo` containing information about the best contraction order.

    .. note::
        It is encouraged for users to maintain the library handle themselves so as to reduce the context initialization time:

        .. code-block:: python

            from cuquantum import cutensornet, NetworkOptions, contract_path

            handle = cutensornet.create()
            network_opts = NetworkOptions(handle=handle, ...)
            path, info = contract_path(..., options=network_opts, ...)
            # ... the same handle can be reused for further calls ...
            # when it's done, remember to destroy the handle
            cutensornet.destroy(handle)

    """

    options = utils.check_or_create_options(configuration.NetworkOptions, options, "network options")

    optimize = utils.check_or_create_options(configuration.OptimizerOptions, optimize, "path optimizer options")

    # Create network.
    with Network(*operands, options=options) as network:

        # Compute path.
        path, opt_info = network.contract_path(optimize=optimize)

    return path, opt_info


def _check_einsum_options(out, dtype, order, casting, optimize):
    """
    Check whether the options provided to the einsum function interface are supported.
    """
    if out is not None:
        message = f"value '{out}' for parameter 'out'."
        raise NotImplementedError(message)

    if dtype is not None:
        message = f"value '{dtype}' for parameter 'dtype'."
        raise NotImplementedError(message)

    if order != 'K':
        message = f"value '{order}' for parameter 'order'."
        raise NotImplementedError(message)

    if casting.lower() != 'safe':
        message = f"value '{casting}' for parameter 'casting'."
        raise NotImplementedError(message)

    if optimize not in (True, False) and not isinstance(optimize, collections.abc.Sequence):
        message = f"""value '{optimize}' for parameter 'optimize'.
Only True or False values are allowed. Alternatively an explicit contraction list from einsum_path
can be provided."""
        raise NotImplementedError(message)


def einsum(*operands, out=None, dtype=None, order='K', casting='safe', optimize=True):
    """
    einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=True)

    A drop-in replacement of :func:`numpy.einsum` for computing the specified tensor contraction using cuTensorNet.

    Not all NumPy options are supported or even used. The :func:`contract` function provides an extensive set of options
    specific to cuTensorNet and is recommended over this function.

    Explicit as well as implicit form is supported for the Einstein summation expression. In addition to the subscript format,
    the interleaved format is also supported as a means of specifying the operands and their mode labels. See :class:`Network`
    for more detail on the types of operands as well as for examples.

    Args:
        subscripts : The mode labels (subscripts) defining the Einstein summation expression as a comma-separated sequence of
            characters. Unicode characters are allowed in the expression thereby expanding the size of the tensor network that
            can be specified using the Einstein summation convention.
        operands : A sequence of tensors (ndarray-like objects). The currently supported types are :class:`numpy.ndarray`,
            :class:`cupy.ndarray`, and :class:`torch.Tensor`.
        out : Not supported in this release.
        dtype : Not supported in this release.
        order : Not supported in this release.
        casting : Not supported in this release.
        optimize :  This parameter specifies options for path optimization. The only values accepted by this interface are `True`,
            `False`, or the contraction path specified in the :func:`numpy.einsum_path` format.

    Returns:
        output:
            A tensor (ndarray-like object) of the same type and on the same device as the operands containing the result of
            the contraction.
    """

    _check_einsum_options(out, dtype, order, casting, optimize)

    # Create network.
    with Network(*operands) as network:

        if optimize is True:
            # Compute path.
            network.contract_path()
        else:
            if optimize is False:
                # Use canonical path.
                path = [(0, 1)] * (network.num_inputs - 1)
            else:
                # Use specified path.
                path = optimize

            # Set path (path validation is done when setting OptimizerOptions).
            optimize = configuration.OptimizerOptions(path=path)
            network.contract_path(optimize=optimize)

        # Skip autotuning since the network is contracted only once.

        # Contraction.
        output = network.contract()

    return output


def einsum_path(*operands, optimize=True):
    """
    einsum_path(subscripts, *operands, optimize=True)

    A drop-in replacement of :func:`numpy.einsum_path` for evaluating the "best" contraction order using cuTensorNet.

    Only a subset of the NumPy options is supported using this interface. The :func:`contract_path` function provides an
    extensive set of options specific to cuTensorNet and is recommended over this function.

    Explicit as well as implicit form is supported for the Einstein summation expression. In addition to the subscript format,
    the interleaved format is also supported as a means of specifying the operands and their mode labels. See :class:`Network`
    for more detail on the types of operands as well as for examples.

    Args:
        subscripts : The mode labels (subscripts) defining the Einstein summation expression as a comma-separated sequence of
            characters. Unicode characters are allowed in the expression thereby expanding the size of the tensor network that
            can be specified using the Einstein summation convention.
        operands : A sequence of tensors (ndarray-like objects). The currently supported types are :class:`numpy.ndarray`,
            :class:`cupy.ndarray`, and :class:`torch.Tensor`.
        optimize : This parameter specifies options for path optimization. The only value allowed with this interface is `True`.

    Returns:
        tuple: A 2-tuple (``path``, ``opt_info``):

            - ``path`` :  A list starting with the string 'einsum_path' and followed by a sequence of pairs of operand ordinals
                representing the best contraction order in the :func:`numpy.einsum_path` format.
            - ``opt_info`` : String representation of an object of type :class:`OptimizerInfo` containing information about
                the best contraction order.
    """

    if optimize is not True:
        message = f"""Invalid value for parameter 'optimize'.
The only allowed value for 'optimize' is True."""
        raise NotImplementedError(message)

    # Create network.
    with Network(*operands) as network:

        # Compute path.
        path, opt_info = network.contract_path()

    return ['einsum_path', *path], str(opt_info)
