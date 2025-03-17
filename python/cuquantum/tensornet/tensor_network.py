# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tensor network contraction with the standard einsum interface using cutensornet.
"""

__all__ = ['contract', 'contract_path', 'einsum', 'einsum_path', 'Network', 'tensor_qualifiers_dtype']

import collections
import dataclasses
import logging
import warnings

import cupy as cp
import numpy as np

from cuquantum.bindings import cutensornet as cutn
from cuquantum.bindings.cutensornet import tensor_qualifiers_dtype
from . import configuration
from . import memory
from ._internal import einsum_parser
from ._internal import grad_torch
from ._internal import optimizer_ifc
from .._internal import formatters
from .._internal import tensor_wrapper
from .._internal import typemaps
from .._internal import utils
from .configuration import MemoryLimitExceeded


class InvalidNetworkState(Exception):
    pass


class Network:

    """
    Network(subscripts, *operands, qualifiers=None, options=None, stream=None)

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
        subscripts: The mode labels (subscripts) defining the Einstein summation expression as a comma-separated sequence of
            characters. Unicode characters are allowed in the expression thereby expanding the size of the tensor network that
            can be specified using the Einstein summation convention.
        operands: A sequence of tensors (ndarray-like objects). The currently supported types are :class:`numpy.ndarray`,
            :class:`cupy.ndarray`, and :class:`torch.Tensor`.
        qualifiers: Specify the tensor qualifiers as a :class:`numpy.ndarray` of :class:`~cuquantum.tensornet.tensor_qualifiers_dtype` objects
            of length equal to the number of operands.
        options: Specify options for the tensor network as a :class:`~cuquantum.NetworkOptions` object. Alternatively, a `dict`
            containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
            the value will be set to the default-constructed ``NetworkOptions`` object.
        stream: Provide the CUDA stream to use for network construction, which is needed for stream-ordered operations such as allocating memory. Acceptable inputs include ``cudaStream_t`` (as
            Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided, the
            current stream will be used.

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

        >>> tn = Network(expr, *operands)

        Find the best contraction order:

        >>> path, info = tn.contract_path({'samples': 500})

        Autotune the network:

        >>> tn.autotune(iterations=5)

        Perform the contraction. The result is of the same type and on the same device as the operands:

        >>> r1 = tn.contract()

        Reset operands to new values:

        >>> operands = [i*operand for i, operand in enumerate(operands, start=1)]
        >>> tn.reset_operands(*operands)

        Get the result of the new contraction:

        >>> r2 = tn.contract()
        >>> from math import factorial
        >>> np.allclose(r2, factorial(len(operands))*r1)
        True

        Finally, free network resources. If this call isn't made, it may hinder further operations (especially if the
        network is large) since the memory will be released only when the object goes out of scope. (*To avoid having
        to explicitly make this call, it is recommended to use the* :class:`Network` *object as a context manager*.)

        >>> tn.free()

        If the operands are on the GPU, they can also be updated using in-place operations. In this case, the call
        to :meth:`reset_operands` can be skipped -- subsequent :meth:`~Network.contract` calls will use the same
        operands (with updated contents). The following example illustrates this using CuPy operands and also demonstrates
        the usage of a :class:`Network` context (so as to skip calling :meth:`free`):

        >>> import cupy as cp
        >>> expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
        >>> shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]
        >>> operands = [cp.random.rand(*shape) for shape in shapes]
        >>>
        >>> with Network(expr, *operands) as tn:
        ...     path, info = tn.contract_path({'samples': 500})
        ...     tn.autotune(iterations=5)
        ...
        ...     # Perform the contraction
        ...     r1 = tn.contract()
        ...
        ...     # Update the operands in place
        ...     for i, operand in enumerate(operands, start=1):
        ...         operand *= i
        ...
        ...     # Perform the contraction with the updated operand values
        ...     r2 = tn.contract()
        ...
        ... # The resources used by the network are automatically released when the context ends.
        >>>
        >>> from math import factorial
        >>> cp.allclose(r2, factorial(len(operands))*r1)
        array(True)

        PyTorch CPU and GPU tensors can be passed as input operands in the same fashion.

        To compute the gradients of the network w.r.t. the input operands (NumPy/CuPy/PyTorch), the :meth:`gradients` method can
        be used. To enable the gradient computation, one should

          1. create the network with the ``qualifiers`` argument
          2. call the :meth:`contract` method prior to the :meth:`gradients` method
          3. seed the :meth:`gradients` method with the output gradient (see the docs for the requirements)

        Below is a minimal example:

        >>> from cuquantum.bindings import cutensornet as cutn
        >>> expr = "ijk,jkl,klm,lmn"
        >>> shapes = ((3, 4, 5), (4, 5, 3), (5, 3, 2), (3, 2, 6))
        >>> operands = [cp.random.rand(*shape) for shape in shapes]
        >>> qualifiers = np.zeros(len(shapes), dtype=cutn.tensor_qualifiers_dtype)
        >>> qualifiers[:]["requires_gradient"] = 1  # request gradients for all input tensors
        >>>
        >>> with Network(expr, *operands, qualifiers=qualifiers) as tn:
        ...     path, info = tn.contract_path()
        ...
        ...     # Perform the contraction
        ...     r = tn.contract()
        ...
        ...     # Perform the backprop
        ...     input_grads = tn.gradients(cp.ones_like(r))
        ...
        >>>

        For PyTorch CPU/GPU tensors with the ``requires_grad`` attribute set up, one does not need to pass the ``qualifiers``
        argument. Note that this :class:`Network` class and its methods are **not** PyTorch operators and do **not** add any
        node to PyTorch's autograd graph. For a native, differentiable PyTorch operator, use the :func:`cuquantum.contract`
        function.

        See :func:`contract` for more examples on specifying the Einstein summation expression as well
        as specifying options for the tensor network and the optimizer.
    """

    def __init__(self, *operands, qualifiers=None, options=None, stream=None):
        """
        __init__(subscripts, *operands, qualifiers=None, options=None, stream=None)
        """

        options = utils.check_or_create_options(configuration.NetworkOptions, options, "network options")
        self.options = options

        # Get cuTensorNet version (as seen at run-time).
        cutn_ver = cutn.get_version()
        cutn_major = cutn_ver // 10000
        cutn_minor = (cutn_ver % 10000) // 100
        cutn_patch = cutn_ver % 100

        # Logger.
        self.logger = options.logger if options.logger is not None else logging.getLogger()
        self.logger.info(f"cuTensorNet version = {cutn_major}.{cutn_minor}.{cutn_patch}")
        self.logger.info("Beginning network creation...")

        # Parse Einsum expression.
        self.operands, self.inputs, self.output, self.has_user_output, \
            self.size_dict, self.mode_map_user_to_ord, self.mode_map_ord_to_user, \
            self.is_interleaved, self.has_ellipses = einsum_parser.parse_einsum(*operands)

        # Infer the library package & device ID the operands belong to.
        self.package = utils.get_operands_package(self.operands)
        self.network_location = 'cuda'
        self.device_id = utils.get_network_device_id(self.operands)
        if self.device_id is None:
            self.package = self.operands[0].name
            if self.package == 'numpy':
                self.package = 'cupy'
            self.network_location = 'cpu'
            self.device_id = options.device_id

        # Allocate device memory (in stream context) if needed.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)

        # Copy operands to device if needed.
        if self.network_location == 'cpu':
            self.operands = tensor_wrapper.to(self.operands, self.device_id, stream_holder)

        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or self.network_location == 'cpu'
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = "This call is non-blocking and will return immediately after the operation is launched on the device."

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

        extents_in = self.extents_in = tuple(o.shape for o in self.operands)
        strides_in = self.strides_in = tuple(o.strides for o in self.operands)
        self.operands_data = utils.get_operands_data(self.operands)
        modes_in = tuple(tuple(m for m in _input) for _input in self.inputs)
        num_modes_in = tuple(len(m) for m in modes_in)
        self.qualifiers_in = utils.check_tensor_qualifiers(qualifiers, cutn.tensor_qualifiers_dtype, num_inputs)

        # For torch tensors, if qualifiers are explicitly passed, we ignore the tensor attrs.
        # Otherwise, we look up the tensor attrs and populate qualifiers.
        if self.package == 'torch' and isinstance(self.qualifiers_in, int):  # = 0
            self.qualifiers_in = np.zeros(num_inputs, dtype=cutn.tensor_qualifiers_dtype)
            self.logger.debug("Checking input tensors' requires_grad attribute")
            for i, t in enumerate(self.operands):
                self.qualifiers_in[i]['requires_gradient'] = self.operands[i].tensor.requires_grad
                self.qualifiers_in[i]['is_conjugate'] = self.operands[i].tensor.is_conj()

        # Check if gradient computation is required
        if isinstance(self.qualifiers_in, np.ndarray):
            self.require_grad = any(self.qualifiers_in['requires_gradient'])
        else:
            self.require_grad = False

        # Create the output in the context of the current stream to work around a performance issue with CuPy's memory pool.
        self.logger.debug("Beginning output tensor creation...")
        self.contraction, self.contraction_output_event, modes_out, extents_out, strides_out = utils.create_output_tensor(
                self.output_class, self.output, self.size_dict, self.device_id, stream_holder, self.data_type)
        self.logger.debug("The output tensor has been created.")

        # Create/set handle.
        if options.handle is not None:
            self.own_handle = False
            self.handle = options.handle
        else:
            self.own_handle = True
            with utils.device_ctx(self.device_id):
                self.handle = cutn.create()

        # Network definition.
        self.network = cutn.create_network_descriptor(self.handle, num_inputs,
                num_modes_in, extents_in, strides_in, modes_in, self.qualifiers_in,  # inputs
                num_modes_out, extents_out, strides_out, modes_out,  # output
                typemaps.NAME_TO_DATA_TYPE[self.data_type], self.compute_type)

        # Keep output extents for creating new tensors, if needed.
        self.extents_out = extents_out
        self.strides_out = strides_out

        # Path optimization attributes.
        self.optimizer_config_ptr, self.optimizer_info_ptr = None, None
        self.optimized = False

        # Workspace attributes.
        self.workspace_desc = cutn.create_workspace_descriptor(self.handle)
        self.workspace_scratch_ptr, self.workspace_scratch_size = None, None
        self.workspace_cache_ptr, self.workspace_cache_size = None, None
        self.workspace_h_scratch_ptr, self.workspace_h_scratch_size = None, None
        self.workspace_h_cache_ptr, self.workspace_h_cache_size = None, None
        self.workspace_scratch_allocated_here, self.workspace_cache_allocated_here = False, False

        # Contraction plan attributes.
        self.plan = None
        self.planned = False

        # Autotuning attributes.
        self.autotune_pref_ptr = None
        self.autotuned = False

        # Attributes to establish stream ordering.
        self.workspace_stream = None
        self.last_compute_event = None

        self.valid_state = True
        self.contracted = False

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

    def _check_valid_operands(self, *args, **kwargs):
        """
        """
        what = kwargs['what']
        if self.operands is None:
            raise RuntimeError(f"{what} cannot be performed if the input operands have been set to None. Use reset_operand() to set the desired input before using performing the {what.lower()}.")

    def _check_optimized(self, *args, **kwargs):
        """
        """
        what = kwargs['what']
        if not self.optimized:
            raise RuntimeError(f"{what} cannot be performed before contract_path() has been called.")

    def _check_planned(self, *args, **kwargs):
        """
        """
        what = kwargs['what']
        if not self.planned:
            raise RuntimeError(f"Internal Error: {what} cannot be performed before planning has been done.")

    def _check_contracted(self, *args, **kwargs):
        """
        This function ensures that gradients are not called before contract() or after the cache workspace has been released without calling contract() again.
        """
        what = kwargs['what']
        if not self.contracted:
            raise RuntimeError(f"{what} cannot be performed before contraction has been done. Note that a new contraction is required once the cache workspace has been released.")

    def _check_qualifiers(self, *args, **kwargs):
        """
        """
        what = kwargs['what']
        # cannot perform equality check (a == 0) if a is a numpy ndarray
        if isinstance(self.qualifiers_in, int):
            raise RuntimeError(f"{what} cannot be performed without creating the Network object with tensor qualifiers")

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
        self.workspace_scratch_ptr = None
        self.workspace_cache_ptr = None
        self.workspace_h_scratch_ptr = None
        self.workspace_h_cache_ptr = None
        self.logger.debug(f"[_free_workspace_memory] The scratch and cache workspace memory has been released.")

        return True

    def _reset_workspace_allocation_tracking(self):
        """
        Reset workspace allocation tracking attributes to False at the end of the methods where workspace memory is
        potentially allocated. This is necessary to prevent any exceptions raised before method entry from using
        stale tracking values.
        """
        self.workspace_scratch_allocated_here = False
        self.workspace_cache_allocated_here = False

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
        self.workspace_scratch_size = None
        self.workspace_cache_size = None
        self.workspace_h_scratch_size = None
        self.workspace_h_cache_size = None

        self._free_plan_resources()

        return True

    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_optimized, "Workspace memory allocation")
    @utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory_perhaps(self, stream_holder, kind):
        assert kind == "scratch" or kind == "cache", "Internal Error."
        assert getattr(self, f"workspace_{kind}_allocated_here") is False, "Internal Error."

        if getattr(self, f"workspace_{kind}_ptr") is not None and getattr(self, f"workspace_h_{kind}_ptr") is not None:
            return

        assert getattr(self, f"workspace_{kind}_size") is not None, "Internal Error."
        assert getattr(self, f"workspace_h_{kind}_size") is not None, "Internal Error."

        self.logger.debug(f"Allocating {kind} workspace for the tensor network computation...")

        # Allocate device workspace.
        device_size = getattr(self, f"workspace_{kind}_size")
        with utils.device_ctx(self.device_id), stream_holder.ctx:
            try:
                setattr(self, f"workspace_{kind}_ptr", self.allocator.memalloc(device_size))
                setattr(self, f"workspace_{kind}_allocated_here", True)
            except TypeError as e:
                message = "The method 'memalloc' in the allocator object must conform to the interface in the "\
                          "'BaseCUDAMemoryManager' protocol."
                raise TypeError(message) from e
        self.workspace_stream = stream_holder.obj
        # Allocate host workspace.
        # TODO: ideally we should use a memory manager, as we did for device memory, but...
        host_size = getattr(self, f"workspace_h_{kind}_size")
        setattr(self, f"workspace_h_{kind}_ptr", np.empty(host_size, dtype=np.int8))
        self.logger.debug("Finished allocating "
                          f"device memory of size {formatters.MemoryStr(device_size)} and "
                          f"host memory of size {formatters.MemoryStr(host_size)} "
                          f"for contraction in the context of stream {self.workspace_stream}.")

        # Set device workspace.
        device_ptr = utils.get_ptr_from_memory_pointer(getattr(self, f"workspace_{kind}_ptr"))
        cutn.workspace_set_memory(self.handle, self.workspace_desc, cutn.Memspace.DEVICE,
                                  cutn.WorkspaceKind.SCRATCH if kind == "scratch" else cutn.WorkspaceKind.CACHE,
                                  device_ptr, device_size)
        # Set host workspace.
        # TODO: ideally we should be manipulating a MemoryPointer object here, but ...
        host_ptr = getattr(self, f"workspace_h_{kind}_ptr").ctypes.data
        cutn.workspace_set_memory(self.handle, self.workspace_desc, cutn.Memspace.HOST,
                                  cutn.WorkspaceKind.SCRATCH if kind == "scratch" else cutn.WorkspaceKind.CACHE,
                                  # WAR: empty numpy arrays still have nonzero ptr addresses
                                  host_ptr if host_size > 0 else 0, host_size)
        self.logger.debug(f"The {kind} workspace memory (device pointer = {device_ptr}, "
                          f"host pointer = {host_ptr}) has been set in the workspace descriptor.")

    def _release_workspace_memory_perhaps(self, kind, release_workspace):
        """
        Free scratch or cache workspace memory if 'release_workspace' is True.
        """
        assert kind == "scratch" or kind == "cache", "Internal Error."
        assert isinstance(release_workspace, bool), "Internal Error."

        if not release_workspace:
            return

        # Establish ordering wrt the computation before releasing cache or scratch workspace.
        if self.last_compute_event is not None:
            self.workspace_stream.wait_event(self.last_compute_event)
            self.logger.debug(f"Established ordering with respect to the computation before releasing the {kind} workspace.")

        if kind == "cache":
            cutn.workspace_purge_cache(self.handle, self.workspace_desc, cutn.Memspace.DEVICE)
            cutn.workspace_purge_cache(self.handle, self.workspace_desc, cutn.Memspace.HOST)
            # Set contracted to False to require a new contraction if the cache workspace is released.
            self.contracted = False
        setattr(self, f"workspace_{kind}_ptr", None)
        setattr(self, f"workspace_h_{kind}_ptr", None)
        self.logger.debug(f"[_release_workspace_memory_perhaps] The {kind} workspace memory has been released.")

    def _release_scratch_memory_perhaps(self, exception=None):
        """
        Free scratch workspace memory if it was allocated in this call (self.workspace_scratch_allocated_here == True) when an exception occurs.
        """
        release_workspace = self.workspace_scratch_allocated_here
        self.logger.debug(f"[_release_scratch_memory_perhaps] The release_workspace flag is set to {release_workspace} based upon the value of 'workspace_scratch_allocated_here'.")
        self._release_workspace_memory_perhaps("scratch", release_workspace)
        return True

    def _release_cache_memory_perhaps(self, exception=None):
        """
        Free cache workspace memory if it was allocated in this call (self.workspace_cache_allocated_here == True) when an exception occurs.
        """
        release_workspace = self.workspace_cache_allocated_here
        self.logger.debug(f"[_release_cache_memory_perhaps] The release_workspace flag is set to {release_workspace} based upon the value of 'workspace_cache_allocated_here'.")
        self._release_workspace_memory_perhaps("cache", release_workspace)
        return True

    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_optimized, "Workspace size calculation")
    def _calculate_workspace_size(self):
        """
        Allocate workspace for cutensornet.
        """

        # Release workspace already allocated, if any, because the new requirements are likely different.
        self.workspace_scratch_ptr = None
        self.workspace_cache_ptr = None
        self.workspace_h_scratch_ptr = None
        self.workspace_h_cache_ptr = None

        cutn.workspace_compute_contraction_sizes(self.handle, self.network, self.optimizer_info_ptr, self.workspace_desc)

        # Deal with device workspaces.
        min_scratch_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.MIN, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
        max_scratch_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.MAX, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
        rec_scratch_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
        min_cache_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.MIN, cutn.Memspace.DEVICE, cutn.WorkspaceKind.CACHE)
        max_cache_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.MAX, cutn.Memspace.DEVICE, cutn.WorkspaceKind.CACHE)

        min_workspace_size = min_scratch_size + self.require_grad * min_cache_size
        if self.memory_limit < min_workspace_size:
            raise MemoryLimitExceeded(self.memory_limit, min_workspace_size, self.device_id)

        if min_cache_size > 0:
            if self.require_grad:
                self.workspace_cache_size = min_cache_size
                self.workspace_scratch_size = min(self.memory_limit - self.workspace_cache_size, max_scratch_size)
            else:
                self.workspace_scratch_size = rec_scratch_size if rec_scratch_size < self.memory_limit else min_scratch_size
                self.workspace_cache_size = min(self.memory_limit - self.workspace_scratch_size, min_cache_size)
        else:
            self.workspace_cache_size = 0
            self.workspace_scratch_size = min(self.memory_limit, max_scratch_size)

        self.logger.info(f"The workspace size requirements range from {formatters.MemoryStr(min_scratch_size + min_cache_size)} to "\
                         f"{formatters.MemoryStr(max_scratch_size + max_cache_size)}.")
        self.logger.info(f"The scratch workspace size has been set to {formatters.MemoryStr(self.workspace_scratch_size)}.")
        self.logger.info(f"The cache workspace size has been set to {formatters.MemoryStr(self.workspace_cache_size)}.")

        # Set workspace size to enable contraction planning. The device pointer will be set later during allocation.
        cutn.workspace_set_memory(
            self.handle, self.workspace_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, 0, self.workspace_scratch_size)
        cutn.workspace_set_memory(
            self.handle, self.workspace_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.CACHE, 0, self.workspace_cache_size)

        # Deal with device workspaces. For now we don't care how much host memory is used.
        self.workspace_h_scratch_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH)
        self.workspace_h_cache_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.HOST, cutn.WorkspaceKind.CACHE)

        # Set workspace size to enable contraction planning. The host pointer will be set later during allocation.
        cutn.workspace_set_memory(
            self.handle, self.workspace_desc, cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH, 0, self.workspace_h_scratch_size)
        cutn.workspace_set_memory(
            self.handle, self.workspace_desc, cutn.Memspace.HOST, cutn.WorkspaceKind.CACHE, 0, self.workspace_h_cache_size)

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

        enum = ConfEnum.COST_FUNCTION_OBJECTIVE
        self._set_opt_config_option('cost_function', enum, optimize.cost_function)

        enum = ConfEnum.SMART_OPTION
        self._set_opt_config_option('smart', enum, optimize.smart)

    @utils.precondition(_check_valid_network)
    @utils.atomic(_free_path_resources, method=True)
    def contract_path(self, optimize=None, **kwargs):
        """
        contract_path(optimize=None)

        Compute the best contraction path together with any slicing that is needed to ensure that the contraction can be
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

        binary_contraction_optimization = len(self.inputs) == 2 and optimize is None
        optimize = utils.check_or_create_options(configuration.OptimizerOptions, optimize, "path optimizer options")

        internal_options = dict()
        internal_options['create_plan'] = utils.Value(True, validator=lambda v: isinstance(v, bool))
        utils.check_and_set_options(internal_options, kwargs)

        if self.optimizer_config_ptr is None:
            self.optimizer_config_ptr = cutn.create_contraction_optimizer_config(self.handle)
        if self.optimizer_info_ptr is None:
            self.optimizer_info_ptr = cutn.create_contraction_optimizer_info(self.handle, self.network)

        opt_info_ifc = optimizer_ifc.OptimizerInfoInterface(self)

        # Special case worth optimizing (when the "optimize" option is not specified), as it's an extremely common use case with a trivial path.
        if binary_contraction_optimization:
            optimize.path = [(0, 1)]

        # Compute path (or set provided path).
        if isinstance(optimize.path, configuration.PathFinderOptions):
            # Set optimizer options.
            self._set_optimizer_options(optimize)
            # Find "optimal" path.
            self.logger.info("Finding optimal path as well as sliced modes...")
            try:
                cutn.contraction_optimize(
                    self.handle, self.network, self.optimizer_config_ptr, self.memory_limit, self.optimizer_info_ptr)
            except cutn.cuTensorNetError as e:
                if 'INTERRUPTED' in str(e):
                    raise KeyboardInterrupt from e
                raise
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
        aux_modes = opt_info_ifc.intermediate_modes
        opt_info = configuration.OptimizerInfo(
            largest_intermediate, opt_cost, path, slices, self.num_slices, aux_modes)

        # If we are not logging, avoid the overhead of creating the string representation of opt_info.
        if self.logger.handlers:
           self.logger.info(f"{opt_info}")

        self.optimized = True

        if internal_options['create_plan']:
            # Calculate workspace size required.
            self._calculate_workspace_size()

            # Create plan.
            self._create_plan()
            self.planned = True
        else:
            self.planned = False

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
    @utils.precondition(_check_planned, "Autotuning")
    @utils.precondition(_check_valid_operands, "Autotuning")
    @utils.atomic(_release_cache_memory_perhaps, method=True)
    @utils.atomic(_release_scratch_memory_perhaps, method=True)
    def autotune(self, *, iterations=3, stream=None, release_workspace=False):
        """Autotune the network to reduce the contraction cost.

        This is an optional step that is recommended if the :class:`Network` object is used to perform multiple contractions.

        Args:
            iterations: The number of iterations for autotuning. See `CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS`.
            stream: Provide the CUDA stream to use for the autotuning operation. Acceptable inputs include ``cudaStream_t``
                (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
                the current stream will be used.
            release_workspace: A value of `True` specifies that the :class:`Network` object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the :class:`Network` object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`autotune`,
                :meth:`contract`, or :meth:`gradients`, but incurs a small overhead due to obtaining and releasing workspace
                memory from and to the package memory pool on every call. The default is `False`.
        """

        message = utils.check_autotune_params(iterations)
        self.logger.info(message)
        if self.autotune_pref_ptr is None:
            self.autotune_pref_ptr = cutn.create_contraction_autotune_preference(self.handle)

        AutoEnum = cutn.ContractionAutotunePreferenceAttribute
        options = {'iterations': (AutoEnum.MAX_ITERATIONS, iterations)}
        self._set_autotune_options(options)

        # Allocate device memory (in stream context) if needed.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self._allocate_workspace_memory_perhaps(stream_holder, "scratch")
        self._allocate_workspace_memory_perhaps(stream_holder, "cache")

        # Check if we still hold an output tensor; if not, create a new one.
        if self.contraction is None:
            self.logger.debug("Beginning output (empty) tensor creation...")
            self.contraction = utils.create_empty_tensor(self.output_class, self.extents_out, self.data_type, self.device_id, stream_holder)
            self.logger.debug("The output (empty) tensor has been created.")
        elif self.contraction_output_event is not None:
            stream_holder.obj.wait_event(self.contraction_output_event)
            self.contraction_output_event = None
            self.logger.debug("Established ordering with output tensor creation event.")

        timing =  bool(self.logger and self.logger.handlers)
        self.logger.info(f"Starting autotuning...")
        self.logger.info(f"{self.call_prologue}")
        with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
            cutn.contraction_autotune(
                self.handle, self.plan, self.operands_data, self.contraction.data_ptr,
                self.workspace_desc, self.autotune_pref_ptr, stream_holder.ptr)

        if elapsed.data is not None:
            self.logger.info(f"The autotuning took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free scratch and cache workspace based on user request.
        self._release_workspace_memory_perhaps("scratch", release_workspace=release_workspace)
        self._release_workspace_memory_perhaps("cache", release_workspace=release_workspace)

        self._reset_workspace_allocation_tracking()
        self.autotuned = True

    @utils.precondition(_check_valid_network)
    def reset_operands(self, *operands, stream=None):
        """Reset the operands held by this :class:`Network` instance.

        This method has two use cases: (1) it can be used to provide new operands for execution when the
           original operands are on the CPU, or (2) it can be used to release the internal reference to the previous operands and make their memory available for
           other use by passing ``None`` for the ``operands`` argument. In this case, this method must be called again to provide the desired operands before another
           call to execution APIs like :meth:`autotune`, :meth:`contract`, or :meth:`gradients`.

        This method is not needed when the operands reside on the GPU and in-place operations are used to update the operand values.

        This method will perform various checks on the new operands to make sure:

            - The shapes, strides, datatypes match those of the old ones.
            - The packages that the operands belong to match those of the old ones.
            - If input tensors are on GPU, the library package and device must match.

        Args:
            operands: See :class:`Network`'s documentation.
            stream: Provide the CUDA stream to use for resetting operands (this is used to copy the operands to the GPU if they are provided on the CPU). Acceptable inputs include ``cudaStream_t``
                (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
                the current stream will be used.
        """

        # Note the we don't need to invalidate cache workspace when setting operands to None, since this will be done when the operands are reset to new ones.
        if len(operands) == 1 and operands[0] is None:
            self.operands_data = None
            self.operands = None
            self.logger.info(f"The operands have been reset to None.")
            return

        if len(operands) != len(self.inputs):
            message = f"Mismatch in the number of operands ({len(operands)} provided, need {len(self.inputs)})."
            raise ValueError(message)

        # Future operations on the workspace stream should be ordered after the computation.
        # Also, we should ensure self.operands is overwritten only after work using them is done.
        if self.last_compute_event is not None:
            self.workspace_stream.wait_event(self.last_compute_event)

        self.logger.info("Resetting operands...")
        # First wrap operands.
        operands = tensor_wrapper.wrap_operands(operands)

        utils.check_attributes_match([self.data_type] * len(self.inputs), [o.dtype for o in operands], "data type")
        utils.check_attributes_match(self.extents_in, [o.shape for o in operands], 'shape')

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)

        package = utils.get_operands_package(operands)
        package = 'cupy' if package == 'numpy' else package   # Handle the NumPy <=> CuPy asymmetry.
        if self.package != package:
            message = f"Library package mismatch: '{self.package}' => '{package}'"
            raise TypeError(message)

        device_id = utils.get_network_device_id(operands)
        if device_id is None:
            if self.operands is None:
                # Copy operands across memory spaces (CPU to GPU).
                self.operands = tensor_wrapper.to(operands, self.device_id, stream_holder)
                # Update the device pointers after copying operands to the GPU.
                self.operands_data = utils.get_operands_data(self.operands)
            else:
                # In-place copy to existing device pointers because the new operands are on the CPU.
                tensor_wrapper.copy_(operands, self.operands, stream_holder)
        else:
            utils.check_attributes_match(self.strides_in, [o.strides for o in operands], 'strides')
            if self.device_id != device_id:
                raise ValueError(f"The new operands must be on the same device ({device_id}) as the original operands "
                                 f"({self.device_id}).")

            # Finally, replace the original data pointers by the new ones.
            self.operands_data = utils.get_operands_data(operands)
            self.operands = operands
        self.logger.info("The operands have been reset.")

        self.contracted = False
        if not self.require_grad:
            return

        # The cache workspace is invalidated.
        cutn.workspace_purge_cache(self.handle, self.workspace_desc, cutn.Memspace.DEVICE)
        cutn.workspace_purge_cache(self.handle, self.workspace_desc, cutn.Memspace.HOST)

    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_valid_operands, "Contraction")
    @utils.precondition(_check_optimized, "Contraction")
    @utils.precondition(_check_planned, "Contraction")
    @utils.atomic(_release_cache_memory_perhaps, method=True)
    @utils.atomic(_release_scratch_memory_perhaps, method=True)
    def contract(self, *, slices=None, stream=None, release_workspace=False):
        """Contract the network and return the result.

        Args:
            slices: Specify the slices to be contracted as Python :class:`range` for contiguous slice IDs or as a Python sequence
                object for arbitrary slice IDs. If not specified, all slices will be contracted.
            stream: Provide the CUDA stream to use for the contraction operation. Acceptable inputs include ``cudaStream_t``
                (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
                the current stream will be used.
            release_workspace: A value of `True` specifies that the :class:`Network` object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the :class:`Network` object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`autotune`,
                :meth:`contract`, or :meth:`gradients`, but incurs a small overhead due to obtaining and releasing workspace
                memory from and to the package memory pool on every call. The default is `False`.

        Returns:
            The result is of the same type and on the same device as the operands.
        """
        # Allocate device memory (in stream context) if needed.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self._allocate_workspace_memory_perhaps(stream_holder, "scratch")
        self._allocate_workspace_memory_perhaps(stream_holder, "cache")

        # Check if we still hold an output tensor; if not, create a new one.
        if self.contraction is None:
            self.logger.debug("Beginning output (empty) tensor creation...")
            self.contraction = utils.create_empty_tensor(
                self.output_class, self.extents_out, self.data_type, self.device_id, stream_holder)
            self.logger.debug("The output (empty) tensor has been created.")
        elif self.contraction_output_event is not None:
            stream_holder.obj.wait_event(self.contraction_output_event)
            self.contraction_output_event = None
            self.logger.debug("Established ordering with output tensor creation event.")

        # Create a slice group for contraction.
        slice_group = None
        if slices is None:
           slice_group = 0
           self.logger.info(f"All the available slices ({self.num_slices}) will be contracted.")
        elif isinstance(slices, range):
           slice_group = cutn.create_slice_group_from_id_range(self.handle, slices.start, slices.stop, slices.step)
           self.logger.info(f"A slice group has been created with start={slices.start}, stop={slices.stop}, and step={slices.step}.")
        elif isinstance(slices, collections.abc.Sequence):
           slice_group = cutn.create_slice_group_from_ids(self.handle, slices, len(slices))
           self.logger.info(f"A slice group has been created from the specified sequence: {formatters.array2string([str(s) for s in slices])}")
        else:
            message = f"The provided 'slices' must be a range object or a sequence object. The object type is {type(slices)}."
            raise TypeError(message)

        timing =  bool(self.logger and self.logger.handlers)
        self.logger.info("Starting network contraction...")
        self.logger.info(f"{self.call_prologue}")
        with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
            cutn.contract_slices(
                self.handle, self.plan, self.operands_data, self.contraction.data_ptr, False,
                self.workspace_desc, slice_group, stream_holder.ptr)

        if elapsed.data is not None:
            self.logger.info(f"The contraction took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free scratch workspace based on user request.
        self._release_workspace_memory_perhaps("scratch", release_workspace=release_workspace)

        # Destroy slice group, if created.
        if slice_group != 0:
           cutn.destroy_slice_group(slice_group)
           self.logger.debug(f"Slice group ({slice_group}) has been destroyed.")

        if self.network_location == 'cpu':
            out = self.contraction.to('cpu', stream_holder=stream_holder)
        else:
            out = self.contraction.tensor

        self.contraction = None    # We cannot overwrite what we've already handed to users.
        self._reset_workspace_allocation_tracking()
        self.contracted = True

        return out

    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_optimized, "Gradient")
    @utils.precondition(_check_planned, "Gradient")
    @utils.precondition(_check_contracted, "Gradient")
    @utils.precondition(_check_qualifiers, "Gradient")
    @utils.precondition(_check_valid_operands, "Gradient")
    @utils.atomic(_release_scratch_memory_perhaps, method=True)
    def gradients(self, output_gradient, *, stream=None, release_workspace=False):
        """Compute the gradients of the network (w.r.t. the input operands whose gradients are required).

        Before calling this method, a full contraction must have been performed (by calling :meth:`contract`), otherwise an
        error is raised.

        Args:
            output_gradient: A tensor of the same package (NumPy/CuPy/PyTorch), shape, dtype, strides, and location (CPU/GPU)
                as the contraction output (as returned by :meth:`contract`), which in turn shares the same properties with the
                input operands. In a chain-rule setting, ``output_gradient`` is the gradient w.r.t. the output tensor.
            stream: Provide the CUDA stream to use for the gradient computation. Acceptable inputs include ``cudaStream_t``
                (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
                the current stream will be used.
            release_workspace: A value of `True` specifies that the :class:`Network` object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the :class:`Network` object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`autotune`,
                :meth:`contract`, or :meth:`gradients`, but incurs a small overhead due to obtaining and releasing workspace
                memory from and to the package memory pool on every call. The default is `False`.

        Returns:
            A sequence of gradient tensors. The result is of the same length and type and on the same device as the input operands.
            For the gradient components that are not requested, ``None`` is returned.

        .. note:: For PyTorch operands, calling this method is **not** tracked by the autograd graph.

        .. warning:: This API is experimental and subject to future changes.
        """
        warnings.warn("Network.gradients() is an experimental API and subject to future changes",
                      stacklevel=2)

        # Allocate scratch memory (in stream context) if needed.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self._allocate_workspace_memory_perhaps(stream_holder, "scratch")

        # At this point, both scratch and cache workspaces are allocated/populated.
        assert self.workspace_scratch_ptr is not None and self.workspace_cache_ptr is not None, "Internal error."
        assert self.workspace_h_scratch_ptr is not None and self.workspace_h_cache_ptr is not None, "Internal error."

        # Future operations on the workspace stream should be ordered after the computation.
        if self.last_compute_event is not None:
            self.workspace_stream.wait_event(self.last_compute_event)
            stream_holder.obj.wait_event(self.last_compute_event)

        # Wrap output_gradient
        output_grad = tensor_wrapper.wrap_operand(output_gradient)
        if output_grad.device_id != self.device_id:
            output_grad = tensor_wrapper.to([output_grad], self.device_id, stream_holder)[0]
        if output_grad.shape != self.extents_out:
            raise ValueError(f"output_gradient shape incorrect (given {output_grad.shape}, expected {self.extents_out})")
        if output_grad.dtype != self.data_type:
            raise ValueError(f"output_gradient dtype incorrect (given {output_grad.dtype}, expected {self.data_type}")
        if output_grad.strides != self.strides_out:
            # output_gradient could be a view, but we need a full buffer for now
            if any(s == 0 for s in output_grad.strides):
                buf = utils.create_empty_tensor(
                    self.output_class, self.extents_out, self.data_type, self.device_id, stream_holder, strides=self.strides_out)
                buf.copy_(output_grad.tensor, stream_holder=stream_holder)
                output_grad = buf
            else:
                raise ValueError(f"output_gradient strides incorrect (given {output_grad.strides}, expected {self.strides_out}")

        # Allocate grad tensors, as needed
        input_grads = []
        for i, extents, strides, requires_grad in zip(
                range(len(self.inputs)), self.extents_in, self.strides_in, self.qualifiers_in['requires_gradient']):
            if requires_grad:
                input_grads.append(
                    utils.create_empty_tensor(self.output_class, extents, self.data_type, self.device_id, stream_holder, strides=strides)
                )
            else:
                input_grads.append(None)
        input_grads_data = utils.get_operands_data(input_grads)

        timing = bool(self.logger and self.logger.handlers)
        self.logger.info("Starting gradient computation...")
        self.logger.info(f"{self.call_prologue}")
        with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
            cutn.compute_gradients_backward(
                self.handle, self.plan, self.operands_data, output_grad.data_ptr,
                input_grads_data, False, self.workspace_desc, stream_holder.ptr)

        if elapsed.data is not None:
            self.logger.info(f"The backprop took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free scratch and cache workspace based on user request.
        self._release_workspace_memory_perhaps("scratch", release_workspace=release_workspace)
        self._release_workspace_memory_perhaps("cache", release_workspace=release_workspace)

        if self.network_location == 'cpu':
            op = lambda t: t.to('cpu', stream_holder=stream_holder) if t is not None else None
        else:
            op = lambda t: t.tensor if t is not None else None

        self._reset_workspace_allocation_tracking()
        return tuple(map(op, input_grads))

    def free(self):
        """Free network resources.

        It is recommended that the :class:`Network` object be used within a context, but if it is not possible then this
        method must be called explicitly to ensure that the network resources are properly cleaned up.
        """

        if not self.valid_state:
            return

        try:
            # Future operations on the workspace stream should be ordered after the computation.
            # The last_compute_event is created by the CUDA execution context (utils.cuda_call_ctx) in an execution method
            # like autotune, contract, or gradients and is used to ensure that workspace memory (scratch or cache) is made
            #  available for another operation only after the operation that uses it is complete.
            if self.last_compute_event is not None:
                self.workspace_stream.wait_event(self.last_compute_event)

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


def contract(*operands, qualifiers=None, options=None, optimize=None, stream=None, return_info=False):
    r"""
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
        qualifiers: Specify the tensor qualifiers as a :class:`numpy.ndarray` of :class:`~cuquantum.tensor_qualifiers_dtype` objects
            of length equal to the number of operands.
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

            from cuquantum.bindings import cutensornet as cutn
            from cuquantum import contract, NetworkOptions

            handle = cutn.create()
            network_opts = NetworkOptions(handle=handle, ...)
            out = contract(..., options=network_opts, ...)
            # ... the same handle can be reused for further calls ...
            # when it's done, remember to destroy the handle
            cutn.destroy(handle)

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

        For PyTorch operands, this function **works like a native PyTorch operator** out of box that will be tracked by the
        autograd graph so as to enable backpropagation. The result ``r`` is a PyTorch tensor on the same device (``dev``) as
        the operands. To enable gradient computation, just set the target operands' ``requires_grad`` attribute to ``True``,
        as usual. If ``stream`` is explicitly passed, the user must establish the stream ordering following the requirements
        outlined in PyTorch's `CUDA Semantics <https://pytorch.org/docs/stable/notes/cuda.html>`_.

    .. doctest::
        :skipif: torch is None

        >>> import torch
        >>> dev = 0
        >>> a = torch.arange(6., device=f'cuda:{dev}').reshape(3, 2)
        >>> a.requires_grad_(True)
        >>> b = torch.arange(6., device=f'cuda:{dev}').reshape(2, 3)
        >>> b.requires_grad_(True)
        >>> r = contract('ij,jk', a, b)
        >>> r.backward(torch.ones_like(r))  # gradient w.r.t self is 1
        >>> a.grad
        tensor([[ 3., 12.],
                [ 3., 12.],
                [ 3., 12.]], device='cuda:0')
        >>> b.grad
        tensor([[6., 6., 6.],
                [9., 9., 9.]], device='cuda:0')
    """

    # Create network.
    network = Network(*operands, qualifiers=qualifiers, options=options, stream=stream)

    # For PyTorch tensors, we ensure contract() is differentiable.
    if network.package == "torch":
        return grad_torch._TorchContract.apply(network, optimize, stream, return_info, *operands)

    with network:

        # Compute path.
        opt_info = network.contract_path(optimize=optimize)

        # Skip autotuning since the network is contracted only once.

        # Contraction.
        output = network.contract(stream=stream)

    if return_info:
        return output, opt_info

    return output


def contract_path(*operands, qualifiers=None, options=None, optimize=None):
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
        qualifiers: Specify the tensor qualifiers as a :class:`numpy.ndarray` of :class:`~cuquantum.tensor_qualifiers_dtype` objects
            of length equal to the number of operands.
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

            from cuquantum.bindings import cutensornet as cutn
            from cuquantum import contract, NetworkOptions

            handle = cutn.create()
            network_opts = NetworkOptions(handle=handle, ...)
            path, info = contract_path(..., options=network_opts, ...)
            # ... the same handle can be reused for further calls ...
            # when it's done, remember to destroy the handle
            cutn.destroy(handle)
    
    .. note::
        Users may use this API to compute path without device memory allocation. 
        One way to achieve this is via dummy :class:`cupy.ndarray` operands, e.g, 
        `path finding without dummy arrays <https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/tensornet/contraction/coarse/example24.py>`_. 
    """

    # Create network.
    with Network(*operands, qualifiers=qualifiers, options=options) as network:

        # Compute path.
        path, opt_info = network.contract_path(optimize=optimize, create_plan=False)

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

    .. note:: For PyTorch operands, calling this method is **not** tracked by the autograd graph.
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
                path = [(0, 1)] * (len(network.inputs) - 1)
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
        path, opt_info = network.contract_path(create_plan=False)

    return ['einsum_path', *path], str(opt_info)
