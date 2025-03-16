# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = ['NetworkState']

import logging

import cupy as cp
import numpy as np

from cuquantum.bindings import cutensornet as cutn

from .configuration import MPSConfig, TNConfig
from .network_operator import NetworkOperator
from ._internal.network_state_utils import (
    EXACT_MPS_EXTENT_LIMIT, 
    STATE_DEFAULT_DTYPE, 
    check_dtype_supported, 
    get_mps_key, 
    state_operands_wrapper, 
    state_result_wrapper, 
    state_labels_wrapper,
)
from .. import memory
from ..tensor_network import Network
from ..circuit_converter import CircuitToEinsum
from ..configuration import NetworkOptions
from ..._internal import formatters, tensor_wrapper, utils
from .._internal.circuit_converter_utils import EMPTY_DICT
from ..._internal.typemaps import NAME_TO_DATA_TYPE


class NetworkState:
    """
    Create an empty tensor network state.

    Args:
        state_mode_extents : A sequence of integers specifying the extents for all state modes.
        dtype : A string specifying the datatype for the network state, currently supports the following data types:
            
            - ``'float32'``
            - ``'float64'``
            - ``'complex64'``
            - ``'complex128'`` (default)
        
        config : The simulation configuration for the state. It can be:

            - A :class:`TNConfig` object for contraction based tensor network simulation (default).
            - A :class:`MPSConfig` object for MPS based tensor network simulation.
            - A `dict` containing the parameters for the :class:`TNConfig` or :class:`MPSConfig` constructor.

        state_labels : Optional, a sequence of different labels corresponding to each state dimension. 
            If provided, users have the option to provide a sequence of these labels as the input arguments for the following APIs including 
            :meth:`apply_tensor_operator`, :meth:`apply_mpo`, :meth:`compute_batched_amplitudes`, :meth:`compute_reduced_density_matrix` and 
            :meth:`compute_sampling`. See the docstring for each of these APIs for more details.
        options : Specify options for the state computation as a :class:`~cuquantum.NetworkOptions` object. 
            Alternatively, a `dict` containing the parameters for the ``NetworkOptions`` constructor can also be provided. 
            If not specified, the value will be set to the default-constructed ``NetworkOptions`` object.
    
    Notes:
        - Currently :class:`NetworkState` only supports pure state representation.
        - If users wish to use a different device than the default current device, it must be explicitly specified via :attr:`NetworkOptions.device_id`.
        - For MPS simulation, currently only open boundary condition is supported.
    
    Examples:

        In this example, we aim to directly perform simulation on a quantum circuit instance using tensor network contraction method.

        >>> from cuquantum.cutensornet.experimental import NetworkState, TNConfig
        >>> import cirq

        Define a random cirq.Circuit, note that qiskit.QuantumCircuit is supported as well using the same API call

        >>> n_qubits = 4
        >>> n_moments = 4
        >>> op_density = 0.9
        >>> circuit = cirq.testing.random_circuit(n_qubits, n_moments, op_density, random_state=2024)

        Use tensor network contraction as the simulation method

        >>> config = TNConfig(num_hyper_samples=4)

        Create the network state object via :meth:`from_circuit` method:

        >>> state = NetworkState.from_circuit(circuit, dtype='complex128', backend='cupy', config=config)

        Compute the amplitude for bitstring 0000

        >>> amplitude = state.compute_amplitude('0000')

        Compute the expectation for a series of Pauli strings with coefficients

        >>> pauli_strings = {'IXIX': 0.4, 'IZIZ': 0.1}
        >>> expec = state.compute_expectation(pauli_strings)

        Compute the reduced density matrix for the first two qubits. 
        Since the backend is specified to ``cupy``, the returned rdm operand will be cupy.ndarray.

        >>> where = (0, 1)
        >>> rdm = state.compute_reduced_density_matrix(where)
        >>> print(f"RDM shape for {where}: {rdm.shape}")
        RDM shape for (0, 1): (2, 2, 2, 2)

        Draw 1000 samples from the state

        >>> shots = 1000
        >>> samples = state.compute_sampling(shots)

        Finally, free network state resources. If this call isn't made, it may hinder further operations (especially if the
        network state is large) since the memory will be released only when the object goes out of scope. (*To avoid having
        to explicitly make this call, it is recommended to use the* :class:`NetworkState` *object as a context manager*.)

        >>> state.free()

        In addition to initializing the state from a circuit instance, users can construct the state by sequentially applying tensor operators with :meth:`apply_tensor_operator` 
        and matrix product operators (MPOs) with :meth:`apply_mpo` or :meth:`apply_network_operator`. 
        Alternatively, simulations can leverage exact or approximate matrix product state (MPS) method by specifing ``options`` as an :class:`MPSConfig` instance.
        More detailed examples can be found in our `NetworkState examples directory <https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/tensornet/experimental/network_state/>`_.
    """
    def __init__(self, state_mode_extents, *, dtype=STATE_DEFAULT_DTYPE, config=None, state_labels=None, options=None):

        options = utils.check_or_create_options(NetworkOptions, options, "network options")
        self.options = options
        self.device_id = self.options.device_id
        if state_labels is not None:
            if any(isinstance(element, int) for element in state_labels):
                raise ValueError("ints are currently not supported in state_labels to avoid potential conflicting usage")
        self.state_labels = list(state_labels) if state_labels is not None else state_labels

        # Get cuTensorNet version (as seen at run-time).
        cutn_ver = cutn.get_version()
        cutn_major = cutn_ver // 10000
        cutn_minor = (cutn_ver % 10000) // 100
        cutn_patch = cutn_ver % 100

        # Logger.
        self.logger = options.logger if options.logger is not None else logging.getLogger()
        self.logger.info(f"cuTensorNet version = {cutn_major}.{cutn_minor}.{cutn_patch}")
        self.logger.info("Beginning network state creation...")
        
        # Set memory limit.
        self.device = cp.cuda.Device(self.device_id)
        self.memory_limit = utils.get_memory_limit(self.options.memory_limit, self.device)
        self.logger.info(f"The memory limit is {formatters.MemoryStr(self.memory_limit)}.")

        self.state_mode_extents = list(state_mode_extents)
        self.n = len(self.state_mode_extents)

        check_dtype_supported(dtype)
        self.dtype = dtype
        self.cuda_dtype = NAME_TO_DATA_TYPE[dtype]

        # internal setup to be determined later
        self.allocator = options.allocator # This may be None at this time
        self.output_location = None
        self.internal_package = None
        self.intermediate_class = None
        self.output_class = None
        self.backend = None
        self.backend_setup = False

        for config_class in (TNConfig, MPSConfig):
            try:
                self.config = utils.check_or_create_options(config_class, config, config_class.__name__)
            except TypeError:
                continue
            else:
                break
        else:
            raise ValueError("method must be either a TNConfig/MPSConfig object or a dict that can be used to construct TNConfig/MPSConfig")
        
        if self.n == 1 and isinstance(self.config, MPSConfig):
            raise ValueError(f"For system with one physical dimension, please switch to tensor network simulation method via TNConfig")
        
        self.operands = {}
        self.owned_network_operators = {}
        self.non_owned_network_operators = {}

        # Create/set handle.
        if options.handle is not None:
            self.own_handle = False
            self.handle = options.handle
        else:
            self.own_handle = True
            with utils.device_ctx(self.device_id):
                self.handle = cutn.create()
        
        # Create the state object
        self.state = cutn.create_state(self.handle, 
            cutn.StatePurity.PURE, self.n, state_mode_extents, self.cuda_dtype)
        
        # Workspace attributes.
        self.workspace_desc = cutn.create_workspace_descriptor(self.handle)
        self.workspace_scratch_ptr, self.workspace_scratch_size = None, None
        self.workspace_cache_ptr, self.workspace_cache_size = None, None
        self.workspace_h_scratch_ptr, self.workspace_h_scratch_size = None, None
        self.workspace_h_cache_ptr, self.workspace_h_cache_size = None, None
        self.workspace_scratch_allocated_here, self.workspace_cache_allocated_here = False, False
        self.workspace_sizes_requirements = {} # a dictionary to cache workspace requirements
        # Attributes to establish stream ordering.
        self.blocking = None # This will be set when operators are applied
        self.workspace_stream = None
        self.last_compute_event = None

        # markers for MPS
        if isinstance(self.config, MPSConfig):
            self.property_prepare_reusable = self.config._is_fixed_extent_truncation()
        else:
            self.property_prepare_reusable = True
        self.state_configured = False
        self.target_state_set = False
        self.state_prepared = False
        self.state_computed = False
        self.initial_state = []
        self.valid_state = True

        self.cached_task_obj = {}
        self.contains_stochastic_channels = False
        self.logger.info("The network state has been created.")
        self.prev_state_key = None
    
    
    def _check_backend_setup(self, *args, **kwargs):
        what = kwargs['what']
        if not self.backend_setup:
            raise RuntimeError(f"{what} cannot be performed before operands or NetworkOperator has been applied.")
    
    def _setup_backend(self, operand):
        # backend setup should only be performed once
        assert not self.backend_setup, "Internal Error"
        self.backend = operand.name
        self.output_class = operand.__class__
        self.output_location = operand.device
        if self.backend == 'numpy':
            self.intermediate_class = tensor_wrapper.CupyTensor
            self.internal_package = 'cupy'
        else:
            self.intermediate_class = operand.__class__
            self.internal_package = self.backend
        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or operand.device == 'cpu'
        if self.allocator is None:
            self.allocator = memory._MEMORY_MANAGER[self.internal_package](self.device_id, self.logger)
        self.backend_setup = True

    def _free_task_object_resources(self, exception=None):
        """
        Free resources allocated in task computation.
        """
        for task_type, key in list(self.cached_task_obj.keys()):
            task_obj = self.cached_task_obj.pop((task_type, key))
            destroy_func = getattr(cutn, f'destroy_{task_type}')
            destroy_func(task_obj)
            self.logger.info(f"The cached {task_type} object has been freed")
        return True

    def free(self):
        """Free state resources.

        It is recommended that the :class:`NetworkState` object can be used within a context, but if it is not possible then this
        method must be called explicitly to ensure that the state resources are properly cleaned up.
        """

        if not self.valid_state:
            return

        try:
            # Future operations on the workspace stream should be ordered after the computation.
            # The last_compute_event is created by the CUDA execution context (utils.cuda_call_ctx) in an execution method
            # is used to ensure that workspace memory (scratch or cache) is made available for another operation only after the operation that uses it is complete.
            if self.last_compute_event is not None:
                self.workspace_stream.wait_event(self.last_compute_event)
            
            self._free_task_object_resources()
            self.owned_network_operators = {}
            self.non_owned_network_operators = {}            

            if self.workspace_desc is not None:
                cutn.destroy_workspace_descriptor(self.workspace_desc)
                self.workspace_desc = None
            self._free_workspace_memory()
            if self.handle is not None and self.own_handle:
                cutn.destroy(self.handle)
                self.handle = None
                self.own_handle = False
        except Exception as e:
            self.logger.critical("Internal error: only part of the network state resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False
        
        self.logger.info("The network state resources have been released.")


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()

    
    _check_valid_network = Network._check_valid_network
    _free_workspace_memory = Network._free_workspace_memory
    _allocate_workspace_memory_perhaps = Network._allocate_workspace_memory_perhaps
    _release_workspace_memory_perhaps = Network._release_workspace_memory_perhaps
    _release_scratch_memory_perhaps = Network._release_scratch_memory_perhaps
    _release_cache_memory_perhaps = Network._release_cache_memory_perhaps
    _reset_workspace_allocation_tracking = Network._reset_workspace_allocation_tracking


    def _calculate_workspace_size(self):
        """
        Allocate workspace for cutensornet.
        """
        # Release workspace already allocated, if any, because the new requirements are likely different.
        self.workspace_scratch_ptr = None
        self.workspace_cache_ptr = None
        self.workspace_h_scratch_ptr = None
        self.workspace_h_cache_ptr = None

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

        if (self.memory_limit < min_scratch_size):
            message = f"""Insufficient memory.
The memory limit specified is {self.memory_limit}, while the minimum workspace size needed is {min_scratch_size}.
"""
            # such failure is due to problem configuration, not due to implementation or runtime factors
            raise MemoryError(message)

        if min_cache_size > 0:
            self.workspace_scratch_size = rec_scratch_size if rec_scratch_size < self.memory_limit else min_scratch_size
            self.workspace_cache_size = min(self.memory_limit - self.workspace_scratch_size, min_cache_size)
        else:
            self.workspace_cache_size = 0
            self.workspace_scratch_size = min(self.memory_limit, max_scratch_size)

        self.logger.info(f"The workspace size requirements range from {formatters.MemoryStr(min_scratch_size + min_cache_size)} to "\
                         f"{formatters.MemoryStr(max_scratch_size + max_cache_size)}.")
        self.logger.info(f"The scratch workspace size has been set to {formatters.MemoryStr(self.workspace_scratch_size)}.")
        self.logger.info(f"The cache workspace size has been set to {formatters.MemoryStr(self.workspace_cache_size)}.")

        # Deal with device workspaces. For now we don't care how much host memory is used.
        self.workspace_h_scratch_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.HOST, cutn.WorkspaceKind.SCRATCH)
        self.workspace_h_cache_size = cutn.workspace_get_memory_size(
            self.handle, self.workspace_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.HOST, cutn.WorkspaceKind.CACHE)
        
        return {
            'scratch_size': self.workspace_scratch_size,
            'cache_size': self.workspace_cache_size,
            'h_scratch_size': self.workspace_h_scratch_size,
            'h_cache_size': self.workspace_h_cache_size
        }

    def _reset_workspace_size_requirement(self, workspace_dict):
        """
        When preparation can be skipped, this API must be called to update the correct workspace size requirement
        """
        for name in ('scratch', 'cache', 'h_scratch', 'h_cache'):
            if getattr(self, f'workspace_{name}_size') != workspace_dict[f'{name}_size']:
                setattr(self, f'workspace_{name}_size', workspace_dict[f'{name}_size'])
                setattr(self, f'workspace_{name}_ptr', None)

    @utils.precondition(_check_valid_network)
    @utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory_perhaps(self, stream_holder, kind):
        return Network._allocate_workspace_memory_perhaps.__wrapped__.__wrapped__.__wrapped__(self, stream_holder, kind)

    
    ###########################################
    # classmethod for one-step initialization #
    ###########################################
    @classmethod
    def from_circuit(cls, circuit, *, dtype=STATE_DEFAULT_DTYPE, backend='cupy', config=None, options=None, stream=None):
        """
        Create a state object from the given circuit.

        Args:
            circuit : A fully parameterized :class:`cirq.Circuit` or :class:`qiskit.QuantumCircuit` object.
            dtype : A string specifying the datatype for the tensor network, currently supports the following data types:
            
                - ``'complex64'``
                - ``'complex128'`` (default)
            
            backend : The backend for all output tensor operands. If not specified, ``cupy`` is used.
            config : The simulation configuration for the state. It can be:

                - A :class:`TNConfig` object for contraction based tensor network simulation (default).
                - A :class:`MPSConfig` object for MPS based tensor network simulation.
                - A `dict` containing the parameters for the :class:`TNConfig` or :class:`MPSConfig` constructor.
            
            options : Specify options for the computation as a :class:`~cuquantum.NetworkOptions` object. Alternatively, a `dict`
                containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
                the value will be set to the default-constructed ``NetworkOptions`` object.
            stream : Provide the CUDA stream to use for state initialization, which is needed for stream-ordered operations such as allocating memory. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        Note:
            - When parsing gates from the circuit object, all gate operands are assumed to be unitary. In the rare case where the target circuit object contains customized non-unitary gates, 
              users are encouraged to use :meth:`apply_tensor_operator` to construct the :class:`NetworkState` object.
        """
        options = utils.check_or_create_options(NetworkOptions, options, "network options")
        with utils.device_ctx(options.device_id):
            converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend)
        return cls.from_converter(converter, config=config, options=options, stream=stream)
    
    
    @classmethod
    def from_converter(cls, converter, *, config=None, options=None, stream=None):
        """
        Create a :class:`NetworkState` object from the given :class:`cuquantum.CircuitToEinsum` converter.

        Args:
            converter : A :class:`cuquantum.CircuitToEinsum` object.
            config : The simulation configuration for the state simulator. It can be:

                - A :class:`TNConfig` object for contraction based tensor network simulation (default).
                - A :class:`MPSConfig` object for MPS based tensor network simulation.
                - A `dict` containing the parameters for the :class:`TNConfig` or :class:`MPSConfig` constructor.
            
            options : Specify options for the state computation as a :class:`~cuquantum.NetworkOptions` object. Alternatively, a `dict`
                containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
                the value will be set to the default-constructed ``NetworkOptions`` object.
            stream : Provide the CUDA stream to use for state initialization, which is needed for stream-ordered operations such as allocating memory. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        """
        dtype = getattr(converter.dtype, '__name__', str(converter.dtype).split('.')[-1])
        state_mode_extents = (2, ) * len(converter.qubits)
        simulator = cls(state_mode_extents, dtype=dtype, config=config, options=options, state_labels=converter.qubits)
        for gate_operand, gate_qubits in converter.gates:
            # all gate operands are assumed to be unitary
            simulator.apply_tensor_operator(gate_qubits, gate_operand, unitary=True, stream=stream)
        return simulator

    def _mark_updated(self, structural=True):
        """
        When the state is changed, either by applying new operators or updating operands, state_computed and norm must be reset. The cached task objects also should be freed. 
        The only exception is when update_tensor_operator is called for a tensor network contraction simulation or an MPS simulation without value based truncation, 
        in which case the cached task objects are still valid and do not need to be freed.
        """
        self.state_computed = False
        if structural:
            self.state_prepared = False
        if structural or (not self.property_prepare_reusable):
            # Task objects must be re-created for the following cases:
            #   1. Structural change made to the underlying tensor network, regardless of simulation methods
            #   2. No structural change, but underlying simulation is an MPS simulation with value based truncation.
            if self.last_compute_event is not None:
                self.workspace_stream.wait_event(self.last_compute_event)
            self._free_task_object_resources()
    
    ###########################################
    #### APIs for customized initialization ###
    ###########################################    
    @state_operands_wrapper(operands_arg_index=1, is_single_operand=False)
    @utils.precondition(_check_valid_network)
    def set_initial_mps(self, mps_tensors, *, stream=None):
        """
        Set the initial state to a non-vacuum state in the MPS form.

        Args:
            mps_tensors : A sequence of tensors (ndarray-like objects) for each MPS operand. 
                The currently supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.
                The modes of each operand are expected to follow the order of ``pkn`` where ``p`` denotes the mode connecting to the previous MPS tensor, 
                ``k`` denotes the ket mode and ``n`` denotes the mode connecting to the next MPS tensor.
                Note that this method currently only support open boundary condition, and ``p`` and ``n`` mode should thus be dropped in the first and last MPS tensor respectively. 
            stream : Provide the CUDA stream to use for setting the initial state to the specified MPS (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        
        Note:
            - This API simply sets the initial state to the provided MPS and does not alter the nature of the simulation method, 
              which is provided via the ``options`` parameter during initialization.
        """
        extents_in = []
        strides_in = []
        state_tensors_in = []
        for o in mps_tensors:
            extents_in.append(o.shape)
            strides_in.append(o.strides)
            state_tensors_in.append(o.data_ptr)
        cutn.state_initialize_mps(self.handle, self.state, cutn.BoundaryCondition.OPEN, 
            extents_in, strides_in, state_tensors_in)
        # keep a reference
        self.initial_state = list(mps_tensors)
        # reset norm / state vector
        self._mark_updated()
        self.logger.info("The initial state has been set to the provided MPS representation.")
        return
    
    
    @state_labels_wrapper(marker_index=1, marker_type='seq')
    @state_labels_wrapper(key='control_modes', marker_type='seq')
    @state_operands_wrapper(operands_arg_index=2, is_single_operand=True)
    @utils.precondition(_check_valid_network)
    def apply_tensor_operator(self, modes, operand, *, control_modes=None, control_values=None, immutable=False, adjoint=False, unitary=False, stream=None):
        """
        Apply a tensor operator to the network state.

        Args:
            modes : A sequence of integers denoting the modes where the tensor operator acts on. 
                If ``state_labels`` has been provided during initialization, ``modes`` can also be provided as a sequence of labels. 
            operand : A ndarray-like object for the tensor operator.
                The modes of the operand is expected to be ordered as ``ABC...abc...``, 
                where ``ABC...`` denotes output bra modes and ``abc...`` denotes input ket modes corresponding to ``modes`` 
            control_modes : A sequence of integers denotes the modes where control operation is acted on (default no control modes). 
                If ``state_labels`` has been provided during initialization, ``control_modes`` can also be provided as a sequence of labels.
            control_values : A sequence of integers specifying the control values corresponding to ``control_modes``. 
                If ``control_modes`` are specified and ``control_values`` are not provided, control values for all control modes will be set as 1.
            immutable : Whether the operator is immutable (default `False`).
            adjoint : Whether the operator should be applied in its adjoint form (default `False`).
            unitary : Whether the operator is unitary (default `False`).
            stream : Provide the CUDA stream to use for applying the tensor operator (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        
        Returns:
            An integer `tensor_id` specifying the location of the input operator.

        Notes:
            - For MPS simulation, the size of ``modes`` shall be restricted to no larger than 2 (two-body operator).
            - For controlled tensor operators, this method currently only supports immutable operators.
        """
        # operand indices (b, a, B, A) required for modes a, b
        operand = operand.T
        if isinstance(self.config, MPSConfig) and operand.ndim > 4:
            raise ValueError(f"MPS simulation only supports one-body and two-body operators, found operator dimension ({operand.ndim})")
        if control_modes is None:
            tensor_id = cutn.state_apply_tensor_operator(self.handle, self.state, len(modes), 
                modes, operand.data_ptr, operand.strides, immutable, adjoint, unitary)
            self.logger.debug(f"The tensor operand has been applied to the state with an ID ({tensor_id}).")
        else:
            if not immutable:
                raise ValueError(f"NetworkState currently only supports immutable controlled tensor operators")
            tensor_id = cutn.state_apply_controlled_tensor_operator(self.handle, self.state, len(control_modes), 
                control_modes, control_values, len(modes), modes, operand.data_ptr, operand.strides, immutable, adjoint, unitary)
            self.logger.debug(f"The controlled tensor operand has been applied to the state with an ID ({tensor_id}).")
        
        assert tensor_id not in self.operands, "Internal Error"
        
        # keep operand alive otherwise cupy will re-use the memory space
        self.operands[tensor_id] = operand, immutable
        # reset norm / state vector
        self._mark_updated()
        return tensor_id
    
    @state_labels_wrapper(marker_index=1, marker_type='seq')
    @state_operands_wrapper(operands_arg_index=2, is_single_operand=False)
    @utils.precondition(_check_valid_network)
    def apply_unitary_tensor_channel(self, modes, operands, probabilities, *, stream=None):
        """
        Apply a unitary tensor channel to the network state.
        For an error channel with non-unitary operators, see
        :meth:`NetworkState.apply_general_tensor_channel`.

        Args:
            modes : A sequence of integers denoting the modes where the tensor operator acts on. 
                If ``state_labels`` has been provided during initialization, ``modes`` can also be provided as a sequence of labels. 
            operands : A sequence of ndarray-like objects for the unitary tensor operators defining the unitary channel.
                The modes of the operand is expected to be ordered as ``ABC...abc...``, 
                where ``ABC...`` denotes output bra modes and ``abc...`` denotes input ket modes corresponding to ``modes`` 
            probabilities : A sequence of positive floats representing the probabilities of each operand.
            stream : Provide the CUDA stream to use for applying the tensor operator (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        
        Returns:
            An integer `channel_id` specifying the location of the unitary channel.

        Notes:
            - For MPS simulation, the size of ``modes`` shall be restricted to no larger than 2 (two-body operator).
        """
        if len(operands) != len(probabilities):
            raise ValueError(f"The number of operands ({len(operands)}) does not matching the size of probabilities ({len(probabilities)})")
        # operand indices (b, a, B, A) required for modes a, b
        operands = [o.T for o in operands]
        tensor_data = [o.data_ptr for o in operands]
        tensor_mode_strides = [o.strides for o in operands]
        if not all(strides == tensor_mode_strides[0] for strides in tensor_mode_strides):
            raise ValueError(f"The strides for all input operands must be the identical.")
        channel_id = cutn.state_apply_unitary_channel(self.handle, 
            self.state, len(modes), modes, len(operands), tensor_data, tensor_mode_strides[0], probabilities)
        # keep operand alive otherwise cupy will re-use the memory space
        self.operands[channel_id] = operands, True
        # reset norm / state vector
        self._mark_updated()
        self.contains_stochastic_channels = True
        return channel_id

    @state_labels_wrapper(marker_index=1, marker_type='seq')
    @state_operands_wrapper(operands_arg_index=2, is_single_operand=False)
    @utils.precondition(_check_valid_network)
    def apply_general_tensor_channel(self, modes, operands, *, stream=None):
        """
        Apply a noise channel to the MPS network state. The noise operators may be non-unitary.
        For a more efficient unitary tensor channel application, see
        :meth:`NetworkState.apply_unitary_tensor_channel`.

        Args:
            modes : A sequence of integers denoting the modes where the tensor operator acts on. 
                If ``state_labels`` has been provided during initialization, ``modes`` can also be provided as a sequence of labels. 
            operands : A sequence of ndarray-like objects for the tensor operators defining the channel.
                The modes of the operand is expected to be ordered as ``ABC...abc...``, 
                where ``ABC...`` denotes output bra modes and ``abc...`` denotes input ket modes corresponding to ``modes`` 
            stream : Provide the CUDA stream to use for applying the tensor operator (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        
        Returns:
            An integer `channel_id` specifying the location of the general channel.

        Notes:
            - This method requires the input channel to be trace-preserving.
              Supplying a non-trace-preserving channel may lead to unexpected results.
            - As of cuTensorNet v2.7.0, this method only supports MPS simulation
              configured with ``gauge_option="free"`` (default).
            - For MPS simulation, the size of ``modes`` shall be restricted to no larger than 2 (two-body operator).
            - The ``channel_id`` cannot be used to update the channel using
              :meth:`NetworkState.update_tensor_operator`.
        """
        # operand indices (b, a, B, A) required for modes a, b
        if isinstance(self.config, TNConfig):
            raise ValueError(f"Noise simulation using general channel is only supported for MPS simulation. Set NetworkState config to MPSConfig to enable MPS simulatino")
        operands = [o.T for o in operands]
        tensor_data = [o.data_ptr for o in operands]
        tensor_mode_strides = [o.strides for o in operands]
        if not all(strides == tensor_mode_strides[0] for strides in tensor_mode_strides):
            raise ValueError(f"The strides for all input operands must be the identical.")
        channel_id = cutn.state_apply_general_channel(self.handle, 
            self.state, len(modes), modes, len(operands), tensor_data, tensor_mode_strides[0])
        # keep operand alive otherwise cupy will re-use the memory space
        self.operands[channel_id] = operands, True
        # reset norm / state vector
        self._mark_updated()
        self.contains_stochastic_channels = True
        return channel_id
    
    @state_operands_wrapper(operands_arg_index=2, is_single_operand=True)
    @utils.precondition(_check_valid_network)
    def update_tensor_operator(self, tensor_id, operand, *, unitary=False, stream=None):
        """
        Update a tensor operator in the state.

        Args:
            tensor_id : An integer specifing the tensor id assigned in :meth:`NetworkState.apply_tensor_operator`.
            operand : A ndarray-like object for the tensor operator. 
                The operand is expected to follow the same mode ordering, data type and strides as the original operand. 
            unitary : Whether the operator is unitary (default `False`).
            stream : Provide the CUDA stream to use for updating tensor operand (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        """
        # operand indices (b, a, B, A) required for modes a, b
        operand = operand.T
        if tensor_id not in self.operands:
            raise RuntimeError(f"operator with ID ({tensor_id}) has not been applied to the network state.")
        prev_operand, immutable = self.operands[tensor_id]
        if immutable:
            raise RuntimeError(f"tensor id ({tensor_id}) has been marked immutable.")
        if operand.strides != prev_operand.strides:
            raise ValueError(f'The new operand must share the same strides as the original operand ({prev_operand.T.strides}), found ({operand.T.strides})')
        cutn.state_update_tensor_operator(self.handle, self.state, tensor_id, operand.data_ptr, unitary)
        self.operands[tensor_id] = operand, immutable
        self.logger.info(f"Tensor operand with ID ({tensor_id}) has been updated.")
        self._mark_updated(structural=False)
        return


    @utils.precondition(_check_valid_network)
    def apply_network_operator(self, network_operator, *, immutable=False, adjoint=False, unitary=False):
        """
        Apply a network operator to the network state.

        Args:
            network_operator : A :class:`NetworkOperator` object for the input network operator. 
                Must contain only one MPO term or one tensor product term.
            immutable : Whether the network operator is immutable (default `False`).
            adjoint : Whether the network operator should be applied in its adjoint form (default `False`).
            unitary : Whether the network operator is unitary (default `False`).
        Returns:
            An integer `network_id` specifying the location of the network operator.
        """
        if tuple(self.state_mode_extents) != tuple(network_operator.state_mode_extents):
            raise ValueError(f"The dimension for the state ({self.state_mode_extents}) not matching that of the network operator ({self.state_mode_extents})")
        if network_operator.dtype != self.dtype:
            raise ValueError(f"Input network operator data type ({network_operator.dtype}) different from network state ({self.dtype})")
        if not self.backend_setup:
            if network_operator.tensor_products:
                operand = network_operator.tensor_products
            elif network_operator.mpos:
                operand = network_operator.mpos[0][0][0]
            else:
                raise RuntimeError(f"Empty NetworkOperator can not be applied to a NetworkState")
            self._setup_backend(operand)
        network_id = cutn.state_apply_network_operator(self.handle, 
            self.state, network_operator.network_operator, immutable, adjoint, unitary)
        self.non_owned_network_operators[network_id] = network_operator
        self.logger.info("Network operator has been applied to the state with an ID ({network_id})")
        # reset norm / state vector
        self._mark_updated()
        return network_id


    @state_labels_wrapper(marker_index=1, marker_type='seq')
    @utils.precondition(_check_valid_network)
    def apply_mpo(self, modes, mpo_tensors, *, immutable=False, adjoint=False, unitary=False, stream=None):
        """
        Apply an MPO operator specified by `mpo_tensors` and `modes` to the network state.

        Args:
            modes : A sequence of integers specifying each mode where the MPO acts on. 
                If ``state_labels`` has been provided during initialization, ``modes`` can also be provided as a sequence of labels. 
            mpo_tensors : A sequence of tensors (ndarray-like objects) for each MPO operand. 
                The currently supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.
                The mode of each operand is expected to follow the order of ``pknb`` where ``p`` denotes 
                the mode connecting to the previous MPO tensor, ``n`` denotes the mode connecting to the 
                next MPO tensor, ``k`` denotes the ket mode and ``b`` denotes the bra mode. 
                Note that currently only MPO with open boundary condition is supported, 
                therefore ``p`` and ``n`` mode should not be present in the first and last MPO tensor respectively. 
                Note that the relative order of bra and ket modes here differs from that of ``operand`` in :meth:`apply_tensor_operator`.
            immutable : Whether the full MPO is immutable (default `False`).
            adjoint : Whether the full MPO should be applied in its adjoint form (default `False`).
            unitary : Whether the full MPO is unitary (default `False`).
            stream : Provide the CUDA stream to use for appending MPO (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        Returns:
            An integer `network_id` specifying the location of the MPO.
        """
        mpo = NetworkOperator(self.state_mode_extents, dtype=self.dtype, options=self.options)
        mpo.append_mpo(1, modes, mpo_tensors, stream=stream)
        network_id = self.apply_network_operator(mpo, immutable=immutable, adjoint=adjoint, unitary=unitary)
        # The `mpo` object is now internally owned by the state
        self.non_owned_network_operators.pop(network_id)
        self.owned_network_operators[network_id] = mpo
        return network_id
    
    ###########################################
    ###### APIs for property computation ######
    ###########################################

    def _maybe_setup_recompute(self, *args, **kwargs):
        """
        When stochastic channels exist in the state, for property computation methods, this API call is needed before calling _compute_target to activate re-computation of MPS for every compute_xxx call.
        """
        if self.contains_stochastic_channels and isinstance(self.config, MPSConfig):
            self.state_computed = False
    
    def _get_current_key(self):
        if isinstance(self.config, TNConfig) or not hasattr(self, 'mps_tensors'):
            return None
        else:
            return get_mps_key(self.mps_tensors)

    def _maybe_configure_state(self):
        if not self.state_configured:
            # configure the state
            self.config._configure_into_state(self.handle, self.state)
            self.state_configured = True

    def _maybe_set_target_state(self, stream):
        self._maybe_configure_state()
        if isinstance(self.config, MPSConfig) and (not self.target_state_set):
            self.prev_state_key = None
            # specify the largest output MPS tensors' sizes
            max_extent = self.config.max_extent
            self.mps_tensors = []
            output_mps_extents = []
            output_mps_strides = []
            combined_extents_left = np.cumprod(self.state_mode_extents[:-1])
            combined_extents_right = np.cumprod(self.state_mode_extents[::-1])[::-1][1:]
            max_extents = np.minimum(combined_extents_left, combined_extents_right)
            if max_extent is not None:
                for i in range(self.n-1):
                    max_extents[i] = min(max_extents[i], max_extent)
            if max_extents.max() > EXACT_MPS_EXTENT_LIMIT:
                raise ValueError
            stream_holder = utils.get_or_create_stream(self.device_id, stream, self.internal_package)
            for i in range(self.n):
                if i == 0:
                    extents = (self.state_mode_extents[i], max_extents[i])
                elif i == self.n - 1:
                    extents = (max_extents[i-1], self.state_mode_extents[i])
                else:
                    extents = (max_extents[i-1], self.state_mode_extents[i], max_extents[i])

                tensor = utils.create_empty_tensor(self.intermediate_class, extents, self.dtype, self.device_id, stream_holder)
                self.mps_tensors.append(tensor)
                output_mps_extents.append(extents)
                output_mps_strides.append(tensor.strides) 
            cutn.state_finalize_mps(self.handle, self.state, 
                cutn.BoundaryCondition.OPEN, output_mps_extents, output_mps_strides)
            self.logger.info(f"The target MPS state has been set.")
        self.target_state_set = True

    def _maybe_compute_state(self, stream, release_workspace):
        self._maybe_set_target_state(stream)
        if not self.state_computed and isinstance(self.config, MPSConfig):
            create_args = ()
            execute_args = (self.workspace_desc, [o.data_ptr for o in self.mps_tensors])
            # record the key from last MPS computation
            self.prev_state_key = self._get_current_key()
            # compute the final MPS tensors
            output = self._compute_target('state', create_args, execute_args, stream, release_workspace)
            if output is None:
                return False
            else:
                # in the case of dynamic extent, update the container.
                extents, strides = output
                for i in range(self.n):
                    extent_in = self.mps_tensors[i].shape
                    extent_out = extents[i]
                    if extent_in != tuple(extent_out):
                        self.mps_tensors[i].update_extents_strides(extent_out, strides[i])
        # mark state as computed
        self.state_computed = True
        return self.state_computed
    
    @utils.atomic(_free_task_object_resources, method=True)
    def _compute_target(self, task, create_args, execute_args, stream, release_workspace, *, config_args=None, caller_name=None, task_key=None):
        if caller_name is None:
            caller_name = task
        if task not in ('marginal', 'expectation', 'accessor', 'state', 'sampler'):
            raise ValueError("only supports marginal, sampler, accessor, expectation and state")
        
        if task == 'state':
            create_func = None
        else:
            # avoid going into infinite loops
            if not self._maybe_compute_state(stream, release_workspace):
                raise ValueError
            create_func = getattr(cutn, f'create_{task}')
        # Allocate device memory (in stream context) if needed.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.internal_package)
        prepare_func = getattr(cutn, f'{task}_prepare')

        if task == 'sampler':
            execute_func = cutn.sampler_sample
        else:
            execute_func = getattr(cutn, f'{task}_compute')
        
        self.logger.info(f"Beginning {caller_name} computation...")

        prepare_needed = True
        if task == 'state': # state vector or MPS computation
            task_obj = self.state
            prepare_needed = not self.state_prepared
        else:
            if task_key is None:
                task_key = (task, create_args)
            if task_key in self.cached_task_obj and (self.property_prepare_reusable or self._get_current_key() == self.prev_state_key):
                # In the following cases, re-creation/preparation of property computation objects can be skipped:
                # 1. Contraction based simulation  -> self.property_prepare_reusable
                # 2. MPS without value based truncation -> self.property_prepare_reusable
                # 3. MPS with value based truncation but at runtime yields the same extents as the previous compute call -> check key against last computation
                task_obj = self.cached_task_obj[task_key]
                self.logger.info(f"Found the same {task} object from the cache")
                prepare_needed = False
            else:
                # make sure when the current stream waits until the last compute event is done and we only cache one compute task object
                if self.cached_task_obj:
                    stream_holder.obj.wait_event(self.last_compute_event)
                self._free_task_object_resources()
                self.cached_task_obj[task_key] = task_obj = create_func(self.handle, self.state, *create_args)
                self.logger.info(f"A new {task} object has been created")
        
        if config_args is not None:
            configure_func = getattr(cutn, f'{task}_configure')
            configure_func(self.handle, task_obj, *config_args)
            # new prepare step needed if the object has be re-configured
            prepare_needed = True
        
        timing =  bool(self.logger and self.logger.handlers)
        if prepare_needed:
            self.logger.info(f"Starting preparing {caller_name} computation with blocking set to {self.blocking}...")
            with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
                prepare_func(self.handle, task_obj, self.memory_limit, self.workspace_desc, stream_holder.ptr) # similar args for marginal and sampler
            if elapsed.data is not None:
                self.logger.info(f"The preparation of {caller_name} computation took {elapsed.data:.3f} ms to complete.")
            else:
                self.logger.info(f"Preparation for {caller_name} has been completed")
            # cache the workspace size requirements
            self.workspace_sizes_requirements[caller_name] = self._calculate_workspace_size()
            if task == 'state':
                self.state_prepared = True
        else:
            self.logger.info(f"Preparation for {caller_name} has been skipped due to cache usage")
            # While the compute object has been prepared in a previous call, 
            # the current workspace size need to be reset to align with the actual requirement
            if caller_name in self.workspace_sizes_requirements:
                workspace_size = self.workspace_sizes_requirements[caller_name]
            else:
                # NOTE: compute_batched_amplitudes may be used to compute the state vector with proper arguments
                equivalent_callers_set = {'batched_amplitudes', 'state vector', 'amplitude'}
                assert caller_name in equivalent_callers_set
                num_fixed_modes = create_args[0]
                if num_fixed_modes == 0:
                    # state vector & batched_amplitudes both support computing the full state vector
                    equivalent_callers_set.remove('amplitude')
                elif num_fixed_modes == self.n:
                    # amplitude & batched_amplitudes both support computing the amplitude
                    equivalent_callers_set.remove('state vector')
                else:
                    raise RuntimeError("Internal Error")
                equivalent_callers_set.remove(caller_name)
                equivalent_caller = equivalent_callers_set.pop()
                assert equivalent_caller in self.workspace_sizes_requirements
                workspace_size = self.workspace_sizes_requirements[equivalent_caller]

            self._reset_workspace_size_requirement(workspace_size)

        self._allocate_workspace_memory_perhaps(stream_holder, "scratch")
        self._allocate_workspace_memory_perhaps(stream_holder, "cache")

        if self.logger.isEnabledFor(logging.INFO):
            info_flops_enum = getattr(cutn, f'{task.capitalize()}Attribute').INFO_FLOPS
            flops_dtype = getattr(cutn, f'{task}_get_attribute_dtype')(info_flops_enum)
            flops = np.zeros(1, dtype=flops_dtype)
            getattr(cutn, f'{task}_get_info')(self.handle, task_obj, info_flops_enum, flops.ctypes.data, flops.dtype.itemsize)
            self.logger.info(f"Total flop count for {caller_name} computation = {flops.item()/1e9} GFlop")

        self.logger.info(f"Starting {caller_name} computation with blocking set to {self.blocking}...")
        with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
            output = execute_func(self.handle, task_obj, *execute_args, stream_holder.ptr)
        if elapsed.data is not None:
            self.logger.info(f"Computation for {caller_name} took {elapsed.data:.3f} ms to complete.")
        else:
            self.logger.info(f"Computation for {caller_name} has been completed")
        
        # Establish ordering wrt the computation and free scratch and cache workspace based on user request.
        self._release_workspace_memory_perhaps("scratch", release_workspace=release_workspace)
        self._release_workspace_memory_perhaps("cache", release_workspace=release_workspace)
        self._reset_workspace_allocation_tracking()

        # for task == 'state', output is a tuple, otherwise None
        return output
    

    def _run_state_accessor(self, caller_name, return_norm, *, fixed_modes=None, stream=None, release_workspace=False):
        if fixed_modes:
            # compute batched amplitudes
            shape = [self.state_mode_extents[q] for q in range(self.n) if q not in fixed_modes]
            num_fixed_modes = len(fixed_modes)
            fixed_modes, fixed_values = zip(*sorted(fixed_modes.items())) 
            fixed_modes = tuple(fixed_modes)
            fixed_values = tuple([int(i) for i in fixed_values])
        else:
            # compute full state vector
            shape = self.state_mode_extents
            num_fixed_modes = fixed_modes = fixed_values = 0
        
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.internal_package)
        amplitudes = utils.create_empty_tensor(self.intermediate_class, shape, self.dtype, self.device_id, stream_holder)

        norm = np.empty(1, dtype=self.dtype)

        create_args = (num_fixed_modes, fixed_modes, tuple(amplitudes.strides))
        if return_norm:
            execute_args = (fixed_values, self.workspace_desc, amplitudes.data_ptr, norm.ctypes.data)
        else:
            execute_args = (fixed_values, self.workspace_desc, amplitudes.data_ptr, 0)
        self._compute_target('accessor', create_args, execute_args, stream, release_workspace, caller_name=caller_name)
        if return_norm:
            return amplitudes, norm.real.item()
        else:
            return amplitudes

    
    @state_result_wrapper(is_scalar=True)
    @utils.precondition(_maybe_setup_recompute)
    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_backend_setup, "Amplitude computation")
    def compute_amplitude(self, bitstring, *, return_norm=False, stream=None, release_workspace=False):
        """
        Compute the probability amplitude of a bitstring.

        Args:
            bitstring : A sequence of integers specifying the desired measured state dimension. 
            return_norm : If true, the squared norm of the state will also be returned.
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
        
        Returns:
            If ``return_norm`` is `False`, a scalar for the bitstring amplitude; otherwise, a 2-tuple consisting of the bitstring of the amplitude 
            and a scalar for the squared norm of the state, i.e, inner product of bra and ket state.
        """
        if len(bitstring) != self.n:
            raise ValueError(f"Length of bitstring is expected to match the dimension of the underlying state ({self.n}), found ({len(bitstring)})")
        fixed_modes = {}
        for i, bit in enumerate(bitstring):
            fixed_modes[i] = int(bit)
        return self._run_state_accessor('amplitude', return_norm, fixed_modes=fixed_modes, stream=stream, release_workspace=release_workspace)
    
    @state_labels_wrapper(marker_index=1, marker_type='dict')
    @state_result_wrapper(is_scalar=False)
    @utils.precondition(_maybe_setup_recompute)
    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_backend_setup, "Batched amplitude computation")
    def compute_batched_amplitudes(self, fixed, *, return_norm=False, stream=None, release_workspace=False):
        """
        Compute the batched amplitudes for a given slice.

        Args:    
            fixed : A dictionary mapping a subset of state dimensions to correponding fixed states. 
                If ``state_labels`` has been provided during initialization, ``fixed`` can also be provided as a dictionary mapping a subset of labels to corresponding fixed states. 
            return_norm : If true, the squared norm of the state will also be returned.
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
        
        Returns:
            If ``return_norm`` is `False`, An ndarray-like object as batched amplitudes. The package and storage location of the ndarray will be the same as 
            the operands provided in :meth:`apply_tensor_operator`, :meth:`apply_mpo` and :meth:`set_initial_mps`; otherwise, a 2-tuple consisting of the batched amplitudes 
            and a scalar for the squared norm of the state, i.e, inner product of bra and ket state.
        """
        return self._run_state_accessor('batched_amplitudes', return_norm, fixed_modes=fixed, stream=stream, release_workspace=release_workspace)
    

    @state_result_wrapper(is_scalar=False)
    @utils.precondition(_maybe_setup_recompute)
    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_backend_setup, "State vector computation")
    def compute_state_vector(self, *, return_norm=False, stream=None, release_workspace=False):
        """
        Compute the state vector.

        Args:
            return_norm : If true, the squared norm of the state will also be returned.
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.

        Returns:
            If ``return_norm`` is `False`, An ndarray-like object as the state vector. The package and storage location of the ndarray will be the same as 
            the operands provided in :meth:`apply_tensor_operator`, :meth:`apply_mpo` and :meth:`set_initial_mps`; otherwise, a 2-tuple consisting of the state vector 
            and a scalar for the squared norm of the state, i.e, inner product of bra and ket state.
        """
        return self._run_state_accessor('state vector', return_norm, fixed_modes={}, stream=stream, release_workspace=release_workspace)

    @state_labels_wrapper(marker_index=1, marker_type='seq')
    @state_labels_wrapper(key='fixed', marker_type='dict')
    @state_result_wrapper(is_scalar=False)
    @utils.precondition(_maybe_setup_recompute)
    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_backend_setup, "Reduced density matrix computation")
    def compute_reduced_density_matrix(self, where, *, fixed=EMPTY_DICT, stream=None, release_workspace=False):
        """
        Compute the reduced density matrix for the given marginal and fixed modes.

        Args:
            where : A sequence of integers for the target modes. 
                If ``state_labels`` has been provided during initialization, ``where`` can also be provided as a sequence of labels. 
            fixed : A dictionary mapping a subset of fixed modes to the fixed value. 
                If ``state_labels`` has been provided during initialization, ``fixed`` can also be provided as a dictionary mapping labels to the corresponding fixed values. 
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
        
        Returns:
            An ndarray-like object as the reduced density matrix. 
            The tensor will follow the modes of ``AB...ab...`` where ``AB...`` and ``ab...`` represents the corresponding output and input marginal modes.
        """
        n_marginal_modes = len(where)
        if fixed:
            n_projected_modes = len(fixed)
            projected_modes, projected_mode_values = zip(*sorted(fixed.items()))
            projected_modes = tuple(projected_modes)
            projected_mode_values = tuple([int(i) for i in projected_mode_values])
        else:
            n_projected_modes = projected_modes = projected_mode_values = 0
        rdm_shape = [self.state_mode_extents[q] for q in where] * 2

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.internal_package)
        rdm = utils.create_empty_tensor(self.intermediate_class, rdm_shape, self.dtype, self.device_id, stream_holder)
        create_args = (n_marginal_modes, tuple(where), n_projected_modes, projected_modes, tuple(rdm.strides))
        execute_args = (projected_mode_values, self.workspace_desc, rdm.data_ptr)
        self._compute_target('marginal', create_args, execute_args, stream, release_workspace)
        return rdm
    
    @utils.precondition(_maybe_setup_recompute)
    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_backend_setup, "Output state")
    def compute_output_state(self, *, stream=None, release_workspace=False, release_operators=False):
        """
        Compute the final output state for the underlying network state object. This method currently is only valid for MPS based simulation.

        Args:
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
            release_operators : A value of `True` will release the reference of all underlying tensor operators and :class:`NetworkOperator` objects. 
                The previous ``tensor_id`` returned by :meth:`apply_tensor_operator`, :meth:`apply_network_operator` and :meth:`apply_mpo` will be invalid.
                If the output state has already been computed, which is an intermediate step in other ``compute_xxx`` methods, the output state will be cached and returned directly. 
                Thus passing ``release_operators=True`` can be used to reset the underlying :class:`NetworkState` object.
    
        Returns:
            When MPS simulation when is specified using the ``options`` argument during object initialization, a sequence of operands representing the underlying 
            MPS state will be returned. The modes of each MPS operand are expected to follow the order of ``pkn`` where ``p`` denotes the mode connecting to the previous MPS tensor, 
            ``k`` denotes the ket mode and ``n`` denotes the mode connecting to the next MPS tensor. Note that ``p`` and ``n`` mode should not be present in the first and last MPS tensor respectively. 
        """
        if isinstance(self.config, TNConfig):
            raise ValueError("For network contraction based state simulation, no explicit state representation is formed, use compute_state_vector for full state vector computation")
        elif isinstance(self.config, MPSConfig):
            self._maybe_compute_state(stream, release_workspace)
            if self.output_location == 'cpu':
                stream_holder = utils.get_or_create_stream(self.device_id, stream, self.internal_package)
                result = [o.to('cpu', stream_holder=stream_holder) for o in self.mps_tensors]
            else:
                result = [o.tensor for o in self.mps_tensors]
            if release_operators:
                # event synchronization
                if self.last_compute_event is not None:
                    if stream is None: 
                        stream = utils.get_or_create_stream(self.device_id, stream, self.internal_package).obj
                    stream.wait_event(self.last_compute_event)
                # release reference to underlying operators and NetworkOperator
                cutn.state_capture_mps(self.handle, self.state)
                self.operands = {}
                self.owned_network_operators = {}
                self.non_owned_network_operators = {}
                self.initial_state = list(self.mps_tensors)
                # mark state as no longer computed & stochastic channels as resolved
                self.contains_stochastic_channels = False
                self._mark_updated()
            return result
        else:
            raise NotImplementedError()
    
    @state_labels_wrapper(key='modes', marker_type='seq')
    @utils.precondition(_maybe_setup_recompute)
    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_backend_setup, "Sampling")
    def compute_sampling(self, nshots, *, modes=None, seed=None, stream=None, release_workspace=False):
        """
        Perform sampling on the given modes.

        Args:
            nshots : The number of samples to collect.
            modes: The target modes to sample on. If not provided, will sample all modes.
                If ``state_labels`` has been provided during initialization, ``modes`` can also be provided as a sequence of labels. 
            seed: A positive integer denoting the random seed to use for generating the samples. If not provided, 
                the generator will continue from the previous seed state or from an unseeded state if no seed was previously set. 
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
        
        Returns:
            A dictionary mapping the bitstring to the corresponding count.
        """
        if modes is None:
            modes = tuple(range(self.n))
        else:
            modes = tuple(modes)
        n_modes = len(modes)
        samples = np.empty((nshots, n_modes), dtype='int64') # equivalent to (n_modes, nshots) in F order

        create_args = (n_modes, modes)
        execute_args = (nshots, self.workspace_desc, samples.ctypes.data)
        if seed is None:
            config_args = None
        else:
            if seed <= 0:
                raise ValueError("seed must be a positive integer")
            attr = cutn.SamplerAttribute.CONFIG_DETERMINISTIC
            val = np.asarray(seed, dtype=cutn.get_sampler_attribute_dtype(attr))
            config_args = (attr, val.ctypes.data, val.dtype.itemsize)
        self._compute_target('sampler', create_args, execute_args, stream, release_workspace, config_args=config_args)
        sampling = {}
        for bitstring, n_sampling in zip(*np.unique(samples, axis=0, return_counts=True)):
            bitstring = np.array2string(bitstring, separator='')[1:-1]
            sampling[bitstring] = n_sampling
        return sampling
    
    @utils.precondition(_maybe_setup_recompute)
    @utils.precondition(_check_valid_network)
    @utils.precondition(_check_backend_setup, "Expectation computation")
    def compute_expectation(self, operators, *, return_norm=False, stream=None, release_workspace=False):
        """
        Compute the expectation value (not normalized) for the given tensor network operator.

        Args:
            operators : The :class:`NetworkOperator` operator object to compute expectation value on. 
                If the underlying state dimensions are all 2 (qubits), it can also be:

                - A single pauli string specifying the pauli operator for each qubit.
                - A dictionary mapping each single pauli string to corresponding coefficient.

            return_norm : If true, the squared norm of the state will also be returned.
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
            
        Returns:
            If ``return_norm`` is `False`, a scalar for the total expectation value; otherwise, a 2-tuple consisting of the total expectation value
            and a scalar for the squared norm of the state, i.e, inner product of bra and ket state.
        
        Note:
            - If user wishes to perform expectation value computation on the same operator multiple times, it is recommended to explicitly provide a :class:`NetworkOperator` object 
              for optimal performance. For detailed examples, please see or `variational expectation example <https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/tensornet/experimental/network_state/generic_states/example04_variational_expectation.py>`_.
            - For pauli operator expectation value computations, this method does not take advantage of lightcone simplification optimization. 
              If user wishes to compute the expectation value on a pauli string operator with many identities in it, consider either using the :meth:`compute_reduced_density_matrix` method or 
              explicitly construct the :class:`NetworkOperator` object with :meth:`NetworkOperator.append_product` for optimal performance.
        """
        if set(self.state_mode_extents) != set([2]):
            assert isinstance(operators, NetworkOperator), f"Pauli operator expectation only supported when all state dimensions equal 2, found ({self.state_mode_extents})."
        if isinstance(operators, str):
            # a single pauli string
            if len(operators) != self.n or (not set(operators).issubset(set('IXYZ'))):
                raise ValueError(f"For pauli expectation computation, operators must be provided as a string made of IXYZ of equal length as the state dimensions, found ({operators})")
            operators = {operators: 1}
        own_network_operators = isinstance(operators, dict)
        if own_network_operators:
            # a pauli string dictionary
            operators = NetworkOperator.from_pauli_strings(operators, dtype=self.dtype, options=self.options, stream=stream)
            self.owned_network_operators[None] = operators
        assert isinstance(operators, NetworkOperator)
        if tuple(self.state_mode_extents) != tuple(operators.state_mode_extents):
            raise ValueError(f"The dimension for the state ({self.state_mode_extents}) not matching that of the network operator ({self.state_mode_extents})")
        expectation_value = np.empty(1, dtype=self.dtype)
        norm = np.empty(1, dtype=self.dtype)
        create_args = (operators.network_operator, )
        # only compute and cache norm when it's has not been computed
        if return_norm:
            execute_args = (self.workspace_desc, expectation_value.ctypes.data, norm.ctypes.data)
        else:
            execute_args = (self.workspace_desc, expectation_value.ctypes.data, 0)
        task_key = ('expectation', operators._get_key())
        self._compute_target('expectation', create_args, execute_args, stream, release_workspace, task_key=task_key)
        output = expectation_value.item()
        if return_norm:
            return output, norm.real.item()
        else:
            return output
