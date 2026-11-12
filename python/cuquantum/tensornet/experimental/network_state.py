# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = ['NetworkState']

import collections.abc
import logging

import numpy as np

from cuquantum.bindings import cutensornet as cutn
from nvmath import memory
from nvmath.internal import utils as nvmath_utils
from nvmath.internal import formatters, tensor_wrapper
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE

from .configuration import MPSConfig, TNConfig
from .network_operator import NetworkOperator
from ._internal.network_state_utils import (
    STATE_DEFAULT_DTYPE,
    check_dtype_supported,
    check_expectation_with_gradients_norm_args,
    get_mps_key,
    get_mps_output_extents,
    host_scalar_to_holder,
    unwrap_output_tensor,
    resolve_unitary_kwarg,
    state_operands_wrapper,
    state_result_wrapper,
    state_labels_wrapper,
)
from ..tensor_network import Network
from ..circuit_converter import CircuitToEinsum
from ..configuration import NetworkOptions
from .._internal.circuit_converter_utils import EMPTY_DICT, split_mixed_fixed, split_mixed_bitstring
from .._internal.decomposition_utils import update_tensor_extents_strides
from nvmath.internal.tensor_wrapper import infer_tensor_package
import importlib
import math


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

        pure_state : Whether the state is pure. ``True`` (default) selects pure-state simulation;
            ``False`` selects mixed-state (density matrix) simulation.

        config : The simulation configuration for the state. It can be:

            - A :class:`TNConfig` object for contraction based tensor network simulation (default).
            - A :class:`MPSConfig` object for MPS based tensor network simulation.
            - A `dict` containing the parameters for the :class:`TNConfig` or :class:`MPSConfig` constructor.

        state_labels : Optional, a sequence of different labels corresponding to each state dimension. 
            If provided, users have the option to provide a sequence of these labels as the input arguments for the following APIs including 
            :meth:`apply_tensor_operator`, :meth:`apply_mpo`, :meth:`compute_batched_amplitudes`, :meth:`compute_reduced_density_matrix` and 
            :meth:`compute_sampling`. See the docstring for each of these APIs for more details.
        options : Specify options for the state computation as a :class:`cuquantum.tensornet.NetworkOptions` object. 
            Alternatively, a `dict` containing the parameters for the ``NetworkOptions`` constructor can also be provided. 
            If not specified, the value will be set to the default-constructed ``NetworkOptions`` object.
    
    Notes:
        - Currently :class:`NetworkState` only supports pure state representation.
        - The state is placed on CUDA device 0 by default. To use a different device, specify it explicitly via :attr:`NetworkOptions.device_id`, and ensure that all input operands reside on that same device.
        - For MPS simulation, currently only open boundary condition is supported.
        - MPS simulation requires at least two state modes. A :class:`NetworkState` with a single state mode and an :class:`MPSConfig` raises a ``ValueError``; use :class:`TNConfig` instead.
    
    Examples:

        In this example, we aim to directly perform simulation on a quantum circuit instance using tensor network contraction method.

        >>> from cuquantum.tensornet.experimental import NetworkState, TNConfig
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
    def __init__(self, state_mode_extents, *, dtype=STATE_DEFAULT_DTYPE, pure_state=True, config=None, state_labels=None, options=None):

        options = nvmath_utils.check_or_create_options(NetworkOptions, options, "network options")
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
        self.memory_limit = nvmath_utils.get_memory_limit_from_device_id(self.options.memory_limit, self.device_id)
        self.logger.info(f"The memory limit is {formatters.MemoryStr(self.memory_limit)}.")

        self.state_mode_extents = list(state_mode_extents)
        self.n = len(self.state_mode_extents)

        check_dtype_supported(dtype)
        self.dtype = dtype
        self.cuda_dtype = NAME_TO_DATA_TYPE[dtype]

        if not isinstance(pure_state, bool):
            raise TypeError(f"pure_state must be a bool, got {type(pure_state)}")
        self.is_pure = pure_state

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
                self.config = nvmath_utils.check_or_create_options(config_class, config, config_class.__name__)
            except TypeError:
                continue
            else:
                break
        else:
            raise ValueError("method must be either a TNConfig/MPSConfig object or a dict that can be used to construct TNConfig/MPSConfig")
        
        if self.n == 1 and isinstance(self.config, MPSConfig):
            raise ValueError(f"For system with one physical dimension, please switch to tensor network simulation method via TNConfig")
        
        self.operands = {}
        self._gradient_tensor_ids = {}  # tensor_id -> operand (gradient uses operand.shape, operand.strides)
        # tensor_id -> last-recorded unitary flag for plain and controlled tensor
        # operators; a flag change on update is a structural modification (plans
        # must be rebuilt)
        self._operator_unitarity = {}
        # ids of diagonal tensor operators (their operands hold only the diagonal entries,
        # so numeric unitarity classification uses the diagonal check)
        self._operator_diagonal = set()
        self.owned_network_operators = {}
        self.non_owned_network_operators = {}

        # Create/set handle.
        if options.handle is not None:
            self.own_handle = False
            self.handle = options.handle
        else:
            self.own_handle = True
            with nvmath_utils.device_ctx(self.device_id):
                self.handle = cutn.create()
        
        # Create the state object
        self.state = cutn.create_state(self.handle, 
            cutn.StatePurity.PURE if self.is_pure else cutn.StatePurity.MIXED,
            self.n, state_mode_extents, self.cuda_dtype)
        
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
        self._output_holds_initial_state = False
        self.state_prepared = False
        self.state_computed = False
        self.initial_state = []
        self.valid_state = True

        self.cached_task_obj = {}
        # key of the computed MPS the cached task object was created for (MPS simulation)
        self._cached_task_state_key = None
        self.contains_stochastic_channels = False
        self.logger.info("The network state has been created.")

        #: Pauli observable memo: :meth:`_pauli_expectation_observable_cache_key` -> :class:`NetworkOperator`
        self._pauli_network_operator_cache = {}

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
            self.internal_package = 'cuda'
            self.intermediate_class = tensor_wrapper._TENSOR_TYPES[self.internal_package]
        else:
            self.intermediate_class = operand.__class__
            self.internal_package = self.backend
        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or operand.device == 'cpu'
        if self.allocator is None:
            self.allocator = memory._MEMORY_MANAGER[self.internal_package](self.device_id, self.logger)
        self.backend_setup = True

    def _get_cutn_task_name(self, task):
        """Map logical task name to cutensornet API base name (e.g. expectation_with_gradients -> expectation)."""
        if task == 'expectation_with_gradients':
            return 'expectation'
        if task == 'marginal_diagonal':
            # MarginalDiagonal shares the same cutensornetStateMarginal_t handle, so prepare/compute/destroy/get_info/configure all use the marginal_* family; only the create function differs.
            return 'marginal'
        return task

    @staticmethod
    def _coefficient_key_for_expectation_cache(coefficient):
        """Hashable real/imag pair; always complex128 (CUTN uses ``cuDoubleComplex`` for term coefficients)."""
        scalar = np.asarray(coefficient, dtype=np.complex128).reshape(-1)[0]
        return (float(np.real(scalar)), float(np.imag(scalar)))

    def _clear_pauli_network_operator_cache(self):
        """Drop memoized Pauli-derived :class:`NetworkOperator` objects (does not mutate CUTN expectation cache dict)."""
        self._pauli_network_operator_cache.clear()

    def _pauli_expectation_observable_cache_key(self, pauli_dict):
        """Sort-and-normalized Pauli (term, coefficient) pairs; memo matches :meth:`NetworkOperator.from_pauli_strings` defaults (including ``remove_identity='auto'``)."""
        # Order by Pauli string only so dict key order does not change the cache entry.
        pairs_sorted_by_term = sorted(pauli_dict.items(), key=lambda item: item[0])
        return tuple(
            (term, self._coefficient_key_for_expectation_cache(coef))
            for term, coef in pairs_sorted_by_term
        )

    def _get_or_build_cached_pauli_network_operator(self, pauli_dict, *, stream=None):
        """
        Return a memoized Pauli-derived :class:`NetworkOperator` for repeated convenience inputs.

        Validates Pauli-string keys like :meth:`compute_expectation`; invalid states are unreachable here because
        expectation entry points require backend setup beforehand.
        Builds via :meth:`NetworkOperator.from_pauli_strings` with defaults (including ``remove_identity='auto'``).
        """
        for term in pauli_dict:
            if len(term) != self.n or (not set(term).issubset(set("IXYZ"))):
                raise ValueError(
                    "For pauli expectation computation, each dict key must be a string made of "
                    f"IXYZ of length {self.n}, got {term!r}."
                )
        cache_key = self._pauli_expectation_observable_cache_key(pauli_dict)
        cached = self._pauli_network_operator_cache.get(cache_key)
        if cached is not None:
            return cached
        op = NetworkOperator.from_pauli_strings(
            pauli_dict,
            backend=self.backend,
            dtype=self.dtype,
            options=self.options,
            stream=stream,
        )
        self._pauli_network_operator_cache[cache_key] = op
        return op

    def _free_task_object_resources(self, exception=None):
        """
        Free resources allocated in task computation.
        """
        for task_type, key in list(self.cached_task_obj.keys()):
            task_obj = self.cached_task_obj.pop((task_type, key))
            cutn_task = self._get_cutn_task_name(task_type)
            destroy_func = getattr(cutn, f'destroy_{cutn_task}')
            destroy_func(task_obj)
            self.logger.info(f"The cached {task_type} object has been freed")
        self._cached_task_state_key = None
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
                self.workspace_stream.wait(self.last_compute_event)
                self.last_compute_event = None
            
            self._free_task_object_resources()
            self._clear_pauli_network_operator_cache()
            self.owned_network_operators = {}
            self.non_owned_network_operators = {}            

            if self.workspace_desc is not None:
                cutn.destroy_workspace_descriptor(self.workspace_desc)
                self.workspace_desc = None
            self._free_workspace_memory()
            if self.state is not None:
                cutn.destroy_state(self.state)
                self.state = None
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

    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory_perhaps(self, stream_holder, kind):
        return Network._allocate_workspace_memory_perhaps.__wrapped__.__wrapped__.__wrapped__(self, stream_holder, kind)

    
    ###########################################
    # classmethod for one-step initialization #
    ###########################################
    @classmethod
    def from_circuit(cls, circuit, *, dtype=STATE_DEFAULT_DTYPE, backend="auto", pure_state=True, config=None, options=None, stream=None):
        """
        Create a state object from the given circuit.

        Args:
            circuit : A fully parameterized :class:`cirq.Circuit` or :class:`qiskit.QuantumCircuit` object.
            dtype : A string specifying the datatype for the tensor network, currently supports the following data types:
            
                - ``'complex64'``
                - ``'complex128'`` (default)
            
            backend : The backend for all output tensor operands. If not specified, ``cupy`` is used when it is available, otherwise ``numpy`` is used.
            pure_state : Whether the state is pure. ``True`` (default) for pure-state simulation;
                ``False`` for mixed-state (density matrix) simulation.
            config : The simulation configuration for the state. It can be:

                - A :class:`TNConfig` object for contraction based tensor network simulation (default).
                - A :class:`MPSConfig` object for MPS based tensor network simulation.
                - A `dict` containing the parameters for the :class:`TNConfig` or :class:`MPSConfig` constructor.
            
            options : Specify options for the computation as a :class:`~cuquantum.tensornet.NetworkOptions` object. Alternatively, a `dict`
                containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
                the value will be set to the default-constructed ``NetworkOptions`` object.
            stream : Provide the CUDA stream to use for state initialization, which is needed for stream-ordered operations such as allocating memory. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
        Note:
            - When parsing gates from the circuit object, all gate operands are assumed to be unitary. In the rare case where the target circuit object contains customized non-unitary gates, 
              users are encouraged to use :meth:`apply_tensor_operator` to construct the :class:`NetworkState` object.
        """
        options = nvmath_utils.check_or_create_options(NetworkOptions, options, "network options")
        if nvmath_utils.infer_object_package(circuit) == 'qiskit':
            parser_options = {'decompose_gates': True, 'check_diagonal': True}
        else:
            # cirq.Circuit does not support decompose_gates option
            parser_options = {'check_diagonal': True}
        with nvmath_utils.device_ctx(options.device_id):
            # CircuitToEinsum auto-detects channels; NetworkState purity (including trajectory
            # simulation of a noisy circuit as a pure state) is controlled by ``pure_state`` here.
            converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend, options=parser_options)
        return cls.from_converter(converter, pure_state=pure_state, config=config, options=options, stream=stream)
    
    
    @classmethod
    def from_converter(cls, converter, *, pure_state=True, config=None, options=None, stream=None):
        """
        Create a :class:`NetworkState` object from the given :class:`cuquantum.tensornet.CircuitToEinsum` converter.

        Args:
            converter : A :class:`cuquantum.tensornet.CircuitToEinsum` object.
            pure_state : Whether the state is pure. ``True`` (default) for pure-state simulation;
                ``False`` for mixed-state (density matrix) simulation.
            config : The simulation configuration for the state simulator. It can be:

                - A :class:`TNConfig` object for contraction based tensor network simulation (default).
                - A :class:`MPSConfig` object for MPS based tensor network simulation.
                - A `dict` containing the parameters for the :class:`TNConfig` or :class:`MPSConfig` constructor.
            
            options : Specify options for the state computation as a :class:`cuquantum.tensornet.NetworkOptions` object. Alternatively, a `dict`
                containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
                the value will be set to the default-constructed ``NetworkOptions`` object.
            stream : Provide the CUDA stream to use for state initialization, which is needed for stream-ordered operations such as allocating memory. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
        """
        dtype = getattr(converter.dtype, '__name__', str(converter.dtype).split('.')[-1])
        state_mode_extents = (2, ) * len(converter.qubits)
        simulator = cls(state_mode_extents, dtype=dtype, pure_state=pure_state, config=config, options=options, state_labels=converter.qubits)

        for entry in converter._gate_entries:
            if entry.kind == 'unitary_channel':
                simulator.apply_unitary_tensor_channel(
                    entry.qubits, entry.operand, list(entry.probabilities), stream=stream)
            elif entry.kind == 'general_channel':
                simulator.apply_general_tensor_channel(entry.qubits, entry.operand, stream=stream)
            else:
                operand = entry.operand
                if entry.is_diagonal:
                    if operand.ndim == 2:
                        operand = operand.diagonal()
                    else:
                        ndim = operand.ndim
                        operand = operand.reshape(2**(ndim//2), 2**(ndim//2)).diagonal().reshape((2,)*(ndim//2))
                    simulator.logger.debug("The operand is diagonal, and is converted to a diagonal tensor.")
                simulator.apply_tensor_operator(
                    entry.qubits, operand, diagonal=entry.is_diagonal, unitary=True, stream=stream)
        return simulator

    @staticmethod
    def _operand_is_unitary(tensor, *, diagonal=False):
        """Numerically classify a native tensor operand as unitary or not.

        ``tensor`` is a native numpy/cupy/torch tensor as passed by the user;
        the check runs wherever the tensor resides via the array interface
        those packages share, with no per-backend branching.
        A non-diagonal operand is reshaped to a square
        matrix over the first/second halves of its modes; since unitarity is
        invariant to permutations within each half and to swapping the two halves
        (transposition), a plain reshape to (D, D) is valid regardless of the
        bra/ket mode ordering. Diagonal operands hold only the diagonal entries,
        which are unitary exactly when all magnitudes equal one.

        The tolerance is intentionally tight (64 * eps(dtype) * sqrt(D)): a
        unitary operand misclassified as non-unitary only costs performance
        (e.g. an unnecessary gauge refresh in simple-update MPS), while the
        reverse misclassification would silently produce wrong results.
        """
        if hasattr(tensor, 'detach'):
            # keep torch autograd out of the classification; no-op for other backends
            tensor = tensor.detach()
        num_elements = math.prod(tensor.shape) if len(tensor.shape) else 1
        # normalized dtype name across backends ('complex128', 'torch.complex128', ...)
        dtype_name = str(tensor.dtype).split('.')[-1]
        eps = float(np.finfo(np.dtype(dtype_name)).eps)
        if diagonal:
            # per-entry magnitude test with no accumulation: the native-precision
            # measurement error (~eps) is well inside the 64*eps threshold
            atol = 64.0 * eps * math.sqrt(num_elements)
            return float(abs(abs(tensor) - 1).max()) <= atol
        dim = math.isqrt(num_elements)
        if dim * dim != num_elements:
            return False
        atol = 64.0 * eps * math.sqrt(dim)
        if dtype_name in ('complex64', 'float32'):
            # Evaluate the Gram residual in double precision so it measures the data's
            # deviation from unitarity rather than fp32 accumulation noise; the
            # tolerance stays keyed to the original dtype's eps.
            if hasattr(tensor, 'astype'):  # numpy/cupy
                tensor = tensor.astype('complex128')
            else:  # torch
                import torch
                tensor = tensor.to(torch.complex128)
        matrix = tensor.reshape(dim, dim)
        gram = matrix @ matrix.conj().T
        # subtract the identity in place (gram is a fresh tensor)
        gram[list(range(dim)), list(range(dim))] -= 1
        return float(abs(gram).max()) <= atol

    def _resolve_apply_unitarity(self, operand, args, kwargs):
        """unitary=None policy for apply_tensor_operator: classify the native
        operand. With control modes the operand is the target block, whose
        unitarity is the whole operator's."""
        control_modes = kwargs.get('control_modes')
        # len() rather than truthiness (ambiguous for array-like sequences); an
        # empty control set denotes a plain (non-controlled) operator, and a
        # controlled operand is never a diagonal-entries tensor
        is_controlled = control_modes is not None and len(control_modes) > 0
        diagonal = (not is_controlled) and bool(kwargs.get('diagonal', False))
        unitary = self._operand_is_unitary(operand, diagonal=diagonal)
        self.logger.debug(f"The operand has been numerically classified as {'unitary' if unitary else 'non-unitary'}.")
        return unitary

    def _resolve_update_unitarity(self, operand, args, kwargs):
        """unitary=None policy for update_tensor_operator: classify the new
        native operand, with diagonal-ness looked up from the recorded
        operator."""
        tensor_id = args[1]
        unitary = self._operand_is_unitary(operand, diagonal=(tensor_id in self._operator_diagonal))
        self.logger.debug(f"The updated operand has been numerically classified as {'unitary' if unitary else 'non-unitary'}.")
        return unitary

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
                self.workspace_stream.wait(self.last_compute_event)
                self.last_compute_event = None
            self._clear_pauli_network_operator_cache()
            self._free_task_object_resources()
    
    ###########################################
    #### APIs for customized initialization ###
    ###########################################    
    @state_operands_wrapper(operands_arg_index=1, is_single_operand=False)
    @nvmath_utils.precondition(_check_valid_network)
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
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
        
        Note:
            - This API simply sets the initial state to the provided MPS and does not alter the nature of the simulation method, 
              which is provided via the ``options`` parameter during initialization.
            - Tensors on the GPU are used without a copy and the object keeps a reference to them. They must not be modified while
              this object uses them; to change the initial state, call this method again.
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
        # output tensors of this object (or views into them) passed back as the initial state must not be overwritten by
        # later computations: detect whether any given tensor lies inside one of the output tensors (allocated at the
        # maximum extents)
        output_ranges = [(o.data_ptr, o.data_ptr + int(np.prod(e)) * o.itemsize)
                         for o, e in zip(getattr(self, 'mps_tensors', []), getattr(self, '_output_mps_max_extents', []))]
        self._output_holds_initial_state = any(lo <= o.data_ptr < hi for o in mps_tensors for lo, hi in output_ranges)
        # reset norm / state vector
        self._mark_updated()
        self.logger.info("The initial state has been set to the provided MPS representation.")
        return
    
    
    @resolve_unitary_kwarg(operand_arg_index=2, resolver_name='_resolve_apply_unitarity')
    @state_labels_wrapper(marker_index=1, marker_type='seq')
    @state_labels_wrapper(key='control_modes', marker_type='seq')
    # operand indices (b, a, B, A) required for modes a, b
    @state_operands_wrapper(operands_arg_index=2, is_single_operand=True, transpose=True)
    @nvmath_utils.precondition(_check_valid_network)
    def apply_tensor_operator(self, modes, operand, *, control_modes=None, control_values=None, immutable=False, adjoint=False, unitary=None, diagonal=False, gradient=None, stream=None):
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
            unitary : Whether the operator is unitary. With ``None`` (default), unitarity is determined
                numerically from the operand at this call (and again at update time when
                :meth:`NetworkState.update_tensor_operator` is called with ``unitary=None``) and the
                determined value is used. Pass an explicit ``True``/``False`` to skip the check;
                the explicit value is taken as the caller's assertion. With ``control_modes``,
                the operand is the target block and the controlled operator is unitary exactly
                when its target block is; classification and update semantics are the same as
                for a plain operator.
            diagonal : Whether the operator is diagonal (default `False`).
            gradient : Whether to register this operator for expectation gradients.
                If ``None`` (default), for PyTorch operands the choice follows ``operand.tensor.requires_grad``;
                for other backends, gradient registration is disabled unless ``gradient`` is explicitly ``True``.
                Use ``True`` / ``False`` to force registration on or off regardless of ``requires_grad``.
                Gradients are not supported for MPS simulation (see :meth:`compute_expectation_with_gradients`).
            stream : Provide the CUDA stream to use for applying the tensor operator (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
        
        Returns:
            An integer `tensor_id` specifying the location of the input operator.

        Notes:
            - For MPS simulation, the size of ``modes`` shall be restricted to no larger than 2 (two-body operator).
        """
        # An empty control set denotes a plain (non-controlled) operator.
        if control_modes is not None and len(control_modes) == 0:
            if control_values is not None and len(control_values) > 0:
                raise ValueError(
                    f"control values ({control_values}) were provided without any control modes; "
                    "pass matching control_modes or omit control_values")
            control_modes = None
            control_values = None
        if isinstance(self.config, MPSConfig) and len(operand.shape) > 4:
            raise ValueError(f"MPS simulation only supports one-body and two-body operators, found operator dimension ({len(operand.shape)})")
        if gradient is None:
            if self.backend == "torch":
                use_gradient = operand.tensor.requires_grad
            else:
                use_gradient = False
        else:
            assert isinstance(gradient, bool)
            use_gradient = gradient
        if use_gradient and (control_modes is not None or diagonal):
            raise ValueError(
                "Gradient registration is only supported for non-controlled, non-diagonal apply_tensor_operator."
            )
        # unitary=None has been resolved to a concrete bool by resolve_unitary_kwarg
        # on the raw native operand, before any operand wrapping or device transfer
        if control_modes is None:
            if diagonal:
                tensor_id = cutn.state_apply_diagonal_tensor_operator(self.handle, self.state, len(modes), 
                    modes, operand.data_ptr, operand.strides, immutable, adjoint, unitary)
                self.logger.debug(f"The diagonal tensor operand has been applied to the state with an ID ({tensor_id}).")
            elif use_gradient:
                # Gradient uses same shape/strides as operand.
                tensor_id = cutn.state_apply_tensor_operator_with_gradient(
                    self.handle, self.state, len(modes), modes,
                    operand.data_ptr, operand.strides, immutable, adjoint, unitary,
                    operand.data_ptr, operand.strides)
                self.logger.debug(f"The tensor operand has been applied to the state with gradient registration (tensor_id={tensor_id}).")
                self._gradient_tensor_ids[tensor_id] = operand
            else:
                tensor_id = cutn.state_apply_tensor_operator(self.handle, self.state, len(modes), 
                    modes, operand.data_ptr, operand.strides, immutable, adjoint, unitary)
                self.logger.debug(f"The tensor operand has been applied to the state with an ID ({tensor_id}).")
        else:
            # Default control_values to 1 for all control modes if not provided (per documentation)
            if control_values is None:
                control_values = tuple(1 for _ in control_modes)
            tensor_id = cutn.state_apply_controlled_tensor_operator(self.handle, self.state, len(control_modes), 
                control_modes, control_values, len(modes), modes, operand.data_ptr, operand.strides, immutable, adjoint, unitary)
            self.logger.debug(f"The controlled tensor operand has been applied to the state with an ID ({tensor_id}).")
        
        assert tensor_id not in self.operands, "Internal Error"
        
        # keep operand alive otherwise cupy will re-use the memory space
        self.operands[tensor_id] = operand, immutable
        # remember the unitarity so update_tensor_operator can detect flag changes,
        # and whether the operand is diagonal so a later numeric classification
        # at update time applies the matching check
        self._operator_unitarity[tensor_id] = bool(unitary)
        if control_modes is None and diagonal:
            self._operator_diagonal.add(tensor_id)
        # reset norm / state vector
        self._mark_updated()
        return tensor_id

    def gradient_tensor_ids(self):
        """
        Tensor operator IDs registered for gradient computation via
        :meth:`apply_tensor_operator`.

        Returns:
            A tuple of integers in ascending ``tensor_id`` order. When the library
            assigns monotonically increasing IDs (the usual case), this matches the
            order of :meth:`apply_tensor_operator` calls that registered gradients.
        """
        return tuple(sorted(self._gradient_tensor_ids))

    @state_labels_wrapper(marker_index=1, marker_type='seq')
    # operand indices (b, a, B, A) required for modes a, b
    @state_operands_wrapper(operands_arg_index=2, is_single_operand=False, transpose=True)
    @nvmath_utils.precondition(_check_valid_network)
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
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
        
        Returns:
            An integer `channel_id` specifying the location of the unitary channel.

        Notes:
            - For MPS simulation, the size of ``modes`` shall be restricted to no larger than 2 (two-body operator).
        """
        if len(operands) != len(probabilities):
            raise ValueError(f"The number of operands ({len(operands)}) does not matching the size of probabilities ({len(probabilities)})")
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
    @state_operands_wrapper(operands_arg_index=2, is_single_operand=False, transpose=True)
    @nvmath_utils.precondition(_check_valid_network)
    def apply_general_tensor_channel(self, modes, operands, *, stream=None):
        """
        Apply a noise channel to the network state. The noise operators may be non-unitary.
        For a more efficient unitary tensor channel application, see
        :meth:`NetworkState.apply_unitary_tensor_channel`.

        Args:
            modes : A sequence of integers denoting the modes where the tensor operator acts on. 
                If ``state_labels`` has been provided during initialization, ``modes`` can also be provided as a sequence of labels. 
            operands : A sequence of ndarray-like objects for the tensor operators defining the channel.
                The modes of the operand is expected to be ordered as ``ABC...abc...``, 
                where ``ABC...`` denotes output bra modes and ``abc...`` denotes input ket modes corresponding to ``modes`` 
            stream : Provide the CUDA stream to use for applying the tensor operator (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
        
        Returns:
            An integer `channel_id` specifying the location of the general channel.

        Notes:
            - This method requires the input channel to be trace-preserving.
              Supplying a non-trace-preserving channel may lead to unexpected results.
            - For pure states with TNConfig, general channels are not supported. Use MPSConfig
              or switch to mixed state simulation (``pure_state=False``).
            - For MPS simulation, the size of ``modes`` shall be restricted to no larger than 2 (two-body operator).
            - The ``channel_id`` cannot be used to update the channel using
              :meth:`NetworkState.update_tensor_operator`.
        """
        # operand indices (b, a, B, A) required for modes a, b
        if isinstance(self.config, TNConfig) and self.is_pure:
            raise ValueError(f"Noise simulation using general channel with TNConfig is only supported for mixed states. "
                             f"Set pure_state=False or use MPSConfig for pure state noise simulation.")
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
    
    @resolve_unitary_kwarg(operand_arg_index=2, resolver_name='_resolve_update_unitarity')
    # operand indices (b, a, B, A) required for modes a, b
    @state_operands_wrapper(operands_arg_index=2, is_single_operand=True, transpose=True)
    @nvmath_utils.precondition(_check_valid_network)
    def update_tensor_operator(self, tensor_id, operand, *, unitary=None, stream=None):
        """
        Update a tensor operator in the state.

        Args:
            tensor_id : An integer specifing the tensor id assigned in :meth:`NetworkState.apply_tensor_operator`.
            operand : A ndarray-like object for the tensor operator.
                The operand is expected to follow the same mode ordering, data type and strides as the original operand.
            unitary : Whether the operator is unitary. With ``None`` (default), unitarity is determined
                numerically from the new operand at this call and the determined value is used.
                Pass an explicit ``True``/``False`` to skip the check; the explicit value is taken as the
                caller's assertion.
                A determined-or-explicit flag change relative to the recorded classification of a plain
                or controlled tensor operator is a structural update: internal preparation and plans
                are rebuilt, which is more expensive than a data-only update with an unchanged flag.
                For a controlled operator the update supplies its target block, and the controlled
                operator is unitary exactly when its target block is. Only MPO, network-operator and
                channel operators keep the classification recorded at application.
            stream : Provide the CUDA stream to use for updating tensor operand (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
        """
        if tensor_id not in self.operands:
            raise RuntimeError(f"operator with ID ({tensor_id}) has not been applied to the network state.")
        prev_operand, immutable = self.operands[tensor_id]
        if immutable:
            raise RuntimeError(f"tensor id ({tensor_id}) has been marked immutable.")
        if operand.strides != prev_operand.strides:
            raise ValueError(f'The new operand must share the same strides as the original operand ({prev_operand.strides[::-1]}), found ({operand.strides[::-1]})')
        # unitary=None has been resolved to a concrete bool by resolve_unitary_kwarg
        # on the raw native operand, before any operand wrapping or device transfer
        cutn.state_update_tensor_operator(self.handle, self.state, tensor_id, operand.data_ptr, unitary)
        self.operands[tensor_id] = operand, immutable
        # A unitarity flag change on a plain or controlled tensor operator is a
        # structural update: the flag feeds structural decisions frozen into
        # prepared plans and cached derived-property task objects (e.g. the
        # gauge-refresh placement), so those must be rebuilt.
        structural = False
        if tensor_id in self._operator_unitarity and self._operator_unitarity[tensor_id] != bool(unitary):
            self._operator_unitarity[tensor_id] = bool(unitary)
            structural = True
        # If this operator has gradient registered, set gradient output to current operand ptr; compute_expectation_with_gradients will set the correct buffers internally when it runs.
        if tensor_id in self._gradient_tensor_ids:
            cutn.state_update_tensor_operator_gradient(self.handle, self.state, tensor_id, operand.data_ptr)
            self._gradient_tensor_ids[tensor_id] = operand
        self.logger.info(f"Tensor operand with ID ({tensor_id}) has been updated.")
        self._mark_updated(structural=structural)
        return


    @nvmath_utils.precondition(_check_valid_network)
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
            raise ValueError(f"The dimension for the state ({self.state_mode_extents}) not matching that of the network operator ({network_operator.state_mode_extents})")
        if network_operator.dtype != self.dtype:
            raise ValueError(f"Input network operator data type ({network_operator.dtype}) different from network state ({self.dtype})")
        if not self.backend_setup:
            operand = network_operator._get_representative_operand()
            if operand is None:
                raise RuntimeError("Empty NetworkOperator can not be applied to a NetworkState")
            self._setup_backend(operand)
        network_id = cutn.state_apply_network_operator(self.handle, 
            self.state, network_operator.network_operator, immutable, adjoint, unitary)
        self.non_owned_network_operators[network_id] = network_operator
        self.logger.info(f"Network operator has been applied to the state with an ID ({network_id})")
        # reset norm / state vector
        self._mark_updated()
        return network_id


    @state_labels_wrapper(marker_index=1, marker_type='seq')
    @nvmath_utils.precondition(_check_valid_network)
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
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
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
            # specify the largest output MPS tensors' sizes
            self._output_mps_max_extents = get_mps_output_extents(
                self.state_mode_extents, self.config.max_extent, np.dtype(self.dtype).itemsize)
            self._allocate_output_mps(stream, self._output_mps_max_extents)
            output_mps_strides = [tensor.strides for tensor in self.mps_tensors]
            cutn.state_finalize_mps(self.handle, self.state,
                cutn.BoundaryCondition.OPEN, self._output_mps_max_extents, output_mps_strides)
            self.logger.info(f"The target MPS state has been set.")
        self.target_state_set = True

    def _allocate_output_mps(self, stream, extents):
        stream_holder = nvmath_utils.get_or_create_stream(self.device_id, stream, self.internal_package)
        # keep the tensors only once all are allocated, so a failed allocation releases the earlier ones
        self.mps_tensors = [self.intermediate_class.empty(
            e, device_id=self.device_id, dtype=self.dtype, stream_holder=stream_holder) for e in extents]

    def _maybe_compute_state(self, stream, release_workspace, *, capture_follows=False):
        """
        ``capture_follows`` indicates that the computed output state becomes the new initial state right after this computation.
        """
        self._maybe_set_target_state(stream)
        if not self.state_computed and isinstance(self.config, MPSConfig):
            if self._output_holds_initial_state and not capture_follows:
                # The output tensors hold the captured initial state, which every later computation reads:
                # compute into new output tensors (the captured ones are released at the next capture).
                self._allocate_output_mps(stream, self._output_mps_max_extents)
                self._output_holds_initial_state = False
            # Otherwise, if this computation is captured right away (capture_follows), it is written in place into the
            # tensors that hold its own initial state, so that capturing after every computation needs no additional
            # memory. The C API does not guarantee that a computation may overwrite its own initial MPS;
            # test_mps_capture_every_computation_in_place covers this.
            create_args = ()
            execute_args = (self.workspace_desc, [o.data_ptr for o in self.mps_tensors])
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
                        update_tensor_extents_strides(self.mps_tensors[i], extent_out, strides[i])
        # mark state as computed
        self.state_computed = True
        return self.state_computed
    
    def _release_workspace_after_failure(self, exception=None):
        """
        Release the workspace memory allocated by a failed computation and reset the allocation tracking, so that the
        object can be used for later computations.
        """
        try:
            self._release_scratch_memory_perhaps(exception)
            self._release_cache_memory_perhaps(exception)
        except Exception as e:
            # keep the original exception
            self.logger.debug(f"Releasing the workspace after a failed computation failed: {e}")
        finally:
            self._reset_workspace_allocation_tracking()
        return True

    @nvmath_utils.atomic(_free_task_object_resources, method=True)
    @nvmath_utils.atomic(_release_workspace_after_failure, method=True)
    def _compute_target(self, task, create_args, execute_args, stream, release_workspace, *, config_args=None, caller_name=None, task_key=None):
        if caller_name is None:
            caller_name = task
        if task not in ('marginal', 'marginal_diagonal', 'expectation', 'accessor', 'state', 'sampler', 'expectation_with_gradients'):
            raise ValueError("only supports marginal, marginal_diagonal, sampler, accessor, expectation, expectation_with_gradients and state")

        cutn_task = self._get_cutn_task_name(task)

        if task == 'state':
            create_func = None
        else:
            # avoid going into infinite loops
            if not self._maybe_compute_state(stream, release_workspace):
                raise ValueError
            if task == 'marginal_diagonal':
                create_func = cutn.create_marginal_diagonal
            else:
                create_func = getattr(cutn, f'create_{cutn_task}')
        # Allocate device memory (in stream context) if needed.
        stream_holder = nvmath_utils.get_or_create_stream(self.device_id, stream, self.internal_package)
        prepare_func = getattr(cutn, f'{cutn_task}_prepare')

        if task == 'expectation_with_gradients':
            execute_func = None  # execute step handled by _execute_expectation_with_gradients
        elif task == 'sampler':
            execute_func = cutn.sampler_sample
        else:
            execute_func = getattr(cutn, f'{cutn_task}_compute')
        
        self.logger.info(f"Beginning {caller_name} computation...")

        prepare_needed = True
        if task == 'state': # state vector or MPS computation
            task_obj = self.state
            prepare_needed = not self.state_prepared
        else:
            if task_key is None:
                task_key = (task, create_args)
            if task_key in self.cached_task_obj and (self.property_prepare_reusable or
                                                     self._get_current_key() == self._cached_task_state_key):
                # In the following cases, re-creation/preparation of property computation objects can be skipped:
                # 1. Contraction based simulation  -> self.property_prepare_reusable
                # 2. MPS without value based truncation -> self.property_prepare_reusable
                # 3. MPS with value based truncation whose computed MPS has the extents it had when the cached object
                #    was created (not only those of the previous computation, which may have changed them back and
                #    forth, e.g. with stochastic channels) -> compare with the key recorded at creation
                task_obj = self.cached_task_obj[task_key]
                self.logger.info(f"Found the same {task} object from the cache")
                prepare_needed = False
            else:
                # make sure when the current stream waits until the last compute event is done and we only cache one compute task object
                if self.cached_task_obj:
                    stream_holder.obj.wait(self.last_compute_event)
                    self.last_compute_event = None
                self._free_task_object_resources()
                self.cached_task_obj[task_key] = task_obj = create_func(self.handle, self.state, *create_args)
                self._cached_task_state_key = self._get_current_key()
                self.logger.info(f"A new {task} object has been created")
        
        if config_args is not None:
            configure_func = getattr(cutn, f'{cutn_task}_configure')
            configure_func(self.handle, task_obj, *config_args)
            # new prepare step needed if the object has be re-configured
            prepare_needed = True
        
        timing =  bool(self.logger and self.logger.handlers)
        if prepare_needed:
            self.logger.info(f"Starting preparing {caller_name} computation with blocking set to {self.blocking}...")
            with nvmath_utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
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
                # For mixed states the projection space is [0, 2N) (independent ket/bra).
                full_projection = self.n if self.is_pure else 2 * self.n
                if num_fixed_modes == 0:
                    # state vector & batched_amplitudes both support computing the full state vector
                    equivalent_callers_set.remove('amplitude')
                elif num_fixed_modes == full_projection:
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
            info_flops_enum = getattr(cutn, f'{cutn_task.capitalize()}Attribute').INFO_FLOPS
            flops_dtype = getattr(cutn, f'{cutn_task}_get_attribute_dtype')(info_flops_enum)
            flops = np.zeros(1, dtype=flops_dtype)
            getattr(cutn, f'{cutn_task}_get_info')(self.handle, task_obj, info_flops_enum, flops.ctypes.data, flops.dtype.itemsize)
            self.logger.info(f"Total flop count for {caller_name} computation = {flops.item()/1e9} GFlop")

        self.logger.info(f"Starting {caller_name} computation with blocking set to {self.blocking}...")
        with nvmath_utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
            if task == 'expectation_with_gradients':
                output = self._execute_expectation_with_gradients(stream_holder, task_obj)
            else:
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
    
    def _translate_label_dict(self, d):
        """Translate a possibly-labelled dict (state-label keys) into integer-keyed form."""
        if not d:
            return {}
        if self.state_labels is None:
            return dict(d)
        if all(isinstance(k, int) for k in d.keys()):
            return dict(d)
        return {self.state_labels.index(k): v for k, v in d.items()}

    def _translate_label_seq(self, seq):
        """Translate a possibly-labelled sequence of mode labels into integer indices."""
        if seq is None:
            return None
        if self.state_labels is None or all(isinstance(item, int) for item in seq):
            return list(seq)
        return [self.state_labels.index(item) for item in seq]

    def _build_accessor_projection(self, fixed_ket, fixed_bra=None):
        """Translate user-facing ket/bra projection dicts into the C-side projection list and output shape.

        For pure states only ``fixed_ket`` is consulted; the projection space is ``[0, N)``.
        For mixed states both are consulted; the projection space is ``[0, 2N)`` with ``[0, N)``
        indexing ket modes and ``[N, 2N)`` indexing bra modes.

        Output mode order is ket-open-asc followed by bra-open-asc (matching the C contract for the mixed-state Accessor).

        Returns:
            (num_projected, projected_modes, projected_values, output_shape)
        """
        n = self.n
        fixed_ket = dict(fixed_ket) if fixed_ket else {}
        for k in fixed_ket:
            if not (isinstance(k, int) and 0 <= k < n):
                raise ValueError(f"ket projection key {k!r} out of range [0, {n})")
        if not self.is_pure:
            fixed_bra = dict(fixed_bra) if fixed_bra else {}
            for k in fixed_bra:
                if not (isinstance(k, int) and 0 <= k < n):
                    raise ValueError(f"bra projection key {k!r} out of range [0, {n})")
            combined = {k: int(v) for k, v in fixed_ket.items()}
            combined.update({k + n: int(v) for k, v in fixed_bra.items()})
            open_ket = [q for q in range(n) if q not in fixed_ket]
            open_bra = [q for q in range(n) if q not in fixed_bra]
            output_shape = ([self.state_mode_extents[q] for q in open_ket]
                            + [self.state_mode_extents[q] for q in open_bra])
        else:
            if fixed_bra:
                raise TypeError("fixed_bra is only supported for pure_state=False")
            combined = {k: int(v) for k, v in fixed_ket.items()}
            open_modes = [q for q in range(n) if q not in fixed_ket]
            output_shape = [self.state_mode_extents[q] for q in open_modes]

        if combined:
            sorted_modes, sorted_values = zip(*sorted(combined.items()))
            num_projected = len(sorted_modes)
            projected_modes = tuple(sorted_modes)
            projected_values = tuple(sorted_values)
        else:
            num_projected = 0
            projected_modes = 0
            projected_values = 0
        return num_projected, projected_modes, projected_values, output_shape

    def _run_state_accessor(self, caller_name, return_norm, *, fixed_modes=None, fixed_bra=None, stream=None, release_workspace=False):
        num_projected, projected_modes, projected_values, shape = self._build_accessor_projection(
            fixed_modes, fixed_bra)

        stream_holder = nvmath_utils.get_or_create_stream(self.device_id, stream, self.internal_package)
        amplitudes = self.intermediate_class.empty(
            shape, device_id=self.device_id, dtype=self.dtype, stream_holder=stream_holder)

        norm = np.empty((), dtype=self.dtype)

        create_args = (num_projected, projected_modes, tuple(amplitudes.strides))
        if return_norm:
            execute_args = (projected_values, self.workspace_desc, amplitudes.data_ptr, norm.ctypes.data)
        else:
            execute_args = (projected_values, self.workspace_desc, amplitudes.data_ptr, 0)
        self._compute_target('accessor', create_args, execute_args, stream, release_workspace, caller_name=caller_name)
        if return_norm:
            norm_out = host_scalar_to_holder(
                self.output_class, self.device_id, self.dtype, norm, stream_holder
            )
            return amplitudes, norm_out
        return amplitudes
    
    @state_result_wrapper()
    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "Amplitude computation")
    def compute_amplitude(self, bitstring, *, return_norm=False, stream=None, release_workspace=False):
        """
        Compute a single element of the underlying state tensor.

        For pure states, returns the complex amplitude :math:`\\langle \\text{bitstring} | \\psi \\rangle`.
        For mixed states, returns the density-matrix element
        :math:`\\langle \\text{ket\\_bitstring} | \\rho | \\text{bra\\_bitstring} \\rangle`.

        Args:
            bitstring : 
                * For pure states, a length-``N`` sequence of integers specifying the desired
                  basis state.
                * For mixed states, either a single length-``N`` sequence -- interpreted
                  symmetrically as ``ket == bra`` to extract a diagonal element (probability) --
                  or a 2-tuple ``(ket_bitstring, bra_bitstring)`` of length-``N`` sequences
                  specifying the row (ket) and column (bra) indices of the density matrix element
                  to compute.

            return_norm : If true, the squared norm of the state will also be returned.
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
        
        Returns:
            If ``return_norm`` is ``False``, a 0-D tensor for the bitstring amplitude on the
            state's backend (the density-matrix element
            :math:`\\langle \\text{ket} | \\rho | \\text{bra} \\rangle` for mixed states).
            If ``return_norm`` is ``True``, a 2-tuple ``(amplitude, norm)`` where ``norm`` is a
            0-D tensor for the squared state norm :math:`\\mathrm{Tr}(\\rho)` (equal to
            :math:`\\langle \\psi | \\psi \\rangle` for pure states).
        """
        n = self.n
        fixed_ket, fixed_bra = self._parse_bitstring_argument(bitstring, n)
        return self._run_state_accessor('amplitude', return_norm,
                                        fixed_modes=fixed_ket, fixed_bra=fixed_bra,
                                        stream=stream, release_workspace=release_workspace)

    def _parse_bitstring_argument(self, bitstring, n):
        """Parse a ``bitstring`` argument into (fixed_ket, fixed_bra) projection dicts.

        For pure states ``bitstring`` must be a length-N sequence of bits; ``fixed_bra`` is ``None``.
        For mixed states ``bitstring`` is either a single length-N sequence (symmetric ``ket == bra``,
        a diagonal element) or a 2-tuple of length-N sequences ``(ket, bra)`` for an off-diagonal
        element, mirroring the underlying mixed-state Accessor C contract.
        """
        if not self.is_pure:
            bs_ket, bs_bra = split_mixed_bitstring(bitstring, n)
            for label, bs in (("ket_bitstring", bs_ket), ("bra_bitstring", bs_bra)):
                if not isinstance(bs, collections.abc.Sequence):
                    raise TypeError(f"`{label}` must be a sequence of bits (e.g. a str or tuple/list)")
                if len(bs) != n:
                    raise ValueError(f"`{label}` must have length {n}, got {len(bs)}")
            fixed_ket = {i: int(b) for i, b in enumerate(bs_ket)}
            fixed_bra = {i: int(b) for i, b in enumerate(bs_bra)}
            return fixed_ket, fixed_bra
        # pure
        if isinstance(bitstring, tuple) and len(bitstring) == 2 and all(
                isinstance(x, collections.abc.Sequence) and not isinstance(x, (str, bytes)) for x in bitstring):
            raise TypeError("the (ket_bitstring, bra_bitstring) tuple form is only valid for "
                            "pure_state=False; pass a single length-N bitstring for pure states")
        if len(bitstring) != n:
            raise ValueError(f"Length of bitstring is expected to match the dimension of the "
                             f"underlying state ({n}), found ({len(bitstring)})")
        fixed_ket = {i: int(b) for i, b in enumerate(bitstring)}
        return fixed_ket, None
    
    @state_labels_wrapper(marker_index=1, marker_type='dict')
    @state_result_wrapper()
    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "Batched amplitude computation")
    def compute_batched_amplitudes(self, fixed, *, return_norm=False, stream=None, release_workspace=False):
        """
        Compute a slice of the underlying state tensor.

        For pure states, returns a slice of the state vector:
        :math:`\\langle \\text{bs} | \\psi \\rangle` over open ket modes.

        For mixed states, returns a slice of the density matrix with the requested ket and bra
        modes projected independently. Output modes are ordered as
        ``(open_ket_modes_asc, open_bra_modes_asc)`` to match the underlying mixed-state Accessor.

        Args:    
            fixed : A dictionary mapping a subset of mode indices ``[0, N)`` to fixed values; modes
                absent from the dictionary are left open. If ``state_labels`` has been provided
                during initialization, keys may also be provided as labels.

                * For pure states, each value is a single fixed index.
                * For mixed states, each value selects the ket and bra index for that mode and may
                  be either

                  - a single index -- shorthand for fixing the ket and bra to the same value (a
                    diagonal/symmetric projection), or
                  - a 2-tuple ``(ket, bra)`` where each entry is an index or ``None``; ``None``
                    leaves that side open, enabling independent ket/bra projection
                    (e.g. ``(0, None)`` fixes only the ket).
            return_norm : If true, the squared norm of the state will also be returned.
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
        
        Returns:
            If ``return_norm`` is `False`, an ndarray-like slice of the state tensor. The package and
            storage location of the ndarray will be the same as the operands provided in
            :meth:`apply_tensor_operator`, :meth:`apply_mpo` and :meth:`set_initial_mps`. Otherwise, a
            2-tuple consisting of the slice and a scalar for the squared norm of the state
            :math:`\\mathrm{Tr}(\\rho)` (equal to :math:`\\langle \\psi | \\psi \\rangle` for pure states).
        """
        fixed_ket, fixed_bra = self._parse_fixed_argument(fixed)
        return self._run_state_accessor('batched_amplitudes', return_norm,
                                        fixed_modes=fixed_ket, fixed_bra=fixed_bra,
                                        stream=stream, release_workspace=release_workspace)

    def _parse_fixed_argument(self, fixed):
        """Parse a ``fixed`` argument into (fixed_ket, fixed_bra) projection dicts.

        For pure states ``fixed`` must be a dict (possibly with labelled keys) mapping modes to a
        single fixed value; ``fixed_bra`` is ``None``.

        For mixed states ``fixed`` is a single dict whose values select the ket and bra index for
        each mode: a scalar fixes both sides to the same value (diagonal projection), while a
        2-tuple ``(ket, bra)`` projects the two sides independently (``None`` leaves that side
        open). This is translated into the ``(fixed_ket, fixed_bra)`` dicts that mirror the
        underlying mixed-state Accessor C contract.
        """
        if not self.is_pure:
            if not isinstance(fixed, collections.abc.Mapping):
                raise TypeError("for pure_state=False, `fixed` must be a dict mapping mode indices "
                                "(or labels) to a single value or a (ket, bra) 2-tuple")
            fixed_ket_in, fixed_bra_in = split_mixed_fixed(fixed)
            return self._translate_label_dict(fixed_ket_in), self._translate_label_dict(fixed_bra_in)
        if isinstance(fixed, tuple):
            raise TypeError("the (fixed_ket, fixed_bra) tuple form is no longer used; pass a single "
                            "dict (pure states accept only scalar values)")
        if not isinstance(fixed, collections.abc.Mapping):
            raise TypeError("`fixed` must be a dict mapping mode indices (or labels) to fixed values")
        return self._translate_label_dict(fixed), None
    
    @state_result_wrapper()
    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "State vector computation")
    def compute_state_vector(self, *, return_norm=False, stream=None, release_workspace=False):
        """
        Compute the state vector.

        This method is only valid for pure states. For mixed states, use
        :meth:`compute_density_matrix` to obtain the full density matrix.

        Args:
            return_norm : If true, the squared norm of the state will also be returned.
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.

        Returns:
            If ``return_norm`` is ``False``, an ndarray-like state vector on the state's backend. The package and storage location of the ndarray will be the same as 
            the operands provided in :meth:`apply_tensor_operator`, :meth:`apply_mpo` and :meth:`set_initial_mps`. If ``return_norm`` is ``True``, a 2-tuple
            ``(state_vector, norm)`` where ``norm`` is a 0-D tensor for the squared state
            2-norm.
        """
        if not self.is_pure:
            raise TypeError(
                "compute_state_vector() is not defined for mixed states; use "
                "compute_density_matrix() to obtain the full density matrix.")
        return self._run_state_accessor('state vector', return_norm, fixed_modes={}, stream=stream, release_workspace=release_workspace)

    def compute_density_matrix(self, *, stream=None, release_workspace=False):
        """
        Compute the full density matrix.

        This is supported for both pure and mixed states. For a pure state this returns
        :math:`\\rho = |\\psi\\rangle\\langle\\psi|`; :meth:`compute_state_vector` is the cheaper
        rank-N alternative in that case.

        Args:
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.

        Returns:
            An ndarray-like density matrix of shape ``(d_0, ..., d_{N-1}, d_0, ..., d_{N-1})`` where the
            first N modes are ket (row) indices and the last N modes are bra (column) indices.
        """
        return self.compute_reduced_density_matrix(
            tuple(range(self.n)), stream=stream, release_workspace=release_workspace)

    @state_labels_wrapper(marker_index=1, marker_type='seq')
    @state_labels_wrapper(key='fixed', marker_type='dict')
    @state_result_wrapper()
    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "Reduced density matrix computation")
    def compute_reduced_density_matrix(self, where, *, fixed=EMPTY_DICT, diagonal=False, stream=None, release_workspace=False):
        r"""
        Compute the reduced density matrix (or, optionally, only its diagonal) for the given marginal and fixed modes.

        Args:
            where : A sequence of integers for the target modes. 
                If ``state_labels`` has been provided during initialization, ``where`` can also be provided as a sequence of labels. 
            fixed : A dictionary mapping a subset of fixed modes to the fixed value. 
                If ``state_labels`` has been provided during initialization, ``fixed`` can also be provided as a dictionary mapping labels to the corresponding fixed values. 
            diagonal : If ``False`` (default), the full reduced density matrix is returned as a rank-``2 * len(where)`` tensor.
                If ``True``, only its diagonal is returned as a rank-``len(where)`` tensor, which avoids materializing the full
                reduced density matrix (backed by ``cutensornetCreateMarginalDiagonal``). These are the bare diagonal entries
                :math:`\rho_{\text{where}}[\text{bs}, \text{bs}]` of the (projected) reduced density matrix; like the full
                reduced density matrix, they are not renormalized.
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
        
        Returns:
            An ndarray-like object. By default, the reduced density matrix following the modes ``AB...ab...`` where ``AB...`` and
            ``ab...`` represent the corresponding output and input marginal modes. If ``diagonal=True``, the rank-``len(where)``
            diagonal of the reduced density matrix following the modes ``AB...``.
        """
        if fixed and not set(fixed).isdisjoint(where):
            raise ValueError(
                "`fixed` modes must be disjoint from the `where` modes; got overlapping "
                f"modes {sorted(set(fixed).intersection(where))}.")
        n_marginal_modes = len(where)
        if fixed:
            n_projected_modes = len(fixed)
            projected_modes, projected_mode_values = zip(*sorted(fixed.items()))
            projected_modes = tuple(projected_modes)
            projected_mode_values = tuple([int(i) for i in projected_mode_values])
        else:
            n_projected_modes = projected_modes = projected_mode_values = 0
        out_shape = [self.state_mode_extents[q] for q in where]
        if not diagonal:
            out_shape = out_shape * 2

        stream_holder = nvmath_utils.get_or_create_stream(self.device_id, stream, self.internal_package)
        out = self.intermediate_class.empty(
            out_shape, device_id=self.device_id, dtype=self.dtype, stream_holder=stream_holder)
        create_args = (n_marginal_modes, tuple(where), n_projected_modes, projected_modes, tuple(out.strides))
        execute_args = (projected_mode_values, self.workspace_desc, out.data_ptr)
        self._compute_target('marginal_diagonal' if diagonal else 'marginal', create_args, execute_args, stream, release_workspace)
        return out

    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "Output state")
    def compute_output_state(self, *, stream=None, release_workspace=False, release_operators=False):
        """
        Compute the final output state for the underlying network state object. This method currently is only valid for MPS based simulation.

        Args:
            stream : Provide the CUDA stream to use for the computation. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
            release_operators : A value of `True` makes the computed output state the new initial state of the underlying :class:`NetworkState` object
                and releases the reference of all underlying tensor operators and :class:`NetworkOperator` objects.
                The previous ``tensor_id`` returned by :meth:`apply_tensor_operator`, :meth:`apply_network_operator` and :meth:`apply_mpo` will be invalid.
                Operators applied afterwards act on this new initial state.
                If the output state has already been computed, which is an intermediate step in other ``compute_xxx`` methods, the output state will be cached and returned directly
                (unless the state contains stochastic channels, which are sampled again by every computation).
                The default is `False`.
    
        Returns:
            For MPS simulation (``MPSConfig`` in ``options``), a sequence of ndarray-like MPS tensors on
            the state's backend (respecting ``output_location``). Each tensor uses mode order ``pkn``:
            ``p`` (left bond), ``k`` (ket), ``n`` (right bond); the first and last tensors omit ``p``
            and ``n`` respectively.

        Note:
            Unless ``output_location`` is ``'cpu'`` (the returned tensors are then copies owned by the caller), the returned tensors
            are owned by this object: they must not be modified and are valid until the state is computed again (also inside other
            ``compute_*`` methods after the state changed), which may overwrite them.
            Copy them to keep them.
        """
        if isinstance(self.config, TNConfig):
            raise ValueError("For network contraction based state simulation, no explicit state representation is formed, use compute_state_vector for full state vector computation")
        elif isinstance(self.config, MPSConfig):
            self._maybe_compute_state(stream, release_workspace, capture_follows=release_operators)
            if self.output_location == 'cpu':
                stream_holder = nvmath_utils.get_or_create_stream(self.device_id, stream, self.internal_package)
                result = [o.to('cpu', stream_holder=stream_holder).tensor for o in self.mps_tensors]
            else:
                result = [o.tensor for o in self.mps_tensors]
            if release_operators:
                # event synchronization
                if self.last_compute_event is not None:
                    stream_holder = nvmath_utils.get_or_create_stream(self.device_id, stream, self.internal_package)
                    stream_holder.obj.wait(self.last_compute_event)
                    self.last_compute_event = None
                # release reference to underlying operators and NetworkOperator
                cutn.state_capture_mps(self.handle, self.state)
                self.operands = {}
                self._gradient_tensor_ids = {}
                self._operator_unitarity = {}
                self._operator_diagonal = set()
                self.owned_network_operators = {}
                self.non_owned_network_operators = {}
                self.initial_state = list(self.mps_tensors)
                # the output tensors now hold the initial state (see _maybe_compute_state)
                self._output_holds_initial_state = True
                # mark state as no longer computed & stochastic channels as resolved
                self.contains_stochastic_channels = False
                self._mark_updated()
            return result
        else:
            raise NotImplementedError()
    
    @state_labels_wrapper(key='modes', marker_type='seq')
    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "Sampling")
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
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
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
        for sample, n_sampling in zip(*np.unique(samples, axis=0, return_counts=True)):
            width = max((len(str(int(bit))) for bit in sample), default=1)
            bitstring = ''.join(str(int(bit)).rjust(width) for bit in sample)
            sampling[bitstring] = n_sampling
        return sampling
    
    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "Expectation computation")
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
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cuda.core.Stream` for NumPy operands, 
                :class:`cupy.cuda.Stream` for CuPy operands, and :class:`torch.cuda.Stream` for PyTorch operands. 
                If a stream is not provided, the current stream will be used.
            release_workspace : A value of `True` specifies that the state object should release workspace memory back to
                the package memory pool on function return, while a value of `False` specifies that the state object
                should retain the memory. This option may be set to `True` if the application performs other operations that consume
                a lot of memory between successive calls to the (same or different) execution API such as :meth:`compute_sampling`,
                :meth:`compute_reduced_density_matrix`, :meth:`compute_amplitude`, :meth:`compute_batched_amplitudes`, or :meth:`compute_expectation`, 
                but incurs a small overhead due to obtaining and releasing workspace memory from and to the package memory pool on every call. 
                The default is `False`.
            
        Returns:
            If ``return_norm`` is ``False``, a 0-D tensor for the expectation value on the state's
            backend. If ``return_norm`` is ``True``, a 2-tuple
            ``(expectation, norm)`` of 0-D tensors, where ``norm`` is the squared state norm
            :math:`\\mathrm{Tr}(\\rho)` (equal to :math:`\\langle \\psi | \\psi \\rangle` for pure
            states). On the PyTorch backend, outputs participate in the autograd graph when
            gradient-registered gate tensors have ``requires_grad=True``; otherwise forward-only
            0-D tensors are returned.
        
        Note:
            - Repeated calls with the same Pauli convenience input (same string keys and coefficients given the state's
              dtype) reuse a memoized :class:`NetworkOperator` so CUTN expectation ``prepare`` can be amortized
              similarly to explicitly reusing one operator instance. The memo is invalidated when the state's cached
              property objects are invalidated (see :meth:`_mark_updated`) and when :meth:`free` runs.
            - For workloads that assemble :class:`NetworkOperator` outside the convenience path,
              see the `variational expectation example <https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/tensornet/experimental/network_state/generic_states/example04_variational_expectation.py>`_.
            - For Pauli workloads that mutate coefficients every call, constructing a fresh :class:`NetworkOperator` may remain appropriate.
            - For pauli operator expectation value computations, this method does not take advantage of lightcone simplification optimization. 
              If user wishes to compute the expectation value on a pauli string operator with many identities in it, consider either using the :meth:`compute_reduced_density_matrix` method or 
              explicitly construct the :class:`NetworkOperator` object with :meth:`NetworkOperator.append_product` for optimal performance.
        """
        if self.backend == "torch":
            from ._internal.grad_torch import _TorchExpectation

            tensor_ids = self.gradient_tensor_ids()
            trainable_tensors = tuple(self._gradient_tensor_ids[tid].tensor for tid in tensor_ids)
            operators = self._ensure_network_operator(operators, stream)
            return _TorchExpectation.apply(
                self,
                operators,
                stream,
                release_workspace,
                return_norm,
                tensor_ids,
                *trainable_tensors,
            )
        return self._compute_expectation(
            operators,
            return_norm=return_norm,
            stream=stream,
            release_workspace=release_workspace,
        )

    @state_result_wrapper()
    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "Expectation computation")
    def _compute_expectation(
        self, operators, *, return_norm=False, stream=None, release_workspace=False
    ):
        """Run CUTN expectation; returns backend 0-D tensor(s)."""
        stream_holder = nvmath_utils.get_or_create_stream(
            self.device_id, stream, self.internal_package
        )
        operators = self._ensure_network_operator(operators, stream)
        expectation_value = np.empty((), dtype=self.dtype)
        norm = np.empty((), dtype=self.dtype)
        create_args = (operators.network_operator,)
        if return_norm:
            execute_args = (self.workspace_desc, expectation_value.ctypes.data, norm.ctypes.data)
        else:
            execute_args = (self.workspace_desc, expectation_value.ctypes.data, 0)
        task_key = ("expectation", operators._get_key())
        self._compute_target(
            "expectation",
            create_args,
            execute_args,
            stream,
            release_workspace,
            task_key=task_key,
        )
        if return_norm:
            return (
                host_scalar_to_holder(
                    self.output_class, self.device_id, self.dtype, expectation_value, stream_holder
                ),
                host_scalar_to_holder(
                    self.output_class, self.device_id, self.dtype, norm, stream_holder
                ),
            )
        return host_scalar_to_holder(
            self.output_class, self.device_id, self.dtype, expectation_value, stream_holder
        )

    def _ensure_network_operator(self, operators, stream):
        """
        Ensure ``operators`` is a :class:`NetworkOperator` (Pauli string / dict / object).
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
            operators = self._get_or_build_cached_pauli_network_operator(operators, stream=stream)
            self.owned_network_operators[None] = operators
        assert isinstance(operators, NetworkOperator)
        if tuple(self.state_mode_extents) != tuple(operators.state_mode_extents):
            raise ValueError(f"The dimension for the state ({self.state_mode_extents}) not matching that of the network operator ({operators.state_mode_extents})")
        return operators

    def _execute_expectation_with_gradients(self, stream_holder, task_obj):
        """
        Execute expectation forward + backward and fill gradient buffers.
        Called by _compute_target when expectation_with_gradients is True.
        Expects self._expectation_grad_run_context to be set to
        (expectation_value, expectation_value_adjoint_arr, return_norm, norm, state_norm_adjoint_arr).
        """
        expectation_value, expectation_value_adjoint_arr, return_norm, norm, state_norm_adjoint_arr = self._expectation_grad_run_context
        gradients_out = {}
        # Allocate gradient tensors with same shape/strides as operands. 
        if self.internal_package == "cuda":
            for tensor_id in self.gradient_tensor_ids():
                operand = self._gradient_tensor_ids[tensor_id]
                ext, strides = operand.shape, operand.strides
                grad = tensor_wrapper._TENSOR_TYPES["numpy"].empty(
                    ext, device_id="cpu", dtype=self.dtype, stream_holder=None, strides=strides)
                grad = grad.to(self.device_id, stream_holder)
                gradients_out[tensor_id] = grad
        else:
            for tensor_id in self.gradient_tensor_ids():
                operand = self._gradient_tensor_ids[tensor_id]
                ext, strides = operand.shape, operand.strides
                grad = self.intermediate_class.empty(
                    ext, device_id=self.device_id, dtype=self.dtype, stream_holder=stream_holder, strides=strides)
                gradients_out[tensor_id] = grad

        for tensor_id, grad in gradients_out.items():
            cutn.state_update_tensor_operator_gradient(self.handle, self.state, tensor_id, grad.data_ptr)

        cutn.expectation_compute_with_gradients_backward(
            self.handle,
            task_obj,
            0,  # overwrite gradients (no accumulation)
            expectation_value_adjoint_arr.ctypes.data,
            state_norm_adjoint_arr.ctypes.data if state_norm_adjoint_arr is not None else 0,
            self.workspace_desc,
            expectation_value.ctypes.data,
            norm.ctypes.data if return_norm else 0,
            stream_holder.ptr,
        )
        if self.backend == "numpy" or self.output_location == 'cpu':
            return {tensor_id: grad.to('cpu', stream_holder=stream_holder).tensor
                    for tensor_id, grad in gradients_out.items()}
        return {tensor_id: grad.tensor for tensor_id, grad in gradients_out.items()}

    @nvmath_utils.precondition(_maybe_setup_recompute)
    @nvmath_utils.precondition(_check_valid_network)
    @nvmath_utils.precondition(_check_backend_setup, "Expectation with gradients computation")
    def compute_expectation_with_gradients(
        self, operators, expectation_value_adjoint, *, return_norm=False, state_norm_adjoint=None, stream=None, release_workspace=False
    ):
        """
        Compute the expectation value of the given operator(s) and gradients with respect to tensor operators
        that were registered for gradients in :meth:`apply_tensor_operator` (PyTorch: typically when the
        operand had ``requires_grad=True``, or when ``gradient=True`` was passed explicitly).

        Args:
            operators : Pauli string, dict of pauli strings, or :class:`NetworkOperator`.
            return_norm : If ``True``, the squared norm of the state is also computed and returned (see :meth:`compute_expectation`).
                Must be ``True`` together with a non-``None`` ``state_norm_adjoint``, or ``False`` together with
                ``state_norm_adjoint=None``.
            expectation_value_adjoint : Scalar adjoint for the expectation value (e.g. 1.0 for d/d(expectation)).
            state_norm_adjoint : Upstream gradient for the squared state 2-norm (scalar in ``self.dtype``).
                Must be non-``None`` exactly when ``return_norm`` is ``True``, and ``None`` when ``return_norm`` is ``False``.
            stream : Execution stream (see :meth:`compute_expectation`).
            release_workspace : Whether to release workspace memory on return (see :meth:`compute_expectation`).

        Returns:
            If ``return_norm`` is ``False``, ``(expectation_value, gradients)`` where
            ``expectation_value`` is a 0-D tensor on the state's backend. If ``return_norm`` is ``True``, ``(expectation_value, norm,
            gradients)`` where ``norm`` is a 0-D tensor for the squared state 2-norm in the same
            form. ``gradients`` maps each ``tensor_id`` from :meth:`gradient_tensor_ids` to a
            tensor with the same shape, dtype as the corresponding registered operand.
            If no operators were registered for gradients, ``gradients`` is empty.

        Note:
            Gradients are not supported for MPS simulation (``MPSConfig`` in ``options``): if operators were registered for
            gradients, this method raises ``cuquantum.bindings.cutensornet.cuTensorNetError`` with status
            ``CUTENSORNET_STATUS_NOT_SUPPORTED``. Expectation values can
            still be computed with :meth:`compute_expectation`.
        """
        check_expectation_with_gradients_norm_args(return_norm, state_norm_adjoint)
        operators = self._ensure_network_operator(operators, stream)
        expectation_value = np.empty((), dtype=self.dtype)
        expectation_value_adjoint_arr = np.asarray(expectation_value_adjoint, dtype=self.dtype)
        state_norm_adjoint_arr = np.asarray(state_norm_adjoint, dtype=self.dtype) if state_norm_adjoint is not None else None
        norm = np.empty((), dtype=self.dtype) if return_norm else None
        self._expectation_grad_run_context = (expectation_value, expectation_value_adjoint_arr, return_norm, norm, state_norm_adjoint_arr)
        try:
            create_args = (operators.network_operator,)
            task_key = ('expectation_with_gradients', operators._get_key())
            gradients_out = self._compute_target(
                'expectation_with_gradients',
                create_args,
                None,
                stream,
                release_workspace,
                task_key=task_key
            )
        finally:
            del self._expectation_grad_run_context
        stream_holder = nvmath_utils.get_or_create_stream(
            self.device_id, stream, self.internal_package
        )
        exp_out = unwrap_output_tensor(
            self,
            host_scalar_to_holder(
                self.output_class, self.device_id, self.dtype, expectation_value, stream_holder
            ),
            stream_holder,
        )
        if return_norm:
            norm_out = unwrap_output_tensor(
                self,
                host_scalar_to_holder(
                    self.output_class, self.device_id, self.dtype, norm, stream_holder
                ),
                stream_holder,
            )
            return exp_out, norm_out, gradients_out
        return exp_out, gradients_out

