# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = ['NetworkOperator']

import logging

from cuquantum.bindings import cutensornet as cutn


from ._internal.network_state_utils import STATE_DEFAULT_DTYPE, check_dtype_supported, create_pauli_operands, state_operands_wrapper
from .. import configuration
from ..._internal import utils, tensor_wrapper
from ..._internal.typemaps import NAME_TO_DATA_TYPE


class NetworkOperator:
    """
    Create a tensor network operator object.

    Args:
        state_mode_extents : A sequence of integers specifying the dimension for all state modes.
        dtype : A string specifying the datatype for the tensor network operator, currently supports the following data types:
            
                - ``'float32'``
                - ``'float64'``
                - ``'complex64'``
                - ``'complex128'`` (default)
        
        options : Specify options for the state computation as a :class:`~cuquantum.NetworkOptions` object. 
            Alternatively, a `dict` containing the parameters for the ``NetworkOptions`` constructor can also be provided. 
            If not specified, the value will be set to the default-constructed ``NetworkOptions`` object.
    """

    def __init__(self, state_mode_extents, dtype=STATE_DEFAULT_DTYPE, options=None):
        """
        __init__(state_mode_extents, dtype='complex128', options=None)
        """
        options = utils.check_or_create_options(configuration.NetworkOptions, options, "network options")
        options._check_compatible_with_state()
        self.options = options

        self.device_id = options.device_id

        # Logger.
        self.logger = options.logger if options.logger is not None else logging.getLogger()
        self.logger.info("Beginning network operator creation...")

        self.state_mode_extents = state_mode_extents
        check_dtype_supported(dtype)
        self.dtype = dtype
        self.cuda_dtype = NAME_TO_DATA_TYPE[dtype]

        # Create/set handle.
        if options.handle is not None:
            self.own_handle = False
            self.handle = options.handle
        else:
            self.own_handle = True
            with utils.device_ctx(self.device_id):
                self.handle = cutn.create()
        
        self.network_operator = cutn.create_network_operator(self.handle, 
            len(self.state_mode_extents), self.state_mode_extents, self.cuda_dtype)
        
        # internal setup to be determined later
        self.internal_package = None
        self.intermediate_class = None
        self.backend = None
        self.backend_setup = False
        
        self.tensor_products = []
        self.mpos = []

        self.valid_state = True
        self.destroy_func = {'handle': cutn.destroy, 
                             'network_operator': cutn.destroy_network_operator}
        self.logger.info("The network operator has been created.")

    def _check_backend_setup(self, *args, **kwargs):
        what = kwargs['what']
        if not self.backend_setup:
            raise RuntimeError(f"{what} cannot be performed before operands has been appended.")
    
    def _setup_backend(self, operand):
        # backend setup should only be performed once
        assert not self.backend_setup, "Internal Error"
        self.backend = operand.name
        if self.backend == 'numpy':
            self.intermediate_class = tensor_wrapper.CupyTensor
            self.internal_package = 'cupy'
        else:
            self.intermediate_class = operand.__class__
            self.internal_package=  self.backend
        self.backend_setup = True
    
    def _get_key(self):
        return (self.network_operator, len(self.tensor_products), len(self.mpos))

    def __del__(self):
        """Free tensor network operator resources."""
        if not self.valid_state:
            return

        # cleanup references
        self.tensor_products = []
        self.mpos = []

        try:
            if self.network_operator is not None:
                self.destroy_func['network_operator'](self.network_operator)
                self.network_operator = None

            if self.handle is not None and self.own_handle:
                self.destroy_func['handle'](self.handle)
                self.handle = None
                self.own_handle = False
        except Exception as e:
            self.logger.critical("Internal error: only part of the network operator resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The network operator resources have been released.")
    

    @state_operands_wrapper(operands_arg_index=3, is_single_operand=False)
    def append_product(self, coefficient, modes, tensors, stream=None):
        """Append a tensor product component to the tensor network operator.

        Args:
            coefficient : A scalar for the cofficient of the full tensor product.
            modes : A nested sequence of integers denoting the modes where each operand in ``tensors`` acts on.
            tensors : A sequence of tensors (ndarray-like objects) for each tensor operand. 
                The currently supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.
                The modes of each operand are expected to be ordered as ``ABC...abc...``, 
                where ``ABC...`` denotes output bra modes and ``abc...`` denotes input ket modes corresponding to ``modes``.
            stream : Provide the CUDA stream to use for appending tensor product (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        """
        # Note that cutensornet.network_operator_append_product requires mode ordering b, a, B, A, 
        # which is different from our standard notation. Therefore we need to transpose it here
        tensors = [o.T for o in tensors]
        num_tensors = len(tensors)
        assert num_tensors == len(modes)
        
        num_modes = []
        tensor_mode_strides = []
        tensor_data = []
    
        for mode, operand in zip(modes, tensors):
            assert len(mode) * 2 == operand.ndim, f"operator dimension must be twice the size of the mode length"
            num_modes.append(len(mode))
            tensor_mode_strides.append(operand.strides)
            tensor_data.append(operand.data_ptr)
        # keep a reference
        self.tensor_products.append((tensors, modes, coefficient))
        cutn.network_operator_append_product(self.handle, 
                self.network_operator, coefficient, num_tensors, 
                num_modes, modes, tensor_mode_strides, tensor_data)
        self.logger.info(f"Tensor product operators with coeff {coefficient} acting on {modes} have been appended.")
        return
    
    @state_operands_wrapper(operands_arg_index=3, is_single_operand=False)
    def append_mpo(self, coefficient, modes, mpo_tensors, stream=None):
        """Append a matrix product operator (MPO) component to the tensor network operator.

        Args:
            coefficient : A scalar for the cofficient of the full tensor product.
            modes : A sequence of integers specifying each mode where the MPO acts on.
            mpo_tensors : A sequence of tensors (ndarray-like objects) for each MPO operand. 
                The currently supported types are :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`.
                The mode of each operand is expected to follow the order of ``pknb`` where ``p`` denotes 
                the mode connecting to the previous MPO tensor, ``n`` denotes the mode connecting to the 
                next MPO tensor, ``k`` denotes the ket mode and ``b`` denotes the bra mode. 
                Note that currently only MPO with open boundary condition is supported, 
                therefore ``p`` and ``n`` mode should not be present in the first and last MPO tensor respectively. 
                Note that the relative order of bra and ket modes here differs from that of ``operand`` in :meth:`append_product`.
            stream : Provide the CUDA stream to use for appending tensor product (this is used to copy the operands to the GPU if they are provided on the CPU). 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        """
        num_tensors = len(mpo_tensors)
        assert num_tensors == len(modes)
        
        tensor_mode_extents = []
        tensor_mode_strides = []
        tensor_data = []
        for operand in mpo_tensors:
            tensor_mode_extents.append(operand.shape)
            tensor_mode_strides.append(operand.strides)
            tensor_data.append(operand.data_ptr)
        # keep a reference
        self.mpos.append((mpo_tensors, modes, coefficient))
        cutn.network_operator_append_mpo(self.handle, 
            self.network_operator, coefficient, num_tensors, 
            modes, tensor_mode_extents, tensor_mode_strides, tensor_data, cutn.BoundaryCondition.OPEN)
        self.logger.info(f"MPO with coeff {coefficient} acting on {modes} have been appended.")
        return    
    

    @classmethod
    def from_pauli_strings(cls, pauli_strings, dtype=STATE_DEFAULT_DTYPE, backend='cupy', options=None, stream=None):
        """
        Generate a tensor network operator object from given pauli strings.

        Args:
            pauli_strings : A dictionary that maps pauli strings to corresponding coefficients. Alternative can be a single pauli string.
            dtype : A string specifying the datatype for the tensor network operator, currently supports the following data types:
            
                - ``'complex64'``
                - ``'complex128'`` (default)

            backend : A string specifying the ndarray backend for the device tensor operands. Currently supports ``'cupy'`` (default) and ``'torch'``.
            options : Specify options for the state computation as a :class:`~cuquantum.NetworkOptions` object. 
                Alternatively, a `dict` containing the parameters for the ``NetworkOptions`` constructor can also be provided. 
                If not specified, the value will be set to the default-constructed ``NetworkOptions`` object.
            stream : Provide the CUDA stream to use for tensor network operator construction, which is needed for stream-ordered operations such as allocating memory. 
                Acceptable inputs include ``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. 
                If a stream is not provided, the current stream will be used.
        """
        if dtype not in {'complex64', 'complex128'}:
            raise ValueError(f"dtype expected to be complex64/complex128, found {dtype}")
        options = utils.check_or_create_options(configuration.NetworkOptions, options, "network options")
        if isinstance(pauli_strings, str):
            pauli_strings = {pauli_strings: 1}
        if not isinstance(pauli_strings, dict):
            raise ValueError(f"pauli_strings must be either a dictionary mapping pauli strings to its coefficient or a single pauli string")

        n_qubits = len(list(pauli_strings.keys())[0])
        state_mode_extents = (2, ) * n_qubits
        operator_obj = cls(state_mode_extents, dtype=dtype, options=options)
        operands_data = create_pauli_operands(pauli_strings, dtype, backend, options.device_id, stream=stream)
        
        for tensors, modes, coefficient in operands_data:
            operator_obj.append_product(coefficient, modes, tensors, stream=stream)
        return operator_obj
