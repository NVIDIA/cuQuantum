# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tensor network contraction and decomposition.
"""

__all__ = ['contract_decompose']

import dataclasses
import logging

from .configuration import ContractDecomposeAlgorithm, ContractDecomposeInfo
from ._internal.utils import is_gate_split, maybe_truncate_qr_output_operands
from ...bindings import cutensornet as cutn
from ..configuration import NetworkOptions
from ..tensor import decompose, SVDInfo
from ..tensor_network import contract
from .._internal import decomposition_utils
from .._internal import einsum_parser
from ..._internal import tensor_wrapper
from ..._internal import utils 



def _gate_split(wrapped_operands, inputs, outputs, size_dict, max_mid_extent, algorithm, options, stream, return_info):
    """
    perform gate split operation by calling ``cutensornetGateSplit``

    Args:
        wrapped_operands : Thin wrapped tensors given the original input operands (not copied to device yet for Numpy ndarrays).
        inputs : A sequence of modes for input tensors in "neutral format" (sequence of sequences).
        outputs : A sequence of modes for output tensors in "neutral format" (sequence of sequences).
        size_dict : A dictionary mapping the modes to the extent.
        algorithm : A ``ContractDecomposeAlgorithm`` object specifying the algorithm for the gate split operation.
        options : Specify options for the operation as a :class:`~cuquantum.NetworkOptions` object.
        max_mid_extent : The maximal mid extent (reduced) expected for the output of the operation.
        stream : Provide the CUDA stream to use for the autotuning operation. Acceptable inputs include ``cudaStream_t``
            (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
            the current stream will be used.
        return_info : If true, information about the contraction and decomposition will also be returned as a ::class:`ContractDecomposeInfo` object.
    """
    # placeholder to help avoid resource leak
    handle = workspace_desc = svd_config = svd_info = None
    input_tensor_descriptors = output_tensor_descriptors = []
    workspaces = dict()
    own_handle = False
    try:
        # Options converted to an internal option
        wrapped_operands, options, own_handle, operands_location, stream_holder = decomposition_utils.parse_decompose_operands_options(
            options, wrapped_operands, stream, allowed_dtype_names=decomposition_utils.DECOMPOSITION_DTYPE_NAMES)
        
        mid_extent = max_mid_extent if algorithm.svd_method.max_extent is None else min(max_mid_extent, algorithm.svd_method.max_extent)

        handle = options.handle
        stream_ptr = stream_holder.ptr  # this exists as we always use the ExternalStream from CuPy internally...

        options.logger.info("Calling specicialized kernel `cutensornetGateSplit` for contraction and decomposition.")
        
        # Create input/output tensor descriptors and empty output operands 
        input_tensor_descriptors, output_operands, output_tensor_descriptors, s, s_ptr = decomposition_utils.create_operands_and_descriptors(
                                    handle, wrapped_operands, size_dict, inputs, outputs, 
                                    mid_extent, algorithm.svd_method, options.device_id, stream_holder, options.logger)

        # Parse SVDConfig
        svd_config = cutn.create_tensor_svd_config(handle)
        decomposition_utils.parse_svd_config(handle, svd_config, algorithm.svd_method, options.logger)

        # Infer GateSplitAlgorithm
        gate_algorithm = cutn.GateSplitAlgo.DIRECT if algorithm.qr_method is False else cutn.GateSplitAlgo.REDUCED

        # Create workspace descriptor
        workspace_desc = cutn.create_workspace_descriptor(handle)
        workspace_ptr = None

        options.logger.debug("Querying workspace size...")
        
        cutn.workspace_compute_gate_split_sizes(handle, 
            *input_tensor_descriptors, *output_tensor_descriptors, 
            gate_algorithm, svd_config, options.compute_type, workspace_desc) 
        
        # Allocate and set workspace
        for mem_space in (cutn.Memspace.DEVICE, cutn.Memspace.HOST):
            pref = cutn.WorksizePref.MIN
            workspace_kind = cutn.WorkspaceKind.SCRATCH
            workspaces[mem_space] = decomposition_utils.allocate_and_set_workspace(options, workspace_desc, 
                    pref, mem_space, workspace_kind, stream_holder, task_name='contract decomposition')

        options.logger.info("Starting contract-decompose (gate split)...")
        timing =  bool(options.logger and options.logger.handlers)
        blocking = options.blocking is True or operands_location == 'cpu'
        if blocking:
            options.logger.info("This call is blocking and will return only after the operation is complete.")
        else:
            options.logger.info("This call is non-blocking and will return immediately after the operation is launched on the device.")

        svd_info = cutn.create_tensor_svd_info(handle)
        with utils.device_ctx(options.device_id), utils.cuda_call_ctx(stream_holder, blocking, timing) as (last_compute_event, elapsed):
            cutn.gate_split(handle, 
                input_tensor_descriptors[0], wrapped_operands[0].data_ptr, 
                input_tensor_descriptors[1], wrapped_operands[1].data_ptr, 
                input_tensor_descriptors[2], wrapped_operands[2].data_ptr, 
                output_tensor_descriptors[0], output_operands[0].data_ptr,
                s_ptr,
                output_tensor_descriptors[1], output_operands[1].data_ptr,
                gate_algorithm, 
                svd_config, 
                options.compute_type,
                svd_info, 
                workspace_desc, 
                stream_ptr)

        if elapsed.data is not None:
            options.logger.info(f"The contract-decompose (gate split) operation took {elapsed.data:.3f} ms to complete.")
            
        svd_info_obj = SVDInfo(**decomposition_utils.get_svd_info_dict(handle, svd_info))

        # Update the operand to reduced_extent if needed
        for (wrapped_tensor, tensor_desc) in zip(output_operands, output_tensor_descriptors):
            wrapped_tensor.reshape_to_match_tensor_descriptor(handle, tensor_desc)
            
        reduced_extent = svd_info_obj.reduced_extent
        if s is not None:
            if reduced_extent != mid_extent:
                s.tensor = s.tensor[:reduced_extent]
    finally:
        # when host workspace is allocated, synchronize stream before return
        if workspaces.get(cutn.Memspace.HOST) is not None:
            stream_holder.obj.synchronize()
        # Free resources
        decomposition_utils._destroy_tensor_descriptors(input_tensor_descriptors)
        decomposition_utils._destroy_tensor_descriptors(output_tensor_descriptors)
        if svd_config is not None:
            cutn.destroy_tensor_svd_config(svd_config)
        if svd_info is not None:
            cutn.destroy_tensor_svd_info(svd_info)
        if workspace_desc is not None:
            cutn.destroy_workspace_descriptor(workspace_desc)

        if own_handle and handle is not None:
            cutn.destroy(handle)

    u, v, s = [decomposition_utils.get_return_operand_data(t, operands_location, stream_holder) for t in output_operands + [s, ]]
    
    if return_info:
        info = ContractDecomposeInfo(qr_method=algorithm.qr_method,
                                    svd_method=algorithm.svd_method,
                                    svd_info=svd_info_obj)
        return u, s, v, info
    else:
        return u, s, v


def contract_decompose(subscripts, *operands, algorithm=None, options=None, optimize=None, stream=None, return_info=False):
    r"""
    Evaluate the compound expression for contraction and decomposition on the input operands. 

    The expression follows a combination of Einstein summation notation for contraction and the decomposition notation for decomposition
    (as in :func:`~cuquantum.cutensornet.tensor.decompose`). 
    The input represents a tensor network that will be contracted to form an intermediate tensor for subsequent decomposition operation, 
    yielding two or three outputs depending on the final decomposition method. 
    The expression requires explicit specification of modes for the input and output tensors (excluding ``S`` for SVD method). 
    The modes for the intermediate tensor are inferred based on the subscripts representing the output modes by using the implicit form of
    the Einstein summation expression (similar to the treatment in ``numpy.einsum`` implicit mode).

    See the notes and examples for clarification.

    Args:
        subscripts : The mode labels (subscripts) defining the contraction and decomposition operation as a comma-separated sequence of
            characters. Unicode characters are allowed in the expression thereby expanding the size of the tensor network that
            can be specified.
        algorithm : Specify the algorithm to perform the contraction and decomposition. Alternatively, 
            a `dict` containing the parameters for the :class:`ContractDecomposeAlgorithm` constructor can be provided. 
            If not specified, the value will be set to the default-constructed ``ContractDecomposeAlgorithm`` object.
        operands : A sequence of tensors (ndarray-like objects). The currently supported types are :class:`numpy.ndarray`,
            :class:`cupy.ndarray`, and :class:`torch.Tensor`.
        options : Specify options for the tensor network as a :class:`~cuquantum.NetworkOptions` object. Alternatively, a `dict`
            containing the parameters for the ``NetworkOptions`` constructor can also be provided. If not specified,
            the value will be set to the default-constructed ``NetworkOptions`` object.
        optimize :  This parameter specifies options for path optimization as an :class:`~cuquantum.OptimizerOptions` object. Alternatively, a
            dictionary containing the parameters for the ``OptimizerOptions`` constructor can also be provided. If not
            specified, the value will be set to the default-constructed ``OptimizerOptions`` object.
        stream: Provide the CUDA stream to use for the autotuning operation. Acceptable inputs include ``cudaStream_t``
            (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
            the current stream will be used.
        return_info : If true, information about the contraction and decomposition will also be returned as a ::class:`ContractDecomposeInfo` object.

    Returns:
        Depending on the decomposition setting specified in ``algorithm``, the results returned may vary:

            - For QR decomposition (default), if ``return_info`` is `False`, the output tensors Q and R (ndarray-like objects) of the same type 
              and on the same device as the input operand are returned as the result of the decomposition. If ``return_info`` is `True`, 
              a 3-tuple of output tensors Q, R and a `ContractDecomposeInfo` object that contains information about the operations will be returned. 
            - For SVD decomposition, if ``return_info`` is `False`, a 3-tuple of output tensors U, S and V (ndarray-like objects) 
              of the same type as the input operand are returned as the result of the decomposition. If ``return_info`` is `True`, 
              a 4-tuple of output tensors U, S, V and a `ContractDecomposeInfo` object that contains information about the operations will be returned. 
              Note, depending on the choice of :attr:`~ContractDecomposeAlgorithm.svd_method.partition`, the returned S operand may be `None`. 
              Also see :attr:`~SVDMethod.partition`. 
    
    Raises:
        :class:`MemoryLimitExceeded`: the memory needed to perform the operation is larger than the ``options.memory_limit``

    The contract and decompose expression adopts a combination of Einstein summation notation for contraction and the decomposition notation
    introduced in :func:`~cuquantum.cutensornet.tensor.decompose`.
    The ``subscripts`` string is a list of subscript labels where each label refers to a mode of the corresponding operand. 
    The subscript labels are separated by either comma or identifier ``->``. 
    The subscript labels before the identifier ``->`` are viewed as inputs, and the ones after are viewed as outputs, respectively.
    The requirements on the subscripts for SVD and QR decomposition are summarized below:

        - For SVD and QR decomposition, the subscripts string is expected to contain more than one input and exactly two output labels (The modes for `S` is not needed in the case of SVD). 
        - One and only one identical mode is expected to exist in the two output mode labels. 
        - The modes for the intermediate tensor which will be decomposed are inferred based on the subscripts representing the output modes by using the implicit form of
          the Einstein summation expression (similar to the treatment in ``numpy.einsum`` implicit mode).
          Therefore, assembling the input modes and the intermediate modes together should result in a valid ``numpy.einsum`` expression (classical or generalized).
 
    Examples:

        >>> # equivalent:
        >>> # q, r = numpy.linalg.qr(numpy.einsum('ij,jk->ik', a, b))
        >>> q, r = contract_decompose('ij,jk->ix,xk', a, b)

        >>> # equivalent:
        >>> # u, s, v = numpy.linalg.svd(numpy.einsum('ij,jk->ik', a, b), full_matrices=False)
        >>> u, s, v = tensor_decompose('ij,jk->ix,xk', a, algorithm={'qr_method':False, 'svd_method': {}})

        For generalization to generic tensor network with multi-dimensional tensors (``a``, ``b``, ``c`` are all rank-4 tensors).
        In this case, the intermediate modes ``ijabe`` is inferred from the output modes ``ixeb`` and ``jax``:

        >>> # equivalent:
        >>> # t = contract('ijc,cad,dbe->ijabe', a, b, c)
        >>> # u, s, v = tensor.decompose('ijabe->ixeb,jax', t, method=SVDMethod())
        >>> u, s, v = contract_decompose('ijc,cad,dbe->ixeb,jax', a, b, c, algorithm={'qr_method': False, 'svd_method': {}})

    If the contract and decompose problem amounts to a **ternary-operand gate split problem** commonly seen in quantum circuit simulation 
    (see :ref:`Gate Split Algorithm<gatesplitalgo>` for details), 
    the user may be able to take advantage of optimized kernels from `cutensornetGateSplit` by placing the gate operand as the last one in the input operands. 
    In this case, QR decomposition can potentially be used to speed up the execution of contraction and SVD. 
    This can be achieved by setting both :attr:`~ContractDecomposeAlgorithm.qr_method` and both :attr:`~ContractDecomposeAlgorithm.svd_method`,
    as demonstrated below.
    
    Example:

        Applying a two-qubit gate to adjacent MPS tensors:

        >>> a, _, b = contract_decompose('ipj,jqk,pqPQ->iPx,xQk', a, b, gate, algorithm={'qr_method':{}, 'svd_method':{}})

    **Broadcasting** is supported for certain cases via ellipsis notation. 
    One may add ellipses in the input modes to represent all the modes that are not explicitly specified in the labels. 
    In such case, an ellipsis is allowed to appear in at most one of the output modes. If an ellipsis appears in one of the output modes, 
    the implicit modes are partitioned onto the corresponding output. If no ellipsis is found in the output, the implicit modes will be summed over 
    to construct the intermediate tensors.
    
    Examples:

        Below are some examples based on two rank-4 tensors ``a`` and ``b``:

        >>> # equivalent:
        >>> # out = contract_decompose('ijab,abcd->ijx,xcd', a, b)
        >>> out = contract_decompose('ijab,ab...->ijx,x...', a, b)  # intermediate modes being "ijcd"

        >>> # equivalent:
        >>> # out = contract_decompose('ijab,abcd->ix,xj', a, b)
        >>> out = contract_decompose('ijab,ab...->ix,xj', a, b)  # intermediate modes being "ij"

        >>> # equivalent:
        >>> # out = contract_decompose('ijab,jkab->ix,xj', a, b)
        >>> out = contract_decompose('ij...,jk...->ix,xj', a, b)  # intermediate modes being "ij"

        >>> # equivalent:
        >>> # out = contract_decompose('ijab,jkab->ixab,xj', a, b)
        >>> out = contract_decompose('ij...,jk...->ix...,xj', a, b)  # intermediate modes being "ijab"

    Note that the number of modes that are implicitly represented by the ellipses must be the same for all occurrences.

    .. note::
        It is encouraged for users to maintain the library handle themselves so as to reduce the context initialization time:

        .. code-block:: python

            from cuquantum.bindings import cutensornet as cutn
            from cuquantum.cutensornet.experimental import contract_decompose

            handle = cutn.create()
            q, r = contract_decompose(..., options={"handle": handle}, ...)
            # ... the same handle can be reused for further calls ...
            # when it's done, remember to destroy the handle
            cutn.destroy(handle)
    """

    algorithm = utils.check_or_create_options(ContractDecomposeAlgorithm, algorithm, "Contract Decompose Algorithm")
    options = utils.check_or_create_options(NetworkOptions, options, "Network Options")

    # Get cuTensorNet version (as seen at run-time)
    cutn_ver = cutn.get_version()
    cutn_major = cutn_ver // 10000
    cutn_minor = (cutn_ver % 10000) // 100
    cutn_patch = cutn_ver % 100

    # Logger
    logger = logging.getLogger() if options.logger is None else options.logger
    logger.info(f"cuTensorNet version = {cutn_major}.{cutn_minor}.{cutn_patch}")
    logger.info("Beginning operands parsing...")
    
    # Parse subscipts and operands
    wrapped_operands, inputs, outputs, size_dict, mode_map_user_to_ord, mode_map_ord_to_user, max_mid_extent = decomposition_utils.parse_decomposition(subscripts, *operands)

    if is_gate_split(inputs, outputs, algorithm):
        # dedicated kernel for GateSplit problem
        return _gate_split(wrapped_operands, inputs, outputs, size_dict, max_mid_extent, algorithm, options, stream, return_info)

    own_handle = False
    try:
        # contraction followed by decomposition
        wrapped_operands, options, own_handle, operands_location, stream_holder = decomposition_utils.parse_decompose_operands_options(
            options, wrapped_operands, stream, allowed_dtype_names=decomposition_utils.DECOMPOSITION_DTYPE_NAMES)
        
        intermediate_modes = einsum_parser.infer_output_mode_labels(outputs)

        intermediate_labels = []
        ellipses_processed = False
        for _modes in intermediate_modes:
            m = mode_map_ord_to_user[_modes]
            if m.startswith('__'): # extra internal modes represented by ellipses
                if not ellipses_processed:
                    m = '...'
                    ellipses_processed = True
                else:
                    continue
            intermediate_labels.append(m)
        intermediate_labels = ''.join(intermediate_labels)
        
        input_modes, output_modes = subscripts.split('->')
        einsum_subscripts = f"{input_modes}->{intermediate_labels}"
        decompose_subscripts = f"{intermediate_labels}->{output_modes}" 
        
        if operands_location == 'cpu':
            # avoid double transfer
            operands = [o.tensor for o in wrapped_operands]
        
        info_dict = {'svd_method': algorithm.svd_method,
                    'qr_method': algorithm.qr_method}
        logger.info("Beginning contraction of the input tensor network...")
        intm_output = contract(einsum_subscripts, *operands, options=options, optimize=optimize, stream=stream, return_info=return_info)
        logger.info("Contraction of the input tensor network is completed.")
        if return_info:
            intm_output, (_, info_dict['optimizer_info']) = intm_output    # Discard the path as it's part of optimizer_info.
        
        #NOTE: The direct integration here is based on splitting the contract_decompose problem into two sub-problems
        #       - 1. contraction. 
        #       - 2. decomposition.
        # If the algorithm is naively applied, one may not find the optimal reduce extent, for example:
        # A[x,y] B[y,z] with input extent x=4, y=2, z=4 -> contract QR decompose -> A[x,k]B[k,z] . 
        # When naively applying the direct algorithm above, the mid extent k in the output will be 4 (QR on a 4x4 matrix).
        # For contract QR decomposition, we manually slice the extents in the outputs.
        # For contract SVD decomposition, we inject max_extent as part of the internal SVDMethod.

        logger.info("Beginning decomposition of the intermediate tensor...")
        decompose_options = dataclasses.asdict(options)
        decompose_options['compute_type'] = None
        if algorithm.qr_method and algorithm.svd_method is False:
            # contract and QR decompose
            results = decompose(
                decompose_subscripts, intm_output, method=algorithm.qr_method, options=decompose_options,
                stream=stream, return_info=False)
            results = maybe_truncate_qr_output_operands(results, outputs, max_mid_extent)
            if operands_location == 'cpu':
                results = [tensor_wrapper.wrap_operand(o).to('cpu', stream_holder=stream_holder) for o in results]
        elif algorithm.svd_method and algorithm.qr_method is False:
            # contract and SVD decompose
            
            use_max_mid_extent = algorithm.svd_method.max_extent is None
            if use_max_mid_extent:
                algorithm.svd_method.max_extent = max_mid_extent
            results = decompose(
                decompose_subscripts, intm_output, method=algorithm.svd_method, options=decompose_options,
                stream=stream, return_info=return_info)
            if use_max_mid_extent:
                # revert back
                algorithm.svd_method.max_extent = None

            if return_info:
                results, info_dict['svd_info'] = results[:-1], results[-1]
            if operands_location == 'cpu':
                results = [o if o is None else tensor_wrapper.wrap_operand(o).to(
                               'cpu', stream_holder=stream_holder)
                           for o in results]
        else:
            raise NotImplementedError("contract_decompose currently doesn't support QR assisted SVD contract decomposition for more than 3 operands")
        logger.info("Decomposition of the intermediate tensor is completed.")
    finally:
        if own_handle and options.handle is not None:
            cutn.destroy(options.handle)
    
    if not return_info:
        return results
    else:
        return *results, ContractDecomposeInfo(**info_dict)
