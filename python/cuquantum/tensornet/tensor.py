# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Computational primitives for tensors
"""

__all__ = ['decompose', 'DecompositionOptions', 'QRMethod', 'SVDInfo', 'SVDMethod']

from dataclasses import dataclass
import logging
import re
from typing import Optional

import numpy

from ..bindings import cutensornet as cutn
from .configuration import NetworkOptions
from ._internal import decomposition_utils
from .._internal import utils 


@dataclass
class DecompositionOptions(NetworkOptions):
    def __post_init__(self):
        super().__post_init__()
        if self.compute_type is not None:
            raise ValueError("The compute_type for decomposition should be None.")
DecompositionOptions.__doc__ = re.sub(":class:`cuquantum.Network` object", ":func:`cuquantum.cutensornet.tensor.decompose` and :func:`cuquantum.cutensornet.experimental.contract_decompose` functions", NetworkOptions.__doc__)


def decompose(
    subscripts, 
    operand, 
    *,
    method=None,
    options=None,
    stream=None,
    return_info=False
):
    r"""

    Perform the tensor decomposition of the operand based on the expression described by ``subscripts``.
    
    The expression adopts a similar notation as Einstein summation (einsum) expression, but in the reversed order.
    Meanwhile the input is now a single operand while the output contains two or three tensors. 
    Unlike einsum expression, the mode labels for all input and output operands must be specified *explicitly*.

    See the notes and examples for clarification. See also :ref:`Tensor Decomposition<approximatedTN>`.

    Args:
        subscripts : The mode labels (subscripts) defining the decomposition as a comma-separated sequence of
            characters. Unicode characters are allowed in the expression thereby expanding the rank (dimensionality) of the tensor that
            can be specified.
        operand : A ndarray-like tensor object. The currently supported types are :class:`numpy.ndarray`,
            :class:`cupy.ndarray`, and :class:`torch.Tensor`.
        method : Specify decomposition method as a :class:`cuquantum.cutensornet.tensor.QRMethod` or a :class:`cuquantum.cutensornet.tensor.SVDMethod` object.
            Alternatively, a `dict` containing the parameters for the ``QRMethod`` or ``SVDMethod`` constructor can also be provided. 
            If not specified, the value will be set to the default-constructed ``QRMethod``. 
            Note that both SVD and QR method operate in reduced fashion, similar to ``full_matrices=False`` for ``numpy.linalg.svd`` and ``reduced=True`` for ``numpy.linalg.qr``.
        options : Specify the computational options for the decomposition as a :class:`cuquantum.cutensornet.tensor.DecompositionOptions` object.
            Alternatively, a `dict` containing the parameters for the ``DecompositionOptions`` constructor can also be provided.
            If not specified, the value will be set to the default-constructed ``DecompositionOptions``.
        stream: Provide the CUDA stream to use for the decomposition. Acceptable inputs include ``cudaStream_t``
            (as Python :class:`int`), :class:`cupy.cuda.Stream`, and :class:`torch.cuda.Stream`. If a stream is not provided,
            the current stream will be used.
        return_info : If true, information about the decomposition will be returned via a :class:`cuquantum.cutensornet.tensor.SVDInfo` object.
            Currently this option is only supported for SVD decomposition (which is specified via ``method``).

    Returns:
        Depending on the decomposition method specified in ``method``, the results returned may vary:

            - For QR decomposition (default), the output tensors Q and R (ndarray-like objects) of the same type 
              and on the same device as the input operand are returned as the result of the decomposition. 
            - For SVD decomposition, if ``return_info`` is `False`, a 3-tuple of output tensors U, S and V (ndarray-like objects) 
              of the same type as the input operand are returned as the result of the decomposition. If ``return_info`` is `True`, 
              a 4-tuple of output tensors U, S, V and a `dict` object that contains information about the decomposition will be returned. 
              Note, depending on the choice of :attr:`cuquantum.cutensornet.tensor.SVDMethod.partition`, the returned S operand may be `None`.
              Also see :attr:`~SVDMethod.partition`. 

    Raises:
        :class:`MemoryLimitExceeded`: the memory needed to perform the operation is larger than the ``options.memory_limit``

    The decomposition expression adopts a similar notation as einsum expression. 
    The ``subscripts`` string is a list of subscript labels where each label refers to a mode of the corresponding operand.
    The subscript labels are separated by either comma or identifier ``->``. 
    The subscript labels before the identifier ``->`` are viewed as input, and the ones after are viewed as outputs, respectively.
    The requirements on the subscripts for SVD and QR decomposition are summarized below:

        - For SVD and QR decomposition, the subscripts string is expected to contain exactly one input and two outputs (the modes for ``s`` is not needed in the case of SVD). 
        - One and only one identical mode is expected to exist in the two output mode labels.
        - When inverted, the decomposition subscript yields a valid einsum subscript that can specify the contraction of the outputs to reproduce the input (modes for ``s`` excluded for SVD).

    Examples:

        >>> # equivalent:
        >>> # q, r = numpy.linalg.qr(a)
        >>> q, r = tensor.decompose('ij->ik,kj', a)

        >>> # equivalent:
        >>> # u, s, v = numpy.linalg.svd(a, full_matrices=False)
        >>> u, s, v = tensor.decompose('ij->ik,kj', a, method=tensor.SVDMethod())

        For generalization to multi-dimensional tensors (here ``a`` is a rank-4 tensor):

        >>> u, s, v = tensor.decompose('ijab->ixb,jax', a, method=tensor.SVDMethod())
        >>> # u is unitary
        >>> identity = cuquantum.contract('ixb,iyb->xy', u, u.conj())
        >>> # re-construct the tensor a by inverting the expression
        >>> a_reconstructed = cuquantum.contract('ixb,x,jax->ijab', u, s, v)

    **Broadcasting** is supported for certain cases via ellipsis notation. 
    One may add an ellipsis in the input to represent all the modes that are not explicitly specified in the labels. 
    The ellipsis *must* also appear in one of the outputs to indicate which output the represented modes will all be partitioned onto.

    Example:

        Given a rank-6 operand ``a``:

        >>> # equivalent:
        >>> # q, r = tensor.decompose('ijabcd->ixj,abcdx', a)
        >>> q, r = tensor.decompose('ij...->ixj,...x', a)
        
    .. note::
        
        It is encouraged for users to maintain the library handle themselves so as to reduce the context initialization time:

        .. code-block:: python

            from cuquantum.bindings import cutensornet as cutn
            from cuquantum.cutensornet.tensor import decompose, QRMethod

            handle = cutn.create()
            q, r = decompose(..., method=QRMethod(), options={"handle": handle}, ...)
            # ... the same handle can be reused for further calls ...
            # when it's done, remember to destroy the handle
            cutn.destroy(handle)

    Below we give more pedagogical examples.

    Examples:

        Use NumPy operands:
        
        >>> from cuquantum.cutensornet.tensor import decompose, SVDMethod
        >>> import numpy as np
        >>> T = np.random.random((4,4,6,6))

        Perform tensor QR decomposition such that T[i,j,a,b] = Q[i,k,a] R[k,j,b]. 
        The results ``q`` and ``r`` are NumPy ndarrays (with the computation performed on the GPU):

        >>> q, r = decompose('ijab->ika,kjb', T)

        Perform exact tensor SVD decomposition such that T[i,j,a,b] = U[i,k,a] S[k] V[k,j,b]: 

        >>> u, s, v = decompose('ijab->ika,kjb', T, method=SVDMethod())

        Perform exact tensor SVD decomposition such that T[i,j,a,b] = US[i,k,a] V[k,j,b] where US[i,k,a] represents the product of U[i,k,a] and S[k]:

        >>> us, _, v = decompose('ijab->ika,kjb', T, method=SVDMethod(partition="U"))

        Perform exact tensor SVD decomposition such that T[i,j,a,b] = U[i,k,a] S[k] V[k,j,b] then normalize the L2 norm of output singular values to 1:

        >>> u, s_normalized, v = decompose('ijab->ika,kjb', T, method=SVDMethod(normalization="L2") )
        >>> print(np.linalg.norm(s_normalized)) # 1.0
        
        Perform truncated SVD decomposition to keep at most 8 singular values.
        Meanwhile, request the runtime information from the SVD truncation with ``return_info=True``.

        >>> u, s, v, info = decompose('ijab->ika,kjb', T, method=SVDMethod(max_extent=8), return_info=True)
        >>> print(s.shape) # (8,)
        >>> print(info)

        We can also perform truncated SVD decomposition with all requirements below:

            - the number of remaining singular values shall not exceed 8.
            - remaining singular values are all larger than 0.01
            - remaining singular values are all larger than 0.1 * largest singular values
            - the L1 norm of the remaining singular values are normalized to 1.
            - the remaining singular values (after truncation and normalization) are equally partitioned onto U and V
        
        >>> method = {"max_extent": 8,
        ...           "abs_cutoff": 0.01,
        ...           "rel_cutoff": 0.1,
        ...           "normalization": "L1",
        ...           "partition": "UV"}
        >>> svd_method = SVDMethod(**method)
        >>> us, _, sv, info = decompose('ijab->ika,kjb', T, method=svd_method, return_info=True)

        Alternatively, the options can be provided as a ``dict`` object:        

        >>> us, _, sv, info = tensor_decompose('ijab->ika,kjb', T, method=method, return_info=True)

        Use CuPy operands. The results ``q`` and ``r`` are CuPy ndarrays on the same device as the input operand, and ``dev`` is any valid
        device ID on your system that you wish to use to store the tensors and perform the decomposition:

        >>> import cupy
        >>> dev = 0
        >>> with cupy.cuda.Device(dev):
        ...     T = cupy.random.random((4,4,6,6))
        >>> q, r = decompose('ijab->ika,kjb', T)

        Use PyTorch operands. The results ``q`` and ``r`` are PyTorch tensors on the same device (``dev``) as the input operand:

    .. doctest::
        :skipif: torch is None

        >>> import torch
        >>> dev = 0
        >>> T = torch.rand(4,4,6,6, device=f'cuda:{dev}')
        >>> q, r = decompose('ijab->ika,kjb', T)
    """
    options = utils.check_or_create_options(DecompositionOptions, options, "decomposition options")

    logger = logging.getLogger() if options.logger is None else options.logger
    logger.info(f"CUDA runtime version = {cutn.get_cudart_version()}")
    logger.info(f"cuTensorNet version = {cutn.MAJOR_VER}.{cutn.MINOR_VER}.{cutn.PATCH_VER}")
    logger.info("Beginning operands parsing...")

    # Infer the correct decomposition method, QRMethod by default
    for method_class in (QRMethod, SVDMethod):
        try:
            method = utils.check_or_create_options(method_class, method, method_class.__name__)
        except TypeError:
            continue
        else:
            break
    else:
        raise ValueError("method must be either a QRMethod/SVDMethod object or a dict that can be used to construct QRMethod/SVDMethod")
    
    # Parse the decomposition expression
    wrapped_operands, inputs, outputs, size_dict, mode_map_user_to_ord, mode_map_ord_to_user, max_mid_extent = decomposition_utils.parse_decomposition(subscripts, operand)

    if len(wrapped_operands) != 1:
        raise ValueError(f"only one input operand expected for tensor.decompose, found {len(wrapped_operands)}")

    # placeholder to help avoid resource leak
    handle = workspace_desc = svd_config = svd_info = None
    input_descriptors = output_descriptors = []
    workspaces = dict()
    own_handle = False
    try:
        # wrap operands to be consistent with options.
        # options is a new instance of DecompositionOptions with all entries initialized
        wrapped_operands, options, own_handle, operands_location, stream_holder = decomposition_utils.parse_decompose_operands_options(
                options, wrapped_operands, stream, allowed_dtype_names=decomposition_utils.DECOMPOSITION_DTYPE_NAMES)
        handle = options.handle
        stream_ptr = stream_holder.ptr  # this exists as we always use the ExternalStream from CuPy internally...

        if isinstance(method, QRMethod):
            mid_extent = max_mid_extent
            if return_info:
                raise ValueError("``return_info`` is only supported for SVDMethod")
        elif isinstance(method, SVDMethod):
            mid_extent = max_mid_extent if method.max_extent is None else min(max_mid_extent, method.max_extent)
        else:
            raise ValueError("method must be either SVDMethod or QRMethod")
        
        # # Create input/output tensor descriptors and empty output operands 
        input_descriptors, output_operands, output_descriptors, s, s_ptr = decomposition_utils.create_operands_and_descriptors(
            options.handle, wrapped_operands, size_dict, inputs, outputs, mid_extent, method, options.device_id, stream_holder, options.logger)

        # Create workspace descriptor
        workspace_desc = cutn.create_workspace_descriptor(handle)
        
        # Compute required workspace size
        if isinstance(method, QRMethod):
            logger.debug("Querying QR workspace size...")
            cutn.workspace_compute_qr_sizes(handle, *input_descriptors, *output_descriptors, workspace_desc)
        elif isinstance(method, SVDMethod):
            svd_config = cutn.create_tensor_svd_config(handle)
            decomposition_utils.parse_svd_config(handle, svd_config, method, logger)
            logger.debug("Querying SVD workspace size...")
            cutn.workspace_compute_svd_sizes(handle, 
                *input_descriptors, *output_descriptors, svd_config, workspace_desc)
        else:
            ValueError("method must be either a QRMethod/SVDMethod object or a dict that can be used to construct QRMethod/SVDMethod")
        
        # Allocate and set workspace
        for mem_space in (cutn.Memspace.DEVICE, cutn.Memspace.HOST):
            pref = cutn.WorksizePref.MIN
            workspace_kind = cutn.WorkspaceKind.SCRATCH
            workspaces[mem_space] = decomposition_utils.allocate_and_set_workspace(options, workspace_desc, 
                pref, mem_space, workspace_kind, stream_holder, task_name='tensor decomposition')
        
        svd_info_obj = None

        # Perform QR/SVD computation
        logger.info("Starting tensor decomposition...")
        blocking = options.blocking is True or operands_location == 'cpu'
        if blocking:
            logger.info("This call is blocking and will return only after the operation is complete.")
        else:
            logger.info("This call is non-blocking and will return immediately after the operation is launched on the device.")
        timing =  bool(logger and logger.handlers)
        if isinstance(method, QRMethod):
            with utils.device_ctx(options.device_id), utils.cuda_call_ctx(stream_holder, blocking, timing) as (last_compute_event, elapsed):
                cutn.tensor_qr(handle, 
                    *input_descriptors, wrapped_operands[0].data_ptr,
                    output_descriptors[0], output_operands[0].data_ptr,
                    output_descriptors[1], output_operands[1].data_ptr,
                    workspace_desc, stream_ptr)

            if elapsed.data is not None:
                logger.info(f"The QR decomposition took {elapsed.data:.3f} ms to complete.")
        elif isinstance(method, SVDMethod):
            svd_info = cutn.create_tensor_svd_info(handle)
            with utils.device_ctx(options.device_id), utils.cuda_call_ctx(stream_holder, blocking, timing) as (last_compute_event, elapsed):
                cutn.tensor_svd(handle, 
                    *input_descriptors, wrapped_operands[0].data_ptr, 
                    output_descriptors[0], output_operands[0].data_ptr, 
                    s_ptr, 
                    output_descriptors[1], output_operands[1].data_ptr, 
                    svd_config,  svd_info, 
                    workspace_desc,  stream_ptr)
            if elapsed.data is not None:
                logger.info(f"The SVD decomposition took {elapsed.data:.3f} ms to complete.")
            svd_info_obj = SVDInfo(**decomposition_utils.get_svd_info_dict(handle, svd_info))

            # update the operand to reduced_extent if needed
            for (wrapped_tensor, tensor_desc) in zip(output_operands, output_descriptors):
                wrapped_tensor.reshape_to_match_tensor_descriptor(handle, tensor_desc)
            reduced_extent = svd_info_obj.reduced_extent
            if s is not None and reduced_extent != mid_extent:
                s.tensor = s.tensor[:reduced_extent]
    finally:
        # when host workspace is allocated, synchronize stream before return
        if workspaces.get(cutn.Memspace.HOST) is not None:
            stream_holder.obj.synchronize()
        # Free resources
        if svd_config is not None:
            cutn.destroy_tensor_svd_config(svd_config)
        if svd_info is not None:
            cutn.destroy_tensor_svd_info(svd_info)
        decomposition_utils._destroy_tensor_descriptors(input_descriptors)
        decomposition_utils._destroy_tensor_descriptors(output_descriptors)
        if workspace_desc is not None:
            cutn.destroy_workspace_descriptor(workspace_desc)

        # destroy handle if owned
        if own_handle and handle is not None:
            cutn.destroy(handle)
        logger.info(f"All resources for the decomposition are freed.")
    
    left_output, right_output, s = [decomposition_utils.get_return_operand_data(
                                        o, operands_location, stream_holder)
                                    for o in output_operands + [s, ]]
    
    if isinstance(method, QRMethod):
        return left_output, right_output
    elif isinstance(method, SVDMethod):
        if return_info:
            return left_output, s, right_output, svd_info_obj
        else:
            return left_output, s, right_output
    else:
        raise NotImplementedError


@dataclass
class QRMethod:
    """A data class for providing QR options to the :func:`cuquantum.cutensornet.tensor.decompose` function."""
    pass


@dataclass
class SVDInfo:

    """A data class for holding information regarding SVD truncation at runtime.

    Attributes:
        full_extent: The total number of singular values after matricization (before truncation). 
        reduced_extent: The number of remaining singular values after truncation. 
        discarded_weight: The discarded weight for the truncation.
        algorithm: The algorithm used in the SVD execution.
        gesvdj_residual: The residual for full gesvdj execution. 
        gesvdj_sweeps: The number of iterations used in the gesvdj execution.
        gesvdp_err_sigma: The error sigma in the gesvdp execution.
    
    .. note::
        
        When the SVD algorithm is set to ``"gesvdr"`` with fixed extent truncation enabled in :class:`cuquantum.cutensornet.tensor.SVDMethod`, 
        the discarded weight will not be computed.
    """

    reduced_extent: int 
    full_extent: int
    discarded_weight: float
    algorithm: str
    gesvdj_residual: Optional[float] = None
    gesvdj_sweeps: Optional[int] = None
    gesvdp_err_sigma: Optional[float] = None

    def __str__(self):
        svd_details = f"Algorithm = {self.algorithm}"      
        if self.gesvdj_residual is not None:
            svd_details += f", residual= {self.gesvdj_residual}"
        if self.gesvdj_sweeps is not None:
            svd_details += f", sweeps = {self.gesvdj_sweeps}"
        if self.gesvdp_err_sigma is not None:
            svd_details += f", sigma error = {self.gesvdp_err_sigma}"

        s = f"""SVD Information at Runtime:
    {svd_details}
    Total number of singular values after matricization = {self.full_extent}
    Number of singular values after truncation = {self.reduced_extent}
    Discarded weight for the truncation = {self.discarded_weight}"""
        return s
    

@dataclass
class SVDMethod:
    """A data class for providing SVD options to the :func:`cuquantum.cutensornet.tensor.decompose` function.

    Attributes:
        max_extent: Keep no more than the largest ``max_extent`` singular values in the output operands (the rest will be truncated). 
        abs_cutoff: Singular values below this value will be trimmed in the output operands. 
        rel_cutoff: Singular values below the product of this value and the largest singular value will be trimmed in the output operands.
        discarded_weight_cutoff: Singular values with discarded weight (square sum dividied by total square sum) below this value will be trimmed.
        partition: Singular values S will be explictly returned by default (``partition=None``). 
            Alternatively, singular values may be factorized onto output tensor U (``partition="U"``), output tensor V (``partition="V"``) or 
            equally onto output tensor U and output tensor V (``partition="UV"``). When any of these three partition schemes is selected,
            the returned ``S`` operand from :func:`cuquantum.cutensornet.tensor.decompose` and
            :func:`cuquantum.cutensornet.experimental.contract_decompose` will be `None`.
        normalization: The specified norm of the singular values (after truncation) will be normalized to 1. 
            Currently supports ``None``, ``"L1"``, ``"L2"`` and ``"LInf"``. 
        algorithm: The SVD algorithm to use. Currently supports ``"gesvd"`` (default), ``"gesvdj"``, ``"gesvdp"`` and ``"gesvdr"``. 
        gesvdj_tol: The tolerance to use when ``algorithm`` is set to ``"gesvdj"``. Default 0 denotes machine precision.
        gesvdj_max_sweeps: The maximal number of sweeps when ``algorithm`` is set to ``"gesvdj"``. Default 0 denotes 100.
        gesvdr_oversampling: The size of oversampling when ``algorithm`` is set to ``"gesvdr"``. Default 0 denotes the lower of 4 times ``max_extent`` and the difference between full rank and ``max_extent``.
        gesvdr_niters: The number of iteration of power method when ``algorithm`` is set to ``"gesvdr"`` and the default (0) is 10.
    
    .. note::

        If multiple truncation paramters are set, e.g, ``max_extent`` and ``discarded_weight_cutoff``, the truncated extent will be determined as the lowest among all.

    .. note::
        
        For detailed explanation on the different SVD algorithms and the corresponding parameters, 
        please refer to `cuSolver documentation page <https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-dense-lapack-function-reference>`_
    
    .. note::
        
        For truncated SVD, currently at least one singular value will be retained in the output even if the truncation parameters are set to trim out all singular values. 
        This behavior may be subject to change in a future release.
        
    """
    max_extent: Optional[int] = None
    abs_cutoff: Optional[float] = 0.0
    rel_cutoff: Optional[float] = 0.0
    discarded_weight_cutoff: Optional[float] = 0.0
    partition: Optional[str] = None
    normalization: Optional[str] = None
    algorithm: Optional[str] = 'gesvd'
    gesvdj_tol: Optional[float] = 0
    gesvdj_max_sweeps: Optional[int] = 0
    gesvdr_oversampling: Optional[int] = 0
    gesvdr_niters: Optional[int] = 0

    def __str__(self):

        svd_details = f"Algorithm = {self.algorithm}"      
        if self.gesvdj_tol is not None:
            svd_details += f", tolerance = {self.gesvdj_tol}"
        if self.gesvdj_max_sweeps is not None:
            svd_details += f", max sweeps = {self.gesvdj_max_sweeps}"
        if self.gesvdr_oversampling is not None:
            svd_details += f", oversampling = {self.gesvdr_oversampling}"
        if self.gesvdr_niters is not None:
            svd_details += f", niters = {self.gesvdr_niters}"
        
        s = f"""SVD Method:
    {svd_details}
    Maxmial number of singular values = {self.max_extent}
    Absolute value cutoff = {self.abs_cutoff} 
    Relative value cutoff = {self.rel_cutoff}
    Discarded weight cutoff = {self.discarded_weight_cutoff}
    Singular values partition = {self.partition}
    Singular values normalization = {self.normalization}"""

        return s
    
    def __post_init__(self):
        if self.algorithm not in ('gesvd', 'gesvdj', 'gesvdr', 'gesvdp'):
            raise ValueError(f"SVD algorithm {self.algorithm} not supported; currently supports gesvd, gesvdj, gesvdr, gesvdp")
        
        if (self.gesvdj_tol !=0 or self.gesvdj_max_sweeps !=0) and self.algorithm != 'gesvdj':
            raise ValueError(f"gesvdj_tol and gesvdj_max_sweeps can only be set when algorithm is set to gesvdj, found algorithm {self.algorithm}")
        
        if (self.gesvdr_oversampling !=0 or self.gesvdr_niters !=0) and self.algorithm != 'gesvdr':
            raise ValueError(f"gesvdr_oversample and gesvdr_niters can only be set when algorithm is set to gesvdr, found algorithm {self.algorithm}")
        
        if self.algorithm == 'gesvdr' and self.discarded_weight_cutoff != 0 and self.max_extent is not None:
            raise ValueError("Discarded weight truncation is not supported for gesvdr algorithm with fixed extent truncation")
    
    def _get_algo_params(self):
        initialized = False
        if self.algorithm in ('gesvdj', 'gesvdr'):
            dtype = cutn.tensor_svd_algo_params_get_dtype(decomposition_utils.SVD_ALGORITHM_MAP[self.algorithm])
            algo_params = numpy.zeros(1, dtype=dtype)
            for name in dtype.names:
                value = getattr(self, f'{self.algorithm}_{name}')
                if value != 0:
                    algo_params[name] = value
                    initialized = True
        if initialized:
            return algo_params
        else:
            return None
