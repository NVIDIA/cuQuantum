# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
A collection of utility functions for decomposition.
"""

import logging

import numpy
import cupy as cp

from ...bindings import cutensornet as cutn
from . import einsum_parser
from ..._internal import formatters
from ..._internal import tensor_wrapper
from ..._internal import typemaps
from ..._internal import utils
from .. import memory
from ..configuration import NetworkOptions, MemoryLimitExceeded


DECOMPOSITION_DTYPE_NAMES = ('float32', 'float64', 'complex64', 'complex128')

#TODO: auto generate the maps below

PARTITION_MAP = {None: cutn.TensorSVDPartition.NONE, 
                'U': cutn.TensorSVDPartition.US, 
                'V': cutn.TensorSVDPartition.SV,
                'UV': cutn.TensorSVDPartition.UV_EQUAL}

NORMALIZATION_MAP = {None: cutn.TensorSVDNormalization.NONE,
                    'L1': cutn.TensorSVDNormalization.L1,
                    'L2': cutn.TensorSVDNormalization.L2,
                    'LInf': cutn.TensorSVDNormalization.LINF}

SVD_ALGORITHM_MAP = {'gesvd': cutn.TensorSVDAlgo.GESVD,
                     'gesvdj': cutn.TensorSVDAlgo.GESVDJ,
                     'gesvdp': cutn.TensorSVDAlgo.GESVDP,
                     'gesvdr': cutn.TensorSVDAlgo.GESVDR}

SVD_ALGORITHM_MAP_TO_STRING = dict((val, key) for key, val in SVD_ALGORITHM_MAP.items())

SVD_METHOD_CONFIG_MAP = {'abs_cutoff': cutn.TensorSVDConfigAttribute.ABS_CUTOFF,
                        'rel_cutoff': cutn.TensorSVDConfigAttribute.REL_CUTOFF, 
                        'partition': cutn.TensorSVDConfigAttribute.S_PARTITION, 
                        'normalization': cutn.TensorSVDConfigAttribute.S_NORMALIZATION,
                        'algorithm': cutn.TensorSVDConfigAttribute.ALGO,
                        'discarded_weight_cutoff': cutn.TensorSVDConfigAttribute.DISCARDED_WEIGHT_CUTOFF}

SVD_INFO_MAP = {'full_extent': cutn.TensorSVDInfoAttribute.FULL_EXTENT,
                'reduced_extent': cutn.TensorSVDInfoAttribute.REDUCED_EXTENT,
                'discarded_weight': cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT,
                'algorithm': cutn.TensorSVDInfoAttribute.ALGO}


def compute_combined_size(size_dict, modes):
    """
    Given the modes, compute the product of all extents using information in size_dict.
    """
    size = 1
    for mode in modes:
        size *= size_dict[mode]
    return size


def parse_decomposition_subscripts(subscripts):
    """
    Parse decomposition expression in string format, retaining ellipses if present.
    """
    input_modes, *output_modes = subscripts.split("->")
    if not output_modes:
        raise ValueError("Output modes must be explicitly specified for decomposition")
    if len(output_modes) > 1:
        raise ValueError("subscripts must contain only 1 ->")
    input_modes = input_modes.split(",")
    output_modes = output_modes[0].split(",")
    if len(output_modes) != 2:
        raise ValueError("subscripts must specify the modes for both left and right tensors")
    return input_modes, output_modes


def compute_mid_extent(size_dict, inputs, outputs):
    """
    Compute the expected mid extent given a size_dict and the modes for both inputs and outputs.
    """
    size_dict = size_dict.copy() # this func will modify it in place
    left_output = set(outputs[0])
    right_output = set(outputs[1])
    shared_mode_out = set(left_output) & set(right_output)
    if len(shared_mode_out) !=1:
        raise ValueError(f"Expect one shared mode in the output tensors, found {len(shared_mode_out)}")
    
    left_output -= shared_mode_out
    right_output -= shared_mode_out

    for _input in inputs:
        left_extent = right_extent = remaining_extent = 1
        left_modes = set()
        right_modes = set()
        for mode in _input:
            extent = size_dict[mode]
            if mode in left_output:
                left_extent *= extent
                left_modes.add(mode)
            elif mode in right_output:
                right_extent *= extent
                right_modes.add(mode)
            else:
                remaining_extent *= extent
        if right_extent * remaining_extent < left_extent:
            # update left modes
            left_mode_collapsed = left_modes.pop()
            size_dict[left_mode_collapsed] = right_extent * remaining_extent
            left_output -= left_modes
        elif left_extent * remaining_extent < right_extent:
            # update right modes
            right_mode_collapsed = right_modes.pop()
            size_dict[right_mode_collapsed] = left_extent * remaining_extent
            right_output -= right_modes
    
    left_extent = compute_combined_size(size_dict, left_output)
    right_extent = compute_combined_size(size_dict, right_output)
    return min(left_extent, right_extent)


def parse_decomposition(subscripts, *operands):
    """
    Parse the generalized decomposition expression in string formats (unicode strings supported). 
    The modes for the outputs must be specified.

    Returns wrapped operands, mapped inputs and output, size dictionary based on internal mode numbers, 
    the forward as well as the reverse mode maps, and the largest mid extent expected for the decomposition.
    """
    inputs, outputs = parse_decomposition_subscripts(subscripts)
    num_operand, num_input = len(operands), len(inputs)
    if num_operand != num_input:
        message = f"""Operand-term mismatch. The number of operands ({num_operand}) must match the number of inputs ({num_input}) specified in the decomposition expression."""
        raise ValueError(message)
    
    morpher = einsum_parser.select_morpher(False)

    # First wrap operands.
    operands = tensor_wrapper.wrap_operands(operands)
    
    inputs = list(einsum_parser.parse_single(_input) for _input in inputs)
    outputs = list(einsum_parser.parse_single(_output) for _output in outputs)
    ellipses_input = any(Ellipsis in _input for _input in inputs)
    num_ellipses_output = sum(Ellipsis in _output for _output in outputs)
    if num_ellipses_output > 1:
        raise ValueError(f"Ellipses found in {num_ellipses_output} output terms, only allowed in one at most.")

    if ellipses_input:
        if num_input == 1 and num_ellipses_output == 0:
            raise ValueError("tensor.decompose does not support reduction operations")
        einsum_parser.check_ellipses(inputs+outputs, morpher)
    else:
        if num_ellipses_output != 0:
            raise ValueError("Invalid ellipsis specification. The output terms contain ellipsis while none of the input terms do.")

    einsum_parser.check_einsum_with_operands(inputs, operands, morpher)

    # Map data to ordinals for cutensornet.
    num_extra_labels = max(len(o.shape) for o in operands) if ellipses_input else 0
    all_modes, _, mode_map_user_to_ord, mode_map_ord_to_user, label_end = einsum_parser.map_modes(inputs + outputs, None, num_extra_labels, morpher)

    mapper = einsum_parser.ModeLabelMapper(mode_map_ord_to_user)
    mapping_morpher = einsum_parser.select_morpher(False, mapper)

    # Replace ellipses with concrete labels
    if ellipses_input:
        if num_input == 1:
            # For tensor.decompose only
            n = len(operands[0].shape) - (len(inputs[0]) -1)
        else:
            num_implicit_modes = set()
            for i, o in enumerate(operands):
                _input = all_modes[i]
                if Ellipsis not in _input:
                    continue

                n = len(o.shape) - (len(_input) - 1)
                assert n >= 0, "Internal error"
                num_implicit_modes.add(n)
            if len(num_implicit_modes) != 1:
                #NOTE: Although we can allow ellipsis denoting different number of modes, 
                # here we disable it due to limited use case if any and potential confusion due to implicit specification.
                raise ValueError(f"Ellipsis for all operands must refer to equal number of modes, found {num_implicit_modes}")
            n = num_implicit_modes.pop()
        
        ellipses_modes = tuple(range(label_end-n, label_end))
        for i, _modes in enumerate(all_modes):
            if Ellipsis not in _modes:
                continue
            s = _modes.index(Ellipsis)
            all_modes[i] = _modes[:s] + ellipses_modes + _modes[s+1:]
    
    inputs = all_modes[:num_input]
    outputs = all_modes[num_input:]

    if num_input == 1:
        contracted_modes_output = set(einsum_parser.infer_output_mode_labels(outputs))
        if contracted_modes_output != set(inputs[0]):
            raise ValueError("The contracted outcome from the right hand side of the expression does not match the input")

    # Create mode-extent map based on internal mode numbers.
    size_dict = einsum_parser.create_size_dict(inputs, operands)

    # Compute the maximally allowed mid extent
    mid_extent = compute_mid_extent(size_dict, inputs, outputs)
    
    return operands, inputs, outputs, size_dict, mode_map_user_to_ord, mode_map_ord_to_user, mid_extent


def get_svd_config_info_scalar_attr(handle, obj_type, obj, attr, svd_algorithm=None):
    """
    Get the data for given attribute of SVDConfig or SVDInfo.
    """
    if obj_type == 'config':
        if attr != cutn.TensorSVDConfigAttribute.ALGO_PARAMS:
            dtype = cutn.tensor_svd_config_get_attribute_dtype(attr)
        else:
            if svd_algorithm not in (cutn.TensorSVDAlgo.GESVDJ, cutn.TensorSVDAlgo.GESVDR):
                return None
            dtype = cutn.tensor_svd_algo_params_get_dtype(svd_algorithm)
        getter = cutn.tensor_svd_config_get_attribute
    elif obj_type == 'info':
        if attr != cutn.TensorSVDInfoAttribute.ALGO_STATUS:
            dtype = cutn.tensor_svd_info_get_attribute_dtype(attr)
        else:
            if svd_algorithm not in (cutn.TensorSVDAlgo.GESVDJ, cutn.TensorSVDAlgo.GESVDP):
                return None
            dtype = cutn.tensor_svd_algo_status_get_dtype(svd_algorithm)
        getter = cutn.tensor_svd_info_get_attribute
    else:
        raise ValueError("object type must be either config or info")
    data = numpy.empty((1,), dtype=dtype)
    getter(handle, obj, attr, data.ctypes.data, data.dtype.itemsize)
    return data


def set_svd_config_scalar_attr(handle, obj, attr, data, svd_algorithm=None):
    """
    Set the data for given attribute of SVDConfig.
    """
    setter = cutn.tensor_svd_config_set_attribute
    if attr != cutn.TensorSVDConfigAttribute.ALGO_PARAMS:
        dtype = cutn.tensor_svd_config_get_attribute_dtype(attr)
    else:
        if svd_algorithm not in (cutn.TensorSVDAlgo.GESVDJ, cutn.TensorSVDAlgo.GESVDR):
            raise ValueError(f"Algorithm specific parameters not supported for {svd_algorithm}")
        dtype = cutn.tensor_svd_algo_params_get_dtype(svd_algorithm)
    if not isinstance(data, numpy.ndarray):
        data = numpy.asarray(data, dtype=dtype)
    setter(handle, obj, attr, data.ctypes.data, data.dtype.itemsize)


def parse_svd_config(handle, svd_config, svd_method, logger=None):
    """
    Given an SVDMethod object, set the corresponding attributes in the SVDConfig.
    """
    svd_algorithm = None
    for method_attr, attr in SVD_METHOD_CONFIG_MAP.items():
        data = getattr(svd_method, method_attr)
        if method_attr == 'partition':
            data = PARTITION_MAP[data]
        elif method_attr == 'normalization':
            data = NORMALIZATION_MAP[data]
        elif method_attr == 'algorithm':
            svd_algorithm = data = SVD_ALGORITHM_MAP[data]
        set_svd_config_scalar_attr(handle, svd_config, attr, data)
        if logger is not None:
            logger.info(f"The SVDConfig attribute '{method_attr}' has been set to {data}.")

    algo_params = svd_method._get_algo_params()
    if algo_params is not None:
        set_svd_config_scalar_attr(handle, svd_config, cutn.TensorSVDConfigAttribute.ALGO_PARAMS, algo_params, svd_algorithm=svd_algorithm)
        if logger is not None:
            logger.info(f"The SVDConfig attribute '{cutn.TensorSVDConfigAttribute.ALGO_PARAMS}' has been set to {algo_params}.")


def get_svd_info_dict(handle, svd_info):
    """
    Parse the information in SVDInfo in a dictionary object.
    """
    info = dict()
    for key, attr in SVD_INFO_MAP.items():
        info[key] = get_svd_config_info_scalar_attr(handle, 'info', svd_info, attr).item()
    svd_algorithm = info['algorithm']
    algo_status = get_svd_config_info_scalar_attr(handle, 'info', svd_info, cutn.TensorSVDInfoAttribute.ALGO_STATUS, svd_algorithm=svd_algorithm)
    info['algorithm'] = SVD_ALGORITHM_MAP_TO_STRING[svd_algorithm]
    if algo_status is not None:
        for name in algo_status.dtype.names:
            key = info['algorithm'] + f'_{name}'
            info[key] = algo_status[name].item()
    return info


def parse_decompose_operands_options(options, wrapped_operands, stream, allowed_dtype_names=None):
    """
    Given initially wrapped tensors and network options, wrap the operands to device and create an internal NetworkOptions object. 
    If cutensornet library handle is not provided in `options`, one will be created in the internal options.
    """
    package = utils.get_operands_package(wrapped_operands)
    operands_location = 'cuda'
    device_id = utils.get_network_device_id(wrapped_operands)
    if device_id is None:
        package = wrapped_operands[0].name
        if package == 'numpy':
            package = 'cupy'
        operands_location = 'cpu'
        device_id = options.device_id
    
    # initialize handle once if not provided
    if options.handle is not None:
        own_handle = False
        handle = options.handle
    else:
        own_handle = True
        with utils.device_ctx(device_id):
            handle = cutn.create()
    
    dtype_name = utils.get_operands_dtype(wrapped_operands)
    if allowed_dtype_names is not None and dtype_name not in allowed_dtype_names:
        raise ValueError(f"dtype {dtype_name} not supported")
    
    # compute_type for decomposition should be None
    if options.__class__.__name__ == 'NetworkOptions':
        compute_type = options.compute_type if options.compute_type is not None else typemaps.NAME_TO_COMPUTE_TYPE[dtype_name]
    else:
        compute_type = None

    stream_holder = utils.get_or_create_stream(options.device_id, stream, package)

    logger = logging.getLogger() if options.logger is None else options.logger
    if operands_location == 'cpu':
        logger.info(f"Begin transferring input data from host to device {device_id}")
        wrapped_operands = tensor_wrapper.to(wrapped_operands, device_id, stream_holder)
        logger.info("Input data transfer finished")

    allocator = options.allocator if options.allocator is not None else memory._MEMORY_MANAGER[package](device_id, logger)
    
    internal_options = options.__class__(device_id=device_id,
                                        logger=logger,
                                        handle=handle,
                                        blocking=options.blocking,
                                        compute_type=compute_type,
                                        memory_limit=options.memory_limit,
                                        allocator=allocator)

    return wrapped_operands, internal_options, own_handle, operands_location, stream_holder


def allocate_and_set_workspace(options: NetworkOptions, workspace_desc, pref, mem_space, workspace_kind, stream_holder, task_name=''):
    """
    Allocate and set the workspace in the workspace descriptor.

    The ``options`` argument should be properly initialized using :func:``create_operands_and_descriptors``.

    Options used: 
        - options.handle
        - options.allocator
        - options.device_id
        - options.logger
        - options.memory_limit
    """
    logger = options.logger
    workspace_size = cutn.workspace_get_memory_size(options.handle, workspace_desc, pref, mem_space, workspace_kind)
    _device = cp.cuda.Device(options.device_id)
    _memory_limit =  utils.get_memory_limit(options.memory_limit, _device)
    if _memory_limit < workspace_size:
        raise MemoryLimitExceeded(_memory_limit, workspace_size, options.device_id)
    # Allocate and set workspace
    if mem_space == cutn.Memspace.DEVICE:
        with utils.device_ctx(options.device_id), stream_holder.ctx:
            try:
                logger.debug(f"Allocating device memory for {task_name}")
                workspace_ptr = options.allocator.memalloc(workspace_size)
            except TypeError as e:
                message = "The method 'memalloc' in the allocator object must conform to the interface in the "\
                        "'BaseCUDAMemoryManager' protocol."
                raise TypeError(message) from e
        
        logger.debug(f"Finished allocating device memory of size {formatters.MemoryStr(workspace_size)} for decomposition in the context of stream {stream_holder.obj}.")
        device_ptr = utils.get_ptr_from_memory_pointer(workspace_ptr)
        cutn.workspace_set_memory(options.handle, workspace_desc, mem_space, workspace_kind, device_ptr, workspace_size)
        logger.debug(f"The workspace memory (device pointer = {device_ptr}) has been set in the workspace descriptor.")
        return workspace_ptr
    elif workspace_size != 0:
        # host workspace
        logger.debug(f"Allocating host memory for {task_name}")
        workspace_host = numpy.empty(workspace_size, dtype=numpy.int8)
        logger.debug(f"Finished allocating host memory of size {formatters.MemoryStr(workspace_size)} for decomposition.")
        cutn.workspace_set_memory(options.handle, workspace_desc, mem_space, workspace_kind, workspace_host.ctypes.data, workspace_size)
        logger.debug(f"The workspace memory (host pointer = {workspace_host.ctypes.data}) has been set in the workspace descriptor.")
        return workspace_host
    else:
        return None


def _destroy_tensor_descriptors(desc_tensors):
    for t in desc_tensors:
        if t is not None:
            cutn.destroy_tensor_descriptor(t)


def create_operands_and_descriptors(
        handle, wrapped_operands, size_dict, inputs, outputs, mid_extent, method, device_id, stream_holder, logger):
    """
    Create empty tensor operands and corresponding tensor descriptors for a decomposition problem.
    """
    # Create input tensor descriptors, output operands and output tensor descriptors
    output_class = wrapped_operands[0].__class__
    dtype_name = wrapped_operands[0].dtype

    # Compute extents for the outputs
    shared_mode_out = list(set(outputs[0]) & set(outputs[1]))[0]
    output_extents = [tuple(size_dict[m] if m != shared_mode_out else mid_extent for m in modes) for modes in outputs]
    
    logger.debug("Creating input tensor descriptors.")
    input_tensor_descriptors = []
    output_tensor_descriptors = []
    try:
        for (t, modes) in zip(wrapped_operands, inputs):
            input_tensor_descriptors.append(t.create_tensor_descriptor(handle, modes))
        logger.debug("The input tensor descriptors have been created.")
        # Create the output in the context of the current stream to work around a performance issue with CuPy's memory pool.    
        logger.debug("Beginning output tensors and descriptors creation...")
        s = None
        s_ptr = 0
        output_operands = []
        with utils.device_ctx(device_id):
            for extent, tensor_modes in zip(output_extents, outputs):
                operand = utils.create_empty_tensor(output_class, extent, dtype_name, device_id, stream_holder)
                output_operands.append(operand)
                output_tensor_descriptors.append(operand.create_tensor_descriptor(handle, tensor_modes))
            
            if hasattr(method, 'partition') and method.partition is None:
                if dtype_name in ['float32', 'complex64']:
                    s_dtype_name = 'float32'
                elif dtype_name in ['float64', 'complex128']:
                    s_dtype_name = 'float64'
                else:
                    raise ValueError(f"{dtype_name} data type not supported")
                s = utils.create_empty_tensor(output_class, (mid_extent, ), s_dtype_name, device_id, stream_holder)
                s_ptr = s.data_ptr
        logger.debug("The output tensors and descriptors have been created.")
    except: 
        _destroy_tensor_descriptors(input_tensor_descriptors)
        _destroy_tensor_descriptors(output_tensor_descriptors)
        raise

    return input_tensor_descriptors, output_operands, output_tensor_descriptors, s, s_ptr


def get_return_operand_data(tensor, target_location, stream_holder):
    """
    Given wrapped tensors, fetch the return operands based on target location.
    """
    if tensor is None: # potentially for s
        return tensor
    if target_location == 'cpu':
        return tensor.to('cpu', stream_holder=stream_holder)
    else: # already on device
        return tensor.tensor
