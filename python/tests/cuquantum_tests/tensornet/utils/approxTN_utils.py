# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# Note: This file must be self-contained and not import private helpers!

import importlib
import logging

try:
    import cupy as cp
except ImportError:
    cp = None
import numpy as np


REL_DEGENERACY_TOLERANCE = 1e-4

class SingularValueDegeneracyError(Exception):
    """Truncation takes places at two or more degenerate singular values."""
    pass

####################################################
################# Helper functions #################
####################################################

def get_stream_pointer(backend):
    if backend == "numpy":
        return 0
    elif backend == "cupy":
        return cp.cuda.get_current_stream().ptr
    elif backend == "torch":
        import torch
        return torch.cuda.current_stream().cuda_stream
    else:
        raise NotImplementedError(f"{backend} not supported")


def infer_backend(obj):
    module = obj.__class__.__module__.split(".")[0]
    return importlib.import_module(module)


def parse_split_expression(split_expression):
    modes_in, modes_out = split_expression.split("->")
    left_modes, right_modes = modes_out.split(",")
    shared_modes = set(left_modes) & set(right_modes)
    # only allow one shared mode in the output
    assert len(shared_modes) == 1, f"the split expr \"{split_expression}\" does not have a unique shared mode"
    shared_mode = list(shared_modes)[0]
    return modes_in, left_modes, right_modes, shared_mode


def get_new_modes(used_modes, num):
    # Note: cannot use _internal.circuit_converter_utils._get_symbol() here, as this
    # module needs to be standalone. We don't need that many symbols here, anyway.
    base_modes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    new_modes = ""
    for mode in base_modes:
        if mode not in used_modes:
            new_modes += mode
        if len(new_modes) == num:
            break
    else:
        raise RuntimeError(f"can't find {num} new modes")
    return new_modes


def prepare_reduced_qr_modes(modes_in, modes_out, new_mode, shared_modes_in):
    """Given the input modes and output modes in a gate problem, generate the modes for QR for the reduced algorithm"""
    modes_q = ""
    modes_r = new_mode
    for mode in modes_in:
        if mode in modes_out and mode not in shared_modes_in: # in case the same mode is used as shared modes in both input and output
            modes_q += mode
        else:
            modes_r += mode
    modes_q += new_mode
    return modes_q, modes_r


# used by cutensornetApprox
def parse_modes_extents(extent_map, split_expression):
    modes_in, left_modes_out, right_modes_out, shared_mode_out = parse_split_expression(split_expression)
    shared_modes_duplicated = False
    input_modes = modes_in.split(",")
    if len(input_modes) == 1:
        left_extent = compute_size(extent_map, left_modes_out.replace(shared_mode_out, ""))
        right_extent = compute_size(extent_map, right_modes_out.replace(shared_mode_out, ""))
    elif len(input_modes) == 3:
        new_modes = get_new_modes(split_expression, 2)
        left_modes_in, right_modes_in, gate_modes = input_modes
        shared_modes_in = set(left_modes_in) & set(right_modes_in)
        modes_qa, modes_ra = prepare_reduced_qr_modes(left_modes_in, left_modes_out, new_modes[0], shared_modes_in)
        modes_qb, modes_rb = prepare_reduced_qr_modes(right_modes_in, right_modes_out, new_modes[1], shared_modes_in)

        # extent for shared mode between qa and ra
        intm_open_extent_a = min(compute_size(extent_map, modes_qa.replace(new_modes[0], "")), 
                                 compute_size(extent_map, modes_ra.replace(new_modes[0], "")))
        # extent for shared mode between qb and rb
        intm_open_extent_b = min(compute_size(extent_map, modes_qb.replace(new_modes[1], "")), 
                                 compute_size(extent_map, modes_rb.replace(new_modes[1], "")))

        intm_modes_out = infer_contracted_output_modes(modes_ra+modes_rb+gate_modes)

        intm_modes_left = "".join([mode for mode in intm_modes_out if mode in left_modes_out and mode != shared_mode_out]) # excluding new_modes[0] (shared mode between qa and ra) and the shared mode between intm_left and intm_right
        intm_modes_right = "".join([mode for mode in intm_modes_out if mode in right_modes_out and mode != shared_mode_out]) # excluding new_modes[1] (shared) and the shared mode between intm_left and intm_right
        
        assert set(infer_contracted_output_modes(intm_modes_left+intm_modes_right+new_modes)) == set(intm_modes_out)
        left_extent = compute_size(extent_map, intm_modes_left) * intm_open_extent_a # multiply by intm_open_extent_a to add back the contribution from shared mode bewteen qa and ra 
        right_extent = compute_size(extent_map, intm_modes_right) * intm_open_extent_b # multiply by intm_open_extent_b to add back the contribution from shared mode bewteen qb and rb 
        shared_modes_duplicated = shared_mode_out in shared_modes_in
    else:
        raise ValueError("Split_expression must be a valid SVD/QR or Gate expression")
    return modes_in, left_modes_out, right_modes_out, shared_mode_out, shared_modes_duplicated, min(left_extent, right_extent)


def infer_contracted_output_modes(modes_in):
    modes_in = modes_in.replace(",","")
    modes_out = "".join([mode for mode in modes_in if modes_in.count(mode)==1])
    return modes_out


def compute_size(size_dict, modes):
    """Given the modes, compute the product of all extents that are recorded in size_dict. Note modes not in size_dict will be neglected."""
    size = 1
    for mode in modes:
        if mode in size_dict:
            size *= size_dict[mode]
    return size


def get_tensordot_axes(modes, shared_mode):
    axes = []
    for ax, mode in enumerate(modes):
        if mode != shared_mode:
            axes.append(ax)
    return [axes, axes]


def reverse_einsum(split_expression, array_left, array_mid, array_right):
    backend = infer_backend(array_left)
    einsum_kwargs = get_einsum_kwargs(backend)
    modes_in, left_modes, right_modes, shared_mode = parse_split_expression(split_expression)
    if modes_in.count(",") == 0:
        modes_out = modes_in
    else:
        modes_out = infer_contracted_output_modes(modes_in)
    if array_mid is None:
        # For QR or SVD with S partitioned onto U or V or both.
        einsum_string = f"{left_modes},{right_modes}->{modes_out}"
        out = backend.einsum(einsum_string, array_left, array_right)
    else:
        # For SVD with singular values explicitly returned
        einsum_string = f"{left_modes},{shared_mode},{right_modes}->{modes_out}"
        out = backend.einsum(einsum_string, array_left, array_mid, array_right, **einsum_kwargs)
    return out


def is_exact_split(**split_options):
    max_extent = split_options.get("max_extent", 0)
    abs_cutoff = split_options.get("abs_cutoff", 0)
    rel_cutoff = split_options.get("rel_cutoff", 0)
    discarded_weight_cutoff = split_options.get("discarded_weight_cutoff", 0)
    normalization = split_options.get("normalization", None)
    return (max_extent == 0 or max_extent is None) and \
            abs_cutoff == 0 and rel_cutoff == 0 and \
            discarded_weight_cutoff == 0 and normalization is None

def is_dw_truncation_only(**split_options):
    max_extent = split_options.get("max_extent", 0)
    abs_cutoff = split_options.get("abs_cutoff", 0)
    rel_cutoff = split_options.get("rel_cutoff", 0)
    return (max_extent == 0 or max_extent is None) and abs_cutoff == 0 and rel_cutoff == 0 


def split_contract_decompose(subscripts):
    if '.' in subscripts:
        raise ValueError("This function does not support ellipses notation.")
    inputs, outputs = subscripts.split('->')
    tmp_subscripts = outputs.replace(",", "")
    intm_modes = "".join(s for s in tmp_subscripts if tmp_subscripts.count(s) == 1)

    contract_subscripts = f"{inputs}->{intm_modes}"
    decompose_subscripts = f"{intm_modes}->{outputs}"
    return contract_subscripts, decompose_subscripts

# NOTE: torch does not have native support on F order
# We here get around this by converting to CuPy/NumPy ndarrays as a workaround
# the overhead for torch tensors on GPU should be minimal as torch tensors support __cuda_array_interface__
def torch_support_wrapper(func):
    def new_func(T, *args, **kwargs):
        backend = infer_backend(T)
        if backend not in (cp, np):  # torch
            if T.device.type == 'cpu':
                out = func(T.numpy(), *args, **kwargs)
            else:
                out = func(cp.asarray(T), *args, **kwargs)
            return backend.as_tensor(out, device=T.device)
        return func(T, *args, **kwargs)
    return new_func

def get_einsum_kwargs(backend):
    if backend in (cp, np):
        return {'optimize': True}
    else:
        return {} # optimize not supported in torch.einsum


####################################
############ Execution #############
####################################

@torch_support_wrapper
def tensor_permute(T, input_modes, output_modes):
    axes = [input_modes.index(i) for i in output_modes]
    return T.transpose(*axes).astype(T.dtype, order="F")


@torch_support_wrapper
def tensor_reshape_fortran_order(T, shape):
    return T.reshape(shape, order='F')


def matrix_qr(T):
    backend = infer_backend(T)
    return backend.linalg.qr(T)


def matrix_svd(
    T, 
    max_extent=0,
    abs_cutoff=0,
    rel_cutoff=0,
    discarded_weight_cutoff=0,
    partition=None,
    normalization=None,
    return_info=True,
    **kwargs,
):
    info = dict()
    backend = infer_backend(T)
    if backend not in (cp, np) and T.device.type != 'cpu':
        u, s, v = backend.linalg.svd(T, full_matrices=False, driver='gesvd')
        if v.is_conj(): # VH from torch.linalg.svd is a view, we need to materialize it
            v = v.resolve_conj()
    else:
        u, s, v = backend.linalg.svd(T, full_matrices=False)
    info["full_extent"] = len(s)
    cutoff = max(abs_cutoff, rel_cutoff*s[0])
    if max_extent == 0 or max_extent is None:
        max_extent = len(s)
    if cutoff != 0:
        reduced_extent = min(max_extent, int((s>cutoff).sum()))
    else:
        reduced_extent = max_extent
    
    if discarded_weight_cutoff != 0:
        s_square_sum = backend.cumsum(s**2, 0)
        if backend not in (cp, np): # torch
            s_square_sum /= s_square_sum[-1].clone()
        else:
            s_square_sum /= s_square_sum[-1]
        dw_reduced_extent = int((s_square_sum<(1-discarded_weight_cutoff)).sum()) + 1
        reduced_extent = min(reduced_extent, dw_reduced_extent)
    reduced_extent = max(reduced_extent, 1)
    info["reduced_extent"] = reduced_extent
    if reduced_extent != len(s):
        sqrt_sum = backend.linalg.norm(s).item() ** 2
        if s[reduced_extent-1] >= REL_DEGENERACY_TOLERANCE * s[0] and s[reduced_extent] >= (1-REL_DEGENERACY_TOLERANCE) * s[reduced_extent-1]:
            raise SingularValueDegeneracyError("Truncation takes places at two or more degenerate singular values.")
        u = u[:, :reduced_extent]
        s = s[:reduced_extent]
        v = v[:reduced_extent]
        reduced_sqrt_sum = backend.linalg.norm(s).item() ** 2
        info["discarded_weight"] = 1 - reduced_sqrt_sum / sqrt_sum
    else:
        info["discarded_weight"] = 0.
    
    if normalization == "L1":
        s /= s.sum()
    elif normalization == "L2":
        s /= backend.linalg.norm(s)
    elif normalization == "LInf":
        s = s / s[0]
    elif normalization is not None:
        raise ValueError

    if partition == "U":
        u = backend.einsum("ab,b->ab", u, s)
        s = None
    elif partition == "V":
        v = backend.einsum("ab,a->ab", v, s)
        s = None
    elif partition == "UV":
        s_sqrt = backend.sqrt(s)
        u = backend.einsum("ab,b->ab", u, s_sqrt)
        v = backend.einsum("ab,a->ab", v, s_sqrt)
        s = None
    elif partition is not None:
        raise ValueError

    if return_info:
        return u, s, v, info
    else:
        return u, s, v


def tensor_decompose(
    split_expression, 
    T, 
    method='qr', 
    return_info=False, 
    **kwargs
):
    modes_in, left_modes, right_modes, shared_mode = parse_split_expression(split_expression)
    left_modes_intm = left_modes.replace(shared_mode, '') + shared_mode
    right_modes_intm = shared_mode + right_modes.replace(shared_mode, '')
    modes_in_intm = left_modes_intm[:-1] + right_modes_intm[1:]
    T_intm = tensor_permute(T, modes_in, modes_in_intm)
    left_shape = T_intm.shape[:len(left_modes)-1]
    right_shape = T_intm.shape[len(left_modes)-1:]
    m = np.prod(left_shape, dtype=np.int64)
    n = np.prod(right_shape, dtype=np.int64)
    T_intm = tensor_reshape_fortran_order(T_intm, (m, n))
    if method.lower() == 'qr':
        if kwargs:
            raise ValueError("QR does not support any options")
        if return_info:
            raise ValueError("No info for tensor QR")
        out_left, out_right = matrix_qr(T_intm)
    elif method.lower() == 'svd':
        out_left, s, out_right, info = matrix_svd(T_intm, return_info=True, **kwargs)
    else:
        raise NotImplementedError(f"{method} not supported")
    T_intm = tensor_reshape_fortran_order(T_intm, (m, n))
    out_left = tensor_reshape_fortran_order(out_left, tuple(left_shape)+(-1,))
    out_right = tensor_reshape_fortran_order(out_right, (-1, ) + tuple(right_shape))
    out_left = tensor_permute(out_left, left_modes_intm, left_modes)
    out_right = tensor_permute(out_right, right_modes_intm, right_modes)
    if method == "qr":
        return out_left, out_right
    else:
        if return_info:
            return out_left, s, out_right, info
        else:
            return out_left, s, out_right


def gate_decompose(
    split_expression, 
    array_a, 
    array_b, 
    array_g, 
    gate_algo="direct", 
    return_info=False, 
    **kwargs
):
    modes_in, left_modes_out, right_modes_out, shared_mode_out = parse_split_expression(split_expression)
    backend = infer_backend(array_a)
    einsum_kwargs = get_einsum_kwargs(backend)
    left_modes_in, right_modes_in, modes_g = modes_in.split(",")
    
    if gate_algo == "direct":
        modes_intm = infer_contracted_output_modes(modes_in)
        T = backend.einsum(f"{modes_in}->{modes_intm}", array_a, array_b, array_g, **einsum_kwargs)
        svd_expression = f"{modes_intm}->{left_modes_out},{right_modes_out}"
        return tensor_decompose(svd_expression, T, method='svd', return_info=return_info, **kwargs)
    elif gate_algo == "reduced":
        new_modes = get_new_modes(split_expression, 2)
        size_dict = dict(zip(left_modes_in, array_a.shape))
        size_dict.update(dict(zip(right_modes_in, array_b.shape)))
        shared_modes_in_ab = set(left_modes_in) & set(right_modes_in)
        modes_qa, modes_ra = prepare_reduced_qr_modes(left_modes_in, left_modes_out, new_modes[0], shared_modes_in_ab)
        modes_qb, modes_rb = prepare_reduced_qr_modes(right_modes_in, right_modes_out, new_modes[1], shared_modes_in_ab)
        skip_qr_a = compute_size(size_dict, modes_qa) <= compute_size(size_dict, modes_ra)
        skip_qr_b = compute_size(size_dict, modes_qb) <= compute_size(size_dict, modes_rb)
        if not skip_qr_a:
            qa, ra = tensor_decompose(f"{left_modes_in}->{modes_qa},{modes_ra}", array_a, method="qr")
        if not skip_qr_b:
            qb, rb = tensor_decompose(f"{right_modes_in}->{modes_qb},{modes_rb}", array_b, method="qr")
        intm_modes_in = f"{left_modes_in if skip_qr_a else modes_ra},{right_modes_in if skip_qr_b else modes_rb},{modes_g}"
        modes_rg = infer_contracted_output_modes(intm_modes_in)
        einsum_string = intm_modes_in + f"->{modes_rg}"
        T = backend.einsum(einsum_string, array_a if skip_qr_a else ra, array_b if skip_qr_b else rb, array_g, **einsum_kwargs)
        modes_rgu = ""
        modes_rgv = shared_mode_out
        for mode in modes_rg:
            if mode in left_modes_out or mode == new_modes[0]:
                modes_rgu += mode
            else:
                modes_rgv += mode
        modes_rgu += shared_mode_out
        svd_expression = f"{modes_rg}->{left_modes_out if skip_qr_a else modes_rgu},{right_modes_out if skip_qr_b else modes_rgv}"
        svd_outputs = tensor_decompose(svd_expression, T, method="svd", return_info=return_info, **kwargs)
        if skip_qr_a:
            u = svd_outputs[0]
        else:
            u = backend.einsum(f"{modes_qa},{modes_rgu}->{left_modes_out}", qa, svd_outputs[0])
        s = svd_outputs[1]
        if skip_qr_b:
            v = svd_outputs[2]
        else:
            v = backend.einsum(f"{modes_qb},{modes_rgv}->{right_modes_out}", qb, svd_outputs[2])
        if return_info:
            return u, s, v, svd_outputs[3]
        else:
            return u, s, v
    else:
        raise ValueError


####################################
########### Verification ###########
####################################

QR_TOLERANCE = {"float32": 1e-5,
                "float64": 1e-13,
                "complex64": 1e-5,
                "complex128": 1e-13}


SVD_TOLERANCE = {"float32": 7e-3,
                 "float64": 1e-13,
                 "complex64": 7e-3,
                 "complex128": 1e-13}


def get_tolerance(task, dtype):
    if hasattr(dtype, "name"):
        dtype = dtype.name
    else:
        dtype = str(dtype).split('.')[-1]
    if task == "qr":
        return QR_TOLERANCE[dtype]
    elif task in ["svd", "gate"]:
        return SVD_TOLERANCE[dtype]
    else:
        raise ValueError


def verify_close(
    array_a, 
    array_b, 
    rtol, 
    scale_by_norm=False, 
    scale_factor=1, 
    error_message=None
):
    backend = infer_backend(array_a)
    diff = backend.linalg.norm(array_a - array_b).item()
    if scale_by_norm:
        diff /= scale_factor * backend.linalg.norm(array_b).item()
    else:
        diff /= scale_factor
    is_close = diff < rtol
    if not is_close:
        array_diff = backend.abs(array_a - array_b).ravel()
        idx = backend.argmax(array_diff)
        if error_message:
            logging.error(error_message)
        else:
            logging.error("Large difference found in input tensors")
        logging.error(f"For a target rtol of {rtol}, diff max: {array_diff.max()} found at idx: {idx} (a[idx]: {array_a.ravel()[idx]}, b[idx]: {array_b.ravel()[idx]})")
    return is_close


def verify_unitary(
    T, 
    modes, 
    shared_mode, 
    rtol, 
    tensor_name="Tensor"
):
    backend = infer_backend(T)
    axes = get_tensordot_axes(modes, shared_mode)
    out = backend.tensordot(T, T.conj(), axes)
    if backend not in (cp, np): # torch
        identity = backend.eye(out.shape[0], device=T.device)
    else:
        identity = backend.eye(out.shape[0])
    error_message = f"{tensor_name} is not unitary"
    return verify_close(out, identity, rtol, False, out.shape[0], error_message)


def verify_upper_triangular(
    T, 
    modes, 
    shared_mode, 
    rtol, 
    tensor_name="Tensor"
):
    backend = infer_backend(T)
    shared_idx = modes.index(shared_mode)
    mid_extent = T.shape[shared_idx]
    axes = [shared_idx] + [idx for idx in range(len(modes)) if idx != shared_idx]
    T_intm = tensor_permute(T, modes, shared_mode+modes.replace(shared_mode, ''))
    T_intm = tensor_reshape_fortran_order(T_intm, (mid_extent, -1))
    error_message = f"{tensor_name} is not upper triangular"
    return verify_close(T_intm, backend.triu(T_intm), rtol, False, mid_extent, error_message)


def verify_split_QR(
    split_expression,
    T,
    array_q,
    array_r,
    array_q_ref,
    array_r_ref
):
    modes_in, left_modes, right_modes, shared_mode = parse_split_expression(split_expression)
    shared_mode_idx = left_modes.index(shared_mode)
    shared_extent = array_q.shape[shared_mode_idx]
    if T is not None:
        reference = T
    else:
        reference = reverse_einsum(split_expression, array_q_ref, None, array_r_ref)

    out = reverse_einsum(split_expression, array_q, None, array_r)
    rtol = get_tolerance("qr", out.dtype)

    is_equal = verify_close(reference, out, rtol, True, scale_factor=shared_extent, error_message="Contracted output is not close to the expected outcome")
    is_unitary = verify_unitary(array_q, left_modes, shared_mode, rtol, tensor_name="Output tensor Q")
    is_upper_triangular = verify_upper_triangular(array_r, right_modes, shared_mode, rtol, tensor_name="Output tensor R")
    return is_equal and is_unitary and is_upper_triangular

def verify_split_SVD(
    split_expression,
    T,
    array_u,
    array_s,
    array_v,
    array_u_ref,
    array_s_ref,
    array_v_ref,
    info=None,
    info_ref=None,
    **split_options
):
    # Note: this functions works for both SVD and Gate (specifying T to be None)
    modes_in, left_modes, right_modes, shared_mode = parse_split_expression(split_expression)
    shared_mode_idx = left_modes.index(shared_mode)
    shared_extent = array_u.shape[shared_mode_idx]
    try:
        max_mid_extent = min(array_u.size, array_v.size) // shared_extent
    except:
        # for torch
        max_mid_extent = min(array_u.numel(), array_v.numel()) // shared_extent 
    max_extent = split_options.pop('max_extent', max_mid_extent)
    if is_exact_split(**split_options) and max_extent == max_mid_extent and T is not None:
        reference = T
    else:
        reference = reverse_einsum(split_expression, array_u_ref, array_s_ref, array_v_ref)
    out = reverse_einsum(split_expression, array_u, array_s, array_v)
    if hasattr(out.dtype, "name"):
        dtype_name = out.dtype.name
    else:
        dtype_name = str(out.dtype).split('.')[-1]
    backend = infer_backend(out)
    rtol = get_tolerance("svd", out.dtype) # Note: tolerance for gate and svd is equal
    if info is not None:
        algorithm = info['algorithm']
    else:
        algorithm = 'gesvd'
    if algorithm == 'gesvdj':
        if dtype_name in ['float64', 'complex128']:
            rtol = 1e-6
        if 'gesvdj_residual' not in info:
            logging.warning("gesvdj_residual not recorded in info; verification may fail due to unknown runtime status")
        else:
            rtol = max(rtol, info['gesvdj_residual'] * max_mid_extent)
    elif algorithm == 'gesvdp':
        if dtype_name in ['float64', 'complex128']:
            rtol = 1e-6
        if 'gesvdp_err_sigma' not in info:
            logging.warning("gesvdp_err_sigma not recorded in info; verification may fail due to unknown runtime status")
        elif info['gesvdp_err_sigma'] > 1e-4:
            logging.warning(f"Large err sigma found for gesvdp: {info['gesvdp_err_sigma']}, skipping verification")
            return True
    elif algorithm == 'gesvdr':
        if dtype_name in ['float64', 'complex128']:
            rtol = 1e-4
        
    is_equal = verify_close(reference, out, rtol, True, scale_factor=shared_extent, error_message="Contracted output is not close to the expected outcome")

    partition = split_options.get("partition", None)
    if partition not in ["U", "V", "UV", None]:
        raise ValueError
    normalization = split_options.get("normalization", None)
    if normalization not in ["L1", "L2", "LInf", None]:
        raise ValueError
        
    is_s_equal = True
    left_tensordot_axes = get_tensordot_axes(left_modes, shared_mode)
    right_tensordot_axes = get_tensordot_axes(right_modes, shared_mode)
    if partition == "U":
        array_s = backend.sqrt(backend.tensordot(array_u, array_u.conj(), left_tensordot_axes,).diagonal().real)
        array_s_ref = backend.sqrt(backend.tensordot(array_u_ref, array_u_ref.conj(), left_tensordot_axes,).diagonal().real)
        array_u = backend.einsum(f"{left_modes},{shared_mode}->{left_modes}", array_u, 1.0/array_s)
    elif partition == "V":
        array_s = backend.sqrt(backend.tensordot(array_v, array_v.conj(), right_tensordot_axes,).diagonal().real)
        array_s_ref = backend.sqrt(backend.tensordot(array_v_ref, array_v_ref.conj(), right_tensordot_axes).diagonal().real)
        array_v = backend.einsum(f"{right_modes},{shared_mode}->{right_modes}", array_v, 1.0/array_s)
    elif partition == "UV":
        array_s = backend.tensordot(array_u, array_u.conj(), left_tensordot_axes).diagonal().real
        array_s1 = backend.tensordot(array_v, array_v.conj(), right_tensordot_axes,).diagonal().real
        array_s_ref = backend.tensordot(array_u_ref, array_u_ref.conj(), left_tensordot_axes).diagonal().real
        is_s_equal = verify_close(array_s, array_s1, rtol, scale_by_norm=True, scale_factor=shared_extent, error_message="Singular values from u and v are not equal")
        array_u = backend.einsum(f"{left_modes},{shared_mode}->{left_modes}", array_u, 1.0/backend.sqrt(array_s))
        array_v = backend.einsum(f"{right_modes},{shared_mode}->{right_modes}", array_v, 1.0/backend.sqrt(array_s))
    
    is_u_unitary = verify_unitary(array_u, left_modes, shared_mode, rtol, tensor_name="Output tensor U")
    is_v_unitary = verify_unitary(array_v, right_modes, shared_mode, rtol, tensor_name="Output tensor V")
    is_s_equal = is_s_equal and verify_close(array_s, array_s_ref, rtol, scale_by_norm=True, scale_factor=shared_extent, error_message="Output singular values not matching reference")
    info_equal = True
    if info is not None and info_ref is not None:
        for attr in ["full_extent", "reduced_extent"]:
            info_equal = info_equal and info[attr] == info_ref[attr]
        # For gesvdr, discarded weight is only computed when fix extent truncation is not enabled
        if info['algorithm'] != 'gesvdr' or max_extent == max_mid_extent:
            info_equal = info_equal and (abs(info["discarded_weight"]-info_ref["discarded_weight"]) < rtol)
        if is_dw_truncation_only(**split_options) and max_extent == max_mid_extent:
            # when only dw is in use, verify that discarded weight is less than the cutoff
            dw_cutn = info["discarded_weight"]
            dw_ref = info_ref["discarded_weight"]
            dw_cutoff = split_options.get('discarded_weight_cutoff', 0)
            if dw_cutn > dw_cutoff:
                logging.error("cutensornet SVD runtime discarded weight {dw_cutn} larger than cutoff {dw_cutoff}")
                return False
            if dw_ref > dw_cutoff:
                logging.error("reference SVD runtime discarded weight {dw_ref} larger than cutoff {dw_cutoff}")
                return False

    if not info_equal:
        info_details = "".join([f"{key}:({info.get(key)}, {info_ref.get(key)}); " for key in info.keys()])
        logging.error(f"SVD Info not matching the reference: {info_details}")
    return is_equal and is_u_unitary and is_v_unitary and is_s_equal and info_equal


def verify_split(
    split_expression,
    method,
    T,
    array_left,
    array_mid,
    array_right,
    array_left_ref,
    array_mid_ref,
    array_right_ref,
    info=None,
    info_ref=None,
    **split_options
):
    if method == "qr":
        return verify_split_QR(split_expression, T, array_left, array_right, array_left_ref, array_right_ref)
    elif method in ["gate", "svd"]:
        return verify_split_SVD(split_expression, T, 
                                array_left, array_mid, array_right, 
                                array_left_ref, array_mid_ref, array_right_ref,
                                info=info, info_ref=info_ref, **split_options)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    np.random.seed(3)
    T = np.random.random([2,2,2,2])
    split_expression = "abcd->axc,xdb"
    method = 'qr'
    
    q, r = tensor_decompose(split_expression, T, method=method)

    assert verify_split(split_expression, method, T, q, None, r, q, None, r)
