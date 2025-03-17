# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


from cuquantum.tensornet._internal.einsum_parser import infer_output_mode_labels


def is_gate_split(inputs, outputs, algo):
    """
    Check if the input and output modes refers to a GateSplit problem.

    Args:
        inputs: Einsum expression in "neutral format" (sequence of sequences) after mapping.
        outputs: Einsum expression in "neutral format" (sequence of sequences) after mapping.
        algo: The algorithm specified for the contract and decompose operations
    """
    if algo.svd_method is False: # contract QR decompose
        return False

    if infer_output_mode_labels(inputs) != infer_output_mode_labels(outputs):
        return False
    
    if len(inputs) == 3:
        # This requires:
        #   1. input A/B/G fully connected
        #   2. The number of opens/uncontracted modes of G parititioned onto output A and B must be non-zero
        a_in, b_in, g, a_out, b_out = map(set, inputs+outputs)
        ab_in = a_in & b_in
        ag_in = a_in & g
        bg_in = b_in & g
        g_open = g - ag_in - bg_in
        ag_out = g_open & a_out
        bg_out = g_open & b_out
        return all([ab_in, ag_in, bg_in, ag_out, bg_out])

    return False


def maybe_truncate_qr_output_operands(operands, modes, mid_extent):
    """
    Given the output operands and modes for QR decomposition, possibly truncate the mid extent of the operands to match the specified mid extent.
    """
    shared_mode = (set(modes[0]) & set(modes[1])).pop()
    truncated_operands = []
    for o, labels in zip(operands, modes):
        idx =  labels.index(shared_mode)
        if o.shape[idx] == mid_extent:
            return operands
        slices = [slice(None)] * o.ndim
        slices[idx] = slice(0, mid_extent)
        truncated_operands.append(o[tuple(slices)])
    return truncated_operands
