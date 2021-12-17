"""
A collection of functions for parsing Einsum expressions.
"""

import numpy as np

from .tensor_wrapper import wrap_operands


def parse_einsum_str(expr):
    """
    Parse einsum expression. Note that no validity checks are performed.

    Return operand as well as output indices if explicit mode or None for implicit mode.
    """
    inputs, output = expr.split('->') if "->" in expr else (expr, None)

    ellipses = '...' in inputs
    if ellipses:
        raise ValueError("Ellipsis broadcasting is not supported.")

    inputs = tuple(tuple(_input) for _input in inputs.split(","))

    return inputs, output


def parse_einsum_interleaved(operand_sublists):
    """
    Parse einsum expression in interleaved format. Note that no validity checks are performed.

    Return operands as well as output indices if explicit mode or None for implicit mode.
    """
    inputs   = list()
    operands = list()

    N = len(operand_sublists) // 2
    for i in range(N):
        operands.append(operand_sublists[2*i])
        inputs.append(operand_sublists[2*i + 1])
    
    N = len(operand_sublists)
    output = operand_sublists[N-1] if N % 2 == 1 else None

    ellipses = [Ellipsis in _input for _input in inputs]
    if any(ellipses):
        raise ValueError("Ellipsis broadcasting is not supported.")

    return operands, inputs, output


def map_modes(user_inputs, user_output):
    """
    Map modes in user-defined inputs and output to ordinals. Create the forward as well as inverse maps.

    Return mapped inputs and output along with the forward and reverse maps.
    """

    ordinal = 0
    mode_map_user_to_ord = dict()
    for modes in user_inputs:
        for mode in modes:
            if mode not in mode_map_user_to_ord:
                mode_map_user_to_ord[mode] = ordinal
                ordinal += 1

    mode_map_ord_to_user = { v : k for k, v in mode_map_user_to_ord.items() }

    inputs = tuple(tuple(mode_map_user_to_ord[m] for m in modes) for modes in user_inputs)

    output = None
    if user_output is not None:
        extra = set(user_output) - set(mode_map_user_to_ord.keys())
        if extra:
            output_modes = "'{}'".format(user_output) if isinstance(user_output, str) else user_output
            message = f"""Extra modes in output.
The specified output modes {output_modes} contain the extra modes: {extra}"""
            raise ValueError(message)
        output = tuple(mode_map_user_to_ord[m] for m in user_output) 

    return inputs, output, mode_map_user_to_ord, mode_map_ord_to_user


def check_einsum_with_operands(user_inputs, operands, interleaved):
    """
    Check that the number of modes in each Einsum term is consistent with the shape of the corresponding operand.
    operands == wrapped
    user_inputs = *before* mapping
    """

    checks = [len(i) == len(o.shape) for i, o in zip(user_inputs, operands)]
    if not all(checks):
        morpher =  (lambda s : tuple(s)) if interleaved else lambda s : "'" + ''.join(s) + "'"
        mismatch = [f"{location}: {morpher(user_inputs[location])} <=> {operands[location].shape}" 
                        for location, predicate in enumerate(checks) if predicate is False]
        mismatch = np.array2string(np.array(mismatch, dtype='object'), separator=', ', formatter={'object': lambda s: s})
        message = f"""Term-operand shape mismatch.
The number of modes in each term of the expression must match the shape of the corresponding operand.
The mismatch in the number of modes as a sequence of "operand position: modes in term <=> operand shape" is: \n{mismatch}"""
        raise ValueError(message)


def create_size_dict(inputs, operands):
    """
    Create size dictionary capturing the extent of each mode.
    inputs = based on renumbered modes.
    """

    size_dict = dict()
    for i, _input in enumerate(inputs):
        for m, mode in enumerate(_input):
            shape = operands[i].shape
            if mode in size_dict:
                if size_dict[mode] == 1:    # Handle broadcasting
                    size_dict[mode] = shape[m]
                elif size_dict[mode] != shape[m]:
                    message = f"""Extent mismatch.
The extent ({shape[m]}) of mode {m} for operand {i} does not match the extent ({size_dict[mode]}) of the same mode found
in previous operand(s)."""
                    raise ValueError(message)
            else:
                size_dict[mode] = shape[m]

    return size_dict


def calculate_mode_frequency(inputs):
    """
    Calculate the number of times a mode appears in the operand list.
    """
    from collections import defaultdict
    mode_frequency = defaultdict(int)

    for index, modes in enumerate(inputs):
        for mode in modes:
            mode_frequency[mode] += 1

    return mode_frequency


def check_classical_einsum(mode_frequency, output, mode_map_user_to_ord, mode_map_ord_to_user):
    """
    Check if classical Einsum. Also infer output indices (all the modes that appear exactly once).
    """

    single_modes = set()
    double_modes = set()
    rest = set()
    for mode, frequency in mode_frequency.items():
        if frequency == 1:
            single_modes.add(mode)
        elif frequency == 2:
            double_modes.add(mode)
        else:
            rest.add(mode)

    if rest:
        rest = tuple(mode_map_ord_to_user[r] for r in rest)
        message = f"""No generalized Einsum support.
These modes appear more than twice: {rest}"""
        raise ValueError(message)

    if output is None:
        # Implicit mode: lexical sort based on user mode labels.
        output = sorted(mode_map_ord_to_user[m] for m in single_modes)
        output = tuple(mode_map_user_to_ord[m] for m in output)
        return output

    output_set = set(output)

    missing = set(mode_map_ord_to_user[m] for m in single_modes - output_set)
    if missing:
        message = f"""No generalized Einsum support.
These single modes must appear in the output: {missing}"""
        raise ValueError(message)

    common = set(mode_map_ord_to_user[c] for c in output_set & double_modes)
    if common:
        message = f"""No generalized Einsum support.
These double modes must not appear in the output: {common}"""
        raise ValueError(message)

    return output


def parse_einsum(*operands):
    """
    Classical Einsum definition: modes that appear twice are summed over and those that appear once must appear in the output.
    Recognizes both string and interleaved formats. Any hashable type is accepted in interleaved format for mode specification, 
    and unicode strings are accepted. If the output is not provided (implicit form or missing output sublist), it will be 
    inferred from the expression.

    Returns wrapped operands, mapped inputs and output, size dictionary based on internal mode numbers, and the forward as 
    well as the reverse mode maps.
    """

    interleaved = False
    if isinstance(operands[0], str):
        inputs, output = parse_einsum_str(operands[0])
        operands = operands[1:]
    else:
        interleaved = True
        operands, inputs, output = parse_einsum_interleaved(operands)

    num_operand, num_input = len(operands), len(inputs)
    if num_operand != num_input:
        message = f"""Operand-term mismatch.
The number of operands ({num_operand}) must match the number of inputs ({num_input}) specified in the Einsum expression."""
        raise ValueError(message)

    if num_operand < 2:
        message = "The network must consist of at least two tensors."
        raise ValueError(message)

    # First wrap operands.
    operands = wrap_operands(operands)

    # Basic check to ensure that the number of modes is consistent with the operand shape.
    check_einsum_with_operands(inputs, operands, interleaved)

    # Map data to ordinals for cutensornet.
    inputs, output, mode_map_user_to_ord, mode_map_ord_to_user = map_modes(inputs, output)

    # Create mode-extent map based on internal mode numbers.
    size_dict = create_size_dict(inputs, operands)

    # Create output modes if not specified.
    mode_frequency = calculate_mode_frequency(inputs)

    # Finally, check if the expression is a classical Einsum. Calculate output indices in implicit mode (output=None).
    output = check_classical_einsum(mode_frequency, output, mode_map_user_to_ord, mode_map_ord_to_user)

    return operands, inputs, output, size_dict, mode_map_user_to_ord, mode_map_ord_to_user


