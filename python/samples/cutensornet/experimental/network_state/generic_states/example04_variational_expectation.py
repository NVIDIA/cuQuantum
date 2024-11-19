# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
An example to show how to leverage the caching feature of contraction based tensor network simulation for a VQC-like workflow.
Consider the case where the task is to compute the expectation value for a fixed Hamiltonian on two states with identical topology but slightly different operands, 
for instance, state_a and state_b below:

                       state_a                          state_b
=======================================================================           
Vacuum:         A   B   C   D   E   F            A   B   C   D   E   F
                |   |   |   |   |   |            |   |   |   |   |   |
one body op     O   O   O   O   O   O            O   O   O   O   O   O
                |   |   |   |   |   |            |   |   |   |   |   |
two body op     XXXXX   XXXXX   XXXXX            YYYYY   YYYYY   YYYYY
                |   |   |   |   |   |            |   |   |   |   |   |
two body op     |   XXXXX   XXXXX   |            |   YYYYY   YYYYY   |
                |   |   |   |   |   |            |   |   |   |   |   |
two body op     XXXXX   XXXXX   XXXXX            YYYYY   YYYYY   YYYYY
                |   |   |   |   |   |            |   |   |   |   |   |
two body op     |   XXXXX   XXXXX   |            |   YYYYY   YYYYY   |
                |   |   |   |   |   |            |   |   |   |   |   |

While we can also compute the expectation value independently on two different states, we may also use the ``NetworkState.update_tensor_operator`` method 
to directly update the two body operators in state_a from X to Y. If ``NetworkState.compute_expectation`` has already been performed on state_a, 
a second call would benefit from the caching feature to avoid the preparation overhead and directly compute the expectation for state_b.

Note that this feature only applies to contraction based simulation and MPS simulation without value based truncations.
"""
import cupy as cp

from cuquantum.cutensornet.experimental import NetworkState, NetworkOperator

# specify the dimensions of the tensor network state
n_state_modes = 6
state_mode_extents = (2, ) * n_state_modes
dtype = 'complex128'

# create random operators
cp.random.seed(4)
random_complex = lambda *args, **kwargs: cp.random.random(*args, **kwargs) + 1.j * cp.random.random(*args, **kwargs)
op_one_body = random_complex((2, 2,))
op_two_body_x = random_complex((2, 2, 2, 2))
op_two_body_y = random_complex((2, 2, 2, 2))

# create an emtpy NetworkState object, by default it will tensor network contraction as simulation method
config = {'max_extent': 2} # also works for contraction based simulation with config=None
state_a = NetworkState(state_mode_extents, dtype=dtype, config=config)
state_b = NetworkState(state_mode_extents, dtype=dtype, config=config)

# apply one body tensor operators to the tensor network state
for i in range(n_state_modes):
    modes_one_body = (i, )
    # op_one_body are fixed, therefore setting immutable to True
    tensor_id_a = state_a.apply_tensor_operator(modes_one_body, op_one_body, immutable=True)
    tensor_id_b = state_b.apply_tensor_operator(modes_one_body, op_one_body, immutable=True)
    assert tensor_id_a == tensor_id_b
    print(f"Apply one body operator to {modes_one_body} of both states, tensor id {tensor_id_a}")

two_body_op_ids = []
# apply two body tensor operators to the tensor network state
for i in range(4):
    for site in range(i, n_state_modes, 2):
        if site + 1 < n_state_modes:
            modes_two_body = (site, site+1)
            # op_two_body differs between state_a and state_b, therefore setting immutable to False
            tensor_id_a = state_a.apply_tensor_operator(modes_two_body, op_two_body_x, immutable=False)
            tensor_id_b = state_b.apply_tensor_operator(modes_two_body, op_two_body_y, immutable=False)
            assert tensor_id_a == tensor_id_b
            print(f"Apply two body operator to {modes_two_body} of both states, tensor id {tensor_id_a}")
            two_body_op_ids.append(tensor_id_a)

# compute the normalized expectation value for a series of Pauli operators
# Note that while pauli_strings can also be provided directly as the operator argument for NetworkState.compute_expectation, 
# caching for expectation value computation can only be properly activated by explicitly providing a NetworkOperator object.
pauli_string = {'IXIXIX': 0.5, 'IYIYIY': 0.2, 'IZIZIZ': 0.3}
operator = NetworkOperator.from_pauli_strings(pauli_string, dtype='complex128')

with state_a, state_b:

    e0 = cp.cuda.Event()
    e1 = cp.cuda.Event()
    e0.record()
    expec_a, norm_a = state_a.compute_expectation(operator, return_norm=True)
    expec_a = expec_a.real / norm_a
    e1.record()
    e1.synchronize()
    print(f"Normalized expectation for state_a from direct computation : {expec_a}, runtime={cp.cuda.get_elapsed_time(e0, e1)} ms")

    expec_b, norm_b = state_b.compute_expectation(operator, return_norm=True)
    expec_b = expec_b.real / norm_b
    e0.record()
    e0.synchronize()
    print(f"Normalized expectation for state_b from direct computation : {expec_b}, runtime={cp.cuda.get_elapsed_time(e1, e0)} ms")

    for tensor_id in two_body_op_ids:
        state_a.update_tensor_operator(tensor_id, op_two_body_y, unitary=False)
        print(f"Update two body operator ({tensor_id}) from X to Y in state_a")

    expec_b_updated, norm_b_updated = state_a.compute_expectation(operator, return_norm=True)
    expec_b_updated = expec_b_updated.real / norm_b_updated
    e1.record()
    e1.synchronize()
    print(f"Normalized expectation for state_b from updating state_a : {expec_b_updated}, runtime={cp.cuda.get_elapsed_time(e0, e1)} ms")
    assert cp.allclose(expec_b, expec_b_updated)
