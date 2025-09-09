# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import numpy as np
from collections import Counter

from cuquantum.tensornet.experimental import NetworkState
from ...utils.circuit_ifc import PropertyComputeHelper
from ...utils.helpers import TensorBackend
def channel_infos_to_full_configuration(channel_info):
    if not channel_info:
        # No channels in the system
        return [None]
    channel_ids, channel_ops = zip(*sorted(channel_info.items()))
    full_configurations = []
    for op_ids in itertools.product(*[range(len(ops)) for ops, _ in channel_ops]):
        channels = []
        for channel_id, (ops, _), op_id in zip(channel_ids, channel_ops, op_ids):
            channels.append([channel_id, op_id, ops[op_id]])
        full_configurations.append(channels)
    return full_configurations

def parse_noisy_factory_state(factory, config):

    """
    outputs: full_configurations, conditional_probabilities, channel_info
    
    The methods perform the following:
        1. generate all configuration of potential channels in full_configuration
        2. create a reference NetworkState with all channels applied as a regular tensor gate (using first channel operand)
        3. compute the "conditional" probabilities for all configurations in conditional_probabilities

        # For example, for a system with 4 channels in total (u, g, u, g) where u denotes a unitary channel while g denotes a general channel

        # 1. Taking the third unitary channel as an example, 
        #    the keys are designed as (None, None, i) where the first two Nones means it's independent of the first two channels
        # 2. Taking the second general channel as an example, 
        #    the keys are designed as (i, j) which denotes the probability of the second general channel at index j given that the first unitary channel is at i
    """
    state_reference = NetworkState(factory.state_dims, dtype=factory.dtype, config=config)
    if factory.initial_mps_dim is not None:
        state_reference.set_initial_mps(factory.get_initial_state())

    conditional_probabilities = dict()        
    channel_info = dict()

    for op, modes, gate_info in factory.sequence:
        if gate_info is None and isinstance(op, (list, tuple)):
            # MPO, apply as it is
            state_reference.apply_mpo(modes, op)
        else:
            if gate_info is None:
                # regular Gate, apply as it is
                tensor_id = state_reference.apply_tensor_operator(modes, op)
            elif 'control_modes' in gate_info:
                assert 'control_values' in gate_info
                # Controlled-Tensor
                # NetworkState currently only support immutable controlled tensors
                tensor_id = state_reference.apply_tensor_operator(modes, op, control_modes=gate_info['control_modes'], control_values=gate_info['control_values'], immutable=True)
            else:
                # Unitary/General Channels
                assert 'probabilities' in gate_info
                probs = gate_info['probabilities']
                is_unitary_channel = probs is not None
                if is_unitary_channel:
                    # For unitary channels, register the conditional probablities
                    for ix, p in enumerate(probs):
                        key = (None, ) * len(channel_info) + (ix, )
                        conditional_probabilities[key] = p
                else:
                    # enumerate all potential configurations (including both unitary & general channels) till this general channel
                    channels_to_update = channel_infos_to_full_configuration(channel_info)

                    for channel in channels_to_update:
                        base_key = []
                        if channel is not None:
                            for channel_id, op_id, channel_op in channel:
                                base_key.append(op_id)
                                # normalize the channels to improve numerical stability
                                probs = channel_info[channel_id][1]
                                if probs is not None:
                                    # Unitary Channel
                                    state_reference.update_tensor_operator(channel_id, channel_op, unitary=True)
                                else:
                                    # General Channel
                                    factor = conditional_probabilities.get(tuple(base_key), 1) ** .5
                                    state_reference.update_tensor_operator(channel_id, channel_op / factor, unitary=False)
                        rdm = state_reference.compute_reduced_density_matrix(modes)
                        if hasattr(rdm, 'numel'):
                            # for torch tensor
                            matrix_dim = int(rdm.numel()**.5)
                        else:
                            matrix_dim = int(rdm.size**.5)
                        rdm = rdm.reshape(matrix_dim, matrix_dim)
                        p_tot = factory.backend.einsum('ii->', rdm).real.item()
                        for ix, general_channel in enumerate(op):
                            general_channel = general_channel.reshape(matrix_dim, matrix_dim)
                            p = factory.backend.einsum('ij,ik,kj->', general_channel, general_channel.conj(), rdm).real.item()
                            key = tuple(base_key) + (ix,)
                            conditional_probabilities[key] = p / p_tot  
                # Now just apply the general channel as a regular tensor gate
                tensor_id = state_reference.apply_tensor_operator(modes, op[0], unitary=is_unitary_channel)
                channel_info[tensor_id] = (op, gate_info['probabilities'])

    full_configurations = channel_infos_to_full_configuration(channel_info)
    return state_reference, full_configurations, conditional_probabilities, channel_info


def get_probability(conditional_probabilities, key):
    # P(key) = P(key[:-1]) * P(key[-1] | key[:-1])
    # where P(key[:-1]) means the probability of key[:-1] being active/selected
    # P(key[-1] | key[:-1]) means the conditional probability that when key[:-1] are active/selected, key[-1] is selected
    if key:
        prev_channels = key[:-1]
        active_channel = key[-1]
        if key in conditional_probabilities:
            # general channel
            conditional_prob = conditional_probabilities[key]
        else:
            # unitary channel
            unitary_key = (None, ) * len(prev_channels) + (active_channel, )
            assert unitary_key in conditional_probabilities
            conditional_prob = conditional_probabilities[unitary_key]
        return conditional_prob * get_probability(conditional_probabilities, prev_channels)
    else:
        return 1.0
        
def compute_full_sv_reference(state_reference, full_configurations, conditional_probabilities, channel_info, force_numpy=False):
    data = []
    for entry in full_configurations:
        active_channels = []
        for (channel_id, op_id, operand) in entry:
            # normalize the channels
            active_channels.append(op_id)
            probs = channel_info[channel_id][1]
            if probs is not None:
                # Unitary Channel
                state_reference.update_tensor_operator(channel_id, operand, unitary=True)
            else:
                # General Channel
                factor = conditional_probabilities.get(tuple(active_channels), 1) ** .5
                state_reference.update_tensor_operator(channel_id, operand / factor, unitary=False)
        p = get_probability(conditional_probabilities, tuple(active_channels))
        output = state_reference.compute_state_vector()
        if force_numpy:
            output = TensorBackend.to_numpy(output)
        data.append([p, output])
    return data

def compute_noisy_sv(factory, config, *, return_probabilities=False, force_numpy=False):
    state_reference, full_configurations, conditional_probabilities, channel_info = parse_noisy_factory_state(factory, config)
    with state_reference:
        data = compute_full_sv_reference(state_reference, full_configurations, conditional_probabilities, channel_info, force_numpy=force_numpy)
    if return_probabilities:
        return data
    else:
        return [output for _, output in data]

def verify_state_sampling(state, modes, nshots, svs, max_trial, **kwargs):
    is_sv_sequence = isinstance(svs, (list, tuple))
    if modes is None or isinstance(modes[0], int):
        normalized_modes = modes
    else:
        normalized_modes = [state.state_labels.index(q) for q in modes]
    overlap_best = 0.
    for i in range(max_trial):
        # seed must be positive integer
        samples = state.compute_sampling(nshots, modes=modes, seed=i+1, **kwargs)
        if is_sv_sequence:
            overlaps = [PropertyComputeHelper.compute_sampling_overlap(samples, sv, target_qubits=normalized_modes) for sv in svs]
            overlap_max = max(overlaps)
        else:
            overlap_max = PropertyComputeHelper.compute_sampling_overlap(samples, svs, target_qubits=normalized_modes)
        if overlap_max > overlap_best:
            overlap_best = overlap_max
        else:
            print(f"WARNING: overlap not improving {overlap_best=} {overlap_max=} as nshots increases!")
        # to reduce test time we set 95% here, but 99% will also work
        if np.round(overlap_max, decimals=2) < 0.95:
            nshots *= 10
            print(f"{overlap_max=}, retry with nshots = {nshots} ...")
        else:
            print(f"{overlap_max=} with nshots = {nshots}")
            break
    else:
        raise AssertionError(f"{overlap_best=} after {max_trial} trials...")
def analyze_trajectory_deviation(traj_results, reference_results, probs):
    """
    traj_results: A sequence of scalar for trajectory simulation results
    reference_results: A sequence of scalar for all reference results
    probs: A sequence of scalar for the probabilities of the configurations corresponding to the reference results
    """
    traj_array = np.asarray(traj_results).reshape(-1, 1)
    reference_array = np.asarray(reference_results).reshape(1, -1)
    deviation = np.abs(reference_array - traj_array)
    args = np.argmin(deviation, axis=1)
    count = Counter(args.tolist())
    prob_deviation = 0
    for i, p in enumerate(probs):
        prob_deviation += np.abs(p - count.get(i, 0) / args.size)
    return deviation.min(axis=1).max(), prob_deviation