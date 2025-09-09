# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import opt_einsum as oe
from nvmath.internal.utils import infer_object_package, get_or_create_stream

from cuquantum.tensornet import contract
from cuquantum.tensornet.experimental import NetworkOperator
from cuquantum.tensornet._internal.circuit_converter_utils import get_pauli_gates

from .helpers import TensorBackend

def get_partial_indices(n, projected_modes=None):
    if projected_modes is None:
        projected_modes = {}
    partial_indices = [slice(None)] * n
    for q, val in projected_modes.items():
        bit = int(val)
        partial_indices[q] = slice(bit, bit+1)
    return tuple(partial_indices)

class PropertyComputeHelper:

    @staticmethod
    def amplitude_from_sv(sv, bitstring):
        index = [int(ibit) for ibit in bitstring]
        return sv[tuple(index)].item()

    @staticmethod
    def batched_amplitudes_from_sv(sv, fixed, *, qubits=None):
        if qubits is not None:
            # requires a mapping from qubits to indices
            qubits = list(qubits)
            fixed = {qubits.index(q): int(bit) for q, bit in fixed.items()}
        n = sv.ndim
        shape = [sv.shape[i] for i in range(n) if i not in fixed]
        sv = sv[get_partial_indices(n, fixed)]
        return sv.reshape(shape)
    
    @staticmethod
    def mpo_expectation_from_sv(sv, mpo_tensors, mpo_modes):
        if infer_object_package(sv) != infer_object_package(mpo_tensors[0]):
            sv = TensorBackend.to_numpy(sv)
            mpo_tensors = [TensorBackend.to_numpy(t) for t in mpo_tensors]
        n = sv.ndim
        mode_frontier = n
        modes = list(range(n))
        current_modes = modes.copy()
        operands = [sv, modes]
        prev_mode = None
        for i, m in enumerate(mpo_modes):
            ket_mode = current_modes[m]
            current_modes[m] = bra_mode = mode_frontier
            mode_frontier += 1
            next_mode = mode_frontier
            mode_frontier += 1
            if i == 0:
                operands += [mpo_tensors[i], (ket_mode, next_mode, bra_mode)]
            elif i == len(mpo_modes) - 1:
                operands += [mpo_tensors[i], (prev_mode, ket_mode, bra_mode)]
            else:
                operands += [mpo_tensors[i], (prev_mode, ket_mode, next_mode, bra_mode)]
            prev_mode = next_mode
        if infer_object_package(sv) != 'torch':
            operands += [sv.conj(), current_modes]
        else:
            operands += [sv.conj().resolve_conj(), current_modes]
        return contract(*operands).item()
    
    @staticmethod
    def product_expectation_from_sv(sv, prod_tensors, prod_modes):
        if infer_object_package(sv) != infer_object_package(prod_tensors[0]):
            sv = TensorBackend.to_numpy(sv)
            prod_tensors = [TensorBackend.to_numpy(t) for t in prod_tensors]
        n = sv.ndim
        mode_frontier = n
        modes = list(range(n))
        current_modes = modes.copy()
        operands = [sv, modes]
        for i, ms in enumerate(prod_modes):
            ket_modes = []
            bra_modes = []
            for m in ms:
                ket_modes.append(current_modes[m])
                bra_modes.append(mode_frontier)
                current_modes[m] = mode_frontier
                mode_frontier += 1
            operands += [prod_tensors[i], ket_modes + bra_modes]
        if infer_object_package(sv) != 'torch':
            operands += [sv.conj(), current_modes]
        else:
            operands += [sv.conj().resolve_conj(), current_modes]
        return contract(*operands).item()

    @staticmethod
    def expectation_from_sv(sv, operator):
        if isinstance(operator, str):
            # operator is a single Pauli string
            n = sv.ndim
            pauli_map = dict(zip(range(n), operator))
            backend_name = infer_object_package(sv)
            pauli_gates, gates_are_diagonal = get_pauli_gates(pauli_map, backend_name, sv.dtype)
            # bra/ket indices
            if backend_name == 'torch':
                inputs = [sv, list(range(n)), sv.conj().resolve_conj(), list(range(n))]
            else:
                inputs = [sv, list(range(n)), sv.conj(), list(range(n))]
            for (o, qs), is_diagonal in zip(pauli_gates, gates_are_diagonal):
                q = qs[0]
                if backend_name == 'torch' and str(sv.device) == 'cpu':
                    o = o.to(device=sv.device)
                if is_diagonal:
                    inputs.extend([o.diagonal(), [q,]])
                else:
                    inputs[3][q] += n # update ket indices
                    inputs.extend([o, [q+n, q]])
            return oe.contract(*inputs).item()
        elif isinstance(operator, NetworkOperator):
            expec = 0
            stream_holder = None
            for (mpo_tensors, modes, coeff) in operator.mpos:
                if mpo_tensors[0].name == "cuda":
                    if stream_holder is None:
                        stream_holder = get_or_create_stream(mpo_tensors[0].device_id, None, 'cuda')
                    mpo_tensors = [o.to('cpu', stream_holder).tensor for o in mpo_tensors]
                else:
                    mpo_tensors = [o.tensor for o in mpo_tensors]
                expec += coeff * PropertyComputeHelper.mpo_expectation_from_sv(sv, mpo_tensors, modes)
            for (prod_tensors, modes, coeff) in operator.tensor_products:
                if prod_tensors[0].name == "cuda":
                    if stream_holder is None:
                        stream_holder = get_or_create_stream(prod_tensors[0].device_id, None, 'cuda')
                    prod_tensors = [o.to('cpu', stream_holder).tensor for o in prod_tensors]
                else:
                    prod_tensors = [o.tensor for o in prod_tensors]
                expec += coeff * PropertyComputeHelper.product_expectation_from_sv(sv, prod_tensors, modes)
            return expec
        else:
            # a dictionary of Pauli strings and coefficients
            expec = 0
            for pauli_string, coeff in operator.items():
                expec += PropertyComputeHelper.expectation_from_sv(sv, pauli_string) * coeff
            return expec

    @staticmethod
    def reduced_density_matrix_from_sv(sv, where, *, fixed=None, qubits=None):
        if qubits is not None:
            # requires a mapping from qubits to indices
            qubits = list(qubits)
            where = [qubits.index(q) for q in where]
            fixed = {qubits.index(q): int(bit) for q, bit in fixed.items()}
        n = sv.ndim
        sv = sv[get_partial_indices(n, projected_modes=fixed)]
        bra_modes = list(range(n))
        ket_modes = [i+n if i in where else i for i in range(n)]
        output_modes = list(where) + [i+n for i in where]
        if infer_object_package(sv) == 'torch':
            inputs = [sv, bra_modes, sv.conj().resolve_conj(), ket_modes]
        else:
            inputs = [sv, bra_modes, sv.conj(), ket_modes]
        inputs.append(output_modes)
        return oe.contract(*inputs)

    @staticmethod
    def probability_from_sv(sv, modes_to_sample):
        backend = infer_object_package(sv)
        p = abs(sv) ** 2
        # convert p to double type in case probs does not add up to 1
        if backend == 'numpy':
            p = p.astype('float64')
        elif backend == 'cupy':
            p = p.get().astype('float64')
        elif backend == 'torch':
            if p.device.type == 'cpu':
                p = p.numpy().astype('float64')
            else:
                p = p.cpu().numpy().astype('float64')
        if modes_to_sample is not None:
            sorted_modes_to_sample = sorted(modes_to_sample)
            axis = [q for q in range(sv.ndim) if q not in modes_to_sample]
            if axis:
                p = p.sum(tuple(axis))
            transpose_order = [sorted_modes_to_sample.index(q) for q in modes_to_sample]
            p = p.transpose(*transpose_order)
        # normalize
        p /= p.sum()
        return p

    @staticmethod
    def compute_sampling_overlap(samples, sv, *, target_qubits=None, qubits=None):
        if qubits is not None and target_qubits is not None:
            modes_to_sample = [qubits.index(q) for q in target_qubits]
        else:
            modes_to_sample = target_qubits
        p = PropertyComputeHelper.probability_from_sv(sv, modes_to_sample)
        distribution = np.zeros(p.shape, dtype=p.dtype)
        for bitstring, count in samples.items():
            index = tuple(int(i) for i in bitstring)
            distribution[index] = count
        nshots = distribution.sum()
        distribution /= nshots
        ovlp = np.minimum(p, distribution).sum()
        return ovlp


class CircuitHelper:

    @staticmethod
    def get_qubits(circuit):
        package = infer_object_package(circuit)
        if package == 'cirq':
            return sorted(circuit.all_qubits())
        elif package == 'qiskit':
            return list(circuit.qubits)

    @staticmethod
    def compute_state_vector(circuit, *, dtype="complex128"):
        package = infer_object_package(circuit)
        if package == 'cirq':
            import cirq
            from cuquantum.tensornet._internal import circuit_parser_utils_cirq
            qubits = sorted(circuit.all_qubits())
            simulator = cirq.Simulator(dtype=dtype)
            circuit = circuit_parser_utils_cirq.remove_measurements(circuit)
            result = simulator.simulate(circuit, qubit_order=qubits)
            statevector = result.state_vector().reshape((2,)*len(qubits))
            return np.asarray(statevector)
        elif package == 'qiskit':
            from qiskit.quantum_info import Statevector
            from cuquantum.tensornet._internal import circuit_parser_utils_qiskit
            qubits = list(circuit.qubits)
            circuit = circuit_parser_utils_qiskit.remove_measurements(circuit)
            # A WAR with Qiskit bug: https://github.com/Qiskit/qiskit/issues/13778
            # This currently does not support precision arg
            result = Statevector.from_instruction(circuit).data
            sv = np.asarray(result).reshape((2,)*circuit.num_qubits)
            # statevector returned by qiskit's simulator is labelled by the inverse of :attr:`qiskit.QuantumCircuit.qubits`
            # this is different from `cirq` and different from the implementation in :class:`CircuitToEinsum`
            sv = sv.transpose(list(range(circuit.num_qubits))[::-1])
            return sv
        else:
            raise ValueError(f"Unsupported package: {package}")

    @staticmethod
    def bitstring_iterator(qudits, num_bitstrings, rng):
        if isinstance(qudits, int):
            state_dims = (2, ) * qudits
        else:
            state_dims = qudits
        for _ in range(num_bitstrings):
            bitstring = ''.join([str(rng.integers(0, dim)) for dim in state_dims])
            yield bitstring

    @staticmethod
    def fixed_iterator(qubits, num_cases, rng, *, lower_bound=1, state_dims=None):
        qubits = list(qubits)
        n_qubits = len(qubits)
        if state_dims is None:
            state_dims = (2, ) * n_qubits
        for _ in range(num_cases):
            qubits_ = list(qubits)
            num_fixed_qubits = rng.integers(lower_bound, n_qubits)
            rng.shuffle(qubits_)
            fixed_qubits = qubits_[:num_fixed_qubits]
            fixed_state_dims = [state_dims[qubits.index(q)] for q in fixed_qubits]
            fixed_values = next(CircuitHelper.bitstring_iterator(fixed_state_dims, 1, rng))
            fixed = dict(zip(fixed_qubits, fixed_values))
            yield fixed

    @staticmethod
    def where_fixed_iterator(qubits, num_cases, rng, *, state_dims=None):
        if state_dims is None:
            state_dims = (2, ) * len(qubits)
        n_qubits = len(qubits)
        for _ in range(num_cases):
            qubits_ = list(qubits)
            rng.shuffle(qubits_)
            num_target_qubits = rng.integers(1, n_qubits)
            where = qubits_[:num_target_qubits]
            fixed_state_dims = [state_dims[qubits.index(q)] for q in qubits_[num_target_qubits:]]
            fixed = next(CircuitHelper.fixed_iterator(qubits_[num_target_qubits:], 1, rng, lower_bound=0, state_dims=fixed_state_dims))
            yield where, fixed

    @staticmethod
    def get_random_pauli_strings(n, num_pauli_strings, rng):
        def _get_pauli_string():
            return ''.join(rng.choice(['I','X', 'Y', 'Z'], n))
    
        if num_pauli_strings is None:
            return _get_pauli_string()
        else:
            # return in dictionary format
            pauli_strings = {}
            for _ in range(num_pauli_strings):
                pauli_string = _get_pauli_string()
                coeff = rng.random() + rng.random() * 1j
                if pauli_string in pauli_strings:
                    pauli_strings[pauli_string] += coeff
                else:
                    pauli_strings[pauli_string] = coeff
            return pauli_strings


def assert_any_with_batch(func):
    def func_with_assert(*args, **kwargs):
        svs = args[0]
        if isinstance(svs, (list, tuple)):
            results = [func(sv, *args[1:], **kwargs) for sv in svs]
            assert any(results)
        else:
            assert func(*args, **kwargs)
    return func_with_assert

class QuantumStateTestHelper:

    @staticmethod
    @assert_any_with_batch
    def verify_state_vector(sv_ref, sv, **kwargs):
        return TensorBackend.verify_close(sv, sv_ref, **kwargs)
    
    @staticmethod
    @assert_any_with_batch
    def verify_amplitude(sv_ref, bitstring, amp, **kwargs):
        amp_ref = PropertyComputeHelper.amplitude_from_sv(sv_ref, bitstring)
        return TensorBackend.verify_close(amp, amp_ref, **kwargs)
    
    @staticmethod
    @assert_any_with_batch
    def verify_batched_amplitudes(sv_ref, fixed, batched_amps, **kwargs):
        batched_amps_ref = PropertyComputeHelper.batched_amplitudes_from_sv(sv_ref, fixed)
        return TensorBackend.verify_close(batched_amps, batched_amps_ref, **kwargs)
    
    @staticmethod
    @assert_any_with_batch
    def verify_expectation(sv_ref, pauli_strings, exp, **kwargs):
        exp_ref = PropertyComputeHelper.expectation_from_sv(sv_ref, pauli_strings)
        return TensorBackend.verify_close(exp, exp_ref, **kwargs)
    
    @staticmethod
    @assert_any_with_batch
    def verify_reduced_density_matrix(sv_ref, where, rdm, fixed=None, **kwargs):
        rdm_ref = PropertyComputeHelper.reduced_density_matrix_from_sv(sv_ref, where, fixed=fixed)
        return TensorBackend.verify_close(rdm, rdm_ref, **kwargs)
