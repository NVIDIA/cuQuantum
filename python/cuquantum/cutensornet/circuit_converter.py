# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
A converter that translates a quantum circuit to tensor network Einsum equations.
"""

__all__ = ['CircuitToEinsum']

import importlib

import numpy as np

from ._internal import circuit_converter_utils as circ_utils

EMPTY_DICT = circ_utils.EMPTY_DICT

class CircuitToEinsum:
    """
    Create a converter object that can generate Einstein summation expressions and tensor operands for a given circuit.

    The supported circuit types include :class:`cirq.Circuit` and :class:`qiskit.QuantumCircuit`. The input circuit must 
    be fully parameterized and can not contain operations that are not well-defined in tensor network simulation, for instance, 
    resetting the quantum state or performing any intermediate measurement. 

    Args:
        circuit : A fully parameterized :class:`cirq.Circuit` or :class:`qiskit.QuantumCircuit` object.
        dtype : The datatype for the output tensor operands. If not specified, double complex is used. 
        backend: The backend for the output tensor operands. If not specified, ``cupy`` is used.
    
    Notes:

      - For :class:`qiskit.QuantumCircuit`, composite gates will be decomposed into either Qiskit standard gates or customized unitary gates.

    Examples:

        Examples using Qiskit:

        >>> import qiskit.circuit.random
        >>> from cuquantum import contract, CircuitToEinsum

        Generate a random quantum circuit:
        
        >>> qc = qiskit.circuit.random.random_circuit(num_qubits=8, depth=7)

        Create a :class:`CircuitToEinsum` object:

        >>> converter = CircuitToEinsum(qc, backend='cupy')

        Find the Einstein summation expression and tensor operands for the state vector:

        >>> expression, operands = converter.state_vector()

        Contract the equation above to compute the state vector:
        
        >>> sv = contract(expression, *operands)
        >>> print(sv.shape)
        (2, 2, 2, 2, 2, 2, 2, 2)

        Find the Einstein summation expression and tensor operands for computing the probability amplitude of bitstring 00000000:

        >>> expression, operands = converter.amplitude('00000000')

        Contract the equation above to compute the amplitude:

        >>> amplitude = contract(expression, *operands)

        Find the Einstein summation expression and tensor operands for computing reduced density matrix on the
        first two qubits with the condition that the last qubit is fixed at state ``1``:

        >>> where = qc.qubits[:2]
        >>> fixed = {qc.qubits[-1]: '1'}
        >>> expression, operands = converter.reduced_density_matrix(where, fixed=fixed)

        Contract the equation above to compute the reduced density matrix:

        >>> rdm = contract(expression, *operands)
        >>> print(rdm.shape)
        (2, 2, 2, 2)

    """
    def __init__(self, circuit, dtype='complex128', backend='cupy'):
        # infer library-specific parser
        self.parser = circ_utils.infer_parser(circuit)

        circuit = self.parser.remove_measurements(circuit)
        self.circuit = circuit
        if isinstance(backend, str):
            backend = importlib.import_module(backend)
        self.backend = backend
        
        if isinstance(dtype, str):
            try:
                dtype = getattr(backend, dtype)
            except AttributeError:
                dtype = getattr(backend, np.dtype(dtype).name)
        self.dtype = dtype

        # unfold circuit metadata
        self._qubits, self.gates = self.parser.unfold_circuit(circuit, dtype=self.dtype, backend=self.backend)
        self.n_qubits = len(self.qubits)
        self._metadata = None
    
    @property
    def qubits(self):
        """A sequence of all qubits in the circuit."""
        return self._qubits
        
    def state_vector(self, fixed=EMPTY_DICT):
        """
        state_vector(fixed=None)

        Generate the Einstein summation expression and tensor operands to compute the statevector for the input circuit.

        Args:    
            fixed: Optional, a dictionary that maps certain qubits to the corresponding fixed states 0 or 1.

        Returns:
            The Einstein summation expression and a list of tensor operands. The order of the output mode labels is consistent with :attr:`CircuitToEinsum.qubits`.
            For :class:`cirq.Circuit`, this order corresponds to all qubits in the circuit sorted in ascending order. 
            For :class:`qiskit.QuantumCircuit`, this order is the same as :attr:`qiskit.QuantumCircuit.qubits`.
        """
        input_mode_labels, input_operands, qubits_frontier = self._get_inputs()
        
        fixed_qubits, fixed_bitstring = circ_utils.parse_fixed_qubits(fixed)
        fixed_mode_labels = [[qubits_frontier[q]] for q in fixed_qubits]    
        mode_labels = input_mode_labels + fixed_mode_labels
        
        operands = input_operands + circ_utils.get_bitstring_tensors(fixed_bitstring, dtype=self.dtype, backend=self.backend)
        output_mode_labels = [qubits_frontier[q] for q in self.qubits if q not in fixed]

        expression = circ_utils.convert_mode_labels_to_expression(mode_labels, output_mode_labels=output_mode_labels)
        return expression, operands 
    
    def amplitude(self, bitstring):
        """Generate the Einstein summation expression and tensor operands to compute the probability amplitude of
        a bitstring for the input circuit.

        Args:    
            bitstring: A sequence of 0/1 specifying the desired measured state. 
                The order of the bitstring is expected to be consistent with :attr:`CircuitToEinsum.qubits`.
                For :class:`cirq.Circuit`, this order corresponds to all qubits in the circuit sorted in ascending order. 
                For :class:`qiskit.QuantumCircuit`, this order is the same as :attr:`qiskit.QuantumCircuit.qubits`.

        Returns:
            The Einstein summation expression and a list of tensor operands
        """
        bitstring = circ_utils.parse_bitstring(bitstring, n_qubits=self.n_qubits)
        input_mode_labels, input_operands, qubits_frontier = self._get_inputs()
        mode_labels = input_mode_labels + [[qubits_frontier[q]] for q in self.qubits]
        output_mode_labels = []

        expression = circ_utils.convert_mode_labels_to_expression(mode_labels, output_mode_labels=output_mode_labels)
        operands = input_operands + circ_utils.get_bitstring_tensors(bitstring, dtype=self.dtype, backend=self.backend)
        return expression, operands 
    
    def reduced_density_matrix(self, where, fixed=EMPTY_DICT, lightcone=True):
        r"""
        reduced_density_matrix(where, fixed=None, lightcone=True)

        Generate the Einstein summation expression and tensor operands to compute the reduced density matrix for
        the input circuit.

        Unitary reverse lightcone cancellation refers to removing the identity formed by a unitary gate (from
        the ket state) and its inverse (from the bra state) when there exists no additional operators
        in-between. One can take advantage of this technique to reduce the effective network size by
        only including the *causal* gates (gates residing in the lightcone).

        Args:    
            where: A sequence of qubits specifying where the density matrix are reduced onto. 
            fixed: Optional, a dictionary that maps certain qubits to the corresponding fixed states 0 or 1.
            lightcone: Whether to apply the unitary reverse lightcone cancellation technique to reduce the number of tensors in density matrix computation.
            
        Returns:
            The Einstein summation expression and a list of tensor operands.
            The mode labels for output of the expression has the same order as the where argument. 
            For example, if where = (:math:`a, b`), the mode labels for the reduced density matrix would be (:math:`a, b, a^{\prime}, b^{\prime}`)
        
        .. seealso:: `unitary reverse lightcone cancellation <https://quimb.readthedocs.io/en/latest/tensor-circuit.html#Unitary-Reverse-Lightcone-Cancellation>`_
        """
        parser = self.parser
        n_qubits = self.n_qubits

        if lightcone:
            coned_qubits = list(where) + list(fixed.keys())
            circuit = parser.get_lightcone_circuit(self.circuit, coned_qubits)
            _, gates = parser.unfold_circuit(circuit, dtype=self.dtype, backend=self.backend)
            # in cirq, the lightcone circuit may only contain a subset of the original qubits
            # It's imperative to use qubits=self.qubits to generate the input tensors
            input_mode_labels, input_operands, qubits_frontier = circ_utils.parse_inputs(self.qubits, gates, self.dtype, self.backend)
        else:
            circuit = self.circuit
            input_mode_labels, input_operands, qubits_frontier = self._get_inputs()
            # avoid inplace modification on metadata
            qubits_frontier = qubits_frontier.copy()

        next_frontier = max(qubits_frontier.values()) + 1

        # handle tensors/mode labels for qubits with fixed state
        fixed_qubits, fixed_bitstring = circ_utils.parse_fixed_qubits(fixed)
        fixed_operands = circ_utils.get_bitstring_tensors(fixed_bitstring, dtype=self.dtype, backend=self.backend)

        mode_labels = input_mode_labels + [[qubits_frontier[ix]] for ix in fixed_qubits]
        for iqubit in fixed_qubits:
            qubits_frontier[iqubit] = next_frontier
            mode_labels.append([next_frontier])
            next_frontier += 1
        operands = input_operands + fixed_operands * 2

        output_mode_labels_info = dict()
        for iqubit in where:
            output_mode_labels_info[iqubit] = [qubits_frontier[iqubit], next_frontier]
            qubits_frontier[iqubit] = next_frontier
            next_frontier += 1

        # inverse circuit
        inverse_circuit  = parser.get_inverse_circuit(circuit)
        _, inverse_gates = parser.unfold_circuit(inverse_circuit, dtype=self.dtype, backend=self.backend)
        igate_mode_labels, igate_operands = circ_utils.parse_gates_to_mode_labels_operands(inverse_gates, 
                                                                                 qubits_frontier, 
                                                                                 next_frontier)
        mode_labels += igate_mode_labels
        operands += igate_operands
        
        mode_labels += [[qubits_frontier[ix]] for ix in self.qubits]
        operands += input_operands[:n_qubits]
        
        output_left_mode_labels = []
        output_right_mode_labels = []
        for iqubits, (left_mode_labels, right_mode_labels) in output_mode_labels_info.items():
            output_left_mode_labels.append(left_mode_labels)
            output_right_mode_labels.append(right_mode_labels)
        output_mode_labels = output_left_mode_labels + output_right_mode_labels
        expression = circ_utils.convert_mode_labels_to_expression(mode_labels, output_mode_labels=output_mode_labels)
        return expression, operands

    def _get_inputs(self):
        """transform the qubits and gates in the circuit to a prelimary Einsum form.

        Returns:
            metadata: A 3-tuple (``mode_labels``, ``operands``, ``qubits_frontier``):

                - ``mode_labels`` :  A list of list of int, each corresponding to the mode labels for the tensor operands.
                - ``operands`` : A list of arrays corresponding to all the qubits and gate tensor operands.
                - ``qubits_frontier`` : A dictionary that maps all qubits to their current mode labels.
        """
        if self._metadata is None:
            self._metadata = circ_utils.parse_inputs(self.qubits, self.gates, self.dtype, self.backend)
        return self._metadata
