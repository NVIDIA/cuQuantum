# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import sys
import types
import warnings

import pytest
import numpy as np

from nvmath.internal import utils, tensor_wrapper
from cuquantum.tensornet import contract, CircuitToEinsum, CirqParserOptions, QiskitParserOptions
from cuquantum.tensornet._internal import circuit_converter_utils

from .utils.circuit_matrix import CircuitMatrix, QiskitCircuitMatrix, CirqCircuitMatrix
from .utils.circuit_ifc import CircuitHelper, PropertyComputeHelper
from .utils.circuit_tester import BaseCircuitToEinsumTester
from .utils.data import ARRAY_BACKENDS
from .utils.helpers import TensorBackend


NUM_TESTS_PER_TASK = 3

class TestParserOptions:

    @pytest.mark.parametrize("check_diagonal", (True, False))
    def test_cirq_parser_options(self, check_diagonal):
        options = CirqParserOptions(check_diagonal=check_diagonal)
        assert options.check_diagonal == check_diagonal
        with pytest.raises(TypeError):
            options = CirqParserOptions(decompose_gates=True)

    @pytest.mark.parametrize("decompose_gates", (True, False))
    @pytest.mark.parametrize("check_diagonal", (True, False))
    def test_qiskit_parser_options(self, decompose_gates, check_diagonal):
        options = QiskitParserOptions(decompose_gates=decompose_gates, check_diagonal=check_diagonal)
        assert options.decompose_gates == decompose_gates
        assert options.check_diagonal == check_diagonal

@pytest.fixture(params=CircuitMatrix.realL0(), scope="class")
def real_circuit(request):
    return request.param

@pytest.fixture(scope="class")
def real_circuit_sv(real_circuit):
    return CircuitHelper.compute_state_vector(real_circuit)


class TestCircuitToEinsumFunctionality:

    @pytest.mark.parametrize("options", (
        None,
        CirqParserOptions(check_diagonal=False),
        {'check_diagonal': True},
        QiskitParserOptions(decompose_gates=True, check_diagonal=False),
        {'decompose_gates': False, 'check_diagonal': True},
    ))
    @pytest.mark.parametrize("circuit", CircuitMatrix.L2())
    def test_circuit_options(self, circuit, options):
        circuit_type = utils.infer_object_package(circuit)
        if circuit_type == 'qiskit':
            expect_type_error = isinstance(options, CirqParserOptions)
        else:
            expect_type_error = (isinstance(options, QiskitParserOptions) or
                (isinstance(options, dict) and 'decompose_gates' in options))
        if expect_type_error:
            context = pytest.raises(TypeError)
        else:
            context = contextlib.nullcontext()
        with context:
            converter = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options=options)
            gates = converter.gates
            gates_are_diagonal = converter._gates_are_diagonal
            assert gates is not None
            assert gates_are_diagonal is not None
            assert len(gates) == len(gates_are_diagonal)
            assert converter.qubits is not None


    @pytest.mark.parametrize("circuit", QiskitCircuitMatrix.L1()) # only qiskit circuits support decompose_gates option
    def test_decompose_gates_option(self, circuit):
        converter_1 = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options={'decompose_gates': True})
        converter_2 = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options={'decompose_gates': False})
        gates_1 = converter_1.gates
        gates_2 = converter_2.gates
        gates_are_diagonal_1 = converter_1._gates_are_diagonal
        gates_are_diagonal_2 = converter_2._gates_are_diagonal
        assert len(gates_1) >= len(gates_2)
        assert len(gates_are_diagonal_1) >= len(gates_are_diagonal_2)

    @pytest.mark.parametrize("circuit", CircuitMatrix.L1())
    def test_check_diagonal_option(self, circuit):
        converter_1 = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options={'check_diagonal': True})
        converter_2 = CircuitToEinsum(circuit, dtype='complex64', backend="numpy", options={'check_diagonal': False})
        gates_1 = converter_1.gates
        gates_2 = converter_2.gates
        gates_are_diagonal_1 = converter_1._gates_are_diagonal
        gates_are_diagonal_2 = converter_2._gates_are_diagonal
        assert len(gates_1) == len(gates_2)
        assert len(gates_are_diagonal_1) == len(gates_are_diagonal_2)
        operands_1 = converter_1.amplitude('0'*len(converter_1.qubits))[1]
        operands_2 = converter_2.amplitude('0'*len(converter_2.qubits))[1]
        assert len(operands_1) == len(operands_2)
        for o1, o2 in zip(operands_1, operands_2):
            if o1.shape == o2.shape:
                assert np.allclose(o1, o2)
            else:
                assert o1.shape == o2.shape[:o2.ndim//2]
                input_modes = [i for i in range(o1.ndim)]
                o2_diag = np.einsum(o2, input_modes*2, input_modes)
                assert np.allclose(o1, o2_diag)

    @pytest.mark.parametrize("circuit", CircuitMatrix.L1())
    def test_batched_amplitudes_marginal_cases(self, circuit):
        converter = CircuitToEinsum(circuit, dtype='complex64', backend="numpy")
        expr, operands = converter.state_vector()
        expr1, operands1 = converter.batched_amplitudes({})
        assert expr == expr1
        for o1, o2 in zip(operands, operands1):
            assert np.allclose(o1, o2)

        qubits = converter.qubits
        bitstring = '0' * len(qubits)
        fixed = dict(zip(qubits, bitstring))
        expr, operands = converter.amplitude(bitstring)
        expr1, operands1 = converter.batched_amplitudes(fixed)
        assert expr == expr1
        for o1, o2 in zip(operands, operands1):
            assert np.allclose(o1, o2)

    @pytest.mark.parametrize("backend", ARRAY_BACKENDS)
    @pytest.mark.parametrize("dtype", ("float32", "float64", "complex64", "complex128"))
    def test_real_circuit(self, real_circuit, real_circuit_sv, backend, dtype):
        converter = CircuitToEinsum(real_circuit, dtype=dtype, backend=backend)
        expr, operands = converter.state_vector()
        sv = contract(expr, *operands)
        assert TensorBackend.verify_close(sv, real_circuit_sv)
        wrapped_operands = tensor_wrapper.wrap_operands(operands)
        assert utils.get_operands_dtype(wrapped_operands) == dtype
        assert utils.get_operands_package(wrapped_operands) == backend

        n_qubits = sv.ndim
        for pauli_char in 'IXZ': # real Pauli strings, e.g, "II", "XX", "ZZ"
            pauli_string = pauli_char * n_qubits
            expr, operands = converter.expectation(pauli_string)
            exp = contract(expr, *operands)
            exp_ref = PropertyComputeHelper.expectation_from_sv(real_circuit_sv, pauli_string)
            assert TensorBackend.verify_close(exp, exp_ref)

        if dtype.startswith("float"):
            with pytest.raises(ValueError) as e:
                expr, operands = converter.expectation('Y' * n_qubits)
            assert "Pauli Y operator" in str(e.value)

    @pytest.mark.parametrize("circuit", CircuitMatrix.complexL0())
    @pytest.mark.parametrize("dtype", ("float32", "float64"))
    def test_negative_complex_circuit(self, circuit, dtype):
        with pytest.raises(RuntimeError) as e:
            CircuitToEinsum(circuit, dtype=dtype)
        assert "imaginary part" in str(e.value)


@pytest.fixture(params=QiskitCircuitMatrix.L2(), scope="class")
def qiskit_circuit(request):
    return request.param

@pytest.fixture(scope="class")
def qiskit_sv(qiskit_circuit):
    return CircuitHelper.compute_state_vector(qiskit_circuit)

@pytest.mark.parametrize("option", (
    None, # default (True, True)
    QiskitParserOptions(decompose_gates=True, check_diagonal=False),
    QiskitParserOptions(decompose_gates=False, check_diagonal=True),
    {'decompose_gates': False, 'check_diagonal': True},
))
class TestQiskitCorrectness(BaseCircuitToEinsumTester):

    num_tests_per_task = 3

    def test_state_vector(self, qiskit_circuit, option, qiskit_sv):
        self._test_state_vector(qiskit_circuit, option, qiskit_sv)

    def test_amplitude(self, qiskit_circuit, option, qiskit_sv):
        self._test_amplitude(qiskit_circuit, option, qiskit_sv)

    def test_batched_amplitudes(self, qiskit_circuit, option, qiskit_sv):
        self._test_batched_amplitudes(qiskit_circuit, option, qiskit_sv)

    @pytest.mark.parametrize("lightcone", (True, False))
    def test_rdm_and_probability(self, qiskit_circuit, option, qiskit_sv, lightcone):
        self._test_rdm_and_probability(qiskit_circuit, option, qiskit_sv, lightcone)

    @pytest.mark.parametrize("lightcone", (True, False))
    def test_expectation(self, qiskit_circuit, option, qiskit_sv, lightcone):
        self._test_expectation(qiskit_circuit, option, qiskit_sv, lightcone)

    def test_backend_dtype_consistency(self, qiskit_circuit, option):
        self._test_backend_dtype_consistency(qiskit_circuit, option)

    def test_auto_backend(self, qiskit_circuit, option):
        self._test_auto_backend(qiskit_circuit, "complex64", option)


@pytest.fixture(params=CirqCircuitMatrix.L2(), scope="class")
def cirq_circuit(request):
    return request.param

@pytest.fixture(scope="class")
def cirq_sv(cirq_circuit):
    return CircuitHelper.compute_state_vector(cirq_circuit)

@pytest.mark.parametrize("option", (
    None, # default (True,)
    CirqParserOptions(check_diagonal=False),
))
class TestCirqCorrectness(BaseCircuitToEinsumTester):

    num_tests_per_task = 3

    def test_state_vector(self, cirq_circuit, option, cirq_sv):
        self._test_state_vector(cirq_circuit, option, cirq_sv)

    def test_amplitude(self, cirq_circuit, option, cirq_sv):
        self._test_amplitude(cirq_circuit, option, cirq_sv)

    def test_batched_amplitudes(self, cirq_circuit, option, cirq_sv):
        self._test_batched_amplitudes(cirq_circuit, option, cirq_sv)

    @pytest.mark.parametrize("lightcone", (True, False))
    def test_rdm_and_probability(self, cirq_circuit, option, cirq_sv, lightcone):
        self._test_rdm_and_probability(cirq_circuit, option, cirq_sv, lightcone)

    @pytest.mark.parametrize("lightcone", (True, False))
    def test_expectation(self, cirq_circuit, option, cirq_sv, lightcone):
        self._test_expectation(cirq_circuit, option, cirq_sv, lightcone)

    def test_backend_dtype_consistency(self, cirq_circuit, option):
        self._test_backend_dtype_consistency(cirq_circuit, option)

    def test_auto_backend(self, cirq_circuit, option):
        self._test_auto_backend(cirq_circuit, "complex64", option)


class TestCircuitToEinsumPurity:
    """Tests for CircuitToEinsum automatic purity detection and pure-state outputs.

    Density-matrix (mixed) mode is auto-enabled when the circuit contains channels; the
    mixed-state projection conventions are validated against an external density-matrix
    reference in :class:`TestCircuitToEinsumChannels`.
    """

    @pytest.mark.parametrize("circuit", CircuitMatrix.L1()[:2])
    def test_density_matrix_pure_is_outer_product(self, circuit):
        """A channel-free circuit is pure; density_matrix() should yield rho = |psi><psi|."""
        sv_ref = CircuitHelper.compute_state_vector(circuit)
        sv_np = TensorBackend.to_numpy(sv_ref)
        n = sv_np.ndim
        rho_ref = np.outer(sv_np.flatten(), sv_np.flatten().conj()).reshape((2,) * (2 * n))

        converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
        assert not converter.is_mixed
        expr, operands = converter.density_matrix()
        dm = contract(expr, *operands)
        np.testing.assert_allclose(dm, rho_ref, atol=1e-10)

    @pytest.mark.parametrize("circuit", CircuitMatrix.L1()[:1])
    def test_state_vector_not_deprecated(self, circuit):
        """state_vector() must not emit a DeprecationWarning."""
        converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            converter.state_vector()
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert not deprecation_warnings, "state_vector() should no longer emit a DeprecationWarning"

    @pytest.mark.parametrize("circuit", CircuitMatrix.L1()[:1])
    def test_channel_free_circuit_is_pure(self, circuit):
        """A channel-free circuit is auto-detected as a pure state."""
        converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
        assert not converter.is_mixed
        assert all(e.kind == 'gate' for e in converter._gate_entries)

    def test_noisy_circuit_is_mixed_and_state_vector_raises(self):
        """A circuit with channels is auto-detected as mixed; state_vector() then raises."""
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Kraus
            from qiskit_aer.noise import depolarizing_error
        except ImportError:
            pytest.skip("qiskit or qiskit_aer not available")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(Kraus(depolarizing_error(0.1, 1)), [0])
        converter = CircuitToEinsum(qc, dtype='complex128', backend='numpy')
        assert converter.is_mixed
        with pytest.raises(TypeError, match="mixed-state"):
            converter.state_vector()


def _embed_operator(op, qubit_indices, n_qubits):
    """Embed a k-qubit operator into the full n-qubit Hilbert space."""
    dim = 2 ** n_qubits
    k = len(qubit_indices)
    result = np.zeros((dim, dim), dtype=complex)
    for row in range(dim):
        for col in range(dim):
            row_bits = [(row >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
            col_bits = [(col >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
            sub_row_bits = [row_bits[i] for i in qubit_indices]
            sub_col_bits = [col_bits[i] for i in qubit_indices]
            other_match = all(row_bits[i] == col_bits[i] for i in range(n_qubits) if i not in qubit_indices)
            if not other_match:
                continue
            sub_row = sum(b << (k - 1 - j) for j, b in enumerate(sub_row_bits))
            sub_col = sum(b << (k - 1 - j) for j, b in enumerate(sub_col_bits))
            result[row, col] = op[sub_row, sub_col]
    return result


def _ref_dm_qiskit(circuit):
    """
    Compute the reference density matrix using Qiskit's DensityMatrix simulator,
    reshaped to match cuquantum's qubit ordering (circuit.qubits order).

    Qiskit DensityMatrix.data uses little-endian (q_{n-1} most significant),
    while cuquantum output follows circuit.qubits order (q_0 first).
    """
    from qiskit.quantum_info import DensityMatrix

    dm = DensityMatrix.from_label('0' * circuit.num_qubits)
    for instruction in circuit.data:
        operation = instruction.operation
        qubit_indices = tuple(circuit.qubits.index(q) for q in instruction.qubits)
        dm = dm.evolve(operation, qubit_indices)
    n = circuit.num_qubits
    rho = dm.data.reshape((2,) * (2 * n))
    ket_perm = list(range(n - 1, -1, -1))
    bra_perm = list(range(2 * n - 1, n - 1, -1))
    return rho.transpose(ket_perm + bra_perm)


def _ref_dm_cirq(circuit):
    """Compute the reference density matrix by gate-by-gate evolution of a Cirq circuit."""
    import cirq
    qubit_list = sorted(circuit.all_qubits())
    n = len(qubit_list)
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    for moment in circuit.moments:
        for operation in moment:
            qubit_indices = [qubit_list.index(q) for q in operation.qubits]
            if cirq.has_kraus(operation) and not cirq.has_unitary(operation):
                rho_new = np.zeros_like(rho)
                for k in cirq.kraus(operation):
                    k_full = _embed_operator(k, qubit_indices, n)
                    rho_new += k_full @ rho @ k_full.conj().T
                rho = rho_new
            else:
                u_full = _embed_operator(cirq.unitary(operation), qubit_indices, n)
                rho = u_full @ rho @ u_full.conj().T
    return rho.reshape((2,) * (2 * n))


def _ref_dm_from_circuit(circuit):
    """
    Compute the reference density matrix for a circuit (Qiskit or Cirq) with channels.
    Returns rho as a (2,)*2n tensor in the same qubit order as CircuitToEinsum.qubits.
    """
    circuit_package = type(circuit).__module__.split('.')[0]
    if circuit_package == 'qiskit':
        return _ref_dm_qiskit(circuit)
    return _ref_dm_cirq(circuit)


def _make_qiskit_depolarizing():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import depolarizing_error
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.append(Kraus(depolarizing_error(0.1, 1)), [0])
    return qc

def _make_qiskit_amplitude_damping():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import amplitude_damping_error
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.append(Kraus(amplitude_damping_error(0.2)), [1])
    return qc

def _make_qiskit_custom_kraus():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    K0 = np.array([[1, 0], [0, np.sqrt(0.7)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(0.3)], [0, 0]], dtype=complex)
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    qc.append(Kraus([K0, K1]), [0])
    return qc

def _make_qiskit_two_qubit_channel():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import depolarizing_error
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.append(Kraus(depolarizing_error(0.05, 2)), [0, 1])
    return qc

def _make_qiskit_channel_type(channel_cls):
    """Factory builder: same physical channel (depolarizing p=0.1) wrapped in different representations."""
    def _make():
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Kraus
        from qiskit_aer.noise import depolarizing_error
        channel = channel_cls(Kraus(depolarizing_error(0.1, 1)))
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(channel, [0])
        return qc
    return _make

def _qiskit_channel_type_factories():
    """Generate factories for each NOISY_CHANNEL_TYPES representation (except Kraus, already covered)."""
    try:
        from qiskit.quantum_info import SuperOp, Choi, Stinespring, Chi, PTM
    except ImportError:
        return {}
    return {
        "qiskit_superop": _make_qiskit_channel_type(SuperOp),
        "qiskit_choi": _make_qiskit_channel_type(Choi),
        "qiskit_stinespring": _make_qiskit_channel_type(Stinespring),
        "qiskit_chi": _make_qiskit_channel_type(Chi),
        "qiskit_ptm": _make_qiskit_channel_type(PTM),
    }

def _make_cirq_depolarizing():
    import cirq
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.DepolarizingChannel(p=0.1).on(q0)])

def _make_cirq_amplitude_damping():
    import cirq
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.AmplitudeDampingChannel(gamma=0.2).on(q1)])

def _make_cirq_custom_kraus():
    import cirq
    K0 = np.array([[1, 0], [0, np.sqrt(0.7)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(0.3)], [0, 0]], dtype=complex)
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.X(q0), cirq.H(q1), cirq.KrausChannel(kraus_ops=[K0, K1]).on(q0)])


# Circuits whose density matrix is *complex* (nonzero <X> and <Y> on qubit 0). The H+T
# combination prepares (|0> + e^{i pi/4}|1>)/sqrt(2); the depolarizing channel makes the
# state mixed (triggers the density-matrix path) while keeping the coherence nonzero. These
# are needed to exercise the Pauli-Y expectation path non-trivially: all circuits in
# _NOISY_CIRCUIT_FACTORIES yield real density matrices, so <Y> = 0 there and a transpose
# error in the Pauli insertion (Y^T = -Y => sign flip) cannot be observed.
def _make_qiskit_complex_phase():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import depolarizing_error
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.t(0)
    qc.h(1)
    qc.append(Kraus(depolarizing_error(0.1, 1)), [0])
    return qc

def _make_cirq_complex_phase():
    import cirq
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.H(q0), cirq.T(q0), cirq.H(q1),
                         cirq.DepolarizingChannel(p=0.1).on(q0)])


_COMPLEX_RHO_CIRCUIT_FACTORIES = {
    "qiskit_complex_phase": _make_qiskit_complex_phase,
    "cirq_complex_phase": _make_cirq_complex_phase,
}


_NOISY_CIRCUIT_FACTORIES = {
    "qiskit_depolarizing": _make_qiskit_depolarizing,
    "qiskit_amplitude_damping": _make_qiskit_amplitude_damping,
    "qiskit_custom_kraus": _make_qiskit_custom_kraus,
    "qiskit_two_qubit_channel": _make_qiskit_two_qubit_channel,
    **_qiskit_channel_type_factories(),
    "cirq_depolarizing": _make_cirq_depolarizing,
    "cirq_amplitude_damping": _make_cirq_amplitude_damping,
    "cirq_custom_kraus": _make_cirq_custom_kraus,
}


class TestCircuitToEinsumChannelsComplexDM:
    """Regression tests for mixed-state Pauli expectation against a *complex* density matrix.

    The fixtures in :data:`_NOISY_CIRCUIT_FACTORIES` all yield real density matrices, so
    ``<Y> = 0`` and the Pauli-Y expectation path is only exercised trivially. The circuits
    here make ``rho`` complex (``<Y> != 0``), which is required to detect a transpose error in
    :func:`build_mixed_expectation_tn`: inserting ``P`` (rather than ``P^T``) between the open
    ket/bra legs computes ``Tr(rho P^T)``, and since ``Y^T = -Y`` this flips the sign of ``<Y>``
    while leaving the symmetric ``X``/``Z`` cases unaffected.
    """

    @pytest.fixture(params=list(_COMPLEX_RHO_CIRCUIT_FACTORIES.keys()))
    def complex_rho_circuit(self, request):
        try:
            return _COMPLEX_RHO_CIRCUIT_FACTORIES[request.param]()
        except ImportError:
            pytest.skip(f"dependencies for {request.param} not available")

    @pytest.mark.parametrize("pauli_string", ('X', 'Y', 'Z'))
    def test_expectation_complex_dm_pauli_variety(self, complex_rho_circuit, pauli_string):
        rho_ref = _ref_dm_from_circuit(complex_rho_circuit)
        converter = CircuitToEinsum(complex_rho_circuit, dtype='complex128', backend='numpy')
        n = len(converter.qubits)
        dim = 2 ** n
        rho_mat = rho_ref.reshape(dim, dim)

        pauli_1q = {'X': np.array([[0, 1], [1, 0]]),
                    'Y': np.array([[0, -1j], [1j, 0]]),
                    'Z': np.array([[1, 0], [0, -1]])}[pauli_string]
        full_op = pauli_1q
        for _ in range(n - 1):
            full_op = np.kron(full_op, np.eye(2))
        exp_ref = np.trace(rho_mat @ full_op)

        # Guard against the fixtures regressing to a real density matrix: a meaningful Y
        # regression requires <Y> != 0 (otherwise a sign flip is masked by -0 == 0).
        if pauli_string == 'Y':
            assert abs(exp_ref) > 1e-6

        ps = pauli_string + 'I' * (n - 1)
        expr, operands = converter.expectation(ps)
        exp_mixed = complex(contract(expr, *operands))
        np.testing.assert_allclose(exp_mixed.real, exp_ref.real, atol=1e-10)
        np.testing.assert_allclose(exp_mixed.imag, exp_ref.imag, atol=1e-10)


class TestCircuitToEinsumChannels:
    """Tests for CircuitToEinsum with noisy channels (auto-detected mixed state, Qiskit + Cirq).

    These also cover the mixed-state ket/bra projection conventions (off-diagonal amplitude,
    independent/one-sided/symmetric batched projections, and fixed-qubit RDM) against an external
    density-matrix reference (:func:`_ref_dm_from_circuit`).
    """

    @pytest.fixture(params=list(_NOISY_CIRCUIT_FACTORIES.keys()))
    def noisy_circuit(self, request):
        try:
            return _NOISY_CIRCUIT_FACTORIES[request.param]()
        except ImportError:
            pytest.skip(f"dependencies for {request.param} not available")

    def test_density_matrix_channels(self, noisy_circuit):
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        expr, operands = converter.density_matrix()
        dm = contract(expr, *operands)
        np.testing.assert_allclose(dm, rho_ref, atol=1e-10)

    def test_amplitude_channels(self, noisy_circuit):
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        n = len(converter.qubits)
        for bs in ['0' * n, '1' * n]:
            expr, operands = converter.amplitude(bs)
            result = complex(contract(expr, *operands))
            idx = tuple(int(b) for b in bs)
            ref_diag = rho_ref[idx + idx]
            np.testing.assert_allclose(result.real, ref_diag.real, atol=1e-10)
            np.testing.assert_allclose(result.imag, 0.0, atol=1e-10)

    @pytest.mark.parametrize("lightcone", (True, False))
    def test_rdm_channels(self, noisy_circuit, lightcone):
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        n = len(converter.qubits)
        dim = 2 ** n
        rho_mat = rho_ref.reshape(dim, dim)
        rdm_ref = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(dim // 2):
                    row = i * (dim // 2) + k
                    col = j * (dim // 2) + k
                    rdm_ref[i, j] += rho_mat[row, col]

        where = converter.qubits[:1]
        expr, operands = converter.reduced_density_matrix(where, lightcone=lightcone)
        rdm = np.asarray(contract(expr, *operands)).reshape(2, 2)
        np.testing.assert_allclose(rdm, rdm_ref, atol=1e-10)

    @pytest.mark.parametrize("lightcone", (True, False))
    def test_expectation_channels(self, noisy_circuit, lightcone):
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        n = len(converter.qubits)
        dim = 2 ** n
        rho_mat = rho_ref.reshape(dim, dim)
        pauli_z_full = np.diag([(-1) ** bin(i).count('1') for i in range(dim)])
        exp_ref = np.trace(rho_mat @ pauli_z_full).real

        pauli_string = 'Z' * n
        expr, operands = converter.expectation(pauli_string, lightcone=lightcone)
        exp_mixed = complex(contract(expr, *operands))
        np.testing.assert_allclose(exp_mixed.real, exp_ref, atol=1e-10)
        np.testing.assert_allclose(exp_mixed.imag, 0.0, atol=1e-10)

    def test_gates_channel_stacking(self, noisy_circuit):
        """.gates must stay type-stable: every operand is a single ndarray-like tensor.

        Channels are stacked along a leading Kraus mode (odd rank ``2k+1``) while gates keep
        even rank ``2k``. The stacked Kraus tensor must be trace preserving
        (``sum_k K_k^dagger K_k == I``), which also confirms unitary channels are folded into
        Kraus form ``sqrt(p_k) K_k``.
        """
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        gates = converter.gates

        n_channels = 0
        for operand, qubits in gates:
            assert not isinstance(operand, list), "gate/channel operand must be a single tensor"
            assert hasattr(operand, 'ndim')
            k = len(qubits)
            if operand.ndim == 2 * k:
                continue  # ordinary gate
            assert operand.ndim == 2 * k + 1, "channel operand must have rank 2k+1"
            n_channels += 1
            m = operand.shape[0]
            dim = 2 ** k
            kraus = np.asarray(operand).reshape(m, dim, dim)
            completeness = sum(K.conj().T @ K for K in kraus)
            np.testing.assert_allclose(completeness, np.eye(dim), atol=1e-10)

        assert n_channels >= 1, "noisy fixture should expose at least one channel via .gates"

    @pytest.mark.parametrize("pauli_string", ('X', 'Y', 'Z'))
    def test_expectation_channels_pauli_variety(self, noisy_circuit, pauli_string):
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        n = len(converter.qubits)
        dim = 2 ** n
        rho_mat = rho_ref.reshape(dim, dim)

        pauli_1q = {'X': np.array([[0, 1], [1, 0]]),
                     'Y': np.array([[0, -1j], [1j, 0]]),
                     'Z': np.array([[1, 0], [0, -1]])}[pauli_string]
        full_op = pauli_1q
        for _ in range(n - 1):
            full_op = np.kron(full_op, np.eye(2))
        exp_ref = np.trace(rho_mat @ full_op)

        ps = pauli_string + 'I' * (n - 1)
        expr, operands = converter.expectation(ps)
        exp_mixed = complex(contract(expr, *operands))
        np.testing.assert_allclose(exp_mixed.real, exp_ref.real, atol=1e-10)
        np.testing.assert_allclose(exp_mixed.imag, exp_ref.imag, atol=1e-10)

    def test_marginal_probability_channels(self, noisy_circuit):
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        n = len(converter.qubits)
        dim = 2 ** n
        rho_mat = rho_ref.reshape(dim, dim)
        probs_ref = np.zeros(2)
        for val in range(2):
            for k in range(dim // 2):
                idx = val * (dim // 2) + k
                probs_ref[val] += rho_mat[idx, idx].real

        where = converter.qubits[:1]
        expr, operands = converter.marginal_probability(where)
        marginal = np.asarray(contract(expr, *operands))
        np.testing.assert_allclose(marginal.real, probs_ref, atol=1e-10)

    @pytest.mark.parametrize("lightcone", (True, False))
    def test_rdm_diagonal_flag_channels(self, noisy_circuit, lightcone):
        """reduced_density_matrix(where, diagonal=True) must (a) equal the diagonal of the full
        RDM and (b) be functionally equivalent to marginal_probability for mixed states."""
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        where = converter.qubits[:1]

        expr_full, ops_full = converter.reduced_density_matrix(where, lightcone=lightcone)
        rdm = np.asarray(contract(expr_full, *ops_full)).reshape(2, 2)

        expr_diag, ops_diag = converter.reduced_density_matrix(where, lightcone=lightcone, diagonal=True)
        diag = np.asarray(contract(expr_diag, *ops_diag))
        assert diag.shape == (2,)
        np.testing.assert_allclose(diag, np.diagonal(rdm), atol=1e-10)

        expr_marg, ops_marg = converter.marginal_probability(where, lightcone=lightcone)
        np.testing.assert_allclose(diag, np.asarray(contract(expr_marg, *ops_marg)), atol=1e-12)
        # operands are identical; only the einsum expression differs (bra collapsed onto ket)
        assert expr_diag == expr_marg

    def test_rdm_diagonal_flag_with_fixed_channels(self, noisy_circuit):
        """diagonal=True must compose with the fixed-qubit projection path and stay equivalent
        to marginal_probability(where, fixed=...)."""
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        q = converter.qubits
        if len(q) < 2:
            pytest.skip("requires at least 2 qubits for a fixed qubit and a separate where qubit")
        where = q[:1]
        fixed = {q[-1]: 0}
        expr_diag, ops_diag = converter.reduced_density_matrix(where, fixed=fixed, diagonal=True)
        diag = np.asarray(contract(expr_diag, *ops_diag))
        expr_marg, ops_marg = converter.marginal_probability(where, fixed=fixed)
        np.testing.assert_allclose(diag, np.asarray(contract(expr_marg, *ops_marg)), atol=1e-12)

    def test_batched_amplitudes_channels(self, noisy_circuit):
        """batched_amplitudes({}) on a noisy mixed-state circuit yields the full density matrix."""
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        n = len(converter.qubits)
        expr, operands = converter.batched_amplitudes({})
        dm = np.asarray(contract(expr, *operands))
        np.testing.assert_allclose(dm, rho_ref.reshape((2,) * (2 * n)), atol=1e-10)

    def test_amplitude_offdiagonal_channels(self, noisy_circuit):
        """amplitude((bs1, bs2)) yields the off-diagonal density-matrix element rho[bs1, bs2]."""
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        n = len(converter.qubits)
        bs1 = '0' * n
        bs2 = '1' + '0' * (n - 1)
        expr, operands = converter.amplitude((bs1, bs2))
        result = complex(contract(expr, *operands))
        idx1 = tuple(int(b) for b in bs1)
        idx2 = tuple(int(b) for b in bs2)
        np.testing.assert_allclose(result, rho_ref[idx1 + idx2], atol=1e-10)

    def test_batched_amplitudes_slice_channels(self, noisy_circuit):
        """A per-key (ket, bra) 2-tuple projects the ket and bra of a qubit independently."""
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        q = converter.qubits
        n = len(q)
        if n < 2:
            pytest.skip("requires at least 2 qubits")
        # fix qubit 0's ket to 0 and bra to 1; remaining ket/bra modes open
        expr, operands = converter.batched_amplitudes({q[0]: (0, 1)})
        sliced = np.asarray(contract(expr, *operands))
        # rho_ref modes: (k_0..k_{n-1}, b_0..b_{n-1}); output is (open_ket, open_bra)
        ref = rho_ref[(0,) + (slice(None),) * (n - 1) + (1,) + (slice(None),) * (n - 1)]
        np.testing.assert_allclose(sliced, ref, atol=1e-10)

    def test_batched_amplitudes_one_sided_channels(self, noisy_circuit):
        """A per-key (val, None) fixes only the ket, leaving the full bra space open."""
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        q = converter.qubits
        n = len(q)
        if n < 2:
            pytest.skip("requires at least 2 qubits")
        expr, operands = converter.batched_amplitudes({q[0]: (0, None)})
        sliced = np.asarray(contract(expr, *operands))
        # ket of qubit 0 fixed to 0 (open ket modes 1..n-1); bra fully open
        ref = rho_ref[(0,) + (slice(None),) * (n - 1) + (slice(None),) * n]
        np.testing.assert_allclose(sliced, ref, atol=1e-10)

    def test_batched_amplitudes_symmetric_channels(self, noisy_circuit):
        """A scalar per-key value fixes ket and bra to the same index (diagonal projection)."""
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        q = converter.qubits
        n = len(q)
        if n < 2:
            pytest.skip("requires at least 2 qubits")
        expr, operands = converter.batched_amplitudes({q[0]: 0})
        sliced = np.asarray(contract(expr, *operands))
        ref = rho_ref[(0,) + (slice(None),) * (n - 1) + (0,) + (slice(None),) * (n - 1)]
        np.testing.assert_allclose(sliced, ref, atol=1e-10)

    @pytest.mark.parametrize("lightcone", (True, False))
    def test_rdm_with_fixed_channels(self, noisy_circuit, lightcone):
        """reduced_density_matrix(where, fixed=...) projects the fixed qubit (ket==bra) and traces the rest."""
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend='numpy')
        q = converter.qubits
        n = len(q)
        if n < 2:
            pytest.skip("requires at least 2 qubits to have a fixed qubit and a separate where qubit")
        where = q[:1]
        fixed = {q[-1]: 0}
        expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=lightcone)
        rdm = np.asarray(contract(expr, *operands))

        # reference from rho_ref (modes k_0..k_{n-1}, b_0..b_{n-1}):
        # fix last qubit's ket and bra to 0, keep qubit 0 open, trace the middle qubits (k_i == b_i).
        sl = [slice(None)] * (2 * n)
        sl[n - 1] = 0
        sl[2 * n - 1] = 0
        sub = rho_ref[tuple(sl)]  # axes: ket qubits 0..n-2, then bra qubits 0..n-2
        m = n - 1  # number of remaining qubits per side
        ket_sub = ['a'] + [chr(ord('c') + j) for j in range(m - 1)]
        bra_sub = ['b'] + [chr(ord('c') + j) for j in range(m - 1)]
        subscript = ''.join(ket_sub) + ''.join(bra_sub) + '->ab'
        ref = np.einsum(subscript, sub)
        np.testing.assert_allclose(rdm, ref, atol=1e-10)

    @pytest.mark.parametrize("backend", ARRAY_BACKENDS)
    def test_density_matrix_channels_backends(self, noisy_circuit, backend):
        """Noisy mixed-state channel TN must contract correctly across array backends.

        The Kraus stacking / conj path in `_build_mixed_core` is backend-specific; this
        guards that channels work on cupy/torch and not only numpy.
        """
        rho_ref = _ref_dm_from_circuit(noisy_circuit)
        converter = CircuitToEinsum(noisy_circuit, dtype='complex128', backend=backend)
        expr, operands = converter.density_matrix()
        dm = TensorBackend.to_numpy(contract(expr, *operands))
        np.testing.assert_allclose(dm, rho_ref, atol=1e-10)

    def test_channels_state_vector_raises_qiskit(self):
        """A channel circuit is auto-detected mixed; state_vector() must raise (Qiskit)."""
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Kraus
            from qiskit_aer.noise import depolarizing_error
            qc = QuantumCircuit(1)
            qc.h(0)
            qc.append(Kraus(depolarizing_error(0.1, 1)), [0])
            converter = CircuitToEinsum(qc, dtype='complex128', backend='numpy')
            assert converter.is_mixed
            with pytest.raises(TypeError, match="mixed-state"):
                converter.state_vector()
        except ImportError:
            pytest.skip("qiskit or qiskit_aer not available")

    def test_channels_state_vector_raises_cirq(self):
        """A general-channel circuit is auto-detected mixed; state_vector() must raise (Cirq)."""
        try:
            import cirq
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.H(q), cirq.AmplitudeDampingChannel(gamma=0.1).on(q)])
            converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
            assert converter.is_mixed
            assert any(e.kind == 'general_channel' for e in converter._gate_entries)
            with pytest.raises(TypeError, match="mixed-state"):
                converter.state_vector()
        except ImportError:
            pytest.skip("cirq not available")

    def test_unitary_channel_detected_cirq(self):
        """Cirq unitary channels (mixtures) are detected and auto-enable mixed mode."""
        try:
            import cirq
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.H(q), cirq.DepolarizingChannel(p=0.1).on(q)])
            converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
            assert converter.is_mixed
            assert any(e.kind == 'unitary_channel' for e in converter._gate_entries)
            with pytest.raises(TypeError, match="mixed-state"):
                converter.state_vector()
        except ImportError:
            pytest.skip("cirq not available")

    def test_unitary_channel_mixed_density_matrix_cirq(self):
        """Cirq unitary channels produce correct density matrices via mixed-state TN."""
        try:
            import cirq
            q0, q1 = cirq.LineQubit.range(2)
            circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.DepolarizingChannel(p=0.1).on(q0)])
            ref = _ref_dm_from_circuit(circuit)
            converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
            assert any(e.kind == 'unitary_channel' for e in converter._gate_entries)
            expr, operands = converter.density_matrix()
            dm = contract(expr, *operands)
            np.testing.assert_allclose(dm, ref, atol=1e-10)
        except ImportError:
            pytest.skip("cirq not available")

    def test_lightcone_preserves_channels_qiskit(self):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Kraus
            from qiskit_aer.noise import depolarizing_error
            qc = QuantumCircuit(3)
            qc.h(0)
            qc.append(Kraus(depolarizing_error(0.1, 1)), [2])

            rho_ref = _ref_dm_qiskit(qc)
            converter = CircuitToEinsum(qc, dtype='complex128', backend='numpy')
            expr, operands = converter.density_matrix()
            dm = contract(expr, *operands)
            np.testing.assert_allclose(dm, rho_ref, atol=1e-10)

            where = converter.qubits[:1]
            expr_lc, ops_lc = converter.reduced_density_matrix(where, lightcone=True)
            expr_no, ops_no = converter.reduced_density_matrix(where, lightcone=False)
            rdm_lc = contract(expr_lc, *ops_lc)
            rdm_no = contract(expr_no, *ops_no)
            np.testing.assert_allclose(rdm_lc, rdm_no, atol=1e-10)
        except ImportError:
            pytest.skip("qiskit or qiskit_aer not available")

    def test_channels_real_dtype_qiskit(self):
        """Real dtype with a real-valued channel (bit-flip) should produce correct results."""
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Kraus
            p = 0.15
            K0 = np.sqrt(1 - p) * np.eye(2)
            K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]])
            qc = QuantumCircuit(2)
            qc.x(0)
            qc.append(Kraus([K0, K1]), [0])
            ref = _ref_dm_from_circuit(qc)
            converter = CircuitToEinsum(qc, dtype='float64', backend='numpy')
            expr, operands = converter.density_matrix()
            dm = contract(expr, *operands)
            np.testing.assert_allclose(dm, ref.real, atol=1e-10)
        except ImportError:
            pytest.skip("qiskit or qiskit_aer not available")

    def test_channels_real_dtype_cirq(self):
        """Real dtype with a real-valued channel (bit-flip) should produce correct results."""
        try:
            import cirq
            p = 0.15
            K0 = np.sqrt(1 - p) * np.eye(2)
            K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]])
            q0, q1 = cirq.LineQubit.range(2)
            circuit = cirq.Circuit([cirq.X(q0), cirq.KrausChannel(kraus_ops=[K0, K1]).on(q0)])
            ref = _ref_dm_from_circuit(circuit)
            converter = CircuitToEinsum(circuit, dtype='float64', backend='numpy')
            expr, operands = converter.density_matrix()
            dm = contract(expr, *operands)
            np.testing.assert_allclose(dm, ref.real, atol=1e-10)
        except ImportError:
            pytest.skip("cirq not available")

    def test_channels_real_dtype_raises_qiskit(self):
        """Real dtype with channels containing imaginary Kraus operators must raise.

        Uses a phase-damping-like channel (K1 ~ S gate) whose Choi matrix is
        inherently complex, so Qiskit's internal round-trip cannot produce an
        all-real Kraus set.
        """
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Kraus
            K0 = np.sqrt(0.9) * np.eye(2, dtype=complex)
            K1 = np.sqrt(0.1) * np.array([[1, 0], [0, 1j]], dtype=complex)
            qc = QuantumCircuit(1)
            qc.append(Kraus([K0, K1]), [0])
            with pytest.raises(RuntimeError, match="channel Kraus operand found to have imaginary part"):
                CircuitToEinsum(qc, dtype='float64', backend='numpy')
        except ImportError:
            pytest.skip("qiskit not available")

    def test_channels_real_dtype_raises_cirq(self):
        """Real dtype with channels containing imaginary Kraus operators must raise."""
        try:
            import cirq
            K0 = np.sqrt(0.9) * np.eye(2, dtype=complex)
            K1 = np.sqrt(0.1) * np.array([[0, -1j], [1j, 0]])
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.KrausChannel(kraus_ops=[K0, K1]).on(q)])
            with pytest.raises(RuntimeError, match="channel Kraus operand found to have imaginary part"):
                CircuitToEinsum(circuit, dtype='float64', backend='numpy')
        except ImportError:
            pytest.skip("cirq not available")

    def test_lightcone_preserves_channels_cirq(self):
        try:
            import cirq
            q0, q1, q2 = cirq.LineQubit.range(3)
            circuit = cirq.Circuit([cirq.H(q0), cirq.DepolarizingChannel(p=0.1).on(q2)])

            rho_ref = _ref_dm_from_circuit(circuit)
            converter = CircuitToEinsum(circuit, dtype='complex128', backend='numpy')
            expr, operands = converter.density_matrix()
            dm = contract(expr, *operands)
            np.testing.assert_allclose(dm, rho_ref, atol=1e-10)

            where = converter.qubits[:1]
            expr_lc, ops_lc = converter.reduced_density_matrix(where, lightcone=True)
            expr_no, ops_no = converter.reduced_density_matrix(where, lightcone=False)
            rdm_lc = contract(expr_lc, *ops_lc)
            rdm_no = contract(expr_no, *ops_no)
            np.testing.assert_allclose(rdm_lc, rdm_no, atol=1e-10)
        except ImportError:
            pytest.skip("cirq not available")


@pytest.mark.parametrize("framework, partial", [("cirq", "qiskit"), ("qiskit", "cirq")])
def test_infer_parser_ignores_partially_initialized_framework(monkeypatch, framework, partial):
    """A framework that another thread is still importing must not block the other."""
    package = pytest.importorskip(framework)
    circuit = package.Circuit() if framework == "cirq" else package.QuantumCircuit(1)
    real = pytest.importorskip(partial)

    class Initializing(types.ModuleType):
        def __getattr__(self, name):
            # The other thread finishes the import after this first look.
            sys.modules[partial] = real
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, partial, Initializing(partial))

    parser = circuit_converter_utils.infer_parser(circuit)

    assert parser.__name__.endswith(f"circuit_parser_utils_{framework}")
