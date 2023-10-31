# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from types import MappingProxyType

try:
    import cirq
    from cuquantum.cutensornet._internal import circuit_parser_utils_cirq
except ImportError:
    cirq = circuit_parser_utils_cirq = None
import cupy as cp
import numpy as np
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
except ImportError:
    torch = None
try:
    import qiskit
    from cuquantum.cutensornet._internal import circuit_parser_utils_qiskit
except ImportError:
    qiskit = circuit_parser_utils_qiskit = None
    
from cuquantum import contract, CircuitToEinsum
from cuquantum import cutensornet as cutn
from cuquantum.cutensornet._internal import circuit_parser_utils_cirq, circuit_parser_utils_qiskit
from cuquantum.cutensornet._internal.circuit_converter_utils import convert_mode_labels_to_expression
from cuquantum.cutensornet._internal.circuit_converter_utils import EINSUM_SYMBOLS_BASE
from cuquantum.cutensornet._internal.circuit_converter_utils import get_pauli_gates
from cuquantum.cutensornet._internal.circuit_converter_utils import parse_gates_to_mode_labels_operands
from cuquantum.cutensornet._internal.decomposition_utils import SVD_ALGORITHM_MAP, NORMALIZATION_MAP
from cuquantum.cutensornet._internal.utils import infer_object_package

from .approxTN_utils import SVD_TOLERANCE, verify_unitary
from .mps_utils import MPS, gen_random_mps, get_mps_tolerance
from .mps_utils import amplitude_from_sv
from .mps_utils import batched_amplitude_from_sv
from .mps_utils import expectation_from_sv
from .mps_utils import reduced_density_matrix_from_sv
from .mps_utils import sample_from_sv
from .test_utils import atol_mapper, rtol_mapper
from .test_cutensornet import manage_resource
from .. import dtype_to_data_type


# note: this implementation would cause pytorch tests being silently skipped
# if pytorch is not available, which is the desired effect since otherwise
# it'd be too noisy
backends = [np, cp]
if torch:
    backends.append(torch)


cirq_circuits = []
qiskit_circuits = []

EMPTY_DICT = MappingProxyType(dict())
GLOBAL_RNG = np.random.default_rng(2023)
DEFAULT_NUM_RANDOM_LAYERS = 2
EXACT_MPS_QUBIT_COUNT_LIMIT = 63 # limit the number of qubits for exact MPS to avoid extent overflowing

STATE_ATTRIBUTE_MAP = {
    'canonical_center' : cutn.StateAttribute.MPS_CANONICAL_CENTER,
    'abs_cutoff' : cutn.StateAttribute.MPS_SVD_CONFIG_ABS_CUTOFF,
    'rel_cutoff' : cutn.StateAttribute.MPS_SVD_CONFIG_REL_CUTOFF,
    'normalization' : cutn.StateAttribute.MPS_SVD_CONFIG_S_NORMALIZATION,
    'discarded_weight_cutoff' : cutn.StateAttribute.MPS_SVD_CONFIG_DISCARDED_WEIGHT_CUTOFF,
    'algorithm' : cutn.StateAttribute.MPS_SVD_CONFIG_ALGO,
    #'algorithm_params' : cutn.StateAttribute.MPS_SVD_CONFIG_ALGO_PARAMS, # NOTE: special treatment required
    'num_hyper_samples' : cutn.StateAttribute.NUM_HYPER_SAMPLES}


def bitstring_generator(n_qubits, nsample=1):
    for _ in range(nsample):
        bitstring = ''.join(np.random.choice(('0', '1'), n_qubits))
        yield bitstring


def where_fixed_generator(qubits, nfix_max, nsite_max=None):
    indices = np.arange(len(qubits))
    for nfix in range(nfix_max):
        for _ in range(2):
            np.random.shuffle(indices)
            fixed_sites = [qubits[indices[ix]] for ix in range(nfix)]
            bitstring = ''.join(np.random.choice(('0', '1'), nfix))
            fixed = dict(zip(fixed_sites, bitstring))
            if nsite_max is None:
                yield fixed
            else:
                for nsite in range(1, nsite_max+1):
                    where = [qubits[indices[ix]] for ix in range(nfix, nfix+nsite)]
                    yield where, fixed


def random_pauli_string_generator(n_qubits, num_strings=4):
    for _ in range(num_strings):
        yield ''.join(np.random.choice(['I','X', 'Y', 'Z'], n_qubits))


################################################
# functions to generate cirq.Circuit for testing
################################################

def get_cirq_random_2q_gate():
    class Random2QGate(cirq.Gate):
        def __init__(self):
            super(Random2QGate, self)
            setattr(self, '_internal_array_', cirq.testing.random_unitary(4))
    
        def _num_qubits_(self):
            return 2
    
        def _unitary_(self):
            return getattr(self, '_internal_array_')
        
        def __pow__(self, power):
            if power == 1:
                return self
            elif power == -1:
                new_gate = Random2QGate()
                unitary = getattr(self, '_internal_array_').T.conj()
                setattr(new_gate, '_internal_array_', unitary)
                return new_gate
            else:
                return NotImplementedError
            
        def _circuit_diagram_info_(self, args):
            return "Q1", "Q2"
        
    return Random2QGate()


def gen_random_layered_cirq_circuit(qubits, num_random_layers=2):
    n_qubits = len(qubits)
    operations = []
    for n in range(num_random_layers):
        for i in range(n%2, n_qubits-1, 2):
            operations.append(get_cirq_random_2q_gate().on(qubits[i], qubits[i+1]))
    return cirq.Circuit(operations)


def cirq_insert_random_layers(circuit, num_random_layers=DEFAULT_NUM_RANDOM_LAYERS):
    if num_random_layers == 0:
        return circuit
    qubits = sorted(circuit.all_qubits())
    circuit = circuit_parser_utils_cirq.remove_measurements(circuit)
    pre_circuit = gen_random_layered_cirq_circuit(qubits, num_random_layers=num_random_layers)
    post_circuit = gen_random_layered_cirq_circuit(qubits, num_random_layers=num_random_layers)
    return pre_circuit.concat_ragged(circuit, post_circuit)


def get_cirq_qft_circuit(n_qubits):
    qubits = cirq.LineQubit.range(n_qubits)
    qreg = list(qubits)[::-1]
    operations = []
    while len(qreg) > 0:
        q_head = qreg.pop(0)
        operations.append(cirq.H(q_head))
        for i, qubit in enumerate(qreg):
            operations.append((cirq.CZ ** (1 / 2 ** (i + 1)))(qubit, q_head))
    circuit = cirq.Circuit(operations)
    return circuit


def get_cirq_random_circuit(n_qubits, n_moments, op_density=0.9, seed=3):
    return cirq.testing.random_circuit(n_qubits, n_moments, op_density, random_state=seed)


N_QUBITS_RANGE = range(7, 9)
N_MOMENTS_RANGE = DEPTH_RANGE = range(5, 7)

if cirq:
    for n_qubits in N_QUBITS_RANGE:
        cirq_circuits.append(get_cirq_qft_circuit(n_qubits))
        for n_moments in N_MOMENTS_RANGE:
            cirq_circuits.append(get_cirq_random_circuit(n_qubits, n_moments))

try:
    from cuquantum_benchmarks.frontends.frontend_cirq import Cirq as cuqnt_cirq
    from cuquantum_benchmarks.benchmarks import qpe, quantum_volume, qaoa
    cirq_generators = [qpe.QPE, quantum_volume.QuantumVolume, qaoa.QAOA]
    config = {'measure': True, 'unfold': True, 'p': 4}
    for generator in cirq_generators:
        for n_qubits in (5, 6):
            seq = generator.generateGatesSequence(n_qubits, config)
            circuit = cuqnt_cirq(n_qubits, config).generateCircuit(seq)
            cirq_circuits.append(circuit)
except:
    pass

cirq_circuits_mps = [cirq_insert_random_layers(circuit) for circuit in cirq_circuits]

#########################################################
# functions to generate qiskit.QuantumCircuit for testing
#########################################################

def get_qiskit_unitary_gate(rng=GLOBAL_RNG, control=None):
    # random unitary two qubit gate
    from qiskit.extensions import UnitaryGate
    m = rng.standard_normal(size=(4, 4)) + 1j*rng.standard_normal(size=(4, 4))
    q, r = np.linalg.qr(m)
    d = np.diag(r)
    q *= d/abs(d)
    gate = UnitaryGate(q)
    if control is None:
        return gate
    else:
        return gate.control(control)


def gen_random_layered_qiskit_circuit(qubits, num_random_layers=DEFAULT_NUM_RANDOM_LAYERS):
    n_qubits = len(qubits)
    circuit = qiskit.QuantumCircuit(qubits)
    for n in range(num_random_layers):
        for i in range(n%2, n_qubits-1, 2):
            circuit.append(get_qiskit_unitary_gate(), qubits[i:i+2])
    return circuit


def qiskit_insert_random_layers(circuit, num_random_layers=DEFAULT_NUM_RANDOM_LAYERS):
    if num_random_layers == 0:
        return circuit
    qubits = circuit.qubits
    circuit.remove_final_measurements()
    pre_circuit = gen_random_layered_qiskit_circuit(qubits, num_random_layers=num_random_layers)
    post_circuit = gen_random_layered_qiskit_circuit(qubits, num_random_layers=num_random_layers)
    circuit.data = pre_circuit.data + circuit.data + post_circuit.data
    return circuit


def get_qiskit_qft_circuit(n_qubits):
    return qiskit.circuit.library.QFT(n_qubits, do_swaps=False).decompose()


def get_qiskit_random_circuit(n_qubits, depth):
    from qiskit.circuit.random import random_circuit
    circuit = random_circuit(n_qubits, depth, max_operands=3)
    return circuit


def get_qiskit_composite_circuit():
    sub_q = qiskit.QuantumRegister(2)
    sub_circ = qiskit.QuantumCircuit(sub_q, name='sub_circ')
    sub_circ.h(sub_q[0])
    sub_circ.crz(1, sub_q[0], sub_q[1])
    sub_circ.barrier()
    sub_circ.id(sub_q[1])
    sub_circ.u(1, 2, -2, sub_q[0])

    # Convert to a gate and stick it into an arbitrary place in the bigger circuit
    sub_inst = sub_circ.to_instruction()

    qr = qiskit.QuantumRegister(3, 'q')
    circ = qiskit.QuantumCircuit(qr)
    circ.h(qr[0])
    circ.cx(qr[0], qr[1])
    circ.cx(qr[1], qr[2])
    circ.append(sub_inst, [qr[1], qr[2]])
    circ.append(sub_inst, [qr[0], qr[2]])
    circ.append(sub_inst, [qr[0], qr[1]])
    return circ


def get_qiskit_nested_circuit():
    qr = qiskit.QuantumRegister(6, 'q')
    circ = qiskit.QuantumCircuit(qr)
    sub_ins = get_qiskit_composite_circuit().to_instruction()
    circ.append(sub_ins, [qr[0], qr[2], qr[4]])
    circ.append(sub_ins, [qr[1], qr[3], qr[5]])
    circ.cx(qr[0], qr[3])
    circ.cx(qr[1], qr[4])
    circ.cx(qr[2], qr[5])
    return circ


def get_qiskit_multi_control_circuit():
    qubits = qiskit.QuantumRegister(5)
    circuit = qiskit.QuantumCircuit(qubits)
    for q in qubits:
        circuit.h(q)
    qs = list(qubits)
    # 3 layers of multi-controlled qubits
    np.random.seed(0)
    rng = np.random.default_rng(1234)
    for i in range(2):
        rng.shuffle(qs)
        ccu_gate = get_qiskit_unitary_gate(rng, control=2)
        circuit.append(ccu_gate, qs[:4])
        for q in qubits:
            if i % 2 == 1:
                circuit.h(q)
            else:
                circuit.x(q)
    circuit.global_phase = 0.5
    circuit.p(0.1, qubits[0])
    return circuit


if qiskit:
    circuit = get_qiskit_composite_circuit()
    qiskit_circuits.append(circuit.copy())
    circuit.global_phase = 0.5
    qiskit_circuits.append(circuit)
    qiskit_circuits.append(get_qiskit_nested_circuit())
    qiskit_circuits.append(get_qiskit_multi_control_circuit())
    for n_qubits in N_QUBITS_RANGE:
        qiskit_circuits.append(get_qiskit_qft_circuit(n_qubits))
        for depth in DEPTH_RANGE:
            qiskit_circuits.append(get_qiskit_random_circuit(n_qubits, depth))

try:
    from cuquantum_benchmarks.frontends.frontend_qiskit import Qiskit as cuqnt_qiskit
    from cuquantum_benchmarks.benchmarks import qpe, quantum_volume, qaoa
    qiskit_generators = [qpe.QPE, quantum_volume.QuantumVolume, qaoa.QAOA]
    config = {'measure': True, 'unfold': True, 'p': 4}
    for generator in qiskit_generators:
        for n_qubits in (5, 6):
            seq = generator.generateGatesSequence(n_qubits, config)
            circuit = cuqnt_qiskit(n_qubits, config).generateCircuit(seq)
            qiskit_circuits.append(circuit)
except:
    pass

qiskit_circuits_mps = [qiskit_insert_random_layers(circuit) for circuit in qiskit_circuits]

def is_converter_mps_compatible(converter):
    for _, qubits in converter.gates:
        if len(qubits) > 2:
            return False
    return True

def compute_histogram_overlap(hist1, hist2, nshots):
    # assuming hist1 & hist2 have the same sample size (=nshots)
    overlap = 0
    for val, count in hist1.items():
        if val not in hist2:
            continue
        overlap += min(hist1[val], hist2[val])
    overlap /= nshots
    return overlap

class _BaseComputeEngine:
    
    @property
    def qubits(self):
        raise NotImplementedError
    
    @property
    def n_qubits(self):
        return len(self.qubits)
    
    @property
    def tolerance(self):
        raise NotImplementedError
    
    def setup_resources(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_sv(self):
        raise NotImplementedError
    
    def get_norm(self):
        if self.norm is None:
            sv = self.get_sv()
            if sv is not None:
                self.norm  = self.backend.linalg.norm(self.get_sv()).item() ** 2
        return self.norm
    
    def get_amplitude(self, bitstring):
        raise NotImplementedError
    
    def get_batched_amplitudes(self, fixed):
        raise NotImplementedError
    
    def get_reduced_density_matrix(self, where, fixed=EMPTY_DICT):
        r"""
        For where = (a, b), reduced density matrix is formulated as:
        :math: `rho_{a,b,a^{\prime},b^{\prime}}  = \sum_{c,d,e,...} SV^{\star}_{a^{\prime}, b^{\prime}, c, d, e, ...} SV_{a, b, c, d, e, ...}`
        """
        raise NotImplementedError
    
    def get_expectation(self, pauli_string):
        raise NotImplementedError
    
    def get_sampling(self, qubits_to_sample=None, seed=None, nshots=5000):
        raise NotImplementedError


class BaseFrameworkComputeEngine(_BaseComputeEngine):
    ###################################################################
    # 
    #      Reference implementation from framework providers.
    #
    #    Simulator APIs inside cirq and qiskit may be subject to change.
    #  Version tests are needed. In cases where simulator API changes,
    #  the methods to be modified are: 
    #    1. `CirqComputeEngine._get_state_vector` 
    #    2. `CirqComputeEngine.get_sampling`
    #    3. `QiskitComputeEngine._get_state_vector`
    #    4. `QiskitComputeEngine.get_sampling`
    #
    ###################################################################

    def __init__(self, circuit, dtype, backend):
        self.circuit = circuit
        self.backend = backend
        self.dtype = dtype
        self.sv = None
        self._tolerance = None
        self.norm = None
    
    @property
    def tolerance(self):
        if self._tolerance is None:
            self._tolerance = {'atol': atol_mapper[self.dtype],
                               'rtol': rtol_mapper[self.dtype]}
        return self._tolerance
    
    def setup_resources(self, *args, **kwargs):
        # No additional resources needed
        pass
    
    def _get_state_vector(self):
        # implementation for different frameworks
        raise NotImplementedError
    
    def get_sv(self):
        if self.sv is None:
            self.sv = self._get_state_vector()
        return self.sv
    
    def get_amplitude(self, bitstring):
        return amplitude_from_sv(self.get_sv(), bitstring)
    
    def get_batched_amplitudes(self, fixed):
        fixed = dict([(self.qubits.index(q), bit) for q, bit in fixed.items()])
        return batched_amplitude_from_sv(self.get_sv(), fixed)
    
    def get_reduced_density_matrix(self, where, fixed=EMPTY_DICT):
        sv = self.get_sv()
        where = [self.qubits.index(q) for q in where]
        fixed = dict([(self.qubits.index(q), bit) for q, bit in fixed.items()])
        return reduced_density_matrix_from_sv(sv, where, fixed=fixed)
    
    def get_expectation(self, pauli_string):
        return expectation_from_sv(self.get_sv(), pauli_string)
    
    def get_sampling(self, qubits_to_sample=None, seed=None, nshots=5000):
        # implementation for different framework providers
        raise NotImplementedError


class CirqComputeEngine(BaseFrameworkComputeEngine):

    @property
    def qubits(self):
        return sorted(self.circuit.all_qubits())
    
    def _get_state_vector(self):
        qubits = self.qubits
        simulator = cirq.Simulator(dtype=self.dtype)
        circuit = circuit_parser_utils_cirq.remove_measurements(self.circuit)
        result = simulator.simulate(circuit, qubit_order=qubits)
        statevector = result.state_vector().reshape((2,)*self.n_qubits)
        if self.backend is torch:
            statevector = torch.as_tensor(statevector, dtype=getattr(torch, self.dtype), device='cuda')
        else:
            statevector = self.backend.asarray(statevector, dtype=self.dtype)
        return statevector
    
    def get_sampling(self, qubits_to_sample=None, seed=None, nshots=5000):
        if qubits_to_sample is None:
            qubits_to_sample = self.qubits
        circuit = circuit_parser_utils_cirq.remove_measurements(self.circuit)
        circuit.append(cirq.measure_each(qubits_to_sample))
        circuit.append(cirq.measure(*qubits_to_sample, key='meas'))
        result = cirq.sample(
            circuit, repetitions=nshots, seed=seed, dtype=getattr(np, self.dtype))
        result = result.histogram(key='meas')
        sampling = {}
        nsamples = 0
        for bitstring, nsample in result.items():
            sampling[int(bitstring)] = nsample
            nsamples += nsample
        assert nsamples == nshots
        return sampling


class QiskitComputeEngine(BaseFrameworkComputeEngine):

    @property
    def qubits(self):
        return list(self.circuit.qubits)

    def _get_precision(self):
        precision = {'complex64': 'single',
                     'complex128': 'double'}[self.dtype]
        return precision
    
    def _get_state_vector(self):
        # requires qiskit >= 0.24.0
        precision = self._get_precision()
        circuit = circuit_parser_utils_qiskit.remove_measurements(self.circuit)
        try:
            # for qiskit >= 0.25.0
            simulator = qiskit.Aer.get_backend('aer_simulator_statevector', precision=precision)
            circuit = qiskit.transpile(circuit, simulator)
            circuit.save_statevector()
            result = simulator.run(circuit).result()
        except:
            # for qiskit 0.24.*
            simulator = qiskit.Aer.get_backend('statevector_simulator', precision=precision)
            result = qiskit.execute(circuit, simulator).result()
        sv = np.asarray(result.get_statevector()).reshape((2,)*circuit.num_qubits)
        # statevector returned by qiskit's simulator is labelled by the inverse of :attr:`qiskit.QuantumCircuit.qubits`
        # this is different from `cirq` and different from the implementation in :class:`CircuitToEinsum`
        sv = sv.transpose(list(range(circuit.num_qubits))[::-1])
        if self.backend is torch:
            sv = torch.as_tensor(sv, dtype=getattr(torch, self.dtype), device='cuda')
        else:
            sv = self.backend.asarray(sv, dtype=self.dtype)
        return sv
    
    def get_sampling(self, qubits_to_sample=None, seed=None, nshots=5000):
        if qubits_to_sample is None:
            qubits_to_sample = self.qubits
        circuit = self.circuit.remove_final_measurements(inplace=False)
        new_creg = circuit._create_creg(len(qubits_to_sample), "meas")
        circuit.add_register(new_creg)
        circuit.measure(qubits_to_sample, new_creg)
        precision = self._get_precision()
        backend = qiskit.Aer.get_backend('qasm_simulator', precision=precision)
        result = backend.run(qiskit.transpile(circuit, backend), shots=nshots, seed=seed).result()
        counts  = result.get_counts(circuit)
        sampling = {}
        nsamples = 0
        for bitstring, nsample in counts.items():
            # little endian from qiskit
            value = int(bitstring[::-1], 2)
            sampling[value] = nsample
            nsamples += nsample
        assert nsamples == nshots
        return sampling


class CircuitToEinsumComputeEngine(_BaseComputeEngine):

    def __init__(self, converter):
        self.converter = converter
        self.backend = self.converter.backend
        if self.backend is torch:
            self.dtype = str(converter.dtype).split('.')[-1]
        else:
            self.dtype = converter.dtype.__name__
        self._tolerance = None
        self.handle = None # Non-owning
        self.sv = None
        self.norm = None
    
    @property
    def qubits(self):
        return list(self.converter.qubits)
    
    @property
    def tolerance(self):
        if self._tolerance is None:
            self._tolerance = {'atol': atol_mapper[self.dtype],
                               'rtol': rtol_mapper[self.dtype]}
        return self._tolerance
    
    def setup_resources(self, *args, **kwargs):
        self.handle = kwargs.get('handle', None)
    
    def _compute_from_converter(self, task, *args, **kwargs):
        assert self.handle is not None, "handle not provided"
        expression, operands = getattr(self.converter, task)(*args, **kwargs)
        return contract(expression, *operands, options={'handle': self.handle})
    
    def get_sv(self):
        if self.sv is None:
            self.sv = self._compute_from_converter('state_vector')
        return self.sv
    
    def get_reduced_density_matrix(self, where, fixed=EMPTY_DICT, lightcone=True):
        return self._compute_from_converter('reduced_density_matrix', where, fixed=fixed, lightcone=lightcone)
    
    def get_sampling(self, qubits_to_sample=None, seed=None, nshots=5000):
        sv = self.get_sv()
        if qubits_to_sample is None:
            modes_to_sample = list(range(self.n_qubits))
        else:
            modes_to_sample = [self.qubits.index(q) for q in qubits_to_sample]
        return sample_from_sv(sv, nshots, modes_to_sample=modes_to_sample, seed=seed)
        
    def get_amplitude(self, bitstring):
        return self._compute_from_converter('amplitude', bitstring)
    
    def get_batched_amplitudes(self, fixed):
        return self._compute_from_converter('batched_amplitudes', fixed)
    
    def get_expectation(self, pauli_string, lightcone=True):
        return self._compute_from_converter('expectation', pauli_string, lightcone=lightcone)
    

class StateComputeEngine(_BaseComputeEngine):
    #####################################################################
    # 
    #      Implementation  from cutensornetState_t APIs.
    #  This reference are only meant to be tested when backend is `cupy`.
    # 
    #  The methods below must have the same API signature with 
    #  their counterer parts in `BastFrameworkComputeEngine` 
    #  (up to the first few arguments being handle and workspace):
    #    1. `StateComputeEngine.get_sv`
    #    2. `StateComputeEngine.get_amplitude`
    #    3. `StateComputeEngine.get_batched_amplitudes`
    #    4. `StateComputeEngine.get_reduced_density_matrix`
    #    5. `StateComputeEngine.get_expectation`
    #    6. `StateComputeEngine.get_sampling`
    #
    #####################################################################

    def __init__(self, converter, **options):
        if converter.backend is not cp:
            raise RuntimeError("This class is only expected to be executed for cupy backend")
        self.converter = converter
        self.state = None
        self.state_computed = False
        self.circuit_state_parsed = False
        if converter.backend is torch:
            self.dtype = str(converter.dtype).split('.')[-1]
        else:
            self.dtype = converter.dtype.__name__
        self.options = options
        self._tolerance = None
        gate_i = cp.asarray([[1,0], [0,1]], dtype=self.dtype, order=np.random.choice(['C', 'F']))
        gate_x = cp.asarray([[0,1], [1,0]], dtype=self.dtype, order=np.random.choice(['C', 'F']))
        gate_y = cp.asarray([[0,-1j], [1j,0]], dtype=self.dtype, order=np.random.choice(['C', 'F']))
        gate_z = cp.asarray([[1,0], [0,-1]], dtype=self.dtype, order=np.random.choice(['C', 'F']))
        self.pauli_map = {'I': gate_i.T,
                          'X': gate_x.T,
                          'Y': gate_y.T,
                          'Z': gate_z.T}
        self.norm = None
        self.sv = None
        self.handle = None # non-owning
        self.workspace = None # non-owning
    
    @property
    def qubits(self):
        return list(self.converter.qubits)
    
    @property
    def tolerance(self):
        if self._tolerance is None:
            self._tolerance = {'atol': atol_mapper[self.dtype],
                               'rtol': rtol_mapper[self.dtype]}
        return self._tolerance

    def __del__(self):
        if self.state is not None:
            cutn.destroy_state(self.state)
    
    def setup_resources(self, *args, **kwargs):
        self.handle = kwargs.get('handle', None)
        self.workspace = kwargs.get('workspace', None)
    
    def _maybe_create_state(self):
        assert self.handle is not None and self.workspace is not None, f"handle or workspace not setted up"
        if self.state is None:
            dtype = dtype_to_data_type[getattr(np, self.dtype)]
            # create the state object
            self.state = cutn.create_state(self.handle, 
                cutn.StatePurity.PURE, self.n_qubits, (2,)*self.n_qubits, dtype)

    def _maybe_parse_state(self):
        self._maybe_create_state()

        if not self.circuit_state_parsed:
            gates = self.converter.gates
            immutable = 0
            adjoint = 0
            unitary = 1 # assuming all gates unitary
            self.operands = []
            for (operand, qubits) in gates:
                n_state_modes = len(qubits)
                state_modes = [self.qubits.index(q) for q in qubits]
                # keep operand alive otherwise cupy will re-use the memory space
                operand = operand.T.astype(operand.dtype, order=np.random.choice(['C', 'F']))
                self.operands.append(operand)
                tensor_mode_strides = [stride_in_bytes//operand.itemsize for stride_in_bytes in operand.strides]
                update_tensor = np.random.choice([True, False], p=[0.1, 0.9])
                if update_tensor:
                    tmp = cp.empty_like(operand)
                    tensor_id = cutn.state_apply_tensor(self.handle, self.state, n_state_modes, 
                        state_modes, tmp.data.ptr, tensor_mode_strides, 
                        immutable, adjoint, unitary)
                    cutn.state_update_tensor(self.handle, self.state, 
                                    tensor_id, operand.data.ptr, unitary)
                else:
                    cutn.state_apply_tensor(self.handle, self.state, n_state_modes, 
                        state_modes, operand.data.ptr, tensor_mode_strides, 
                        immutable, adjoint, unitary)
            self.circuit_state_parsed = True
    
    def _maybe_parse_options(self):
        if self.options:
            raise NotImplementedError
    
    def _maybe_compute_state(self):
        # Implement this for different type of simulators
        # For tensor network simulator, final state is not computed
        # For other types of simulator, final state must be explictly computed and stored
        if not self.state_computed:
            self._maybe_parse_state()
            self._maybe_parse_options()
            self.state_computed = True
    
    def _compute_target(self, task, create_args, execute_args, stream):
        if task != 'state':
            # avoid going into infinite loops
            self._maybe_compute_state()
        if task == 'marginal':
            create_func = cutn.create_marginal
            configure_func = cutn.marginal_configure
            hyper_sample_attr = cutn.MarginalAttribute.OPT_NUM_HYPER_SAMPLES
            num_hyper_samples_dtype = cutn.marginal_get_attribute_dtype(hyper_sample_attr)
            prepare_func = cutn.marginal_prepare
            execute_func = cutn.marginal_compute
            destroy_func = cutn.destroy_marginal
        elif task == 'sampler':
            create_func = cutn.create_sampler
            configure_func = cutn.sampler_configure
            hyper_sample_attr = cutn.SamplerAttribute.OPT_NUM_HYPER_SAMPLES
            num_hyper_samples_dtype = cutn.sampler_get_attribute_dtype(hyper_sample_attr)
            prepare_func = cutn.sampler_prepare
            execute_func = cutn.sampler_sample
            destroy_func = cutn.destroy_sampler
        elif task == 'accessor':
            create_func = cutn.create_accessor
            configure_func = cutn.accessor_configure
            hyper_sample_attr = cutn.AccessorAttribute.OPT_NUM_HYPER_SAMPLES
            num_hyper_samples_dtype = cutn.accessor_get_attribute_dtype(hyper_sample_attr)
            prepare_func = cutn.accessor_prepare
            execute_func = cutn.accessor_compute
            destroy_func = cutn.destroy_accessor
        elif task == 'expectation':
            create_func = cutn.create_expectation
            configure_func = cutn.expectation_configure
            hyper_sample_attr = cutn.ExpectationAttribute.OPT_NUM_HYPER_SAMPLES
            num_hyper_samples_dtype = cutn.accessor_get_attribute_dtype(hyper_sample_attr)
            prepare_func = cutn.expectation_prepare
            execute_func = cutn.expectation_compute
            destroy_func = cutn.destroy_expectation
        elif task == 'state':
            # full state_vector computation does not need to destroy state
            create_func = None
            configure_func = cutn.state_configure
            hyper_sample_attr = cutn.StateAttribute.NUM_HYPER_SAMPLES
            num_hyper_samples_dtype = cutn.state_get_attribute_dtype(hyper_sample_attr)
            prepare_func = cutn.state_prepare
            execute_func = cutn.state_compute
            destroy_func = None
        else:
            raise ValueError("only supports marginal, sampler, accessor, expectation and state")
        
        dev = cp.cuda.Device()
        free_mem = dev.mem_info[0]
        scratch_size = free_mem // 2 # maximal usage of 50% device memory
        if create_func is None: # state vector computation
            task_obj = self.state
        else:
            task_obj = create_func(self.handle, self.state, *create_args)
        num_hyper_samples = np.asarray(8, dtype=num_hyper_samples_dtype)
        configure_func(self.handle, task_obj, hyper_sample_attr, 
            num_hyper_samples.ctypes.data, num_hyper_samples.dtype.itemsize)
        prepare_func(self.handle, task_obj, scratch_size, self.workspace, stream.ptr) # similar args for marginal and sampler

        for memspace in (cutn.Memspace.DEVICE, cutn.Memspace.HOST):
            workspace_size = cutn.workspace_get_memory_size(self.handle, 
                        self.workspace, cutn.WorksizePref.RECOMMENDED, 
                        memspace, cutn.WorkspaceKind.SCRATCH)
            workspace_ptr = None
            if memspace == cutn.Memspace.DEVICE:
                if workspace_size > scratch_size:
                    destroy_func(task_obj)
                    return None
                else:
                    workspace_ptr = cp.cuda.alloc(workspace_size).ptr
            else:
                workspace_ptr = np.empty(workspace_size, dtype=np.int8).ctypes.data
            if workspace_size != 0:
                cutn.workspace_set_memory(self.handle, 
                    self.workspace, memspace, 
                    cutn.WorkspaceKind.SCRATCH, workspace_ptr, workspace_size)
        
        output = execute_func(self.handle, task_obj, *execute_args, stream.ptr)
        stream.synchronize()
        if destroy_func is not None:
            destroy_func(task_obj)
        if isinstance(output, tuple):
            return output
        else:
            return True

    def _run_state_accessor(self, bitstring=None, fixed=None):
        if bitstring is not None:
            # compute a single bitstring amplitude
            assert fixed is None
            shape = 1
            num_fixed_modes = self.n_qubits
            fixed_modes = list(range(self.n_qubits))
            fixed_values = [int(i) for i in bitstring]
        elif fixed is not None:
            # compute batched amplitudes
            shape = (2,) * (self.n_qubits - len(fixed))
            num_fixed_modes = len(fixed)
            fixed_modes = []
            fixed_values = []
            for q, bit in fixed.items():
                fixed_modes.append(self.qubits.index(q))
                fixed_values.append(int(bit))
        else:
            # compute full state vector
            shape = (2, ) * self.n_qubits
            num_fixed_modes = fixed_modes = fixed_values = 0
        
        amplitudes = cp.empty(shape, dtype=self.dtype, order=np.random.choice(('C', 'F')))
        amplitudes_strides = [stride_in_bytes // amplitudes.itemsize for stride_in_bytes in amplitudes.strides]
        norm = np.empty(1, dtype=self.dtype)

        create_args = (num_fixed_modes, fixed_modes, amplitudes_strides)
        execute_args = (fixed_values, self.workspace, amplitudes.data.ptr, norm.ctypes.data)
        stream = cp.cuda.get_current_stream()
        if self._compute_target('accessor', create_args, execute_args, stream):
            if self.norm is None:
                self.norm = norm.item()
            else:
                assert np.allclose(self.norm, norm.item(), **self.tolerance)
            return amplitudes
        else:
            return None

    def get_sv(self):
        if self.sv is None:
            self.sv = self._run_state_accessor()
        return self.sv
    
    def get_reduced_density_matrix(self, where, fixed=EMPTY_DICT):
        n_marginal_modes = len(where)
        marginal_modes = [self.qubits.index(q) for q in where]
        if fixed:
            n_projected_modes = len(fixed)
            projected_modes = []
            projected_mode_values = []
            for q, bit in fixed.items():
                projected_modes.append(self.qubits.index(q))
                projected_mode_values.append(int(bit))
        else:
            n_projected_modes = projected_modes = projected_mode_values = 0
        
        rdm = cp.empty((2,2)*n_marginal_modes, 
                dtype=self.dtype, order=np.random.choice(['C', 'F']))
        rdm_strides = [s // rdm.itemsize for s in rdm.strides]
        stream = cp.cuda.get_current_stream()

        create_args = (n_marginal_modes, marginal_modes, n_projected_modes, projected_modes, rdm_strides)
        execute_args = (projected_mode_values, self.workspace, rdm.data.ptr)
        if self._compute_target('marginal', create_args, execute_args, stream):
            return rdm
        else:
            return None
 
    def get_sampling(self, qubits_to_sample=None, seed=None, nshots=5000):
        if qubits_to_sample is None:
            qubits_to_sample = self.qubits
        n_modes_to_sample = len(qubits_to_sample)
        modes_to_sample = [self.qubits.index(q) for q in qubits_to_sample]
        samples = np.empty((nshots, n_modes_to_sample), 
                    dtype='int64', order='C') # equivalent to (n_modes, nshots) in F order
        stream = cp.cuda.get_current_stream()

        create_args = (n_modes_to_sample, modes_to_sample)
        execute_args = (nshots, self.workspace, samples.ctypes.data)
        if self._compute_target('sampler', create_args, execute_args, stream):
            sampling = {}
            for bitstring, n_sampling in zip(*np.unique(samples, axis=0, return_counts=True)):
                bitstring = np.array2string(bitstring, separator='')[1:-1]
                sampling[int(bitstring, 2)] = n_sampling
            return sampling
        else:
            return None
        
    def get_amplitude(self, bitstring):
        return self._run_state_accessor(bitstring=bitstring)
    
    def get_batched_amplitudes(self, fixed):
        return self._run_state_accessor(fixed=fixed)
    
    # cutensornet State APIs can not compute a single expectation. 
    # Here we compute the sum of all Pauli strings
    def get_expectation_sum(self, pauli_strings):
        if not isinstance(pauli_strings, dict):
            raise ValueError("pauli_strings is expected to be a map from paul strings to coefficients")
        dtype = dtype_to_data_type[getattr(np, self.dtype)]
        hamiltonian = cutn.create_network_operator(self.handle, 
                self.n_qubits, (2,)*self.n_qubits, dtype)
        for pauli_string, coefficient in pauli_strings.items():
            num_tensors = 0
            num_modes = []
            state_modes = []
            tensor_mode_strides = []
            tensor_data = []
            for q, pauli_char in enumerate(pauli_string):
                if pauli_char == 'I': continue
                operand = self.pauli_map[pauli_char]
                num_tensors += 1
                num_modes.append(1)
                state_modes.append([q])
                tensor_mode_strides.append([stride_in_bytes//operand.itemsize for stride_in_bytes in operand.strides])
                tensor_data.append(operand.data.ptr)
            if num_tensors == 0: 
                # pauli string being IIIIII
                num_tensors = self.n_qubits
                num_modes = [1,] * num_tensors
                state_modes = list(range(num_tensors))
                operand = self.pauli_map['I']
                tensor_data = [operand.data.ptr] * num_tensors
                tensor_mode_strides = [stride_in_bytes//operand.itemsize for stride_in_bytes in operand.strides] * num_tensors
            cutn.network_operator_append_product(self.handle, 
                hamiltonian, coefficient, num_tensors, 
                num_modes, state_modes, tensor_mode_strides, tensor_data)
        
        expectation_value = np.empty(1, dtype=self.dtype)
        norm = np.empty(1, dtype=self.dtype)
        create_args = (hamiltonian, )
        execute_args = (self.workspace, expectation_value.ctypes.data, norm.ctypes.data)
        stream = cp.cuda.get_current_stream()
        if self._compute_target('expectation', create_args, execute_args, stream):
            output = expectation_value.item()
            if self.norm is None:
                self.norm = norm.item()
            else:
                assert np.allclose(self.norm, norm.item(), **self.tolerance)
        else:
            output = None
        cutn.destroy_network_operator(hamiltonian)
        return output


class SVStateComputeEngine(StateComputeEngine):

    def _maybe_compute_state(self):
        # Implement this for different type of simulators
        # For tensor network simulator, final state is not computed
        # For other types of simulator, final state must be explictly computed and stored
        if not self.state_computed:
            self._maybe_parse_state()
            self._maybe_parse_options()
            order = np.random.choice(('C', 'F'))
            sv = cp.empty((2,) * self.n_qubits, dtype=self.dtype, order=order)
            stream = cp.cuda.get_current_stream()
            create_args = ()
            execute_args = (self.workspace, [sv.data.ptr])
            output = self._compute_target('state', create_args, execute_args, stream)
            if output:
                extents = output[0][0]
                strides = [s * sv.dtype.itemsize for s in output[1][0]]
                if order == 'F':
                    self.sv = sv
                else:
                    self.sv = cp.ndarray(extents, 
                            dtype=sv.dtype, memptr=sv.data, strides=strides)
                self.state_computed = True
            else:
                self.sv = None
    
    def get_sv(self):
        self._maybe_compute_state()
        return self.sv


class MPSStateComputeEngine(StateComputeEngine):

    @property
    def tolerance(self):
        if self._tolerance is None:
            # tolerance for double precision is increase
            self._tolerance = get_mps_tolerance(self.dtype)
        return self._tolerance

    def _maybe_parse_options(self):
        self._maybe_create_state()
        # parse max extent
        max_extent = self.options.get('max_extent', None)
        if max_extent is None:
            if self.n_qubits > EXACT_MPS_QUBIT_COUNT_LIMIT:
                raise ValueError(f"Exact MPS will encounter overflow with n_qubits={self.n_qubits}")
            else:
                max_extent = 2**EXACT_MPS_QUBIT_COUNT_LIMIT
        self.mps_tensors = []
        prev_extent = 1
        output_mps_extents = []
        output_mps_strides = []
        for i in range(self.n_qubits):
            next_extent = min(max_extent, 2**(i+1), 2**(self.n_qubits-i-1))
            if i==0:
                extents = (2, next_extent)
            elif i !=self.n_qubits - 1:
                extents = (prev_extent, 2, next_extent)
            else:
                extents = (prev_extent, 2)
            prev_extent = next_extent
            tensor = cp.empty(extents, dtype=self.dtype, order=np.random.choice(['C', 'F']))
            self.mps_tensors.append(tensor)
            output_mps_extents.append(extents)
            output_mps_strides.append([stride_in_bytes // tensor.itemsize for stride_in_bytes in tensor.strides])
        cutn.state_finalize_mps(self.handle, self.state, 
            cutn.BoundaryCondition.OPEN, output_mps_extents, output_mps_strides)
        
        algorithm = 'gesvd'
        for key, value in self.options.items():
            if key in STATE_ATTRIBUTE_MAP:
                attr = STATE_ATTRIBUTE_MAP[key]
                dtype = cutn.state_get_attribute_dtype(attr)
                if key == 'algorithm':
                    algorithm = value
                    value = SVD_ALGORITHM_MAP[value]
                elif key == 'normalization':
                    value = NORMALIZATION_MAP[value]
                elif key == 'canonical_center' and value is None:
                    continue
                value = np.asarray(value, dtype=dtype)
                cutn.state_configure(self.handle, self.state, attr, value.ctypes.data, value.dtype.itemsize)

        if algorithm in ('gesvdj', 'gesvdr'):
            dtype = cutn.tensor_svd_algo_params_get_dtype(SVD_ALGORITHM_MAP[algorithm])
            algo_params = np.zeros(1, dtype=dtype)

            for name in dtype.names:
                value = self.options.get(f'{algorithm}_{name}', 0)
                if value != 0:
                    algo_params[name] = value
            cutn.state_configure(self.handle, self.state, 
                cutn.StateAttribute.MPS_SVD_CONFIG_ALGO_PARAMS, 
                algo_params.ctypes.data, algo_params.dtype.itemsize)
    
    def _maybe_compute_state(self):
        # Implement this for different type of simulators
        # For tensor network simulator, final state is not computed
        # For other types of simulator, final state must be explictly computed and stored
        if not self.state_computed:
            self._maybe_parse_state()
            self._maybe_parse_options()
            stream = cp.cuda.get_current_stream()
            create_args = ()
            execute_args = (self.workspace, [o.data.ptr for o in self.mps_tensors])
            output = self._compute_target('state', create_args, execute_args, stream)
            if output is None:
                return False
            else:
                extents, strides = output
                for i in range(self.n_qubits):
                    extent_in = self.mps_tensors[i].shape
                    extent_out = extents[i]
                    if extent_in != tuple(extent_out):
                        tensor_strides = [s * self.mps_tensors[i].dtype.itemsize for s in strides[i]]
                        self.mps_tensors[i] = cp.ndarray(extent_out, 
                            dtype=self.mps_tensors[i].dtype, memptr=self.mps_tensors[i].data, strides=tensor_strides)
                self.state_computed = True

    def check_canonicalization(self):
        self._maybe_compute_state()
        center = self.options.get('canonical_center', None)
        if center is None:
            return
        for i in range(self.n_qubits):
            if i == 0:
                modes = 'pj'
            elif i == self.n_qubits - 1:
                modes = 'ip'
            else:
                modes = 'ipj'
            if i == center:
                continue
            if i < center:
                shared_mode = 'j'
            elif i > center:
                shared_mode = 'i'
            else:
                continue
            verify_unitary(self.mps_tensors[i], modes, shared_mode, 
                SVD_TOLERANCE[self.dtype], tensor_name=f"Site {i} canonicalization")


class BaseTester:
    
    @property
    def reference_engine(self):
        raise NotImplementedError
    
    @property
    def target_engines(self):
        raise NotImplementedError
    
    @property
    def all_engines(self):
        return [self.reference_engine] + self.target_engines
    
    def test_misc(self):
        raise NotImplementedError
    
    def test_norm(self):
        norm1 = self.reference_engine.get_norm()
        for engine in self.target_engines:
            norm2 = engine.get_norm()
            message = f"{engine.__class__.__name__} maxDiff={abs(norm1-norm2)}"
            assert np.allclose(norm1, norm2, **engine.tolerance), message
    
    def test_state_vector(self):
        sv1 = self.reference_engine.get_sv()
        for engine in self.target_engines:
            sv2 = engine.get_sv()
            message = f"{engine.__class__.__name__} maxDiff={abs(sv1-sv2).max()}"
            assert self.backend.allclose(sv1, sv2, **engine.tolerance), message
    
    def test_amplitude(self):
        for bitstring in bitstring_generator(self.n_qubits, self.nsample):    
            amp1 = self.reference_engine.get_amplitude(bitstring)
            for engine in self.target_engines:
                amp2 = engine.get_amplitude(bitstring)
                message = f"{engine.__class__.__name__} maxDiff={abs(amp1-amp2).max()}"
                assert self.backend.allclose(amp1, amp2, **engine.tolerance), message
    
    def test_batched_amplitudes(self):
        for fixed in where_fixed_generator(self.qubits, self.nfix_max):
            batched_amps1 = self.reference_engine.get_batched_amplitudes(fixed)
            for engine in self.target_engines:
                batched_amps2 = engine.get_batched_amplitudes(fixed)
                message = f"{engine.__class__.__name__} maxDiff={abs(batched_amps1-batched_amps2).max()}"
                assert self.backend.allclose(batched_amps1, batched_amps2, **engine.tolerance), message
    
    def test_reduced_density_matrix(self):
        for where, fixed in where_fixed_generator(self.qubits, self.nfix_max, nsite_max=self.nsite_max):
            operands1 = self.converter.reduced_density_matrix(where, fixed=fixed, lightcone=True)[1]
            operands2 = self.converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)[1]
            assert len(operands1) <= len(operands2) + 2 # potential phase handling for qiskit Circuit            
            
            rdm1 = self.reference_engine.get_reduced_density_matrix(where, fixed=fixed)
            if isinstance(self.reference_engine, CircuitToEinsumComputeEngine):
                # by default CircuitToEinsumComputeEngine.get_reduced_density_matrix uses lightcone=True
                rdm2 = self.reference_engine.get_reduced_density_matrix(where, fixed=fixed, lightcone=False)
                assert self.backend.allclose(rdm1, rdm2, **self.reference_engine.tolerance)
            
            # comparision with different references
            for engine in self.target_engines:
                rdm2 = engine.get_reduced_density_matrix(where, fixed=fixed)
                message = f"{engine.__class__.__name__} maxDiff={abs(rdm1-rdm2).max()}"
                assert self.backend.allclose(rdm1, rdm2, **engine.tolerance), message
    
    def test_expectation(self):
        full_expectation = 0.
        pauli_strings = dict()
        for pauli_string in random_pauli_string_generator(self.n_qubits, 6):
            coefficient = np.random.random(1).item() + 1j * np.random.random(1).item()
            if pauli_string not in pauli_strings:
                pauli_strings[pauli_string] = coefficient
            else:
                # in case duplicate pauli string is reproduced by the random generator
                pauli_strings[pauli_string] += coefficient
            operands1 = self.converter.expectation(pauli_string, lightcone=True)[1]
            operands2 = self.converter.expectation(pauli_string, lightcone=False)[1]
            assert len(operands1) <= len(operands2) + 2 # potential phase handling for qiskit Circuit
            
            expec1 = self.reference_engine.get_expectation(pauli_string)
            if isinstance(self.reference_engine, CircuitToEinsumComputeEngine):
                expec2 = self.reference_engine.get_expectation(pauli_string, lightcone=False)
                assert self.backend.allclose(expec1, expec2, **self.reference_engine.tolerance)
            
            full_expectation += coefficient * expec1
                
            for engine in self.target_engines:
                if not isinstance(engine, StateComputeEngine):
                    expec2 = engine.get_expectation(pauli_string)
                    message = f"{engine.__class__.__name__} maxDiff={abs(expec1-expec2).max()}"
                    assert self.backend.allclose(expec1, expec2, **engine.tolerance), message
        
        for engine in self.target_engines:
            if isinstance(engine, StateComputeEngine):
                expec2 = engine.get_expectation_sum(pauli_strings)
                message = f"{engine.__class__.__name__} maxDiff={abs(full_expectation-expec2).max()}"
                assert self.backend.allclose(full_expectation, expec2, **engine.tolerance), message
    
    def test_sampling(self):
        full_qubits = list(self.qubits)
        np.random.shuffle(full_qubits)
        selected_qubits = full_qubits[:len(full_qubits)//2]

        for engine in self.target_engines:
            for qubits_to_sample in (None, selected_qubits):
                seed = self.seed
                nshots = self.nshots
                max_try = 3
                overlap_best = 0.

                for counter in range(1, max_try+1):
                    # build a histogram for the reference impl
                    hist_ref = self.reference_engine.get_sampling(qubits_to_sample=qubits_to_sample, seed=seed, nshots=self.nshots)

                    # do the same for cutensornet sampling
                    hist_cutn = engine.get_sampling(qubits_to_sample=qubits_to_sample, seed=seed, nshots=self.nshots)

                    # compute overlap of the histograms (cutn vs ref)
                    overlap = compute_histogram_overlap(hist_cutn, hist_ref, self.nshots)
                    if overlap > overlap_best:
                        overlap_best = overlap
                    else:
                        print(f"WARNING: overlap not improving {counter=} {overlap_best=} {overlap=} as nshots increases!")
                    
                    # to reduce test time we set 95% here, but 99% will also work
                    if np.round(overlap, decimals=2) < 0.95:
                        self.nshots *= 10
                        print(f"retry with nshots = {self.nshots} ...")
                    else:
                        self.nshots = nshots  # restore
                        break
                else:
                    self.nshots = nshots  # restore
                    assert False, f"{overlap_best=} after {counter} retries..."
    
    @manage_resource("handle")
    @manage_resource("workspace")
    def run_tests(self):
        resources = {'handle': self.handle, 'workspace': self.workspace}
        # share cutensornet resources for all compute engines
        for engine in self.all_engines:
            engine.setup_resources(**resources)
        
        self.test_state_vector()
        self.test_amplitude()
        self.test_batched_amplitudes()
        self.test_reduced_density_matrix()
        self.test_expectation()
        self.test_norm()
        self.test_misc()
        if self.backend is cp:
            # sampling only needed to be tested for cupy backend
            self.test_sampling()


class CircuitToEinsumTester(BaseTester):
    def __init__(self, circuit, dtype, backend, nsample, nsite_max, nfix_max, nshots=5000, seed=1024):
        self.circuit = circuit
        self.converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend)
        self.backend = backend
        self.qubits = list(self.converter.qubits)
        self.n_qubits = self.converter.n_qubits
        self.dtype = dtype
        self.sv = None
        self.nsample = nsample
        self.nsite_max = max(1, min(nsite_max, self.n_qubits-1))
        self.nfix_max = max(min(nfix_max, self.n_qubits-nsite_max-1), 0)
        self.nshots = nshots
        self.seed = seed
        
        self._reference_engine = CircuitToEinsumComputeEngine(self.converter)

        # Framework provider as reference
        if qiskit and isinstance(circuit, qiskit.QuantumCircuit):
            self._target_engines = [QiskitComputeEngine(circuit, dtype, backend)]
        elif cirq and isinstance(circuit, cirq.Circuit):
            self._target_engines = [CirqComputeEngine(circuit, dtype, backend)]
        else:
            raise ValueError(f"circuit type {type(circuit)} not supported")
        
        if backend == cp:
            # Tensor network state simulator
            self._target_engines.append(StateComputeEngine(self.converter))
            # SV state simulator
            self._target_engines.append(SVStateComputeEngine(self.converter))
            # MPS simulators are only functioning if no multicontrol gates exist in the circuit. 
            if is_converter_mps_compatible(self.converter):
                # MPS state simulator
                self._target_engines.append(MPSStateComputeEngine(self.converter))
                # reference MPS implementation
                self._target_engines.append(MPS.from_converter(self.converter))
    
    @property
    def reference_engine(self):
        return self._reference_engine
    
    @property
    def target_engines(self):
        return self._target_engines

    def test_misc(self):
        self.test_qubits()
        self.test_gates()
        norm = self.reference_engine.get_norm()
        assert np.allclose(norm, 1, **self.reference_engine.tolerance)
    
    def test_qubits(self):
        assert len(self.qubits) == self.n_qubits
    
    def test_gates(self):
        for (gate_operand, qubits) in self.converter.gates:
            assert gate_operand.ndim == len(qubits) * 2
            assert infer_object_package(gate_operand) == self.backend.__name__


class ApproximateMPSTester(BaseTester):
    def __init__(self, converter, nsample, nsite_max, nfix_max, nshots=5000, seed=1024, **mps_options):
        self.converter = converter
        self.backend = converter.backend
        if self.backend is not cp:
            raise ValueError("This tester is only meant for cupy testing")
        self.qubits = list(self.converter.qubits)
        self.n_qubits = self.converter.n_qubits
        self.dtype = self.converter.dtype.__name__
        self.sv = None
        self.norm = None
        self.nsample = nsample
        self.nsite_max = max(1, min(nsite_max, self.n_qubits-1))
        self.nfix_max = max(min(nfix_max, self.n_qubits-nsite_max-1), 0)
        self.nshots = nshots
        self.seed = seed
        self.mps_options = mps_options
        if not is_converter_mps_compatible(self.converter):
            raise ValueError("circuit contains gates acting on more than 2 qubits")
        self._reference_engine = MPS.from_converter(self.converter, **self.mps_options)
        self._target_engines = [MPSStateComputeEngine(self.converter, **self.mps_options)]
    
    @property
    def reference_engine(self):
        return self._reference_engine
    
    @property
    def target_engines(self):
        return self._target_engines
    
    def test_misc(self):
        for engine in self.all_engines:
            engine.check_canonicalization()