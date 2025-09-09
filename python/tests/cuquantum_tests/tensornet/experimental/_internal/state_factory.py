# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import numpy as np

from cuquantum.tensornet import contract, NetworkOptions
from cuquantum.tensornet.experimental import NetworkState, NetworkOperator
from cuquantum.tensornet.experimental._internal.network_state_utils import get_pauli_map

from ...utils.data import ARRAY_BACKENDS
from ...utils.helpers import TensorBackend, get_dtype_name

def get_random_network_operator(state_dims, rng, backend, *, num_repeats=2, dtype='complex128', options=None):
    if isinstance(options, dict):
        device_id = options.get('device_id', None)
    elif isinstance(options, NetworkOptions):
        device_id = options.device_id
    else:
        device_id = None
    if backend == "numpy":
        device_id = None
    backend = TensorBackend(backend=backend, device_id=device_id)    
    operator_obj = NetworkOperator(state_dims, dtype=dtype, options=options)
    def get_random_modes():
        num_rand_modes = rng.integers(2, len(state_dims))
        rand_modes = list(range(len(state_dims)))
        rng.shuffle(rand_modes)
        return rand_modes[:num_rand_modes]

    # adding tensor product
    for _ in range(num_repeats):
        coefficient = rng.random(1).item()
        if dtype.startswith('complex'):
            coefficient += 1j * rng.random(1).item()
        prod_modes = get_random_modes()
        prod_tensors = []
        for i in prod_modes:
            shape = (state_dims[i], ) * 2
            t = backend.random(shape, dtype, rng)
            prod_tensors.append(t)
        prod_modes = [[i] for i in prod_modes]
        operator_obj.append_product(coefficient, prod_modes, prod_tensors)
    # adding MPOs
    for _ in range(num_repeats):
        coefficient = rng.random(1).item()
        if dtype.startswith('complex'):
            coefficient += 1j * rng.random(1).item()
        mpo_modes = get_random_modes()
        num_mpo_modes = len(mpo_modes)
        mpo_tensors = []
        bond_prev = None
        for i, m in enumerate(mpo_modes):
            bond_next = rng.integers(2, 5)
            dim = state_dims[m]
            if i == 0:
                shape = (dim, bond_next, dim)
            elif i == num_mpo_modes - 1:
                shape = (bond_prev, dim, dim)
            else:
                shape = (bond_prev, dim, bond_next, dim)
            t = backend.random(shape, dtype, rng)
            mpo_tensors.append(t)
            bond_prev = bond_next
        operator_obj.append_mpo(coefficient, mpo_modes, mpo_tensors)
    return operator_obj

def create_vqc_states(config, backend, *, with_control=False):
    # specify the dimensions of the tensor network state
    n_state_modes = 6
    state_mode_extents = (2, ) * n_state_modes
    dtype = 'complex128'

    assert backend in {'numpy', 'cupy'}
    module = importlib.import_module(backend)
    # create random operators
    module.random.seed(8) # seed is chosen such that the value based truncation in config will yield different output MPS extents for state_a and state_b
    random_complex = lambda *args, **kwargs: module.random.random(*args, **kwargs) + 1.j * module.random.random(*args, **kwargs)
    op_one_body = random_complex((2, 2,))
    if with_control:
        op_two_body_x = random_complex((2, 2,))
        op_two_body_y = random_complex((2, 2,))
    else:
        op_two_body_x = random_complex((2, 2, 2, 2))
        op_two_body_y = random_complex((2, 2, 2, 2))

    state_a = NetworkState(state_mode_extents, dtype=dtype, config=config)
    state_b = NetworkState(state_mode_extents, dtype=dtype, config=config)

    # apply one body tensor operators to the tensor network state
    for i in range(n_state_modes):
        modes_one_body = (i, )
        # op_one_body are fixed, therefore setting immutable to True
        tensor_id_a = state_a.apply_tensor_operator(modes_one_body, op_one_body, immutable=True)
        tensor_id_b = state_b.apply_tensor_operator(modes_one_body, op_one_body, immutable=True)
        assert tensor_id_a == tensor_id_b

    two_body_op_ids = []
    # apply two body tensor operators to the tensor network state
    for i in range(0, n_state_modes, 2):
        if with_control:
            control_mode = (i, )
            target_mode = (i+1, )  
        else:
            control_mode = None
            target_mode = (i, i+1)
            
        tensor_id_a = state_a.apply_tensor_operator(target_mode, op_two_body_x, control_modes=control_mode, control_values=0, immutable=False)
        tensor_id_b = state_b.apply_tensor_operator(target_mode, op_two_body_y, control_modes=control_mode, control_values=0, immutable=False)
        assert tensor_id_a == tensor_id_b
        two_body_op_ids.append(tensor_id_a)
    
    pauli_string = {'IXIXIX': 0.5, 'IYIYIY': 0.2, 'IZIZIZ': 0.3, 'IIIIII': 0.1, 'ZIZIZI': 0.4, 'XIXIXI': 0.2}
    operator = NetworkOperator.from_pauli_strings(pauli_string, dtype='complex128', backend=backend)

    return (state_a, op_two_body_x), (state_b, op_two_body_y), operator, two_body_op_ids

def apply_factory_sequence(network_state, sequence):
    tensor_ids = []
    for op, modes, gate_info in sequence:
        if gate_info is None:
            if isinstance(op, (list, tuple)):
                # MPO
                tensor_id = network_state.apply_mpo(modes, op)
            else:
                # GATE
                tensor_id = network_state.apply_tensor_operator(modes, op)
        else:
            if 'probabilities' in gate_info:
                is_unitary_channel = gate_info['probabilities'] is not None
                if is_unitary_channel:
                    tensor_id = network_state.apply_unitary_tensor_channel(modes, op, gate_info['probabilities'])
                else:
                    tensor_id = network_state.apply_general_tensor_channel(modes, op)
            else:
                assert 'control_modes' in gate_info
                assert 'control_values' in gate_info
                # Controlled-Tensor
                # NetworkState currently only support immutable controlled tensors
                tensor_id = network_state.apply_tensor_operator(modes, op, control_modes=gate_info['control_modes'], control_values=gate_info['control_values'], immutable=True)
        tensor_ids.append(tensor_id)
    return tensor_ids


class StateFactory:
    def __init__(
        self, 
        qudits, 
        dtype, 
        layers, 
        rng,
        *,
        backend='numpy',
        adjacent_double_layer=True,
        mpo_bond_dim=None,
        mpo_num_sites=None,
        mpo_geometry="adjacent-ordered",
        ct_target_place="last", # Controlled-Tensor: ct
        initial_mps_dim=None,
    ):
        if isinstance(qudits, (int, np.integer)):
            self.num_qudits = qudits
            self.state_dims = (2, ) * self.num_qudits
        else:
            self.num_qudits = len(qudits)
            self.state_dims = qudits

        if backend not in ARRAY_BACKENDS:
            raise ValueError(f"Backend {backend} not supported")
        self.backend = TensorBackend(backend=backend)
        self.dtype = get_dtype_name(dtype)

        dims = set(self.state_dims)
        if len(dims) == 1 and dims.pop() == 2:
            # unitary/general channel only supported for qubits
            assert set(layers).issubset(set('SDCMuUgG'))
        else:
            assert set(layers).issubset(set('SDCM'))
        self.layers = layers

        self.rng = rng
        self.adjacent_double_layer = adjacent_double_layer

        # settings for MPO layer
        self.mpo_bond_dim = mpo_bond_dim
        if mpo_num_sites is None:
            self.mpo_num_sites = self.num_qudits
        else:
            mpo_num_sites = min(self.num_qudits, mpo_num_sites)
            assert mpo_num_sites >= 2
            self.mpo_num_sites = mpo_num_sites
        assert mpo_geometry in {"adjacent-ordered", "random-ordered", "random"}
        self.mpo_geometry = mpo_geometry
        assert ct_target_place in {"first", "middle", "last"}
        self.ct_target_place = ct_target_place
        self.initial_mps_dim = initial_mps_dim
        self.psi = None
        self._sequence = []

    def get_initial_state(self):
        if self.psi is None:
            self.psi = []
            if self.initial_mps_dim is None:
                for d in self.state_dims:
                    t = self.backend.zeros(d, dtype=self.dtype)
                    t[0] = 1
                    self.psi.append(t)
            else:
                virtual_dim = self.initial_mps_dim
                for i, d in enumerate(self.state_dims):
                    if i==0:
                        shape = (d, virtual_dim)
                    elif i == self.num_qudits - 1:
                        shape = (virtual_dim, d)
                    else:
                        shape = (virtual_dim, d, virtual_dim)
                    self.psi.append(self.backend.random(shape, self.dtype, self.rng))
        return self.psi

    def __str__(self):
        return f"StateFactory(num_qudits={self.num_qudits}, dtype={self.dtype}, layers={self.layers}, rng={self.rng.bit_generator.state}, backend={self.backend.name}, adjacent_double_layer={self.adjacent_double_layer}, mpo_bond_dim={self.mpo_bond_dim}, mpo_num_sites={self.mpo_num_sites}, mpo_geometry={self.mpo_geometry}, ct_target_place={self.ct_target_place}, initial_mps_dim={self.initial_mps_dim})"

    @property
    def sequence(self):
        if not self._sequence:
            self._generate_raw_sequence()
        return self._sequence

    def _generate_raw_sequence(self):
        double_layer_offset = 0
        for layer in self.layers:
            if layer == 'S':
                self._append_single_qudit_layer()
            elif layer == 'D':
                self._append_double_qudit_layer(offset=double_layer_offset)
                # intertwine double layers
                double_layer_offset = 1 - double_layer_offset
            elif layer == 'C':
                self._append_controlled_tensor_mpo_layer()
            elif layer == 'M':
                self._append_mpo_layer()
            elif layer.upper() == 'U':
                self._append_unitary_channel_layer(dense=layer=='U')
            elif layer.upper() == 'G':
                self._append_general_channel_layer(dense=layer=='G')
            else:
                raise ValueError(f"layer type {layer} not supported")
    
    def append_sequence(self, sequence):
        self._sequence.append(sequence)
    
    def _get_pauli_map(self):
        is_complex = 'complex' in self.dtype
        if not hasattr(self, 'pauli_map'):
            complex_dtype = self.dtype if is_complex else {'float32': 'complex64', 'float64': 'complex128'}[self.dtype]
            self.pauli_map = get_pauli_map(self.backend.name, complex_dtype)
            if not is_complex:
                self.pauli_map = {p: self.pauli_map[p].real for p in 'IXZ'}
            pauli_keys = list(self.pauli_map.keys())
            for p1 in pauli_keys:
                for p2 in pauli_keys:
                    self.pauli_map[p1+p2] = self.backend.einsum('Aa,Bb->ABab', self.pauli_map[p1], self.pauli_map[p2])
        return self.pauli_map
    
    def _append_unitary_channel_layer(self, *, dense=True):
        is_complex = 'complex' in self.dtype
        pauli_map = self._get_pauli_map()
        qudits = list(range(self.num_qudits))
        self.rng.shuffle(qudits)
        if dense:
            self._sequence.append(([pauli_map['I'], pauli_map['X']], (qudits[0], ), {'probabilities': [0.95, 0.05]}))
            if is_complex:
                self._sequence.append(([pauli_map[p] for p in 'IXYZ'], (qudits[1], ), {'probabilities': [0.7, 0.15, 0.1, 0.05]}))
            else:
                self._sequence.append(([pauli_map[p] for p in 'IXZ'], (qudits[1], ), {'probabilities': [0.75, 0.15, 0.1]}))
        
        if (dense and self.num_qudits >= 4) or (not dense):
            target_qudits = qudits[2:4] if dense else qudits[:2]
            if is_complex:
                operands = [self.pauli_map[p] for p in ('XY', 'YZ', 'ZX')]
            else:
                operands = [self.pauli_map[p] for p in ('IX', 'IZ', 'XZ')]
            gate_info = {'probabilities': [0.7, 0.2, 0.1]}
            self._sequence.append((operands, target_qudits, gate_info))
        return  

    def _append_general_channel_layer(self, *, dense=True):
        is_complex = 'complex' in self.dtype
        pauli_map = self._get_pauli_map()
        qudits = list(range(self.num_qudits))
        self.rng.shuffle(qudits)
        if dense:
            g0, g1 = self.rng.random(2)
            dampling_channel = [
                self.backend.asarray([[1, 0], [0, np.sqrt(1 - g0)]], dtype=self.dtype),
                self.backend.asarray([[0, np.sqrt(g0)], [0, 0]], dtype=self.dtype)
            ]
            if is_complex:
                depolarizing_channel = [
                    (1 - g1) ** .5 * pauli_map['I'],
                    (g1 / 3.) ** .5 * pauli_map['X'],
                    (g1 / 3.) ** .5 * pauli_map['Y'],
                    (g1 / 3.) ** .5 * pauli_map['Z'],
                ]
            else:
                depolarizing_channel = [
                    (1 - g1) ** .5 * pauli_map['I'],
                    (g1 / 2.) ** .5 * pauli_map['X'],
                    (g1 / 2.) ** .5 * pauli_map['Z'],
                ]
            
            gate_info = {'probabilities': None}
            self._sequence.append((dampling_channel, (qudits[0], ), gate_info)) 
            self._sequence.append((depolarizing_channel, (qudits[1], ), gate_info)) 
        
        if (dense and self.num_qudits >= 4) or (not dense):
            target_qudits = qudits[2:4] if dense else qudits[:2]
            g = self.rng.random()
            if is_complex:
                operands = [
                    (1 - g) **.5 * pauli_map['XY'], 
                    (0.6 * g) ** .5 * pauli_map['YZ'], 
                    (0.4 * g) ** .5 * pauli_map['ZX'],
                ]
            else:
                operands = [
                    (1 - g) **.5 * pauli_map['IX'], 
                    (0.6 * g) ** .5 * pauli_map['IZ'], 
                    (0.4 * g) ** .5 * pauli_map['XZ'],
                ]
            gate_info = {'probabilities': None}
            self._sequence.append((operands, target_qudits, gate_info))
        return   
    
    def _append_single_qudit_layer(self):
        for i in range(self.num_qudits):
            shape = (self.state_dims[i], ) * 2
            t = self.backend.random(shape, self.dtype, self.rng)
            t = t + t.conj().T
            t /= self.backend.norm(t)
            self._sequence.append((t, (i,), None))
    
    def _append_double_qudit_layer(self, offset=0):
        for i in range(offset, self.num_qudits-1, 2):
            j = i + 1 if self.adjacent_double_layer else self.rng.integers(i+1, self.num_qudits)
            shape = (self.state_dims[i], self.state_dims[j])* 2
            t = self.backend.random(shape, self.dtype, self.rng)
            try:
                t = t + t.conj().transpose(2,3,0,1)
            except TypeError:
                t = t + t.conj().permute(2,3,0,1)
            t /= self.backend.norm(t)
            self._sequence.append((t, (i, j), None))
    
    def _append_mpo_layer(self):
        if self.mpo_geometry == "adjacent-ordered":
            start_site = self.rng.integers(0, self.num_qudits-self.mpo_num_sites+1)
            modes = list(range(start_site, start_site+self.mpo_num_sites))
        elif self.mpo_geometry in {"random-ordered", "random"}:
            modes = list(range(self.num_qudits))
            self.rng.shuffle(modes)
            modes = modes[:self.mpo_num_sites]
            if self.mpo_geometry == "random-ordered":
                modes = sorted(modes)
        else:
            raise ValueError(f"mpo geometry {self.mpo_geometry} not supported")
        mpo_tensors = []
        for i, q in enumerate(modes):
            phys_dim = self.state_dims[q]
            if i == 0:
                shape = (phys_dim, self.mpo_bond_dim, phys_dim)
                transpose_order = (2,1,0)
            elif i == self.mpo_num_sites - 1:
                shape = (self.mpo_bond_dim, phys_dim, phys_dim)
                transpose_order = (0,2,1)
            else:
                shape = (self.mpo_bond_dim, phys_dim, ) * 2
                transpose_order = (0,3,2,1)
            t = self.backend.random(shape, self.dtype, self.rng)
            try:
                t = t + t.conj().transpose(*transpose_order)
            except TypeError:
                t = t + t.conj().permute(*transpose_order)
            t /= self.backend.norm(t)
            mpo_tensors.append(t)
        self._sequence.append((mpo_tensors, modes, None))

    def _append_controlled_tensor_mpo_layer(self):
        if self.mpo_geometry == "adjacent-ordered":
            start_site = self.rng.integers(0, self.num_qudits-self.mpo_num_sites+1)
            modes = list(range(start_site, start_site+self.mpo_num_sites))
        elif self.mpo_geometry in {"random-ordered", "random"}:
            modes = list(range(self.num_qudits))
            self.rng.shuffle(modes)
            modes = modes[:self.mpo_num_sites]
            modes = sorted(modes)
        else:
            raise ValueError(f"controlled-tensor-mpo geometry {self.mpo_geometry} not supported")
        
        target_modes = []
        control_modes = []
        if self.ct_target_place == "first":
            target_modes = [modes[0]]
            control_modes = modes[1:]
        elif self.ct_target_place == "middle":
            if(len(modes) < 3):
                raise ValueError(f"To apply target in the middle, #qubits should be > 2")
            idx = self.rng.integers(1, len(modes)-1)
            target_modes = [modes[idx]]
            modes.pop(idx)
            control_modes = modes
        elif self.ct_target_place == "last":
            target_modes = [modes[-1]]
            control_modes = modes[:-1]
        else:
            raise ValueError(f"controlled-tensor target place {self.ct_target_place} not supported")
        
        # control values
        control_values = []
        control_modes = sorted(control_modes)
        for cm in control_modes:
            v = self.rng.integers(0, self.state_dims[cm])
            control_values.append(v)
        # target data
        target_phys_dim = self.state_dims[target_modes[0]] # Currently we support only single-qubit target
        shape = (target_phys_dim, target_phys_dim)
        transpose_order = (1, 0)
        t = self.backend.random(shape, self.dtype, self.rng)
        try:
            t = t + t.conj().transpose(*transpose_order)
        except TypeError:
            t = t + t.conj().permute(*transpose_order)
        t /= self.backend.norm(t)
        gate_info = {'control_modes': control_modes, 'control_values': control_values}
        self._sequence.append((t, target_modes, gate_info))

    def compute_control_tensor(self, control_dim, control_val, rank, direction):
        c1_rank3 = self.backend.asarray([1, 0, 0, 0,  0, 0, 0, 1]).reshape(2, 2, 2)
        c0_rank3_dwon = self.backend.asarray([0, 0, 1, 0,  0, 1, 0, 0]).reshape(2, 2, 2)
        c0_rank3_up = self.backend.asarray([0, 0, 0, 1,  1, 0, 0, 0]).reshape(2, 2, 2)
        c1_rank4_down = self.backend.asarray([1, 0, 0, 0,  0, 1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 1]).reshape(2, 2, 2, 2)
        c1_rank4_up = self.backend.asarray([1, 0, 1, 0,  0, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 1]).reshape(2, 2, 2, 2)
        c0_rank4_down = self.backend.asarray([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0]).reshape(2, 2, 2, 2)
        c0_rank4_up = self.backend.asarray([1, 0, 0, 0,  0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 0, 0]).reshape(2, 2, 2, 2)
        
        if control_dim == 2:
            if rank == "rank-3":
                if control_val == 1:
                    return c1_rank3
                elif control_val == 0 and direction == "down":
                     return c0_rank3_dwon
                elif control_val == 0 and direction == "up":
                        return c0_rank3_up
                else:
                    raise ValueError(f" Failed! Computing control tensor with rank {rank} and direction {direction} is not supported.")
            
            
            elif rank == "rank-4":
                if control_val == 1 and direction == "down":
                    return c1_rank4_down
                elif control_val == 1 and direction == "up":
                    return c1_rank4_up
                elif control_val == 0 and direction == "down":
                    return c0_rank4_down
                elif control_val == 0 and direction == "up":
                    return c0_rank4_up
                else:
                    raise ValueError(f" Failed! Computing control tensor with rank {rank} and direction {direction} is not supported.")
            else:
                raise ValueError(f" Failed! Control tensor can be rank-3 or rank-4, {rank} is not supported.")

        else: # It is a qudit with arbitrary extent/dimension
            if rank == "rank-3":
                if direction == "down":
                    shape = (control_dim, 2, control_dim)
                    ctrl_tensor = self.backend.zeros(shape)
                    for i in range(control_dim):
                        if i == control_val:
                            b = 1
                        else:
                            b = 0
                        ctrl_tensor[i][b][i] = 1
                    return ctrl_tensor
                elif direction == "up":
                    shape = (2, control_dim, control_dim)
                    ctrl_tensor = self.backend.zeros(shape)
                    for i in range(control_dim):
                        if i == control_val:
                            b = 1
                        else:
                            b = 0
                        ctrl_tensor[b][i][i] = 1
                    return ctrl_tensor
                else:
                    raise ValueError(f" Failed! Computing control tensor with direction {direction} is not supported.")

            elif rank == "rank-4":
                if direction == "down":
                    shape = (2, control_dim, 2, control_dim)
                    ctrl_tensor = self.backend.zeros(shape)
                    for b1 in range(2):
                        for i in range(control_dim):
                            if b1 == 1 and i == control_val:
                                b2 = 1
                            else:
                                b2 = 0
                            ctrl_tensor[b1][i][b2][i] = 1
                    return ctrl_tensor
                elif direction == "up":
                    shape = (2, control_dim, 2, control_dim)
                    ctrl_tensor = self.backend.zeros(shape)
                    for b2 in range(2):
                        for i in range(control_dim):
                            if b2 == 1 and i == control_val:
                                b1 = 1
                            else:
                                b1 = 0
                            ctrl_tensor[b1][i][b2][i] = 1
                    return ctrl_tensor
                else:
                    raise ValueError(f" Failed! Computing control tensor with direction {direction} is not supported.")

            else:
                raise ValueError(f" Failed! Control tensor can be rank-3 or rank-4, {rank} is not supported.")

    def compute_ct_mpo_tensors(self, control_modes, control_vals, target_modes, target_data):
        # Note: Input target_data has mode ABC...abc where ABCD are bra modes and abcd are ket modes
        # output MPO tensors (provide to apply_mpo) should have mode panA where p and n refers to modes connecting to previous and next site
        tensors = []
        modes = control_modes + target_modes
        modes = sorted(modes)
        n_qudits = len(modes)
        target_tensor_applied = 0
        for i, q in enumerate(modes):
            if i == 0: # rank-3
                if q == target_modes[0]: # target tensor
                    target = target_data.reshape(1, -1)[0]
                    t_dim = self.state_dims[q]
                    c0 = [0] * (t_dim ** 2)
                    for i in range(t_dim):
                        idx = i + (i * t_dim)
                        c0[idx] = 1
                    c0 = self.backend.asarray(c0)
                    t_temp = self.backend.vstack([c0, target])
                    # modes (n, A, a), i.e, virtual, bra, ket
                    t_temp = t_temp.reshape(2, t_dim, t_dim)
                    # convert to modes (a, n, A), i.e, ket, virtual, bra
                    try:
                        t_temp = t_temp.transpose(2, 0, 1)
                    except TypeError:
                        t_temp = t_temp.permute(2, 0, 1)
                    t = self.backend.asarray(t_temp, dtype=self.dtype)
                    target_tensor_applied = 1
                    tensors.append(t)
                else: # control tensor
                    c = self.compute_control_tensor(self.state_dims[q], control_vals[0], "rank-3", "down")
                    c = self.backend.asarray(c, dtype=self.dtype)
                    tensors.append(c)

            elif i == n_qudits-1: # rank-3
                if q == target_modes[0]: # target tensor
                    target = target_data.reshape(1, -1)[0]
                    t_dim = self.state_dims[q]
                    c0 = [0] * (t_dim ** 2)
                    for i in range(t_dim):
                        idx = i + (i * t_dim)
                        c0[idx] = 1
                    c0 = self.backend.asarray(c0)
                    # modes (p, A, a), i.e, virtual, bra, ket
                    t_temp = self.backend.vstack([c0, target])
                    t_temp = t_temp.reshape(2, t_dim, t_dim)
                    # convert to modes (p, a, A), i.e, virtual, ket, bra
                    try:
                        t_temp = t_temp.transpose(0, 2, 1)
                    except TypeError:
                        t_temp = t_temp.permute(0, 2, 1)
                    t = self.backend.asarray(t_temp, dtype=self.dtype)
                    target_tensor_applied = 1
                    tensors.append(t)
                else: # control tensor
                    c = self.compute_control_tensor(self.state_dims[q], control_vals[i-1], "rank-3", "up")
                    c = self.backend.asarray(c, dtype=self.dtype)
                    tensors.append(c)

            else: # rank-4
                if q == target_modes[0]: # target tensor
                    target = target_data.reshape(1, -1)[0]
                    t_dim = self.state_dims[q]
                    c0 = [0] * (t_dim ** 2)
                    for i in range(t_dim):
                        idx = i + (i * t_dim)
                        c0[idx] = 1
                    c0 = self.backend.asarray(c0)
                    t_temp = self.backend.vstack([c0, c0, c0, target])
                    # modes (p, n, A, a)
                    t_temp = t_temp.reshape(2, 2, t_dim, t_dim)
                    # convert to (p, a, n, A)
                    try:
                        t_temp = t_temp.transpose(0, 3, 1, 2)
                    except TypeError:
                        t_temp = t_temp.permute(0, 3, 1, 2)
                    t = self.backend.asarray(t_temp, dtype=self.dtype)
                    target_tensor_applied = 1
                    tensors.append(t)
                else: # control tensor
                    if target_tensor_applied == 1:
                        c = self.compute_control_tensor(self.state_dims[q], control_vals[i-1], "rank-4", "up")
                        c = self.backend.asarray(c, dtype=self.dtype)
                        tensors.append(c)
                    else:
                        c = self.compute_control_tensor(self.state_dims[q], control_vals[i], "rank-4", "down")
                        c = self.backend.asarray(c, dtype=self.dtype)
                        tensors.append(c)

        return tensors


    def get_sv_contraction_expression(self):
        operands = []
        # initial qudit mode
        qudit_modes = list(range(self.num_qudits))
        mode_frontier = self.num_qudits

        initial_state = self.get_initial_state()
        if self.initial_mps_dim is None:
            for i, t in enumerate(initial_state):
                operands += [t, [qudit_modes[i]]]            
        else:
            prev_mode = None
            for i, t in enumerate(initial_state):
                if i == 0:
                    modes = (qudit_modes[i], mode_frontier)
                elif i == self.num_qudits - 1:
                    modes = (prev_mode, qudit_modes[i])
                else:
                    modes = (prev_mode, qudit_modes[i], mode_frontier)
                prev_mode = mode_frontier
                mode_frontier += 1
                operands += [t, modes]
        
        for op, qudits, gate_info in self.sequence:
            if gate_info is not None:
                if 'control_values' in gate_info and 'control_modes' in gate_info:
                    # convert control tensor into MPO
                    ctrl_modes, ctrl_vals = gate_info['control_modes'], gate_info['control_values']
                    op = self.compute_ct_mpo_tensors(ctrl_modes, ctrl_vals, qudits, op)
                    qudits = qudits + ctrl_modes
                    qudits = sorted(qudits)
                    gate_info = None
                else:
                    raise RuntimeError("Not the expected code path")
            n_qudits = len(qudits)
            if isinstance(op, (list, tuple)):
                # for MPO
                prev_mode = None
                for i, q in enumerate(qudits):
                    mode_in = qudit_modes[q]
                    qudit_modes[q] = mode_frontier 
                    mode_frontier += 1
                    if i == 0:
                        modes = [mode_in, mode_frontier, qudit_modes[q]]
                        prev_mode = mode_frontier
                        mode_frontier += 1
                    elif i == n_qudits - 1:
                        modes = [prev_mode, mode_in, qudit_modes[q]]
                    else:
                        modes = [prev_mode, mode_in, mode_frontier, qudit_modes[q]]
                        prev_mode = mode_frontier
                        mode_frontier +=1
                    operands += [op[i], modes]
            else:
                modes_in = [qudit_modes[q] for q in qudits]
                modes_out = []
                for q in qudits:
                    qudit_modes[q] = mode_frontier
                    modes_out.append(mode_frontier)
                    mode_frontier +=1
                operands += [op, modes_out + modes_in]
        
        operands.append(qudit_modes)
        return operands
    
    def compute_state_vector(self):
        operands = self.get_sv_contraction_expression()
        return contract(*operands)

    def to_network_state(self, *, config=None, options=None):
        network_state = NetworkState(self.state_dims, dtype=self.dtype, config=config, options=options)
        if self.initial_mps_dim is not None:
            network_state.set_initial_mps(self.get_initial_state())
        apply_factory_sequence(network_state, self.sequence)
        return network_state
