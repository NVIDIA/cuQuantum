#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: IBM 127-qubit Kicked Ising Simulation using cuPauliProp

This example demonstrates cuPauliProp simulation of the IBM 127-qubit kicked Ising 
experiment, as presented in Nature volume 618, pages 500–505 (2023). Specifically, 
we simulate the Z_62 20-Trotter-step experiment of Fig 4. b), the only circuit with 
a full 127-qubit lightcone, at an X-rotation angle of PI/4, finding agreement with 
the error-mitigated experimental results.

For simplicity, we do not simulate the error channels nor twirling process, though 
inhomogeneous one and two qubit Pauli channels are supported by cuPauliProp and can 
accelerate simulation.
"""

import numpy as np
import cupy as cp
from cuquantum.bindings import cupauliprop
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
import time


# Sphinx: #2
# ========================================================================
# Memory usage
# ========================================================================

# Each Pauli expansion has two pre-allocated GPU buffers, storing packed
# integers (which encode Pauli strings) and corresponding coefficients.
# As much memory can be dedicated as your hardware allows, while the min-
# imum required is specific and very sensitive to the simulated circuit,
# studied observable, and the chosen truncation hyperparameters.
# Some operations also require additional workspace memory which is also
# ideally pre-allocated, and can be established using the API 'Prepare'
# functions (e.g. cupaulipropPauliExpansionViewPrepareTraceWithZeroState).
# In this demo, we dedicate either (a percentage of) the entirety of GPU
# memory uniformly between the required memory buffers, or instead use a
# fixed hardcoded amount which has been prior tested to be consistent with
# our other simulation parameters (like truncations); these choices are
# toggled via USE_MAX_VRAM below.

# True to use MAX_VRAM_PERCENT of VRAM, False to use fixed memories below
USE_MAX_VRAM = False
MAX_VRAM_PERCENT = 90  # 0-100%

FIXED_EXPANSION_PAULI_MEM = 16 * (1 << 20)  # bytes = 16 MiB
FIXED_EXPANSION_COEF_MEM = 4 * (1 << 20)    # bytes = 4 MiB
FIXED_WORKSPACE_MEM = 20 * (1 << 20)        # bytes = 20 MiB


# Sphinx: #3
# ========================================================================
# Circuit preparation (Trotterised Ising on IBM heavy hex topology)
# ========================================================================

# This demo simulates the circuits experimentally executed by IBM in article
# 'Nature volume 618, pages 500–505 (2023)'. This is a circuit Trotterising
# the evolution operator of a 2D transverse-field Ising model, but where the
# prescribed ZZ rotations have a fixed angle of -pi/2, and where the X angles
# are arbitrarily set/swept; later, we will fix the X angle to be pi/4. The
# Hamiltonian ZZ interactions are confined to a heavy-hex topology, matching
# the connectivity of the IBM Eagle processor 'ibm_kyiv', as we fix below.

NUM_CIRCUIT_QUBITS = 127
NUM_ROTATIONS_PER_LAYER = 48
NUM_PAULIS_PER_X_ROTATION = 1
NUM_PAULIS_PER_Z_ROTATION = 2

PI = np.pi
ZZ_ROTATION_ANGLE = -PI / 2.0

# Indices of ZZ-interacting qubits which undergo the first (red) Trotter round
ZZ_QUBITS_RED = np.array([
    [  2,   1],  [ 33,  39], [ 59,  60], [ 66,  67], [ 72,  81], [118, 119],
    [ 21,  20],  [ 26,  25], [ 13,  12], [ 31,  32], [ 70,  74], [122, 123],
    [ 96,  97],  [ 57,  56], [ 63,  64], [107, 108], [103, 104], [ 46,  45],
    [ 28,  35],  [  7,   6], [ 79,  78], [  5,   4], [109, 114], [ 62,  61],
    [ 58,  71],  [ 37,  52], [ 76,  77], [  0,  14], [ 36,  51], [106, 105],
    [ 73,  85],  [ 88,  87], [ 68,  55], [116, 115], [ 94,  95], [100, 110],
    [ 17,  30],  [ 92, 102], [ 50,  49], [ 83,  84], [ 48,  47], [ 98,  99],
    [  8,   9],  [121, 120], [ 23,  24], [ 44,  43], [ 22,  15], [ 53,  41]
], dtype=np.int32)

# Indices of ZZ-interacting qubits which undergo the second (blue) Trotter round
ZZ_QUBITS_BLUE = np.array([
    [ 53,  60], [123, 124], [ 21,  22], [ 11,  12], [ 67,  68], [  2,   3],
    [ 66,  65], [122, 121], [110, 118], [  6,   5], [ 94,  90], [ 28,  29],
    [ 14,  18], [ 63,  62], [111, 104], [100,  99], [ 45,  44], [  4,  15],
    [ 20,  19], [ 57,  58], [ 77,  71], [ 76,  75], [ 26,  27], [ 16,   8],
    [ 35,  47], [ 31,  30], [ 48,  49], [ 69,  70], [125, 126], [ 89,  74],
    [ 80,  79], [116, 117], [114, 113], [ 10,   9], [106,  93], [101, 102],
    [ 92,  83], [ 98,  91], [ 82,  81], [ 54,  64], [ 96, 109], [ 85,  84],
    [ 87,  86], [108, 112], [ 34,  24], [ 42,  43], [ 40,  41], [ 39,  38]
], dtype=np.int32)

# Indices of ZZ-interacting qubits which undergo the third (green) Trotter round
ZZ_QUBITS_GREEN = np.array([
    [ 10,  11], [ 54,  45], [111, 122], [ 64,  65], [ 60,  61], [103, 102],
    [ 72,  62], [  4,   3], [ 33,  20], [ 58,  59], [ 26,  16], [ 28,  27],
    [  8,   7], [104, 105], [ 73,  66], [ 87,  93], [ 85,  86], [ 55,  49],
    [ 68,  69], [ 89,  88], [ 80,  81], [117, 118], [101, 100], [114, 115],
    [ 96,  95], [ 29,  30], [106, 107], [ 83,  82], [ 91,  79], [  0,   1],
    [ 56,  52], [ 90,  75], [126, 112], [ 36,  32], [ 46,  47], [ 77,  78],
    [ 97,  98], [ 17,  12], [119, 120], [ 22,  23], [ 24,  25], [ 43,  34],
    [ 42,  41], [ 40,  39], [ 37,  38], [125, 124], [ 50,  51], [ 18,  19]
], dtype=np.int32)


# Sphinx: #4
# ========================================================================
# Circuit construction
# ========================================================================

def get_x_rotation_layer(handle, x_rotation_angle):
    """
    Create a layer of single-qubit X rotations on every qubit.
    """
    layer = []
    paulis = np.array([cupauliprop.PauliKind.PAULI_X], dtype=np.int32)
    
    for i in range(NUM_CIRCUIT_QUBITS):
        qubit_index = np.array([i], dtype=np.int32)
        oper = cupauliprop.create_pauli_rotation_gate_operator(
            handle, x_rotation_angle, NUM_PAULIS_PER_X_ROTATION, 
            qubit_index, paulis)
        layer.append(oper)
    
    return layer


def get_zz_rotation_layer(handle, topology):
    """
    Create a layer of two-qubit ZZ rotations on the specified topology.
    """
    layer = []
    paulis = np.array([cupauliprop.PauliKind.PAULI_Z, 
                       cupauliprop.PauliKind.PAULI_Z], dtype=np.int32)
    
    for i in range(NUM_ROTATIONS_PER_LAYER):
        qubit_indices = topology[i]
        oper = cupauliprop.create_pauli_rotation_gate_operator(
            handle, ZZ_ROTATION_ANGLE, NUM_PAULIS_PER_Z_ROTATION,
            qubit_indices, paulis)
        layer.append(oper)
    
    return layer


def get_ibm_heavy_hex_ising_circuit(handle, x_rotation_angle, num_trotter_steps):
    """
    Construct the full IBM heavy-hex Ising circuit.
    """
    circuit = []
    
    for n in range(num_trotter_steps):
        layer_x = get_x_rotation_layer(handle, x_rotation_angle)
        layer_red_zz = get_zz_rotation_layer(handle, ZZ_QUBITS_RED)
        layer_blue_zz = get_zz_rotation_layer(handle, ZZ_QUBITS_BLUE)
        layer_green_zz = get_zz_rotation_layer(handle, ZZ_QUBITS_GREEN)
        
        circuit.extend(layer_x)
        circuit.extend(layer_red_zz)
        circuit.extend(layer_blue_zz)
        circuit.extend(layer_green_zz)
    
    return circuit


# Sphinx: #5
# ========================================================================
# Observable preparation
# ========================================================================

def get_pauli_string_as_packed_integers(paulis, qubits):
    """
    Convert a Pauli string to packed integer representation for cuPauliProp.
    
    Args:
        paulis: List of Pauli operators (I, X, Y, Z)
        qubits: List of qubit indices
    
    Returns:
        numpy array of packed integers encoding the Pauli string
    """
    assert len(paulis) == len(qubits)
    assert max(qubits) < NUM_CIRCUIT_QUBITS
    
    num_packed_ints = cupauliprop.get_num_packed_integers(NUM_CIRCUIT_QUBITS)
    
    # A single Pauli string is composed of separate X and Z masks
    out = np.zeros(num_packed_ints * 2, dtype=np.uint64)
    x_ptr = out[:num_packed_ints]
    z_ptr = out[num_packed_ints:]
    
    # Process one input (pauli, qubit) pair at a time
    num_bits_per_packed_int = 64  # sizeof(uint64_t) * 8
    
    for i in range(len(qubits)):
        int_ind = qubits[i] // num_bits_per_packed_int
        bit_ind = qubits[i] % num_bits_per_packed_int
        
        # Overwrite a bit of either the X or Z masks (or both when pauli==Y)
        if paulis[i] in [cupauliprop.PauliKind.PAULI_X, cupauliprop.PauliKind.PAULI_Y]:
            x_ptr[int_ind] = x_ptr[int_ind] | (1 << bit_ind)
        if paulis[i] in [cupauliprop.PauliKind.PAULI_Z, cupauliprop.PauliKind.PAULI_Y]:
            z_ptr[int_ind] = z_ptr[int_ind] | (1 << bit_ind)
    
    return out


# Sphinx: #6
# ========================================================================
# Main
# ========================================================================

def main():
    print("cuPauliProp IBM Heavy-hex Ising Example")
    print("=" * 39)
    print()
    
    
    # Sphinx: #7
    # ========================================================================
    # Library setup
    # ========================================================================
    
    device_id = 0
    cp.cuda.Device(device_id).use()
    
    # Work on the default stream
    handle = cupauliprop.create()
    
    
    # Sphinx: #8
    # ========================================================================
    # Decide memory usage
    # ========================================================================
    
    # As outlined in the 'Memory usage' section above, we here either uniformly
    # allocate all (or a high percentage of) available memory between the needed
    # memory buffers, or use the pre-decided fixed values. This demo will create
    # a total of two Pauli expansions (each of which accepts two separate buffers
    # to store Pauli strings and their corresponding coefficients; these have
    # different sizes) and one workspace, hence we arrange for an allocation of
    # five buffers in total.
    
    if USE_MAX_VRAM:
        # Query currently available device memory
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        total_usable_mem = int(free_mem * MAX_VRAM_PERCENT / 100)
        
        # Divide it between the three instances (two expansions, one workspace)
        instance_mem = total_usable_mem // 3
        
        # Determine the ideal ratio between an expansion's Pauli and coef buffers
        num_packed_ints = cupauliprop.get_num_packed_integers(NUM_CIRCUIT_QUBITS)
        pauli_mem_per_term = 2 * num_packed_ints * 8  # 8 bytes per uint64
        coef_mem_per_term = 8  # sizeof(double)
        total_mem_per_term = pauli_mem_per_term + coef_mem_per_term
        
        expansion_pauli_mem = (instance_mem * pauli_mem_per_term) // total_mem_per_term
        expansion_coef_mem = (instance_mem * coef_mem_per_term) // total_mem_per_term
        workspace_mem = instance_mem
        
        total_used_mem = 2 * expansion_pauli_mem + 2 * expansion_coef_mem + workspace_mem
        print(f"Dedicated memory: {MAX_VRAM_PERCENT}% of {free_mem} B free = {total_used_mem} B")
    else:
        # Use pre-decided buffer sizes
        expansion_pauli_mem = FIXED_EXPANSION_PAULI_MEM
        expansion_coef_mem = FIXED_EXPANSION_COEF_MEM
        workspace_mem = FIXED_WORKSPACE_MEM
        
        total_used_mem = 2 * expansion_pauli_mem + 2 * expansion_coef_mem + workspace_mem
        print(f"Dedicated memory: {total_used_mem} B = 60 MiB")
    
    print(f"  expansion Pauli buffer: {expansion_pauli_mem} B")
    print(f"  expansion coef buffer:  {expansion_coef_mem} B")
    print(f"  workspace buffer:       {workspace_mem} B")
    print()
    
    
    # Sphinx: #9
    # ========================================================================
    # Pauli expansion preparation
    # ========================================================================
    
    # Create GPU buffers for two Pauli expansions, which will serve as 'input' and
    # 'output' to the out-of-place cuPauliProp API. Note that the capacities of
    # these buffers constrain the maximum number of Pauli strings maintained
    # during simulation, and ergo inform the accuracy of the simulation. The
    # sufficient buffer sizes are specific to the simulated system, and we here
    # choose a surprisingly small capacity as admitted by the studied circuit.
    
    d_in_expansion_pauli_buffer = cp.cuda.alloc(expansion_pauli_mem)
    d_in_expansion_coef_buffer = cp.cuda.alloc(expansion_coef_mem)
    d_out_expansion_pauli_buffer = cp.cuda.alloc(expansion_pauli_mem)
    d_out_expansion_coef_buffer = cp.cuda.alloc(expansion_coef_mem)
    
    # Prepare the X and Z masks which encode the experimental observable Z_62,
    # which has a coefficient of unity, as seen in Figure 4. b) of the IBM work.
    print("Observable: Z_62")
    num_observable_terms = 1
    observable_coef = 1.0
    observable_paulis = [cupauliprop.PauliKind.PAULI_Z]
    observable_qubits = [62]
    
    # Create observable in host memory using numpy arrays
    num_packed_ints = cupauliprop.get_num_packed_integers(NUM_CIRCUIT_QUBITS)
    h_observable_pauli_buffer = np.zeros((1, num_packed_ints * 2), dtype=np.uint64, order="C")
    h_observable_coef_buffer = np.array([[observable_coef]], dtype=np.float64, order="C")
    
    # Encode the observable Pauli string into the host buffer
    observable_packed_ints = get_pauli_string_as_packed_integers(observable_paulis, observable_qubits)
    h_observable_pauli_buffer[0, :] = observable_packed_ints
    
    # Create a host Pauli expansion containing the observable
    h_expansion = cupauliprop.create_pauli_expansion(
        handle, NUM_CIRCUIT_QUBITS,
        h_observable_pauli_buffer.ctypes.data, h_observable_pauli_buffer.nbytes,
        h_observable_coef_buffer.ctypes.data, h_observable_coef_buffer.nbytes,
        NAME_TO_DATA_TYPE['float64'], num_observable_terms, 1, 1)
    
    h_observable_view = cupauliprop.pauli_expansion_get_contiguous_range(
        handle, h_expansion, 0, num_observable_terms)
    
    # Create two Pauli expansions on device. Because we begin from a real observable 
    # coefficient, and our circuit is completely positive and trace preserving, it is 
    # sufficient to use strictly real coefficients in our expansions, informing dataType 
    # below. We indicate that the single prepared term in the input expansion is technically 
    # unique, and the terms sorted, which permits cuPauliProp to use automatic optimisations 
    # during simulation.
    
    in_expansion = cupauliprop.create_pauli_expansion(
        handle, NUM_CIRCUIT_QUBITS,
        d_in_expansion_pauli_buffer.ptr, expansion_pauli_mem,
        d_in_expansion_coef_buffer.ptr, expansion_coef_mem,
        NAME_TO_DATA_TYPE['float64'], 0, 0, 0)
    
    out_expansion = cupauliprop.create_pauli_expansion(
        handle, NUM_CIRCUIT_QUBITS,
        d_out_expansion_pauli_buffer.ptr, expansion_pauli_mem,
        d_out_expansion_coef_buffer.ptr, expansion_coef_mem,
        NAME_TO_DATA_TYPE['float64'], 0, 0, 0)
    
    # Copy observable from host to device using populateFromView
    cupauliprop.pauli_expansion_populate_from_view(handle, h_observable_view, in_expansion)
    
    # Clean up host resources
    cupauliprop.destroy_pauli_expansion_view(h_observable_view)
    cupauliprop.destroy_pauli_expansion(h_expansion)
    print()
    
    
    # Sphinx: #10
    # ========================================================================
    # Workspace preparation
    # ========================================================================
    
    # Some API functions require additional workspace memory which we bind to a
    # workspace descriptor. Ordinarily we use the 'Prepare' functions to precisely
    # bound upfront the needed workspace memory, but in this simple demo, we
    # instead use a workspace memory which we prior know to be sufficient.
    
    workspace = cupauliprop.create_workspace_descriptor(handle)
    
    d_workspace_buffer = cp.cuda.alloc(workspace_mem)
    cupauliprop.workspace_set_memory(
        handle, workspace,
        cupauliprop.Memspace.DEVICE, 
        cupauliprop.WorkspaceKind.WORKSPACE_SCRATCH,
        d_workspace_buffer.ptr, workspace_mem)
    
    # Note that the 'prepare' functions which check the required workspace memory
    # will detach this buffer, requiring we re-call SetMemory above. We can avoid
    # this repeated re-attachment by use of a second, bufferless workspace descri-
    # ptor which we pass to the 'prepare' functions in lieu of this one.
    
    
    # Sphinx: #11
    # ========================================================================
    # Truncation parameter preparation
    # ========================================================================
    
    # The Pauli propagation simulation technique has memory and runtime costs which
    # (for generic circuits) grow exponentially with the circuit length, which we
    # curtail through "truncation"; dynamic discarding of Pauli strings in our
    # expansion which are predicted to contribute negligibly to the final output
    # expectation value. This file demonstrates simultaneous usage of two truncation
    # techniques; discarding of Pauli strings with an absolute coefficient less than
    # 0.0001, or a "Pauli weight" (the number of non-identity operators in the string)
    # exceeding eight.
    
    coef_trunc_params = cupauliprop.CoefficientTruncationParams()
    coef_trunc_params.cutoff = 1e-4
    
    weight_trunc_params = cupauliprop.PauliWeightTruncationParams()
    weight_trunc_params.cutoff = 8
    
    # Create truncation strategy structs
    coef_strategy = cupauliprop.TruncationStrategy()
    coef_strategy.strategy = cupauliprop.TruncationStrategyKind.TRUNCATION_STRATEGY_COEFFICIENT_BASED
    coef_strategy.param_struct = coef_trunc_params.ptr
    
    weight_strategy = cupauliprop.TruncationStrategy()
    weight_strategy.strategy = cupauliprop.TruncationStrategyKind.TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED
    weight_strategy.param_struct = weight_trunc_params.ptr
    
    num_trunc_strats = 2
    truncation_strategies = [coef_strategy, weight_strategy]
    
    print(f"Coefficient truncation threshold: {coef_trunc_params.cutoff}")
    print(f"Pauli weight truncation threshold: {weight_trunc_params.cutoff}")
    print()
    
    # It is not necessary to perform truncation after every gate, since the
    # Pauli expansion size may not have grown substantially, and attempting
    # to truncate may incur superfluous memory enumeration costs. In this
    # demo, we choose to truncate only after every tenth applied gate. Note
    # deferring truncation requires additional expansion memory; choosing to
    # truncate after every gate shrinks this demo's costs to 20 MiB total.
    
    num_gates_between_truncations = 10
    
    
    # Sphinx: #12
    # ========================================================================
    # Back-propagation of the observable through the circuit
    # ========================================================================
    
    # We now simulate the observable operator being back-propagated through the
    # adjoint circuit, mapping the input expansion (initialised to Z_62) to a
    # final output expansion containing many weighted Pauli strings. We use the
    # heavy-hex fixed-angle Ising circuit with 20 total repetitions, fixing the
    # angle of the X rotation gates to PI/4. Our simulation therefore corresponds
    # to the middle datum of Fig. 4 b) of the IBM manuscript, for which MPS and
    # isoTNS siulation techniques showed the greatest divergence from experiment.
    
    x_rotation_angle = PI / 4.0
    num_trotter_steps = 20
    circuit = get_ibm_heavy_hex_ising_circuit(handle, x_rotation_angle, num_trotter_steps)
    
    print(f"Circuit: 127 qubit IBM heavy-hex Ising circuit, with...")
    print(f"  Trotter steps: {num_trotter_steps}")
    print(f"  Total gates:   {len(circuit)}")
    print(f"  Rx angle:      {x_rotation_angle} (i.e. PI/4)")
    
    # Constrain that every intermediate output expansion contains unique Pauli
    # strings (forbidding duplicates), but permit the retained strings to be
    # unsorted. This combination gives cuPauliProp the best chance of automatically
    # selecting optimal internal functions and postconditions for the simulation.
    adjoint = True
    make_sorted = False
    keep_duplicates = False
    
    # Begin timing before any gates are applied
    start_time = time.time()
    max_num_terms = 0
    
    # Iterate the circuit in reverse to effect the adjoint of the total circuit
    for gate_ind in range(len(circuit) - 1, -1, -1):
        gate = circuit[gate_ind]
        
        # Create a view of the current input expansion, selecting all currently
        # contained terms. For very large systems, we may have alternatively
        # chosen a smaller view of the partial state to work around memory limits.
        num_expansion_terms = cupauliprop.pauli_expansion_get_num_terms(handle, in_expansion)
        in_view = cupauliprop.pauli_expansion_get_contiguous_range(
            handle, in_expansion, 0, num_expansion_terms)
        
        # Track the intermediate expansion size, for our curiousity
        if num_expansion_terms > max_num_terms:
            max_num_terms = num_expansion_terms
        
        # Choose whether or not to perform truncations after this gate
        num_passed_trunc_strats = num_trunc_strats if (gate_ind % num_gates_between_truncations == 0) else 0
        
        # Check the expansion and workspace memories needed to apply the current gate
        req_expansion_pauli_mem, req_expansion_coef_mem = \
            cupauliprop.pauli_expansion_view_prepare_operator_application(
                handle, in_view, gate, make_sorted, keep_duplicates,
                num_passed_trunc_strats, truncation_strategies if num_passed_trunc_strats > 0 else None,
                workspace_mem, workspace)
        
        req_workspace_mem = cupauliprop.workspace_get_memory_size(
            handle, workspace,
            cupauliprop.Memspace.DEVICE,
            cupauliprop.WorkspaceKind.WORKSPACE_SCRATCH)
        
        # Verify that our existing buffers and workspace have sufficient memory
        assert req_expansion_pauli_mem <= expansion_pauli_mem
        assert req_expansion_coef_mem <= expansion_coef_mem
        assert req_workspace_mem <= workspace_mem
        
        # Beware that cupaulipropPauliExpansionViewPrepareOperatorApplication() above
        # detaches the memory buffer from the workspace, which we here re-attach.
        cupauliprop.workspace_set_memory(
            handle, workspace,
            cupauliprop.Memspace.DEVICE,
            cupauliprop.WorkspaceKind.WORKSPACE_SCRATCH,
            d_workspace_buffer.ptr, workspace_mem)
        
        # Apply the gate upon the prepared view of the input expansion, evolving the
        # Pauli strings pointed to within, truncating the result. The input expansion
        # is unchanged while the output expansion is entirely overwritten.
        cupauliprop.pauli_expansion_view_compute_operator_application(
            handle, in_view, out_expansion, gate,
            adjoint, make_sorted, keep_duplicates,
            num_passed_trunc_strats, truncation_strategies if num_passed_trunc_strats > 0 else None,
            workspace)
        
        # Free the temporary view since it points to the old input expansion, whereas
        # we will subsequently treat the modified output expansion as the next input
        cupauliprop.destroy_pauli_expansion_view(in_view)
        
        # Treat outExpansion as the input in the next gate application
        in_expansion, out_expansion = out_expansion, in_expansion
    
    # Restore outExpansion to being the final output for clarity
    in_expansion, out_expansion = out_expansion, in_expansion
    
    
    # Sphinx: #13
    # ========================================================================
    # Evaluation of the expectation value
    # ========================================================================
    
    # The output expansion is now a proxy for the observable back-propagated
    # through to the front of the circuit (though having discarded components
    # which negligibly influence the subsequent overlap). The expectation value
    # of the IBM experiment is the overlap of the output expansion with the
    # zero state, i.e. Tr(outExpansion * |0><0|), as we now compute.
    
    # Obtain a view of the full output expansion (we'll free it in 'Clean up')
    num_out_terms = cupauliprop.pauli_expansion_get_num_terms(handle, in_expansion)
    out_view = cupauliprop.pauli_expansion_get_contiguous_range(
        handle, in_expansion, 0, num_out_terms)
    
    # Check that the existing workspace memory is sufficient to compute the trace
    cupauliprop.pauli_expansion_view_prepare_trace_with_zero_state(
        handle, out_view, workspace_mem, workspace)
    
    req_workspace_mem = cupauliprop.workspace_get_memory_size(
        handle, workspace,
        cupauliprop.Memspace.DEVICE,
        cupauliprop.WorkspaceKind.WORKSPACE_SCRATCH)
    assert req_workspace_mem <= workspace_mem
    
    # Beware that we must now reattach the buffer to the workspace
    cupauliprop.workspace_set_memory(
        handle, workspace,
        cupauliprop.Memspace.DEVICE,
        cupauliprop.WorkspaceKind.WORKSPACE_SCRATCH,
        d_workspace_buffer.ptr, workspace_mem)
    
    # Compute the trace; the main and final output of this simulation!
    expec = np.zeros(1, dtype=np.float64)
    cupauliprop.pauli_expansion_view_compute_trace_with_zero_state(
        handle, out_view, expec.ctypes.data, workspace)
    
    # End timing after trace is evaluated
    end_time = time.time()
    duration = end_time - start_time
    
    print()
    print(f"Expectation value:       {expec[0]}")
    print(f"Final number of terms:   {num_out_terms}")
    print(f"Maximum number of terms: {max_num_terms}")
    print(f"Runtime:                 {duration} seconds")
    print()
    
    
    # Sphinx: #14
    # ========================================================================
    # Clean up
    # ========================================================================
    
    cupauliprop.destroy_pauli_expansion_view(out_view)
    
    for gate in circuit:
        cupauliprop.destroy_operator(gate)
    
    cupauliprop.destroy_workspace_descriptor(workspace)
    
    cupauliprop.destroy_pauli_expansion(in_expansion)
    cupauliprop.destroy_pauli_expansion(out_expansion)
    
    cupauliprop.destroy(handle)


if __name__ == "__main__":
    main()

