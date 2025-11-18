/*
 * Copyright (c) 2025-2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file ibm_kicked_ising.cpp
 * @brief An example demonstrating cuPauliProp simulation of the IBM 127-qubit kicked Ising experiment,
 *        as presented in Nature volume 618, pages 500–505 (2023). Specifically, we simulate the Z_62
 *        20-Trotter-step experiment of Fig 4. b), the only circuit with a full 127-qubit lightcone, 
 *        at an X-rotation angle of PI/4, finding agreement with the error-mitigated experimental results.
 *        For simplicity, we do not simulate the error channels nor twirling process, though inhomogeneous
 *        one and two qubit Pauli channels are supported by cuPauliProp and can accelerate simulation.
 */

#include <cupauliprop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <chrono>


// Sphinx: #1
// ========================================================================
// CUDA and cuPauliProp error handling
// ========================================================================

#define HANDLE_CUPP_ERROR(x)                               \
{                                                          \
  const auto err = x;                                      \
  if (err != CUPAULIPROP_STATUS_SUCCESS)                   \
  {                                                        \
    printf("cuPauliProp error in line %d\n", __LINE__);    \
    fflush(stdout);                                        \
    std::abort();                                          \
  }                                                        \
};


#define HANDLE_CUDA_ERROR(x)                                \
{                                                           \
  const auto err = x;                                       \
  if (err != cudaSuccess)                                   \
  {                                                         \
    const char * error = cudaGetErrorString(err);           \
    printf("CUDA Error: %s in line %d\n", error, __LINE__); \
    fflush(stdout);                                         \
    std::abort();                                           \
  }                                                         \
};


// Sphinx: #2
// ========================================================================
// Memory usage
// ========================================================================

// Each Pauli expansion has two pre-allocated GPU buffers, storing packed
// integers (which encode Pauli strings) and corresponding coefficients.
// As much memory can be dedicated as your hardware allows, while the min-
// imum required is specific and very sensitive to the simulated circuit,
// studied observable, and the chosen truncation hyperparameters.
// Some operations also require additional workspace memory which is also
// ideally pre-allocated, and can be established using the API 'Prepare'
// functions (e.g. cupaulipropPauliExpansionViewPrepareTraceWithZeroState).
// In this demo, we dedicate either (a percentage of) the entirety of GPU
// memory uniformly between the required memory buffers, or instead use a
// fixed hardcoded amount which has been prior tested to be consistent with
// our other simulation parameters (like truncations); these choices are
// toggled via USE_MAX_VRAM below.

// =true to use MAX_VRAM_PERCENT of VRAM, and =false to use fixed memories below
bool USE_MAX_VRAM = false;
double MAX_VRAM_PERCENT = 90; // 0-100%

size_t FIXED_EXPANSION_PAULI_MEM = 16 * (1LLU << 20); // bytes = 16 MiB
size_t FIXED_EXPANSION_COEF_MEM  =  4 * (1LLU << 20); // bytes = 4  MiB
size_t FIXED_WORKSPACE_MEM       = 20 * (1LLU << 20); // bytes = 20 MiB


// Sphinx: #3
// ========================================================================
// Circuit preparation (Trotterised Ising on IBM heavy hex topology)
// ========================================================================

// This demo simulates the circuits experimentally executed by IBM in article
// 'Nature volume 618, pages 500–505 (2023)'. This is a circuit Trotterising
// the evolution operator of a 2D transverse-field Ising model, but where the
// prescribed ZZ rotations have a fixed angle of -pi/2, and where the X angles
// are arbitrarily set/swept; later, we will fix the X angle to be pi/4. The
// Hamiltonian ZZ interactions are confined to a heavy-hex topology, matching
// the connectivity of the IBM Eagle processor 'ibm_kyiv', as we fix below.

const int NUM_CIRCUIT_QUBITS = 127;
const int NUM_ROTATIONS_PER_LAYER = 48;
const int NUM_PAULIS_PER_X_ROTATION = 1;
const int NUM_PAULIS_PER_Z_ROTATION = 2;

const double PI = 3.14159265358979323846;
const double ZZ_ROTATION_ANGLE = - PI / 2.0;

// Indices of ZZ-interacting qubits which undergo the first (red) Trotter round
const int32_t ZZ_QUBITS_RED[NUM_ROTATIONS_PER_LAYER][NUM_PAULIS_PER_Z_ROTATION] = {
  {  2,   1},  { 33,  39}, { 59,  60}, { 66,  67}, { 72,  81}, {118, 119},
  { 21,  20},  { 26,  25}, { 13,  12}, { 31,  32}, { 70,  74}, {122, 123},
  { 96,  97},  { 57,  56}, { 63,  64}, {107, 108}, {103, 104}, { 46,  45},
  { 28,  35},  {  7,   6}, { 79,  78}, {  5,   4}, {109, 114}, { 62,  61},
  { 58,  71},  { 37,  52}, { 76,  77}, {  0,  14}, { 36,  51}, {106, 105},
  { 73,  85},  { 88,  87}, { 68,  55}, {116, 115}, { 94,  95}, {100, 110},
  { 17,  30},  { 92, 102}, { 50,  49}, { 83,  84}, { 48,  47}, { 98,  99},
  {  8,   9},  {121, 120}, { 23,  24}, { 44,  43}, { 22,  15}, { 53,  41}
};

// Indices of ZZ-interacting qubits which undergo the second (blue) Trotter round
const int32_t ZZ_QUBITS_BLUE[NUM_ROTATIONS_PER_LAYER][NUM_PAULIS_PER_Z_ROTATION] = {
  { 53,  60}, {123, 124}, { 21,  22}, { 11,  12}, { 67,  68}, {  2,   3},
  { 66,  65}, {122, 121}, {110, 118}, {  6,   5}, { 94,  90}, { 28,  29},
  { 14,  18}, { 63,  62}, {111, 104}, {100,  99}, { 45,  44}, {  4,  15},
  { 20,  19}, { 57,  58}, { 77,  71}, { 76,  75}, { 26,  27}, { 16,   8},
  { 35,  47}, { 31,  30}, { 48,  49}, { 69,  70}, {125, 126}, { 89,  74},
  { 80,  79}, {116, 117}, {114, 113}, { 10,   9}, {106,  93}, {101, 102},
  { 92,  83}, { 98,  91}, { 82,  81}, { 54,  64}, { 96, 109}, { 85,  84},
  { 87,  86}, {108, 112}, { 34,  24}, { 42,  43}, { 40,  41}, { 39,  38}
};

// Indices of ZZ-interacting qubits which undergo the third (green) Trotter round
const int32_t ZZ_QUBITS_GREEN[NUM_ROTATIONS_PER_LAYER][NUM_PAULIS_PER_Z_ROTATION] = {
  { 10,  11}, { 54,  45}, {111, 122}, { 64,  65}, { 60,  61}, {103, 102},
  { 72,  62}, {  4,   3}, { 33,  20}, { 58,  59}, { 26,  16}, { 28,  27},
  {  8,   7}, {104, 105}, { 73,  66}, { 87,  93}, { 85,  86}, { 55,  49},
  { 68,  69}, { 89,  88}, { 80,  81}, {117, 118}, {101, 100}, {114, 115},
  { 96,  95}, { 29,  30}, {106, 107}, { 83,  82}, { 91,  79}, {  0,   1},
  { 56,  52}, { 90,  75}, {126, 112}, { 36,  32}, { 46,  47}, { 77,  78},
  { 97,  98}, { 17,  12}, {119, 120}, { 22,  23}, { 24,  25}, { 43,  34},
  { 42,  41}, { 40,  39}, { 37,  38}, {125, 124}, { 50,  51}, { 18,  19}
};


// Sphinx: #4
// ========================================================================
// Circuit construction
// ========================================================================

// Each 'step' of the Trotter circuit alternates a layer of single-qubit X
// rotations on every qubit, then a sequence of two-qubit Z rotations on the
// heavy-hex topology, upon qubit pairs in the red, blue and green lists 
// above. Note that ZZ rotations about -pi/2 are actually Clifford, though
// we still here treat them like a generic Pauli rotation. The functions
// below construct a Trotter circuit with a variable number of steps.


std::vector<cupaulipropQuantumOperator_t> getXRotationLayer(
  cupaulipropHandle_t handle, double xRotationAngle
) {  
  std::vector<cupaulipropQuantumOperator_t> layer(NUM_CIRCUIT_QUBITS);

  const cupaulipropPauliKind_t paulis[NUM_PAULIS_PER_X_ROTATION] = {CUPAULIPROP_PAULI_X};

  for (int32_t i=0; i<NUM_CIRCUIT_QUBITS; i++) {
    HANDLE_CUPP_ERROR(cupaulipropCreatePauliRotationGateOperator(
      handle, xRotationAngle, NUM_PAULIS_PER_X_ROTATION, &i, paulis, &layer[i]));
  }

  return layer;
}


std::vector<cupaulipropQuantumOperator_t> getZZRotationLayer(
  cupaulipropHandle_t handle,
  const int32_t topology[NUM_ROTATIONS_PER_LAYER][NUM_PAULIS_PER_Z_ROTATION]
) {
  std::vector<cupaulipropQuantumOperator_t> layer(NUM_ROTATIONS_PER_LAYER);

  const cupaulipropPauliKind_t paulis[NUM_PAULIS_PER_Z_ROTATION] = {
    CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z};

  for (uint32_t i=0; i<NUM_ROTATIONS_PER_LAYER; i++) {
    HANDLE_CUPP_ERROR(cupaulipropCreatePauliRotationGateOperator(
      handle, ZZ_ROTATION_ANGLE, NUM_PAULIS_PER_Z_ROTATION, topology[i], paulis, &layer[i]));
  }

  return layer;
}


std::vector<cupaulipropQuantumOperator_t> getIBMHeavyHexIsingCircuit(
  cupaulipropHandle_t handle, double xRotationAngle, int numTrotterSteps
) {
  std::vector<cupaulipropQuantumOperator_t> circuit;

  for (int n=0; n<numTrotterSteps; n++) {
    auto layerX       = getXRotationLayer (handle, xRotationAngle);
    auto layerRedZZ   = getZZRotationLayer(handle, ZZ_QUBITS_RED);
    auto layerBlueZZ  = getZZRotationLayer(handle, ZZ_QUBITS_BLUE);
    auto layerGreenZZ = getZZRotationLayer(handle, ZZ_QUBITS_GREEN);
    
    circuit.insert(circuit.end(), layerX.begin(),       layerX.end());
    circuit.insert(circuit.end(), layerRedZZ.begin(),   layerRedZZ.end());
    circuit.insert(circuit.end(), layerBlueZZ.begin(),  layerBlueZZ.end());
    circuit.insert(circuit.end(), layerGreenZZ.begin(), layerGreenZZ.end());
  }

  return circuit;
}


// Sphinx: #5
// ========================================================================
// Observable preparation
// ========================================================================

// This demo simulates the IBM circuit via back-propagating the measurement
// observable through the adjoint circuit. As such, we encode the measured
// observable into our initial Pauli expansion, in the format recognised by
// cuPauliProp. Pauli strings are represented with "packed integers" wherein
// every bit encodes a Pauli operator upon a corresponding qubit. We maintain
// two masks which separately encode the position of X and Z Pauli operators,
// indicated by a set bit at the qubit index, with a common set bit encoding
// a Y Pauli operator. Simulating more qubits than exist bits in the packed
// integer type (64) requires using multiple packed integers for each X and Z
// mask. We store a Pauli string's constituent X and Z masks contiguously in
// a single array, where the final mask of each per-string is padded with zero
// bits to be an integer multiple of the packed integer size (64 bits).

// The below function accepts a single Pauli string (i.e. a tensor product of
// the given Pauli operators at the specified qubit indices) and returns the
// sequence of packed integers which encode it as per the cuPauliProp API;
// this sequence can be copied directly to the GPU buffer of a Pauli expansion.

std::vector<cupaulipropPackedIntegerType_t> getPauliStringAsPackedIntegers(
  std::vector<cupaulipropPauliKind_t> paulis, 
  std::vector<uint32_t> qubits
) {
  assert(paulis.size() == qubits.size());
  assert(*std::max_element(qubits.begin(), qubits.end()) < NUM_CIRCUIT_QUBITS);

  int32_t numPackedInts;
  HANDLE_CUPP_ERROR(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));

  // A single Pauli string is composed of separate X and Z masks, one after the other
  std::vector<cupaulipropPackedIntegerType_t> out(numPackedInts * 2, 0);
  auto xPtr = &out[0];
  auto zPtr = &out[numPackedInts];

  // Process one input (pauli, qubit) pair at a time
  for (auto i=0; i<qubits.size(); i++) {

    // The qubit corresponds to a specific bit of a specific packed integer
    auto numBitsPerPackedInt = 8 * sizeof(cupaulipropPackedIntegerType_t);
    auto intInd = qubits[i] / numBitsPerPackedInt;
    auto bitInd = qubits[i] % numBitsPerPackedInt;

    // Overwrite a bit of either the X or Z masks (or both when pauli==Y)
    if (paulis[i] == CUPAULIPROP_PAULI_X || paulis[i] == CUPAULIPROP_PAULI_Y)
      xPtr[intInd] = xPtr[intInd] | (1ULL << bitInd);
    if (paulis[i] == CUPAULIPROP_PAULI_Z || paulis[i] == CUPAULIPROP_PAULI_Y)
      zPtr[intInd] = zPtr[intInd] | (1ULL << bitInd);
  }

  return out;
}


// Sphinx: #6
// ========================================================================
// Main
// ========================================================================

// Simulation of the IBM utility experiment proceeds as follows. We setup the
// cuPauliProp library, attaching a new stream, then proceed to creating two
// Pauli expansions (since the API is out-of-place, as elaborated upon below).
// One expansion is initialised to the measured observable of the IBM circuit.
// We prepare workspace memory, fix truncation hyperparameters, then create
// the circuit as a list of cuPauliProp operators. We process the circuit in
// reverse, adjointing each operation, applied upon a newly prepared view of
// the input expansion, each time checking our dynamically growing memory costs
// have not exceeded our budgets. Thereafter we compute the overlap between the
// final back-propagated observable and the experimental initial state (the all-
// zero state), producing an estimate of the experimental expectation value.
// Finally, we free all allocated memory like good citizens.


int main(int argc, char** argv) {
  std::cout << "cuPauliProp IBM Heavy-hex Ising Example" << std::endl;
  std::cout << "=======================================" << std::endl << std::endl;


  // Sphinx: #7
  // ========================================================================
  // Library setup
  // ========================================================================

  int deviceId = 0;
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));

  // Work on the default stream; use cupaulipropSetStream() to perform all
  // cuPauliProp operations on a specific stream for asynchronous usage
  cupaulipropHandle_t handle;
  HANDLE_CUPP_ERROR(cupaulipropCreate(&handle));


  // Sphinx: #8
  // ========================================================================
  // Decide memory usage
  // ========================================================================

  // As outlined in the 'Memory usage' section above, we here either uniformly
  // allocate all (or a high percentage of) available memory between the needed
  // memory buffers, or use the pre-decided fixed values. This demo will create
  // a total of two Pauli expansions (each of which accepts two separate buffers
  // to store Pauli strings and their corresponding coefficients; these have
  // different sizes) and one workspace, hence we arrange for an allocation of
  // five buffers in total.

  size_t expansionPauliMem;
  size_t expansionCoefMem;
  size_t workspaceMem;
  size_t totalUsedMem;

  if (USE_MAX_VRAM) {

    // Find usable device memory
    size_t totalFreeMem, totalGlobalMem;
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&totalFreeMem, &totalGlobalMem));
    size_t totalUsableMem = static_cast<size_t>(totalFreeMem * MAX_VRAM_PERCENT/100);

    // Divide it between the three instances (two expansions, one workspace)
    size_t instanceMem = totalUsableMem / 3;

    // Determine the ideal ratio between an expansion's Pauli and coef buffers
    int32_t numPackedInts;
    HANDLE_CUPP_ERROR(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPackedInts));
    size_t pauliMemPerTerm = 2 * numPackedInts * sizeof(cupaulipropPackedIntegerType_t);
    size_t coefMemPerTerm = sizeof(double);
    size_t totalMemPerTerm = pauliMemPerTerm + coefMemPerTerm;

    expansionPauliMem = (instanceMem * pauliMemPerTerm) / totalMemPerTerm;
    expansionCoefMem  = (instanceMem * coefMemPerTerm ) / totalMemPerTerm;
    workspaceMem      = instanceMem;

    totalUsedMem = 2*expansionPauliMem + 2*expansionCoefMem + workspaceMem;
    std::cout << "Dedicated memory: " << MAX_VRAM_PERCENT << "% of " << totalFreeMem;
    std::cout << " B free = " << totalUsedMem << " B, divided into..." << std::endl;

  } else {

    // Use pre-decided buffer sizes
    expansionPauliMem = FIXED_EXPANSION_PAULI_MEM;
    expansionCoefMem  = FIXED_EXPANSION_COEF_MEM;
    workspaceMem      = FIXED_WORKSPACE_MEM;

    totalUsedMem = 2*expansionPauliMem + 2*expansionCoefMem + workspaceMem;
    std::cout << "Dedicated memory: " << totalUsedMem << " B = 60 MiB, divided into..." << std::endl;
  }

  std::cout << "  expansion Pauli buffer: " << expansionPauliMem << " B" << std::endl;
  std::cout << "  expansion coef buffer:  " << expansionCoefMem << " B" << std::endl;
  std::cout << "  workspace buffer:       " << workspaceMem << " B\n" << std::endl;


  // Sphinx: #9
  // ========================================================================
  // Pauli expansion preparation
  // ========================================================================

  // Create buffers for two Pauli expansions, which will serve as 'input' and
  // 'output' to the out-of-place cuPauliProp API. Note that the capacities of
  // these buffers constrain the maximum number of Pauli strings maintained
  // during simulation, and ergo inform the accuracy of the simulation. The
  // sufficient buffer sizes are specific to the simulated system, and we here
  // choose a surprisingly small capacity as admitted by the studied circuit.

  void * d_inExpansionPauliBuffer;
  void * d_outExpansionPauliBuffer;
  void * d_inExpansionCoefBuffer;
  void * d_outExpansionCoefBuffer;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_inExpansionPauliBuffer,  expansionPauliMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_inExpansionCoefBuffer,   expansionCoefMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_outExpansionPauliBuffer, expansionPauliMem));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_outExpansionCoefBuffer,  expansionCoefMem));

  // Prepare the X and Z masks which encode the experimental observable Z_62,
  // which has a coefficient of unity, as seen in Figure 4. b) of the IBM work.
  std::cout << "Observable: Z_62\n" << std::endl;
  int64_t numObservableTerms = 1;
  double observableCoef = 1.0;
  std::vector<cupaulipropPauliKind_t> observablePaulis = {CUPAULIPROP_PAULI_Z};
  std::vector<uint32_t> observableQubits = {62};

  // Overwrite the 'input' Pauli expansion buffers with the observable data
  auto observablePackedInts = getPauliStringAsPackedIntegers(observablePaulis, observableQubits);
  size_t numObservableBytes = observablePackedInts.size() * sizeof(observablePackedInts[0]);
  HANDLE_CUDA_ERROR(cudaMemcpy(
    d_inExpansionPauliBuffer, observablePackedInts.data(), numObservableBytes, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(
    d_inExpansionCoefBuffer, &observableCoef, sizeof(observableCoef), cudaMemcpyHostToDevice));

  // Create two Pauli expansions, which will serve as 'input and 'output' to the API.
  // Because we begin from a real observable coefficient, and our circuit is completely
  // positive and trace preserving, it is sufficient to use strictly real coefficients
  // in our expansions, informing dataType below. We indicate that the single prepared
  // term in the input expansion is technically unique, and the terms sorted, which
  // permits cuPauliProp to use automatic optimisations during simulation.

  cupaulipropPauliExpansion_t inExpansion;
  cupaulipropPauliExpansion_t outExpansion;

  int32_t isSorted = 1;
  int32_t isUnique = 1;
  cudaDataType_t dataType = CUDA_R_64F;

  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion( // init to above
    handle, NUM_CIRCUIT_QUBITS,
    d_inExpansionPauliBuffer, expansionPauliMem,
    d_inExpansionCoefBuffer,  expansionCoefMem,
    dataType, numObservableTerms, isSorted, isUnique, 
    &inExpansion));
  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion( // init to empty
    handle, NUM_CIRCUIT_QUBITS,
    d_outExpansionPauliBuffer, expansionPauliMem,
    d_outExpansionCoefBuffer,  expansionCoefMem,
    dataType, 0, 0, 0,
    &outExpansion));

  
  // Sphinx: #10
  // ========================================================================
  // Workspace preparation
  // ========================================================================

  // Some API functions require additional workspace memory which we bind to a
  // workspace descriptor. Ordinarily we use the 'Prepare' functions to precisely
  // bound upfront the needed workspace memory, but in this simple demo, we
  // instead use a workspace memory which we prior know to be sufficient.

  // Create a workspace
  cupaulipropWorkspaceDescriptor_t workspace;
  HANDLE_CUPP_ERROR(cupaulipropCreateWorkspaceDescriptor(handle, &workspace));

  void* d_workspaceBuffer;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_workspaceBuffer, workspaceMem));
  HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    d_workspaceBuffer, workspaceMem));
  
  // Note that the 'prepare' functions which check the required workspace memory
  // will detach this buffer, requiring we re-call SetMemory above. We can avoid
  // this repeated re-attachment by use of a second, bufferless workspace descri-
  // ptor which we pass to the 'prepare' functions in lieu of this one.


  // Sphinx: #11
  // ========================================================================
  // Truncation parameter preparation
  // ========================================================================

  // The Pauli propagation simulation technique has memory and runtime costs which
  // (for generic circuits) grow exponentially with the circuit length, which we
  // curtail through "truncation"; dynamic discarding of Pauli strings in our
  // expansion which are predicted to contribute negligibly to the final output
  // expectation value. This file demonstrates simultaneous usage of two truncation
  // techniques; discarding of Pauli strings with an absolute coefficient less than
  // 0.0001, or a "Pauli weight" (the number of non-identity operators in the string)
  // exceeding eight. For example, given a Pauli expansion containing:
  //    0.1 XYZXYZII + 1E-5 ZIIIIIII + 0.2 XXXXYYYY,
  // our truncation parameters below would see the latter two strings discarded due
  // to coefficient and weight truncation respectively.

  cupaulipropCoefficientTruncationParams_t coefTruncParams;
  coefTruncParams.cutoff = 1E-4;

  cupaulipropPauliWeightTruncationParams_t weightTruncParams;
  weightTruncParams.cutoff = 8;

  const uint32_t numTruncStrats = 2;
  cupaulipropTruncationStrategy_t truncStrats[] = {
    {
      CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED,
      &coefTruncParams
    },
    {
      CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED,
      &weightTruncParams
    }
  };

  // It is not necessary to perform truncation after every gate, since the
  // Pauli expansion size may not have grown substantially, and attempting
  // to truncate may incur superfluous memory enumeration costs. In this
  // demo, we choose to truncate only after every tenth applied gate. Note
  // deferring truncation requires additional expansion memory; choosing to
  // truncate after every gate shrinks this demo's costs to 20 MiB total.
  const int numGatesBetweenTruncations = 10;

  std::cout << "Coefficient truncation threshold:  " << coefTruncParams.cutoff << std::endl;
  std::cout << "Pauli weight truncation threshold: " << weightTruncParams.cutoff << std::endl;
  std::cout << "Truncation performed after every:  " << numGatesBetweenTruncations << " gates\n" << std::endl;


  // Sphinx: #12
  // ========================================================================
  // Back-propagation of the observable through the circuit
  // ========================================================================

  // We now simulate the observable operator being back-propagated through the
  // adjoint circuit, mapping the input expansion (initialised to Z_62) to a
  // final output expansion containing many weighted Pauli strings. We use the
  // heavy-hex fixed-angle Ising circuit with 20 total repetitions, fixing the
  // angle of the X rotation gates to PI/4. Our simulation therefore corresponds
  // to the middle datum of Fig. 4 b) of the IBM manuscript, for which MPS and
  // isoTNS siulation techniques showed the greatest divergence from experiment.

  double xRotationAngle = PI / 4.;
  int numTrotterSteps = 20;
  auto circuit = getIBMHeavyHexIsingCircuit(handle, xRotationAngle, numTrotterSteps);

  std::cout << "Circuit: 127 qubit IBM heavy-hex Ising circuit, with..." << std::endl;
  std::cout << "  Trotter steps: " << numTrotterSteps << std::endl;
  std::cout << "  Total gates:   " << circuit.size() << std::endl;
  std::cout << "  Rx angle:      " << xRotationAngle << " (i.e. PI/4)\n" << std::endl;

  // Constrain that every intermediate output expansion contains unique Pauli
  // strings (forbidding duplicates), but permit the retained strings to be
  // unsorted. This combination gives cuPauliProp the best chance of automatically
  // selecting optimal internal functions and postconditions for the simulation.
  uint32_t adjoint = true;
  uint32_t makeSorted = false;
  uint32_t keepDuplicates = false;

  std::cout << "Imposed postconditions:" << std::endl;
  if (makeSorted) {
    std::cout << "  Pauli strings will be sorted." << std::endl;
  }
  if (!keepDuplicates) {
    std::cout << "  Pauli strings will be unique." << std::endl;
  }
  if (keepDuplicates && !makeSorted) {
    std::cout << "No postconditions imposed on Pauli strings." << std::endl;
  }
  std::cout << std::endl;

  // Begin timing before any gates are applied
  auto startTime = std::chrono::high_resolution_clock::now();
  int64_t maxNumTerms = 0;

  // Iterate the circuit in reverse to effect the adjoint of the total circuit
  for (int gateInd=circuit.size()-1; gateInd >= 0; --gateInd) {
    cupaulipropQuantumOperator_t gate = circuit[gateInd];

    // Create a view of the current input expansion, selecting all currently
    // contained terms. For very large systems, we may have alternatively
    // chosen a smaller view of the partial state to work around memory limits.
    cupaulipropPauliExpansionView_t inView;
    int64_t numExpansionTerms;
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(handle, inExpansion, &numExpansionTerms));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(
      handle, inExpansion, 0, numExpansionTerms, &inView));

    // Track the intermediate expansion size, for our curiousity
    if (numExpansionTerms > maxNumTerms)
      maxNumTerms = numExpansionTerms;

    // Choose whether or not to perform truncations after this gate
    int numPassedTruncStrats = (gateInd % numGatesBetweenTruncations == 0)? numTruncStrats : 0;

    // Check the expansion and workspace memories needed to apply the current gate
    int64_t reqExpansionPauliMem;
    int64_t reqExpansionCoefMem;
    int64_t reqWorkspaceMem;
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareOperatorApplication(
      handle, inView, gate, makeSorted, keepDuplicates,
      numPassedTruncStrats, numPassedTruncStrats > 0 ? truncStrats : nullptr,
      workspaceMem,
      &reqExpansionPauliMem, &reqExpansionCoefMem, workspace));
    HANDLE_CUPP_ERROR(cupaulipropWorkspaceGetMemorySize(
      handle, workspace,
      CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
      &reqWorkspaceMem));

    // Verify that our existing buffers and workspace have sufficient memory
    assert(reqExpansionPauliMem <= expansionPauliMem);
    assert(reqExpansionCoefMem  <= expansionCoefMem);
    assert(reqWorkspaceMem      <= workspaceMem);

    // Beware that cupaulipropPauliExpansionViewPrepareOperatorApplication() above
    // detaches the memory buffer from the workspace, which we here re-attach.
    HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
      handle, workspace,
      CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
      d_workspaceBuffer, workspaceMem));

    // Apply the gate upon the prepared view of the input expansion, evolving the
    // Pauli strings pointed to within, truncating the result. The input expansion
    // is unchanged while the output expansion is entirely overwritten.
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeOperatorApplication(
      handle, inView, outExpansion, gate, 
      adjoint, makeSorted, keepDuplicates,
      numPassedTruncStrats, numPassedTruncStrats > 0 ? truncStrats : nullptr,
      workspace));

    // Free the temporary view since it points to the old input expansion, whereas
    // we will subsequently treat the modified output expansion as the next input
    HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(inView));

    // Treat outExpansion as the input in the next gate application
    std::swap(inExpansion, outExpansion);
  }

  // Restore outExpansion to being the final output for clarity
  std::swap(inExpansion, outExpansion);


  // Sphinx: #13
  // ========================================================================
  // Evaluation of the the expectation value of observable
  // ========================================================================

  // The output expansion is now a proxy for the observable back-propagated
  // through to the front of the circuit (though having discarded components
  // which negligibly influence the subsequent overlap). The expectation value
  // of the IBM experiment is the overlap of the output expansion with the
  // zero state, i.e. Tr(outExpansion * |0><0|), as we now compute.

  // Obtain a view of the full output expansion (we'll free it in 'Clean up')
  cupaulipropPauliExpansionView_t outView;
  int64_t numOutTerms;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(handle, inExpansion, &numOutTerms));
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(
    handle, inExpansion, 0, numOutTerms, &outView));

  // Check that the existing workspace memory is sufficient to compute the trace 
  int64_t reqWorkspaceMem;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(
    handle, outView, workspaceMem, workspace));
  HANDLE_CUPP_ERROR(cupaulipropWorkspaceGetMemorySize(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    &reqWorkspaceMem));
  assert(reqWorkspaceMem <= workspaceMem);

  // Beware that we must now reattach the buffer to the workspace
  HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
    handle, workspace,
    CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH,
    d_workspaceBuffer, workspaceMem));

  // Compute the trace; the main and final output of this simulation!
  double expec;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeTraceWithZeroState(
    handle, outView, &expec, workspace));

  // End timing after trace is evaluated
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
  auto durationSecs = (duration.count() / 1e6);

  std::cout << "Expectation value:       " << expec << std::endl;
  std::cout << "Final number of terms:   " << numOutTerms << std::endl;
  std::cout << "Maximum number of terms: " << maxNumTerms << std::endl;
  std::cout << "Runtime:                 " << durationSecs << " seconds\n" << std::endl;


  // Sphinx: #14
  // ========================================================================
  // Clean up
  // ========================================================================

  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(outView));
 
  for (auto & gate : circuit) {
    HANDLE_CUPP_ERROR(cupaulipropDestroyOperator(gate));
  }

  HANDLE_CUPP_ERROR(cupaulipropDestroyWorkspaceDescriptor(workspace));
  HANDLE_CUDA_ERROR(cudaFree(d_workspaceBuffer));

  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(inExpansion));
  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(outExpansion));

  HANDLE_CUDA_ERROR(cudaFree(d_inExpansionPauliBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_outExpansionPauliBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_inExpansionCoefBuffer));
  HANDLE_CUDA_ERROR(cudaFree(d_outExpansionCoefBuffer));

  HANDLE_CUPP_ERROR(cupaulipropDestroy(handle));

  return EXIT_SUCCESS;
}
