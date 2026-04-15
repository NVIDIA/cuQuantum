/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file kicked_ising_backward_diff_example.cpp
 * @brief An example demonstrating reverse-mode automatic differentiation of the IBM
 *        127-qubit kicked Ising experiment, as presented in Nature volume 618, pages
 *        500-505 (2023), using the cuPauliProp backward differentiation API.
 *        We simulate the Z_62 20-Trotter-step experiment of Fig 4. b) with X-rotation
 *        angle PI/4 and ZZ-rotation angle -PI/2, computing the two-element gradient
 *        [d<Z>/d(x_angle), d<Z>/d(zz_angle)] via O(1)-memory reverse-mode AD with
 *        tape replay. For simplicity, we do not simulate error channels nor twirling,
 *        though inhomogeneous Pauli channels are supported by cuPauliProp.
 */

#include <cupauliprop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#define HANDLE_CUPP_ERROR(x)                                              \
{                                                                         \
  const auto err = x;                                                     \
  if (err != CUPAULIPROP_STATUS_SUCCESS) {                                \
    std::cerr << "cuPauliProp error at line " << __LINE__ << std::endl;   \
    std::abort();                                                         \
  }                                                                       \
}

#define HANDLE_CUDA_ERROR(x)                                              \
{                                                                         \
  const auto err = x;                                                     \
  if (err != cudaSuccess) {                                               \
    std::cerr << "CUDA error at line " << __LINE__ << ": "                \
              << cudaGetErrorString(err) << std::endl;                    \
    std::abort();                                                         \
  }                                                                       \
}

namespace {

constexpr int NUM_CIRCUIT_QUBITS = 127;
constexpr int NUM_ROTATIONS_PER_LAYER = 48;
constexpr double PI = 3.14159265358979323846;

constexpr size_t EXPANSION_PAULI_MEM = 16 * (1ULL << 20);
constexpr size_t EXPANSION_COEF_MEM = 4 * (1ULL << 20);
constexpr size_t WORKSPACE_MEM = 32 * (1ULL << 20);

const int32_t ZZ_QUBITS_RED[NUM_ROTATIONS_PER_LAYER][2] = {
  {2, 1}, {33, 39}, {59, 60}, {66, 67}, {72, 81}, {118, 119},
  {21, 20}, {26, 25}, {13, 12}, {31, 32}, {70, 74}, {122, 123},
  {96, 97}, {57, 56}, {63, 64}, {107, 108}, {103, 104}, {46, 45},
  {28, 35}, {7, 6}, {79, 78}, {5, 4}, {109, 114}, {62, 61},
  {58, 71}, {37, 52}, {76, 77}, {0, 14}, {36, 51}, {106, 105},
  {73, 85}, {88, 87}, {68, 55}, {116, 115}, {94, 95}, {100, 110},
  {17, 30}, {92, 102}, {50, 49}, {83, 84}, {48, 47}, {98, 99},
  {8, 9}, {121, 120}, {23, 24}, {44, 43}, {22, 15}, {53, 41},
};

const int32_t ZZ_QUBITS_BLUE[NUM_ROTATIONS_PER_LAYER][2] = {
  {53, 60}, {123, 124}, {21, 22}, {11, 12}, {67, 68}, {2, 3},
  {66, 65}, {122, 121}, {110, 118}, {6, 5}, {94, 90}, {28, 29},
  {14, 18}, {63, 62}, {111, 104}, {100, 99}, {45, 44}, {4, 15},
  {20, 19}, {57, 58}, {77, 71}, {76, 75}, {26, 27}, {16, 8},
  {35, 47}, {31, 30}, {48, 49}, {69, 70}, {125, 126}, {89, 74},
  {80, 79}, {116, 117}, {114, 113}, {10, 9}, {106, 93}, {101, 102},
  {92, 83}, {98, 91}, {82, 81}, {54, 64}, {96, 109}, {85, 84},
  {87, 86}, {108, 112}, {34, 24}, {42, 43}, {40, 41}, {39, 38},
};

const int32_t ZZ_QUBITS_GREEN[NUM_ROTATIONS_PER_LAYER][2] = {
  {10, 11}, {54, 45}, {111, 122}, {64, 65}, {60, 61}, {103, 102},
  {72, 62}, {4, 3}, {33, 20}, {58, 59}, {26, 16}, {28, 27},
  {8, 7}, {104, 105}, {73, 66}, {87, 93}, {85, 86}, {55, 49},
  {68, 69}, {89, 88}, {80, 81}, {117, 118}, {101, 100}, {114, 115},
  {96, 95}, {29, 30}, {106, 107}, {83, 82}, {91, 79}, {0, 1},
  {56, 52}, {90, 75}, {126, 112}, {36, 32}, {46, 47}, {77, 78},
  {97, 98}, {17, 12}, {119, 120}, {22, 23}, {24, 25}, {43, 34},
  {42, 41}, {40, 39}, {37, 38}, {125, 124}, {50, 51}, {18, 19},
};

struct GateRecord {
  cupaulipropQuantumOperator_t op{};
  bool isX{};
};

std::vector<cupaulipropPackedIntegerType_t> getPauliStringAsPackedIntegers(
    const std::vector<cupaulipropPauliKind_t>& paulis,
    const std::vector<int32_t>& qubits) {
  int32_t numPacked = 0;
  HANDLE_CUPP_ERROR(cupaulipropGetNumPackedIntegers(NUM_CIRCUIT_QUBITS, &numPacked));
  std::vector<cupaulipropPackedIntegerType_t> out(2 * static_cast<size_t>(numPacked), 0);
  auto* x = out.data();
  auto* z = out.data() + numPacked;
  for (size_t i = 0; i < paulis.size(); ++i) {
    const int32_t intInd = qubits[i] / 64;
    const int32_t bitInd = qubits[i] % 64;
    if (paulis[i] == CUPAULIPROP_PAULI_X || paulis[i] == CUPAULIPROP_PAULI_Y) x[intInd] |= (1ULL << bitInd);
    if (paulis[i] == CUPAULIPROP_PAULI_Z || paulis[i] == CUPAULIPROP_PAULI_Y) z[intInd] |= (1ULL << bitInd);
  }
  return out;
}

void appendXLayer(cupaulipropHandle_t handle, double angle, std::vector<GateRecord>& out) {
  const cupaulipropPauliKind_t paulis[1] = {CUPAULIPROP_PAULI_X};
  for (int32_t q = 0; q < NUM_CIRCUIT_QUBITS; ++q) {
    cupaulipropQuantumOperator_t gate{};
    HANDLE_CUPP_ERROR(cupaulipropCreatePauliRotationGateOperator(handle, angle, 1, &q, paulis, &gate));
    out.push_back({gate, true});
  }
}

void appendZZLayer(cupaulipropHandle_t handle, double angle, const int32_t topology[NUM_ROTATIONS_PER_LAYER][2], std::vector<GateRecord>& out) {
  const cupaulipropPauliKind_t paulis[2] = {CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z};
  for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; ++i) {
    cupaulipropQuantumOperator_t gate{};
    HANDLE_CUPP_ERROR(cupaulipropCreatePauliRotationGateOperator(handle, angle, 2, topology[i], paulis, &gate));
    out.push_back({gate, false});
  }
}

std::vector<GateRecord> buildCircuit(cupaulipropHandle_t handle, double xAngle, double zzAngle, int trotterSteps) {
  std::vector<GateRecord> circuit;
  circuit.reserve(static_cast<size_t>(trotterSteps) * (NUM_CIRCUIT_QUBITS + 3 * NUM_ROTATIONS_PER_LAYER));
  for (int s = 0; s < trotterSteps; ++s) {
    appendXLayer(handle, xAngle, circuit);
    appendZZLayer(handle, zzAngle, ZZ_QUBITS_RED, circuit);
    appendZZLayer(handle, zzAngle, ZZ_QUBITS_BLUE, circuit);
    appendZZLayer(handle, zzAngle, ZZ_QUBITS_GREEN, circuit);
  }
  return circuit;
}

void reattachWorkspace(cupaulipropHandle_t handle, cupaulipropWorkspaceDescriptor_t workspace, void* dWorkspace) {
  HANDLE_CUPP_ERROR(cupaulipropWorkspaceSetMemory(
      handle, workspace, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWorkspace, WORKSPACE_MEM));
}

}  // namespace

int main() {
  std::cout << "cuPauliProp IBM Heavy-hex Ising Backward Diff Example" << std::endl;
  std::cout << "=====================================================" << std::endl;
  std::cout << std::endl;

  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle{};
  HANDLE_CUPP_ERROR(cupaulipropCreate(&handle));

  // ========================================================================
  // Memory budget
  // ========================================================================
  const size_t totalUsedMem =
      4 * EXPANSION_PAULI_MEM + 4 * EXPANSION_COEF_MEM + WORKSPACE_MEM;
  std::cout << "Dedicated memory: " << totalUsedMem << " B" << std::endl;
  std::cout << "  expansion Pauli buffer: " << EXPANSION_PAULI_MEM << " B  (x4)" << std::endl;
  std::cout << "  expansion coef buffer:  " << EXPANSION_COEF_MEM  << " B  (x4)" << std::endl;
  std::cout << "  workspace buffer:       " << WORKSPACE_MEM << " B" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // Observable
  // ========================================================================
  std::cout << "Observable: Z_62\n" << std::endl;

  void *dInXZ = nullptr, *dInCoef = nullptr, *dOutXZ = nullptr, *dOutCoef = nullptr;
  void *dCot0XZ = nullptr, *dCot0Coef = nullptr, *dCot1XZ = nullptr, *dCot1Coef = nullptr;
  HANDLE_CUDA_ERROR(cudaMalloc(&dInXZ, EXPANSION_PAULI_MEM));
  HANDLE_CUDA_ERROR(cudaMalloc(&dInCoef, EXPANSION_COEF_MEM));
  HANDLE_CUDA_ERROR(cudaMalloc(&dOutXZ, EXPANSION_PAULI_MEM));
  HANDLE_CUDA_ERROR(cudaMalloc(&dOutCoef, EXPANSION_COEF_MEM));
  HANDLE_CUDA_ERROR(cudaMalloc(&dCot0XZ, EXPANSION_PAULI_MEM));
  HANDLE_CUDA_ERROR(cudaMalloc(&dCot0Coef, EXPANSION_COEF_MEM));
  HANDLE_CUDA_ERROR(cudaMalloc(&dCot1XZ, EXPANSION_PAULI_MEM));
  HANDLE_CUDA_ERROR(cudaMalloc(&dCot1Coef, EXPANSION_COEF_MEM));

  auto observablePacked = getPauliStringAsPackedIntegers({CUPAULIPROP_PAULI_Z}, {62});
  double observableCoef = 1.0;
  HANDLE_CUDA_ERROR(cudaMemcpy(dInXZ, observablePacked.data(),
      observablePacked.size() * sizeof(observablePacked[0]), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(dInCoef, &observableCoef,
      sizeof(observableCoef), cudaMemcpyHostToDevice));

  cupaulipropPauliExpansion_t inExpansion{}, outExpansion{}, cot0Expansion{}, cot1Expansion{};
  const auto sortOrder = CUPAULIPROP_SORT_ORDER_NONE;
  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion(
      handle, NUM_CIRCUIT_QUBITS, dInXZ, EXPANSION_PAULI_MEM, dInCoef, EXPANSION_COEF_MEM,
      CUDA_R_64F, 1, sortOrder, 0, &inExpansion));
  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion(
      handle, NUM_CIRCUIT_QUBITS, dOutXZ, EXPANSION_PAULI_MEM, dOutCoef, EXPANSION_COEF_MEM,
      CUDA_R_64F, 0, sortOrder, 0, &outExpansion));
  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion(
      handle, NUM_CIRCUIT_QUBITS, dCot0XZ, EXPANSION_PAULI_MEM, dCot0Coef, EXPANSION_COEF_MEM,
      CUDA_R_64F, 0, sortOrder, 0, &cot0Expansion));
  HANDLE_CUPP_ERROR(cupaulipropCreatePauliExpansion(
      handle, NUM_CIRCUIT_QUBITS, dCot1XZ, EXPANSION_PAULI_MEM, dCot1Coef, EXPANSION_COEF_MEM,
      CUDA_R_64F, 0, sortOrder, 0, &cot1Expansion));

  // ========================================================================
  // Workspace
  // ========================================================================
  cupaulipropWorkspaceDescriptor_t workspace{};
  HANDLE_CUPP_ERROR(cupaulipropCreateWorkspaceDescriptor(handle, &workspace));
  void* dWorkspace = nullptr;
  HANDLE_CUDA_ERROR(cudaMalloc(&dWorkspace, WORKSPACE_MEM));
  reattachWorkspace(handle, workspace, dWorkspace);

  // ========================================================================
  // Truncation parameters
  // ========================================================================
  cupaulipropCoefficientTruncationParams_t coefTrunc{1e-4};
  cupaulipropPauliWeightTruncationParams_t weightTrunc{8};
  cupaulipropTruncationStrategy_t truncStrats[2] = {
      {CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefTrunc},
      {CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightTrunc}};

  std::cout << "Coefficient truncation threshold:  " << coefTrunc.cutoff << std::endl;
  std::cout << "Pauli weight truncation threshold: " << weightTrunc.cutoff << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // Circuit construction
  // ========================================================================
  const double xAngle = PI / 4.0;
  const double zzAngle = -PI / 2.0;
  const int numTrotterSteps = 20;
  auto circuit = buildCircuit(handle, xAngle, zzAngle, numTrotterSteps);

  std::cout << "Circuit: 127 qubit IBM heavy-hex Ising circuit, with..." << std::endl;
  std::cout << "  Trotter steps: " << numTrotterSteps << std::endl;
  std::cout << "  Total gates:   " << circuit.size() << std::endl;
  std::cout << "  Rx angle:      " << xAngle << " (i.e. PI/4)" << std::endl;
  std::cout << "  Rzz angle:     " << zzAngle << " (i.e. -PI/2)" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // Forward pass: back-propagate observable through adjoint circuit
  // ========================================================================
  auto startTime = std::chrono::high_resolution_clock::now();
  int64_t maxNumTerms = 0;

  for (int i = static_cast<int>(circuit.size()) - 1; i >= 0; --i) {
    int64_t inTerms = 0, reqXZ = 0, reqCoef = 0, reqWs = 0;
    cupaulipropPauliExpansionView_t inView{};
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(handle, inExpansion, &inTerms));
    if (inTerms > maxNumTerms) maxNumTerms = inTerms;
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(handle, inExpansion, 0, inTerms, &inView));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, inView, circuit[i].op, sortOrder, 0, 2, truncStrats, WORKSPACE_MEM, &reqXZ, &reqCoef, workspace));
    HANDLE_CUPP_ERROR(cupaulipropWorkspaceGetMemorySize(
        handle, workspace, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &reqWs));
    assert(reqXZ <= static_cast<int64_t>(EXPANSION_PAULI_MEM));
    assert(reqCoef <= static_cast<int64_t>(EXPANSION_COEF_MEM));
    assert(reqWs <= static_cast<int64_t>(WORKSPACE_MEM));
    reattachWorkspace(handle, workspace, dWorkspace);
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, inView, outExpansion, circuit[i].op, 1, sortOrder, 0, 2, truncStrats, workspace, stream));
    HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(inView));
    std::swap(inExpansion, outExpansion);
  }

  auto fwdEndTime = std::chrono::high_resolution_clock::now();
  auto fwdDuration = std::chrono::duration_cast<std::chrono::microseconds>(fwdEndTime - startTime);
  double fwdSecs = fwdDuration.count() / 1e6;

  // ========================================================================
  // Trace evaluation (expectation value)
  // ========================================================================
  int64_t finalTerms = 0;
  cupaulipropPauliExpansionView_t finalView{};
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(handle, inExpansion, &finalTerms));
  if (finalTerms > maxNumTerms) maxNumTerms = finalTerms;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(handle, inExpansion, 0, finalTerms, &finalView));
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, finalView, WORKSPACE_MEM, workspace));
  reattachWorkspace(handle, workspace, dWorkspace);
  double s = 0.0, p = 0.0;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, finalView, &s, &p, workspace, stream));
  const double expec = s * std::exp2(p);

  std::cout << "Forward pass completed in " << fwdSecs << " seconds" << std::endl;
  std::cout << "  Final number of terms:   " << finalTerms << std::endl;
  std::cout << "  Maximum number of terms: " << maxNumTerms << std::endl;
  std::cout << "  Expectation value:       " << expec << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // Backward: seed cotangent from trace
  // ========================================================================
  const double cotS = std::exp2(p);
  const double cotP = s * std::exp2(p) * std::log(2.0);
  int64_t reqCotXZ = 0, reqCotCoef = 0;
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff(
      handle, finalView, WORKSPACE_MEM, &reqCotXZ, &reqCotCoef, workspace));
  assert(reqCotXZ <= static_cast<int64_t>(EXPANSION_PAULI_MEM));
  assert(reqCotCoef <= static_cast<int64_t>(EXPANSION_COEF_MEM));
  reattachWorkspace(handle, workspace, dWorkspace);
  HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff(
      handle, finalView, &cotS, &cotP, cot0Expansion, workspace, stream));
  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(finalView));

  // ========================================================================
  // Backward pass: reverse-mode AD with tape replay
  // ========================================================================
  double gradX = 0.0, gradZZ = 0.0;
  auto bwdStartTime = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < circuit.size(); ++i) {
    int64_t inTerms = 0;
    cupaulipropPauliExpansionView_t inView{};
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(handle, inExpansion, &inTerms));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(handle, inExpansion, 0, inTerms, &inView));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, inView, circuit[i].op, sortOrder, 0, 2, truncStrats, WORKSPACE_MEM, &reqCotXZ, &reqCotCoef, workspace));
    reattachWorkspace(handle, workspace, dWorkspace);
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, inView, outExpansion, circuit[i].op, 0, sortOrder, 0, 2, truncStrats, workspace, stream));
    HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(inView));
    std::swap(inExpansion, outExpansion);

    int64_t cotTerms = 0, reqXZ = 0, reqCoef = 0, reqWs = 0;
    cupaulipropPauliExpansionView_t viewIn{}, cotOutView{};
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(handle, inExpansion, &inTerms));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(handle, inExpansion, 0, inTerms, &viewIn));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetNumTerms(handle, cot0Expansion, &cotTerms));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionGetContiguousRange(handle, cot0Expansion, 0, cotTerms, &cotOutView));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff(
        handle, viewIn, cotOutView, circuit[i].op, sortOrder, 0, 2, truncStrats, WORKSPACE_MEM, &reqXZ, &reqCoef, workspace));
    HANDLE_CUPP_ERROR(cupaulipropWorkspaceGetMemorySize(
        handle, workspace, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &reqWs));
    assert(reqXZ <= static_cast<int64_t>(EXPANSION_PAULI_MEM));
    assert(reqCoef <= static_cast<int64_t>(EXPANSION_COEF_MEM));
    assert(reqWs <= static_cast<int64_t>(WORKSPACE_MEM));
    reattachWorkspace(handle, workspace, dWorkspace);

    double gateGrad = 0.0;
    HANDLE_CUPP_ERROR(cupaulipropQuantumOperatorAttachCotangentBuffer(
        handle, circuit[i].op, &gateGrad, sizeof(double), CUDA_R_64F, CUPAULIPROP_MEMSPACE_HOST));
    HANDLE_CUPP_ERROR(cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff(
        handle, viewIn, cotOutView, cot1Expansion, circuit[i].op, 1, sortOrder, 0, 2, truncStrats, workspace, stream));
    HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(viewIn));
    HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansionView(cotOutView));
    if (circuit[i].isX) gradX += gateGrad; else gradZZ += gateGrad;
    std::swap(cot0Expansion, cot1Expansion);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto bwdDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - bwdStartTime);
  auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
  double bwdSecs = bwdDuration.count() / 1e6;
  double totalSecs = totalDuration.count() / 1e6;

  // ========================================================================
  // Results
  // ========================================================================
  std::cout << "Backward pass completed in " << bwdSecs << " seconds" << std::endl;
  std::cout << std::endl;
  std::cout << "Expectation value:       " << expec << std::endl;
  std::cout << "d<Z_62>/d(x_angle):      " << gradX << std::endl;
  std::cout << "d<Z_62>/d(zz_angle):     " << gradZZ << std::endl;
  std::cout << "Final number of terms:   " << finalTerms << std::endl;
  std::cout << "Maximum number of terms: " << maxNumTerms << std::endl;
  std::cout << "Total runtime:           " << totalSecs << " seconds" << std::endl;
  std::cout << std::endl;

  // ========================================================================
  // Clean up
  // ========================================================================
  for (const auto& gate : circuit) HANDLE_CUPP_ERROR(cupaulipropDestroyOperator(gate.op));
  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(inExpansion));
  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(outExpansion));
  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(cot0Expansion));
  HANDLE_CUPP_ERROR(cupaulipropDestroyPauliExpansion(cot1Expansion));
  HANDLE_CUPP_ERROR(cupaulipropDestroyWorkspaceDescriptor(workspace));
  HANDLE_CUPP_ERROR(cupaulipropDestroy(handle));

  HANDLE_CUDA_ERROR(cudaFree(dWorkspace));
  HANDLE_CUDA_ERROR(cudaFree(dInXZ));
  HANDLE_CUDA_ERROR(cudaFree(dInCoef));
  HANDLE_CUDA_ERROR(cudaFree(dOutXZ));
  HANDLE_CUDA_ERROR(cudaFree(dOutCoef));
  HANDLE_CUDA_ERROR(cudaFree(dCot0XZ));
  HANDLE_CUDA_ERROR(cudaFree(dCot0Coef));
  HANDLE_CUDA_ERROR(cudaFree(dCot1XZ));
  HANDLE_CUDA_ERROR(cudaFree(dCot1Coef));
  return EXIT_SUCCESS;
}

