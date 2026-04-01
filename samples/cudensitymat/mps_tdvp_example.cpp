/* Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// MPS-TDVP time propagation example.
//
// Demonstrates propagating a pure MPS state under a transverse-field Ising
// Hamiltonian encoded as an MPO, using the split-scope TDVP method with
// Krylov subspace exponentiation.
//
// Workflow:
//  1. Build Hamiltonian as MPO (nearest-neighbor ZZ + transverse X field)
//  2. Create input and output MPS states (cudensitymatCreateStateMPS)
//  3. Initialize input MPS to a Neel state |0101...>
//  4. Create TDVP time propagation object
//  5. Configure TDVP / Krylov parameters
//  6. Prepare propagation and allocate workspace
//  7. Time-stepping loop
//  8. Clean up all resources

#include <cudensitymat.h>
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <numeric>
#include <chrono>
#include <iostream>
#include <cassert>


using Complex = std::complex<double>;
constexpr cudaDataType_t DATA_TYPE = CUDA_C_64F;

constexpr bool verbose = true;

// --- Simulation parameters ---
constexpr int32_t  NUM_SITES    = 40;
constexpr int64_t  PHYS_DIM     = 2;
constexpr int64_t  MAX_BOND_DIM = 128;
constexpr int64_t  MPO_BOND_DIM = 3;
constexpr int32_t  NUM_STEPS    = 3;
constexpr double   DT           = 0.01;
constexpr double   J_COUPLING   = 1.0;
constexpr double   H_FIELD      = 0.5;


// ============================================================================
// Transverse-field Ising MPO builder
// ============================================================================
//
// H = -J * sum_{i} Z_i Z_{i+1}  +  h * sum_{i} X_i
//
// Standard MPO representation with bond dimension 3.
// Site tensor mode ordering follows cuDensityMat convention (column-major):
//   Left boundary (site 0):     [phys_ket(d), right_bond(bR), phys_bra(d)]
//   Interior sites:             [left_bond(bL), phys_ket(d), right_bond(bR), phys_bra(d)]
//   Right boundary (site N-1):  [left_bond(bL), phys_ket(d), phys_bra(d)]
//
// The bulk MPO matrix (indexed by bond dimensions aL, aR) is:
//
//        |  I      0     0  |
//   W =  |  Z      0     0  |
//        | h*X   -J*Z    I  |
//
// Left boundary row-vector  (d x 3 x d):  W_L = [ h*X  -J*Z  I ]
// Right boundary col-vector (3 x d x d):  W_R = [ I; Z; h*X ]^T
//
// Pauli matrices: I = diag(1,1),  X = [[0,1],[1,0]],  Z = diag(1,-1)

struct IsingMPO {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void*> gpuPtrs;

  void build() {
    const Complex zero{0.0, 0.0}, one{1.0, 0.0};
    const Complex mone{-1.0, 0.0};
    const Complex jc{-J_COUPLING, 0.0};
    const Complex hc{H_FIELD, 0.0};

    // Pauli matrices stored column-major: mat[col*2 + row]
    auto I_mat = [&](int r, int c) -> Complex { return (r == c) ? one : zero; };
    auto X_mat = [&](int r, int c) -> Complex { return (r != c) ? one : zero; };
    auto Z_mat = [&](int r, int c) -> Complex { return (r == c) ? ((r == 0) ? one : mone) : zero; };

    hostTensors.resize(NUM_SITES);
    gpuPtrs.resize(NUM_SITES, nullptr);

    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t bL = (site == 0) ? 1 : MPO_BOND_DIM;
      const int64_t bR = (site == NUM_SITES - 1) ? 1 : MPO_BOND_DIM;
      const int64_t vol = bL * PHYS_DIM * PHYS_DIM * bR;
      hostTensors[site].assign(vol, zero);

      // Library mode ordering (column-major):
      //   [left_bond, phys_ket, right_bond, phys_bra]
      // Boundary sites omit the missing bond; bL=1 or bR=1 makes
      // the degenerate dimension transparent in the flat index.
      auto idx = [&](int64_t aL, int64_t ket, int64_t aR, int64_t bra) -> int64_t {
        return aL + bL * (ket + PHYS_DIM * (aR + bR * bra));
      };

      if (NUM_SITES == 1) {
        for (int bra = 0; bra < PHYS_DIM; ++bra)
          for (int ket = 0; ket < PHYS_DIM; ++ket)
            hostTensors[site][idx(0, ket, 0, bra)] = hc * X_mat(bra, ket);
      } else if (site == 0) {
        // Left boundary: row-vector [h*X, -J*Z, I]
        for (int bra = 0; bra < PHYS_DIM; ++bra)
          for (int ket = 0; ket < PHYS_DIM; ++ket) {
            hostTensors[site][idx(0, ket, 0, bra)] = hc * X_mat(bra, ket);
            hostTensors[site][idx(0, ket, 1, bra)] = jc * Z_mat(bra, ket);
            hostTensors[site][idx(0, ket, 2, bra)] = I_mat(bra, ket);
          }
      } else if (site == NUM_SITES - 1) {
        // Right boundary: col-vector [I; Z; h*X]^T
        for (int bra = 0; bra < PHYS_DIM; ++bra)
          for (int ket = 0; ket < PHYS_DIM; ++ket) {
            hostTensors[site][idx(0, ket, 0, bra)] = I_mat(bra, ket);
            hostTensors[site][idx(1, ket, 0, bra)] = Z_mat(bra, ket);
            hostTensors[site][idx(2, ket, 0, bra)] = hc * X_mat(bra, ket);
          }
      } else {
        // Bulk:
        //   row 0: [I, 0, 0]
        //   row 1: [Z, 0, 0]
        //   row 2: [h*X, -J*Z, I]
        for (int bra = 0; bra < PHYS_DIM; ++bra)
          for (int ket = 0; ket < PHYS_DIM; ++ket) {
            hostTensors[site][idx(0, ket, 0, bra)] = I_mat(bra, ket);
            hostTensors[site][idx(1, ket, 0, bra)] = Z_mat(bra, ket);
            hostTensors[site][idx(2, ket, 0, bra)] = hc * X_mat(bra, ket);
            hostTensors[site][idx(2, ket, 1, bra)] = jc * Z_mat(bra, ket);
            hostTensors[site][idx(2, ket, 2, bra)] = I_mat(bra, ket);
          }
      }

      gpuPtrs[site] = createInitializeArrayGPU(hostTensors[site]);
    }
  }

  void destroy() {
    for (auto & ptr : gpuPtrs) {
      if (ptr) { destroyArrayGPU(ptr); ptr = nullptr; }
    }
  }
};


// ============================================================================
// Build a Neel-state MPS  |0,1,0,1,...>  with given bond dimensions
// ============================================================================
// Each MPS tensor A[site] has shape (bondL, phys, bondR) in column-major.
// For a product state, only the (0,sigma,0) slice is nonzero:
//   A[0,sigma,0] = delta(sigma, site%2)
// Remaining bond indices are zero-padded.

struct NeelMPS {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void*> gpuPtrs;

  void build(const std::vector<int64_t>& bondDims) {
    const Complex zero{0.0, 0.0}, one{1.0, 0.0};

    hostTensors.resize(NUM_SITES);
    gpuPtrs.resize(NUM_SITES, nullptr);

    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t bL = (site == 0) ? 1 : bondDims[site - 1];
      const int64_t bR = (site == NUM_SITES - 1) ? 1 : bondDims[site];
      const int64_t vol = bL * PHYS_DIM * bR;
      hostTensors[site].assign(vol, zero);

      // T[aL, sigma, aR]  column-major
      auto idx = [&](int64_t aL, int64_t sigma, int64_t aR) -> int64_t {
        return aL + bL * (sigma + PHYS_DIM * aR);
      };

      const int64_t neelSpin = site % 2;  // 0 for even sites, 1 for odd
      hostTensors[site][idx(0, neelSpin, 0)] = one;

      gpuPtrs[site] = createInitializeArrayGPU(hostTensors[site]);
    }
  }

  void destroy() {
    for (auto & ptr : gpuPtrs) {
      if (ptr) { destroyArrayGPU(ptr); ptr = nullptr; }
    }
  }
};


// ============================================================================
// Example workflow
// ============================================================================

void exampleWorkflow(cudensitymatHandle_t handle)
{
  // --- 1. Build the transverse-field Ising MPO ---
  IsingMPO mpo;
  mpo.build();
  if (verbose)
    std::cout << "Built transverse-field Ising MPO (bond dim " << MPO_BOND_DIM << ")\n";

  const std::vector<int64_t> spaceShape(NUM_SITES, PHYS_DIM);
  std::vector<int64_t> mpoBondDims(NUM_SITES - 1, MPO_BOND_DIM);

  cudensitymatMatrixProductOperator_t mpoHandle;
  std::vector<cudensitymatWrappedTensorCallback_t> mpoCallbacks(NUM_SITES, cudensitymatTensorCallbackNone);
  std::vector<cudensitymatWrappedTensorGradientCallback_t> mpoGradCallbacks(NUM_SITES, cudensitymatTensorGradientCallbackNone);
  HANDLE_CUDM_ERROR(cudensitymatCreateMatrixProductOperator(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpoBondDims.data(),
                      DATA_TYPE,
                      mpo.gpuPtrs.data(),
                      mpoCallbacks.data(),
                      mpoGradCallbacks.data(),
                      &mpoHandle));
  if (verbose)
    std::cout << "Created MPO handle\n";

  // --- 2. Build the Operator from the MPO ---
  cudensitymatOperatorTerm_t operatorTerm;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      &operatorTerm));

  std::vector<int32_t> modesActedOn(NUM_SITES);
  std::iota(modesActedOn.begin(), modesActedOn.end(), 0);
  std::vector<int32_t> modeDuality(NUM_SITES, 0);
  std::vector<int32_t> mpoConjugation = {0};

  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendMPOProduct(handle,
                      operatorTerm,
                      1,
                      &mpoHandle,
                      mpoConjugation.data(),
                      modesActedOn.data(),
                      modeDuality.data(),
                      make_cuDoubleComplex(1.0, 0.0),
                      cudensitymatScalarCallbackNone,
                      cudensitymatScalarGradientCallbackNone));

  cudensitymatOperator_t hamiltonian;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      &hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                      hamiltonian,
                      operatorTerm,
                      0,
                      make_cuDoubleComplex(1.0, 0.0),
                      cudensitymatScalarCallbackNone,
                      cudensitymatScalarGradientCallbackNone));
  if (verbose)
    std::cout << "Constructed Hamiltonian operator from MPO\n";

  // --- 3. Create input and output MPS states ---
  const int64_t batchSize = 1;

  // For OBC, cap bond dimensions by the exact Hilbert space dimension on each side.
  std::vector<int64_t> mpsBondDims(NUM_SITES - 1);
  for (int32_t i = 0; i < NUM_SITES - 1; ++i) {
    int64_t leftDim = 1;
    for (int32_t j = 0; j <= i; ++j) leftDim *= spaceShape[j];
    int64_t rightDim = 1;
    for (int32_t j = i + 1; j < NUM_SITES; ++j) rightDim *= spaceShape[j];
    mpsBondDims[i] = std::min({MAX_BOND_DIM, leftDim, rightDim});
  }

  cudensitymatState_t stateIn, stateOut;
  HANDLE_CUDM_ERROR(cudensitymatCreateStateMPS(handle,
                      CUDENSITYMAT_STATE_PURITY_PURE,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpsBondDims.data(),
                      DATA_TYPE,
                      batchSize,
                      &stateIn));
  HANDLE_CUDM_ERROR(cudensitymatCreateStateMPS(handle,
                      CUDENSITYMAT_STATE_PURITY_PURE,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpsBondDims.data(),
                      DATA_TYPE,
                      batchSize,
                      &stateOut));

  // Query number of MPS components (= NUM_SITES tensors)
  int32_t numComponents = 0;
  HANDLE_CUDM_ERROR(cudensitymatStateGetNumComponents(handle, stateIn, &numComponents));
  assert(numComponents == NUM_SITES);

  // Query storage sizes for each MPS tensor
  std::vector<std::size_t> componentSizes(numComponents);
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(handle,
                      stateIn, numComponents, componentSizes.data()));

  if (verbose) {
    std::cout << "MPS state has " << numComponents << " components, sizes (bytes):";
    for (auto s : componentSizes) std::cout << " " << s;
    std::cout << "\n";
  }

  // --- 4. Allocate GPU storage and initialise Neel MPS for stateIn ---
  NeelMPS neelMps;
  neelMps.build(mpsBondDims);

  // Allocate empty storage for stateOut
  std::vector<void*> stateOutBufs(numComponents, nullptr);
  for (int32_t c = 0; c < numComponents; ++c)
    stateOutBufs[c] = createArrayGPU<Complex>(componentSizes[c] / sizeof(Complex));

  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      stateIn, numComponents,
                      neelMps.gpuPtrs.data(), componentSizes.data()));
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      stateOut, numComponents,
                      stateOutBufs.data(), componentSizes.data()));

  if (verbose)
    std::cout << "Initialized MPS states (Neel state |0101...>)\n";

  // --- 5. Create TDVP time propagation object ---
  cudensitymatTimePropagation_t timeProp;
  HANDLE_CUDM_ERROR(cudensitymatCreateTimePropagation(handle,
                      hamiltonian,
                      1,  // Hermitian
                      CUDENSITYMAT_PROPAGATION_SCOPE_SPLIT,
                      CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV,
                      &timeProp));
  if (verbose)
    std::cout << "Created TDVP time propagation object\n";

  // --- 6. Configure TDVP and Krylov parameters ---
  // TDVP config: 2nd-order, 1-site
  cudensitymatTimePropagationScopeSplitTDVPConfig_t tdvpConfig;
  HANDLE_CUDM_ERROR(cudensitymatCreateTimePropagationScopeSplitTDVPConfig(handle, &tdvpConfig));
  {
    const int32_t order = 2;
    HANDLE_CUDM_ERROR(cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute(handle,
                        tdvpConfig,
                        CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_TDVP_ORDER,
                        &order, sizeof(order)));
  }
  HANDLE_CUDM_ERROR(cudensitymatTimePropagationConfigure(handle,
                      timeProp,
                      CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_TDVP_CONFIG,
                      &tdvpConfig, sizeof(tdvpConfig)));

  // Krylov config
  cudensitymatTimePropagationApproachKrylovConfig_t krylovConfig;
  HANDLE_CUDM_ERROR(cudensitymatCreateTimePropagationApproachKrylovConfig(handle, &krylovConfig));
  {
    const int32_t maxDim = 10;
    HANDLE_CUDM_ERROR(cudensitymatTimePropagationApproachKrylovConfigSetAttribute(handle,
                        krylovConfig,
                        CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_MAX_DIM,
                        &maxDim, sizeof(maxDim)));
    const double tol = 1e-8;
    HANDLE_CUDM_ERROR(cudensitymatTimePropagationApproachKrylovConfigSetAttribute(handle,
                        krylovConfig,
                        CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_TOLERANCE,
                        &tol, sizeof(tol)));
  }
  HANDLE_CUDM_ERROR(cudensitymatTimePropagationConfigure(handle,
                      timeProp,
                      CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_CONFIG,
                      &krylovConfig, sizeof(krylovConfig)));
  if (verbose)
    std::cout << "Configured TDVP (order=2, 1-site) with Krylov (max_dim=10, tol=1e-8)\n";

  // --- 7. Prepare propagation and allocate workspace ---
  cudensitymatWorkspaceDescriptor_t workspaceDescr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspaceDescr));

  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.95);
  if (verbose)
    std::cout << "Available workspace memory (bytes) = " << freeMem << "\n";

  HANDLE_CUDM_ERROR(cudensitymatTimePropagationPrepare(handle,
                      timeProp,
                      stateIn,
                      stateOut,
                      CUDENSITYMAT_COMPUTE_64F,
                      freeMem,
                      workspaceDescr,
                      0x0));
  if (verbose)
    std::cout << "Prepared time propagation\n";

  // Query and allocate scratch workspace
  std::size_t scratchSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(handle,
                      workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      &scratchSize));
  void * scratchBuf = nullptr;
  if (scratchSize > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&scratchBuf, scratchSize));
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(handle,
                        workspaceDescr,
                        CUDENSITYMAT_MEMSPACE_DEVICE,
                        CUDENSITYMAT_WORKSPACE_SCRATCH,
                        scratchBuf, scratchSize));
  }
  if (verbose)
    std::cout << "Scratch workspace (bytes) = " << scratchSize << "\n";

  // --- 8. Time-stepping loop ---
  if (verbose) {
    std::cout << "\nStarting TDVP propagation: " << NUM_STEPS << " steps, dt = " << DT << "\n";
    std::cout << "Hamiltonian: H = -" << J_COUPLING << " sum Z_i Z_{i+1} + "
              << H_FIELD << " sum X_i\n\n";
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  const auto wallStart = std::chrono::high_resolution_clock::now();

  for (int32_t step = 0; step < NUM_STEPS; ++step) {
    const double currentTime = step * DT;

    HANDLE_CUDM_ERROR(cudensitymatTimePropagationCompute(handle,
                        timeProp,
                        DT,      // timeStepReal
                        0.0,     // timeStepImag
                        currentTime,
                        batchSize,
                        0,       // numParams
                        nullptr, // params
                        stateIn,
                        stateOut,
                        workspaceDescr,
                        0x0));

    // Swap input/output for next step (double-buffering)
    std::swap(stateIn, stateOut);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  const auto wallEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed = wallEnd - wallStart;
  if (verbose)
    std::cout << "\nTotal propagation wall time (sec) = " << elapsed.count() << "\n";

  // --- 9. Clean up ---
  if (scratchBuf)
    HANDLE_CUDA_ERROR(cudaFree(scratchBuf));
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspaceDescr));
  HANDLE_CUDM_ERROR(cudensitymatDestroyTimePropagation(timeProp));
  HANDLE_CUDM_ERROR(cudensitymatDestroyTimePropagationApproachKrylovConfig(krylovConfig));
  HANDLE_CUDM_ERROR(cudensitymatDestroyTimePropagationScopeSplitTDVPConfig(tdvpConfig));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(operatorTerm));
  HANDLE_CUDM_ERROR(cudensitymatDestroyMatrixProductOperator(mpoHandle));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(stateOut));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(stateIn));

  for (auto buf : stateOutBufs)
    destroyArrayGPU(buf);
  neelMps.destroy();
  mpo.destroy();

  if (verbose)
    std::cout << "Destroyed all resources\n";
}


int main(int argc, char ** argv)
{
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  if (verbose)
    std::cout << "Set active device\n";

  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
  if (verbose)
    std::cout << "Created library handle\n";

  exampleWorkflow(handle);

  HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  if (verbose)
    std::cout << "Destroyed library handle\n";

  HANDLE_CUDA_ERROR(cudaDeviceReset());
  return 0;
}
