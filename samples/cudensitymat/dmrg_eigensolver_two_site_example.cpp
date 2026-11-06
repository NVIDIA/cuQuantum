/* Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Two-site DMRG eigensolver example.
//
// Computes the ground state of the 1-D transverse-field Ising model
//   H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i   (open boundary)
// with the 2-site DMRG solver (SCOPE_SPLIT + APPROACH_KRYLOV + SMALLEST_REAL)
// on an MPS state.  Two adjacent MPS sites are merged, locally eigensolved,
// SVD-split with adaptive bond truncation, and synced back per step.
//
// The chain is blocked: each MPS site carries a pair of qubits (phys dim 4,
// basis {|00>, |01>, |10>, |11>}), so 3 MPS sites model a 6-qubit chain.  A
// small SVD MAX_EXTENT cap and a partial-fill initial bond layout (current
// bond extents strictly between 1 and the buffer maximum) exercise the 2-site
// memory-layout and adaptive-growth contracts.
//
// 3 MPS sites is intentionally small: both bonds are boundary-adjacent.

// Workflow:
//  1. Build the blocked-TFIM MPO (bond dim 3)
//  2. Wrap the MPO into a Hamiltonian operator
//  3. Create an initial MPS state (noisy product) + partial-fill bond layout
//  4. Create the EigenDecomposition (SMALLEST_REAL + SCOPE_SPLIT + KRYLOV)
//  5. Configure DMRG (NUM_SITES=2 + SVD config), Krylov, Prepare, attach
//  6. Compute the ground-state pair
//  7. Report the energy and the adapted (current vs max) bond extents
//  8. Clean up

#include <cudensitymat.h>
#include "helpers.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <cassert>


using Complex = std::complex<double>;
constexpr cudaDataType_t kDataType = CUDA_C_64F;

constexpr bool verbose = true;

// --- Simulation parameters ---
constexpr int32_t  NUM_SITES   = 3;     // blocked MPS sites (== 6 qubits); small by design
constexpr int64_t  PHYS_DIM    = 4;     // 2 qubits per blocked site
constexpr int64_t  MPO_BOND    = 3;     // standard TFIM MPO bond dim
constexpr int64_t  MAX_BOND    = 8;     // MPS bond-dim buffer cap
constexpr int32_t  NUM_DMRG_SITES = 2;  // 2-site DMRG
constexpr int64_t  SVD_MAX_EXTENT = 3;  // modest SVD truncation cap (< buffer extent)
constexpr int64_t  INIT_CURRENT_BOND = 2; // partial fill: strictly between 1 and the maximum
constexpr int32_t  MAX_SWEEPS  = 30;
constexpr int32_t  KRYLOV_MAX_DIM = 5;  // local Krylov subspace cap
constexpr double   ENERGY_TOL  = 1e-10;
constexpr double   RESIDUAL    = 1e-6;
constexpr double   NOISE_EPS   = 1e-2;
constexpr uint64_t RNG_SEED    = 1729;

constexpr double   J_COUPLING  = 1.0;   // ZZ coupling
constexpr double   H_FIELD     = 0.5;   // transverse field

// Exact ground-state energy for the 6-qubit OBC TFIM (J=1, h=0.5), from dense
// diagonalization of the 2^6 x 2^6 Hamiltonian.  Printed reference.
constexpr double   E0_EXACT    = -5.522029570800221;


// ============================================================================
// Blocked-TFIM MPO  (phys dim 4 per site, bond dimension 3)
// ============================================================================
//
// Each MPS site carries two qubits (q_left, q_right), basis index i = 2*q_left
// + q_right.  The W-matrix carries the standard 3-state TFIM finite automaton:
// column 0 holds the finish operators, the bottom row the start operators, plus
// the on-site block term B = -J*ZZ - h*(Xleft + Xright).
// Site tensor layout (aL, ket, aR, bra): idx = aL + bL*(ket + d*(aR + bR*bra)).

struct BlockedTFIMMPO {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void*> gpuPtrs;

  // 4x4 operators on a 2-qubit block; entry (bra, ket).
  static Complex I4_(int r, int c) { return (r == c) ? Complex(1, 0) : Complex(0, 0); }
  static Complex Zleft_(int r, int c) {
    if (r != c) return Complex(0, 0);
    return (r < 2) ? Complex(1, 0) : Complex(-1, 0);     // Z on first qubit: diag(+,+,-,-)
  }
  static Complex Zright_(int r, int c) {
    if (r != c) return Complex(0, 0);
    return ((r & 1) == 0) ? Complex(1, 0) : Complex(-1, 0);   // Z on second qubit: diag(+,-,+,-)
  }
  static Complex ZZ_(int r, int c) {
    if (r != c) return Complex(0, 0);
    const int rL = (r >> 1) & 1, rR = r & 1;
    return (rL == rR) ? Complex(1, 0) : Complex(-1, 0);  // Z⊗Z: diag(+,-,-,+)
  }
  static Complex Xleft_(int r, int c)  { return (r == (c ^ 0b10)) ? Complex(1, 0) : Complex(0, 0); }
  static Complex Xright_(int r, int c) { return (r == (c ^ 0b01)) ? Complex(1, 0) : Complex(0, 0); }
  // Within-block on-site Hamiltonian B = -J*ZZ - h*(Xleft + Xright).
  static Complex B_(int r, int c) {
    return -Complex(J_COUPLING, 0) * ZZ_(r, c)
           - Complex(H_FIELD, 0) * (Xleft_(r, c) + Xright_(r, c));
  }

  void build() {
    const Complex zero{0.0, 0.0};
    hostTensors.assign(NUM_SITES, {});
    gpuPtrs.assign(NUM_SITES, nullptr);

    // Left boundary site 0 — shape (bL=1, phys=4, bR=3, phys=4).
    {
      const int64_t bL = 1, bR = MPO_BOND;
      hostTensors[0].assign(bL * PHYS_DIM * bR * PHYS_DIM, zero);
      auto idx = [&](int64_t aL, int64_t ket, int64_t aR, int64_t bra) {
        return aL + bL * (ket + PHYS_DIM * (aR + bR * bra));
      };
      for (int bra = 0; bra < PHYS_DIM; ++bra) {
        for (int ket = 0; ket < PHYS_DIM; ++ket) {
          hostTensors[0][idx(0, ket, 0, bra)] = B_(bra, ket);
          hostTensors[0][idx(0, ket, 1, bra)] = Zright_(bra, ket);
          hostTensors[0][idx(0, ket, 2, bra)] = I4_(bra, ket);
        }
      }
      gpuPtrs[0] = createInitializeArrayGPU(hostTensors[0]);
    }

    // Bulk sites — shape (bL=3, phys=4, bR=3, phys=4).
    for (int32_t site = 1; site < NUM_SITES - 1; ++site) {
      const int64_t bL = MPO_BOND, bR = MPO_BOND;
      hostTensors[site].assign(bL * PHYS_DIM * bR * PHYS_DIM, zero);
      auto idx = [&](int64_t aL, int64_t ket, int64_t aR, int64_t bra) {
        return aL + bL * (ket + PHYS_DIM * (aR + bR * bra));
      };
      for (int bra = 0; bra < PHYS_DIM; ++bra) {
        for (int ket = 0; ket < PHYS_DIM; ++ket) {
          hostTensors[site][idx(0, ket, 0, bra)] = I4_(bra, ket);
          hostTensors[site][idx(1, ket, 0, bra)] = -Complex(J_COUPLING, 0) * Zleft_(bra, ket);
          hostTensors[site][idx(2, ket, 0, bra)] = B_(bra, ket);
          hostTensors[site][idx(2, ket, 1, bra)] = Zright_(bra, ket);
          hostTensors[site][idx(2, ket, 2, bra)] = I4_(bra, ket);
        }
      }
      gpuPtrs[site] = createInitializeArrayGPU(hostTensors[site]);
    }

    // Right boundary site N-1 — shape (bL=3, phys=4, bR=1, phys=4).
    {
      const int32_t last = NUM_SITES - 1;
      const int64_t bL = MPO_BOND, bR = 1;
      hostTensors[last].assign(bL * PHYS_DIM * bR * PHYS_DIM, zero);
      auto idx = [&](int64_t aL, int64_t ket, int64_t aR, int64_t bra) {
        return aL + bL * (ket + PHYS_DIM * (aR + bR * bra));
      };
      for (int bra = 0; bra < PHYS_DIM; ++bra) {
        for (int ket = 0; ket < PHYS_DIM; ++ket) {
          hostTensors[last][idx(0, ket, 0, bra)] = I4_(bra, ket);
          hostTensors[last][idx(1, ket, 0, bra)] = -Complex(J_COUPLING, 0) * Zleft_(bra, ket);
          hostTensors[last][idx(2, ket, 0, bra)] = B_(bra, ket);
        }
      }
      gpuPtrs[last] = createInitializeArrayGPU(hostTensors[last]);
    }
  }

  void destroy() {
    for (auto & ptr : gpuPtrs) {
      if (ptr) { destroyArrayGPU(ptr); ptr = nullptr; }
    }
  }
};


// ============================================================================
// Noisy product initial MPS  ( |00,00,...> + epsilon * Gaussian noise )
// ============================================================================
//
// A low-energy ferromagnetic product seed gives DMRG a generic, non-exact
// starting point.

struct NoisyProductMPS {

  std::vector<std::vector<Complex>> hostTensors;
  std::vector<void*> gpuPtrs;

  void build(const std::vector<int64_t>& bondDims, double epsilon, uint64_t seed) {
    const Complex zero{0.0, 0.0}, one{1.0, 0.0};
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, epsilon);

    hostTensors.assign(NUM_SITES, {});
    gpuPtrs.assign(NUM_SITES, nullptr);

    for (int32_t site = 0; site < NUM_SITES; ++site) {
      const int64_t bL  = (site == 0) ? 1 : bondDims[site - 1];
      const int64_t bR  = (site == NUM_SITES - 1) ? 1 : bondDims[site];
      const int64_t vol = bL * PHYS_DIM * bR;
      hostTensors[site].assign(vol, zero);

      auto idx = [&](int64_t aL, int64_t sigma, int64_t aR) {
        return aL + bL * (sigma + PHYS_DIM * aR);
      };

      // Product seed: every block in qubit-pair state |00> (index 0).
      hostTensors[site][idx(0, 0, 0)] = one;
      // Per-element Gaussian noise (Re, Im) iid N(0, epsilon^2).
      for (auto & v : hostTensors[site]) {
        v += Complex(noise(rng), noise(rng));
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
// Example workflow
// ============================================================================

void exampleWorkflow(cudensitymatHandle_t handle)
{
  if (verbose) {
    std::cout << "Two-site DMRG eigensolver example\n"
              << "  Model:           blocked transverse-field Ising chain (phys dim 4)\n"
              << "  H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i  (open boundary)\n"
              << "  N (MPS sites)   = " << NUM_SITES      << "   (== " << 2 * NUM_SITES << " qubits)\n"
              << "  d (phys dim)    = " << PHYS_DIM       << "\n"
              << "  J (coupling)    = " << J_COUPLING     << "\n"
              << "  h (field)       = " << H_FIELD        << "\n"
              << "  chi (bond cap)  = " << MAX_BOND       << "\n"
              << "  DMRG num_sites  = " << NUM_DMRG_SITES << "\n"
              << "  SVD max_extent  = " << SVD_MAX_EXTENT << "\n"
              << "  max sweeps      = " << MAX_SWEEPS     << "\n"
              << "  energy tol      = " << ENERGY_TOL     << "\n\n";
  }

  // --- 1. Build the blocked-TFIM MPO ---
  BlockedTFIMMPO mpo;
  mpo.build();
  if (verbose) std::cout << "Built blocked-TFIM MPO (bond dim " << MPO_BOND << ")\n";

  const std::vector<int64_t> spaceShape(NUM_SITES, PHYS_DIM);
  std::vector<int64_t> mpoBondDims(NUM_SITES - 1, MPO_BOND);

  cudensitymatMatrixProductOperator_t mpoHandle = nullptr;
  std::vector<cudensitymatWrappedTensorCallback_t> mpoCB(
      NUM_SITES, cudensitymatTensorCallbackNone);
  std::vector<cudensitymatWrappedTensorGradientCallback_t> mpoGCB(
      NUM_SITES, cudensitymatTensorGradientCallbackNone);
  HANDLE_CUDM_ERROR(cudensitymatCreateMatrixProductOperator(handle,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpoBondDims.data(),
                      kDataType,
                      mpo.gpuPtrs.data(),
                      mpoCB.data(),
                      mpoGCB.data(),
                      &mpoHandle));
  if (verbose) std::cout << "Created MPO handle\n";

  // --- 2. Wrap the MPO into an Operator (one term, one MPO product) ---
  cudensitymatOperatorTerm_t operatorTerm = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                      NUM_SITES, spaceShape.data(), &operatorTerm));

  std::vector<int32_t> modesActedOn(NUM_SITES);
  std::iota(modesActedOn.begin(), modesActedOn.end(), 0);
  std::vector<int32_t> modeDuality(NUM_SITES, 0);
  std::vector<int32_t> mpoConj = {0};
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendMPOProduct(handle,
                      operatorTerm,
                      1, &mpoHandle, mpoConj.data(),
                      modesActedOn.data(), modeDuality.data(),
                      make_cuDoubleComplex(1.0, 0.0),
                      cudensitymatScalarCallbackNone,
                      cudensitymatScalarGradientCallbackNone));

  cudensitymatOperator_t hamiltonian = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle,
                      NUM_SITES, spaceShape.data(), &hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                      hamiltonian, operatorTerm, 0,
                      make_cuDoubleComplex(1.0, 0.0),
                      cudensitymatScalarCallbackNone,
                      cudensitymatScalarGradientCallbackNone));
  if (verbose) std::cout << "Created Hamiltonian operator from MPO\n";

  // --- 3. Build the initial MPS state ---
  //
  // Buffer (maximum) bond dims are capped at the natural Schmidt-rank bound.
  std::vector<int64_t> mpsBondDims(NUM_SITES - 1);
  for (int32_t i = 0; i < NUM_SITES - 1; ++i) {
    int64_t leftDim = 1;
    for (int32_t j = 0; j <= i; ++j) leftDim *= spaceShape[j];
    int64_t rightDim = 1;
    for (int32_t j = i + 1; j < NUM_SITES; ++j) rightDim *= spaceShape[j];
    mpsBondDims[i] = std::min<int64_t>({MAX_BOND, leftDim, rightDim});
  }

  cudensitymatState_t mpsState = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateStateMPS(handle,
                      CUDENSITYMAT_STATE_PURITY_PURE,
                      NUM_SITES,
                      spaceShape.data(),
                      CUDENSITYMAT_BOUNDARY_CONDITION_OPEN,
                      mpsBondDims.data(),
                      kDataType,
                      /*batchSize=*/1,
                      &mpsState));

  NoisyProductMPS mps;
  mps.build(mpsBondDims, NOISE_EPS, RNG_SEED);

  int32_t numComponents = 0;
  HANDLE_CUDM_ERROR(cudensitymatStateGetNumComponents(handle, mpsState, &numComponents));
  std::vector<std::size_t> componentSizes(numComponents);
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(handle, mpsState,
                      numComponents, componentSizes.data()));
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle, mpsState,
                      numComponents, mps.gpuPtrs.data(), componentSizes.data()));
  if (verbose) std::cout << "Built and attached noisy-product MPS (seed=" << RNG_SEED << ")\n";

  // --- 3b. Partial-fill bond layout: set initial CURRENT bond extents strictly
  // between 1 and the buffer maximum, exercising the 2-site partial-fill layout.
  const int32_t numBonds = NUM_SITES - 1;
  std::vector<int64_t> initialCurrentBonds(numBonds, INIT_CURRENT_BOND);
  HANDLE_CUDM_ERROR(cudensitymatStateMPSSetCurrentBondExtents(handle, mpsState,
                      initialCurrentBonds.data()));
  if (verbose) {
    std::cout << "Set initial current bond extents (partial fill) = {";
    for (int32_t b = 0; b < numBonds; ++b)
      std::cout << initialCurrentBonds[b] << (b + 1 < numBonds ? "," : "");
    std::cout << "}  (buffer max = {";
    for (int32_t b = 0; b < numBonds; ++b)
      std::cout << mpsBondDims[b] << (b + 1 < numBonds ? "," : "");
    std::cout << "})\n";
  }

  // --- 4. Create EigenDecomposition (SCOPE_SPLIT + APPROACH_KRYLOV + SMALLEST_REAL) ---
  cudensitymatEigenDecomposition_t eigen = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateEigenDecomposition(handle,
                      hamiltonian,
                      /*isHermitian=*/1,
                      CUDENSITYMAT_EIGEN_SPECTRUM_SMALLEST_REAL,
                      CUDENSITYMAT_EIGEN_SCOPE_SPLIT,
                      CUDENSITYMAT_EIGEN_APPROACH_KRYLOV,
                      &eigen));
  if (verbose) std::cout << "Created EigenDecomposition object\n";

  // --- 5. Configure DMRG attributes (NUM_SITES=2, SVD config, MAX_SWEEPS, ENERGY_TOLERANCE).
  // Sub-config values are deep-copied at Configure, so the handles can be freed afterward.
  cudensitymatEigenDecompositionScopeSplitDMRGConfig_t dmrgCfg = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateEigenDecompositionScopeSplitDMRGConfig(handle, &dmrgCfg));
  {
    const int32_t dmrgNumSites = NUM_DMRG_SITES;
    HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
        handle, dmrgCfg,
        CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_NUM_SITES,
        &dmrgNumSites, sizeof(dmrgNumSites)));

    // SVD config: a modest MAX_EXTENT cap plus value cutoffs. The cap is below
    // the buffer maximum, so the truncated current bond extents adapt below it.
    cudensitymatSVDConfig_t svd = nullptr;
    HANDLE_CUDM_ERROR(cudensitymatCreateSVDConfig(handle, &svd));
    const int64_t maxExtent = SVD_MAX_EXTENT;
    const double absCutoff = 1e-12, relCutoff = 1e-12;
    HANDLE_CUDM_ERROR(cudensitymatSVDConfigSetAttribute(handle, svd,
        CUDENSITYMAT_SVD_CONFIG_MAX_EXTENT, &maxExtent, sizeof(maxExtent)));
    HANDLE_CUDM_ERROR(cudensitymatSVDConfigSetAttribute(handle, svd,
        CUDENSITYMAT_SVD_CONFIG_ABS_CUTOFF, &absCutoff, sizeof(absCutoff)));
    HANDLE_CUDM_ERROR(cudensitymatSVDConfigSetAttribute(handle, svd,
        CUDENSITYMAT_SVD_CONFIG_REL_CUTOFF, &relCutoff, sizeof(relCutoff)));
    HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
        handle, dmrgCfg,
        CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_SVD_CONFIG, &svd, sizeof(svd)));
    HANDLE_CUDM_ERROR(cudensitymatDestroySVDConfig(svd));   // safe once attached (deep-cloned)

    const int32_t maxSweeps = MAX_SWEEPS;
    HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
        handle, dmrgCfg,
        CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_MAX_SWEEPS,
        &maxSweeps, sizeof(maxSweeps)));
    const double energyTol = ENERGY_TOL;
    HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionScopeSplitDMRGConfigSetAttribute(
        handle, dmrgCfg,
        CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_ENERGY_TOLERANCE,
        &energyTol, sizeof(energyTol)));
  }
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionConfigure(handle, eigen,
                      CUDENSITYMAT_EIGEN_SPLIT_SCOPE_DMRG_CONFIG,
                      &dmrgCfg, sizeof(dmrgCfg)));
  HANDLE_CUDM_ERROR(cudensitymatDestroyEigenDecompositionScopeSplitDMRGConfig(dmrgCfg));
  dmrgCfg = nullptr;
  if (verbose) std::cout << "Configured DMRG sub-config (num_sites=" << NUM_DMRG_SITES
                         << ", svd_max_extent=" << SVD_MAX_EXTENT
                         << ", max_sweeps=" << MAX_SWEEPS << ", energy_tol=" << ENERGY_TOL << ")\n";

  // --- 5b. Configure the Krylov sub-config.
  cudensitymatEigenDecompositionApproachKrylovConfig_t krylovCfg = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateEigenDecompositionApproachKrylovConfig(handle, &krylovCfg));
  {
    const int32_t krylovMaxDim = KRYLOV_MAX_DIM;
    HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionApproachKrylovConfigSetAttribute(
        handle, krylovCfg,
        CUDENSITYMAT_EIGEN_APPROACH_KRYLOV_MAX_DIM,
        &krylovMaxDim, sizeof(krylovMaxDim)));
  }
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionConfigure(handle, eigen,
                      CUDENSITYMAT_EIGEN_APPROACH_KRYLOV_CONFIG,
                      &krylovCfg, sizeof(krylovCfg)));
  HANDLE_CUDM_ERROR(cudensitymatDestroyEigenDecompositionApproachKrylovConfig(krylovCfg));
  krylovCfg = nullptr;
  if (verbose) std::cout << "Configured Krylov sub-config (max_dim=" << KRYLOV_MAX_DIM << ")\n";

  // --- 6. Prepare + attach workspace ---
  cudensitymatWorkspaceDescriptor_t workspaceDescr = nullptr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspaceDescr));

  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.85);
  if (verbose) std::cout << "Available workspace memory (bytes) = " << freeMem << "\n";

  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionPrepare(handle, eigen,
                      /*maxEigenStates=*/1, mpsState, CUDENSITYMAT_COMPUTE_64F,
                      freeMem, workspaceDescr, /*stream=*/0x0));

  std::size_t scratchSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(handle, workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      &scratchSize));
  void * scratchBuf = nullptr;
  if (scratchSize > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&scratchBuf, scratchSize));
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(handle, workspaceDescr,
                        CUDENSITYMAT_MEMSPACE_DEVICE,
                        CUDENSITYMAT_WORKSPACE_SCRATCH,
                        scratchBuf, scratchSize));
  }
  // Provide the host scratch workspace reported by the prepare call, if any.
  std::size_t hostScratchSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(handle, workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_HOST,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      &hostScratchSize));
  void* hostScratchBuf = nullptr;
  if (hostScratchSize > 0) {
    hostScratchBuf = std::malloc(hostScratchSize);
    if (hostScratchBuf == nullptr) {
      std::cerr << "Failed to allocate " << hostScratchSize << " bytes of host scratch workspace\n";
      std::abort();
    }
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(handle, workspaceDescr,
                        CUDENSITYMAT_MEMSPACE_HOST,
                        CUDENSITYMAT_WORKSPACE_SCRATCH,
                        hostScratchBuf, hostScratchSize));
  }
  if (verbose) std::cout << "Prepared DMRG plan; scratch workspace = " << scratchSize
                         << " bytes (device), " << hostScratchSize << " bytes (host)\n";

  // --- 7. Compute the ground-state pair (eigenvalue, MPS) ---
  void * eigenvalueGpu = createArrayGPU<Complex>(1);
  double residual = RESIDUAL;
  cudensitymatState_t eigenstatesArr[1] = { mpsState };

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDM_ERROR(cudensitymatEigenDecompositionCompute(handle, eigen,
                      /*time=*/0.0,
                      /*batchSize=*/1,
                      /*numParams=*/0, /*params=*/nullptr,
                      /*numEigenStates=*/1, eigenstatesArr,
                      eigenvalueGpu, &residual,
                      workspaceDescr, /*stream=*/0x0));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  cuDoubleComplex eigenvalueHost{0.0, 0.0};
  HANDLE_CUDA_ERROR(cudaMemcpy(&eigenvalueHost, eigenvalueGpu,
                               sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  // Query the adapted (current) bond extents after Compute.
  std::vector<int64_t> finalCurrentBonds(numBonds, 0);
  HANDLE_CUDM_ERROR(cudensitymatStateMPSGetCurrentBondExtents(handle, mpsState,
                      finalCurrentBonds.data()));

  const double eDmrg     = eigenvalueHost.x;
  const double energyErr = std::abs(eDmrg - E0_EXACT);
  const bool   converged = (residual < RESIDUAL);

  std::cout << "\n=================================================================\n"
            << "DMRG ground-state energy : " << std::scientific << std::setprecision(12)
            << eDmrg << "\n"
            << "Exact (dense ED, N=" << 2 * NUM_SITES << " qubits) : " << E0_EXACT << "\n"
            << "Energy error             : " << std::scientific << std::setprecision(3)
            << energyErr << "\n"
            << "Final residual           : " << residual << "\n";
  std::cout << "Bond extents (current/max):";
  for (int32_t b = 0; b < numBonds; ++b)
    std::cout << " " << finalCurrentBonds[b] << "/" << mpsBondDims[b];
  std::cout << "\n"
            << "Result                   : " << (converged ? "PASS" : "FAIL") << "\n"
            << "=================================================================\n\n";

  // --- 8. Cleanup (reverse Create order) ---
  destroyArrayGPU(eigenvalueGpu);
  if (scratchBuf) HANDLE_CUDA_ERROR(cudaFree(scratchBuf));
  if (hostScratchBuf) std::free(hostScratchBuf);
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspaceDescr));
  HANDLE_CUDM_ERROR(cudensitymatDestroyEigenDecomposition(eigen));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(mpsState));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(hamiltonian));
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(operatorTerm));
  HANDLE_CUDM_ERROR(cudensitymatDestroyMatrixProductOperator(mpoHandle));
  mps.destroy();
  mpo.destroy();

  if (verbose) std::cout << "Destroyed all resources\n";
}


int main(int /*argc*/, char ** /*argv*/)
{
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  if (verbose) std::cout << "Set active CUDA device 0\n";

  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
  if (verbose) std::cout << "Created cuDensityMat library handle\n";

  exampleWorkflow(handle);

  HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  if (verbose) std::cout << "Destroyed cuDensityMat library handle\n";

  HANDLE_CUDA_ERROR(cudaDeviceReset());
  return 0;
}
