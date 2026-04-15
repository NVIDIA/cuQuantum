/*
 * Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 * </blockquote>}
 */

/**
 * @file
 * \brief This file contains all public declarations of the cuDensityMat library.
 */

#pragma once

#include <library_types.h>    // CUDA data types
#include <cuComplex.h>        // CUDA complex numbers
#include <cuda_runtime_api.h> // CUDA runtime API
#include <stdint.h>           // C integer types

// LIBRARY VERSION

#define CUDENSITYMAT_MAJOR 0 //!< cuDensityMat major version.
#define CUDENSITYMAT_MINOR 5 //!< cuDensityMat minor version.
#define CUDENSITYMAT_PATCH 1 //!< cuDensityMat patch version.
#define CUDENSITYMAT_VERSION (CUDENSITYMAT_MAJOR * 10000 + CUDENSITYMAT_MINOR * 100 + CUDENSITYMAT_PATCH)


// MACRO CONSTANTS

/**
 * \brief The maximal length of the name for a user-provided memory pool.
 */
#define CUDENSITYMAT_ALLOCATOR_NAME_LEN 64


#if defined(__cplusplus)
#include <cstdint>
#include <cstdio>

extern "C" {
#else
#include <stdint.h>
#include <stdio.h>

#endif // defined(__cplusplus)


// CONSTANTS AND ENUMERATIONS

/**
 * \defgroup constenums Constants and Enumerations
 * \{
 */

/**
 * \brief Return status of the library API functions.
 *
 * \details All library API functions return a status
 * which can take one of the following values.
 */
typedef enum
{
  /** The operation has completed successfully. */
  CUDENSITYMAT_STATUS_SUCCESS                   = 0,
  /** The library is not initialized. */
  CUDENSITYMAT_STATUS_NOT_INITIALIZED           = 1,
  /** Resource allocation failed inside the library. */
  CUDENSITYMAT_STATUS_ALLOC_FAILED              = 3,
  /** An invalid parameter value was passed to a function (normally indicates a user error). */
  CUDENSITYMAT_STATUS_INVALID_VALUE             = 7,
  /** The GPU device is either not ready or the target architecture is not supported. */
  CUDENSITYMAT_STATUS_ARCH_MISMATCH             = 8,
  /** The GPU program failed to execute. This is often caused by a CUDA kernel launch failure on the GPU. */
  CUDENSITYMAT_STATUS_EXECUTION_FAILED          = 13,
  /** An internal library error has occurred. */
  CUDENSITYMAT_STATUS_INTERNAL_ERROR            = 14,
  /** The requested operation is not supported. */
  CUDENSITYMAT_STATUS_NOT_SUPPORTED             = 15,
  /** An error occurred inside a user callback function. */
  CUDENSITYMAT_STATUS_CALLBACK_ERROR            = 16,
  /** A call to the cuBLAS library did not succeed. */
  CUDENSITYMAT_STATUS_CUBLAS_ERROR              = 17,
  /** An unknown CUDA error has occurred. */
  CUDENSITYMAT_STATUS_CUDA_ERROR                = 18,
  /** The provided workspace buffer is insufficient. */
  CUDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE    = 19,
  /** The CUDA driver version is insufficient. */
  CUDENSITYMAT_STATUS_INSUFFICIENT_DRIVER       = 20,
  /** An error occurred during file I/O. */
  CUDENSITYMAT_STATUS_IO_ERROR                  = 21,
  /** The dynamically linked cuTENSOR library is incompatible. */
  CUDENSITYMAT_STATUS_CUTENSOR_VERSION_MISMATCH = 22,
  /** Drawing GPU device memory from a memory pool is requested, but the memory pool has not been set. */
  CUDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR       = 23,
  /** A call to the cuTENSOR library did not succeed. */
  CUDENSITYMAT_STATUS_CUTENSOR_ERROR            = 24,
  /** A call to the cuSOLVER library did not succeed. */
  CUDENSITYMAT_STATUS_CUSOLVER_ERROR            = 25,
  /** GPU device memory pool operation failure. */
  CUDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR    = 26,
  /** Distributed communication service failure. */
  CUDENSITYMAT_STATUS_DISTRIBUTED_FAILURE       = 27,
  /** Operation interrupted by the user and cannot recover or complete. */
  CUDENSITYMAT_STATUS_INTERRUPTED               = 28,
  /** A call to the cuTensorNet library did not succeed. */
  CUDENSITYMAT_STATUS_CUTENSORNET_ERROR         = 29,
} cudensitymatStatus_t;

/**
 * @brief Supported compute types.
*/
typedef enum
{
  CUDENSITYMAT_COMPUTE_32F = (1U << 2U),  ///< single-precision floating-point compute type
  CUDENSITYMAT_COMPUTE_64F = (1U << 4U),  ///< double-precision floating-point compute type
} cudensitymatComputeType_t;

/**
 * \brief Supported providers of the distributed communication service.
 */
typedef enum
{
  CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE    = 0,  ///< No communication service provider (single-GPU execution)
  CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI     = 1,  ///< MPI communication service
  CUDENSITYMAT_DISTRIBUTED_PROVIDER_NCCL    = 2,  ///< NCCL communication service (experimental)
//CUDENSITYMAT_DISTRIBUTED_PROVIDER_NVSHMEM = 3,  ///< NVSHMEM communication service
} cudensitymatDistributedProvider_t;

/**
 * \brief Supported target devices for user-defined callbacks. 
 */
typedef enum
{
  CUDENSITYMAT_CALLBACK_DEVICE_CPU,  ///< CPU-side callback function
  CUDENSITYMAT_CALLBACK_DEVICE_GPU   ///< GPU-side callback function
} cudensitymatCallbackDevice_t;

/**
 * \brief Supported differentiation directions.
 */
typedef enum
{
//CUDENSITYMAT_DIFFERENTIATION_DIR_FORWARD  = 0,  ///< Forward differentiation
  CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD = 1   ///< Backward differentiation
} cudensitymatDifferentiationDir_t;

/**
 * \brief Quantum state purity (pure or mixed state).
 */
typedef enum
{
  CUDENSITYMAT_STATE_PURITY_PURE,  ///< Pure quantum state (aka state-vector)
  CUDENSITYMAT_STATE_PURITY_MIXED  ///< Mixed quantum state (aka density matrix)
} cudensitymatStatePurity_t;

/**
 * \brief Elementary operator sparsity kind.
 */
typedef enum
{
  CUDENSITYMAT_OPERATOR_SPARSITY_NONE          = 0,  ///< No sparsity (dense tensor)
  CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL = 1,  ///< Multi-diagonal sparsity (one or multiple non-zero diagonals)
} cudensitymatElementaryOperatorSparsity_t;

/**
 * \brief Kinds of the operator extreme eigen-spectrum computation.
 */
typedef enum
{
  CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST = 0,       ///< Compute the largest by magnitude eigen-values of the operator
  CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST = 1,      ///< Compute the smallest by magnitude eigen-values of the operator
  CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL = 2,  ///< Compute the largest by the real part eigen-values of the operator
  CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL = 3, ///< Compute the smallest by the real part eigen-values of the operator
} cudensitymatOperatorSpectrumKind_t;

/**
 * \brief Configuration options for the operator extreme eigen-spectrum computation.
 */
typedef enum
{
  CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION = 0,  ///< int32_t: Configures the max ratio of the number of Krylov subspace blocks to the number of requested eigen-pairs (defaults to 5)
  CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS = 1,   ///< int32_t: Configures the max number of restarted iterations of the block Krylov algorithm (defaults to 20)
  CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MIN_BLOCK_SIZE = 2, ///< int32_t: Configures the min block size of the block Krylov algorithm (defaults to 1)
} cudensitymatOperatorSpectrumConfig_t;

/**
 * \brief This enum lists supported boundary conditions for supported state factorizations.
 */
typedef enum
{
  CUDENSITYMAT_BOUNDARY_CONDITION_OPEN = 0,     ///< Open boundary condition
//CUDENSITYMAT_BOUNDARY_CONDITION_PERIODIC = 1  ///< Periodic boundary condition
} cudensitymatBoundaryCondition_t;

/**
 * \brief Time propagation scope (full vs split evolution).
 */
typedef enum
{
//CUDENSITYMAT_PROPAGATION_SCOPE_FULL = 0,  ///< Full propagation in a single scope
  CUDENSITYMAT_PROPAGATION_SCOPE_SPLIT = 1, ///< Split propagation (e.g., operator splitting)
} cudensitymatTimePropagationScopeKind_t;

/*
 * \brief Full-scope kind for time propagation.
 *
typedef enum
{
  CUDENSITYMAT_PROPAGATION_SCOPE_FULL_EXACT = 0, ///< Exact (full) propagation (default for dense states)
} cudensitymatTimePropagationScopeFullKind_t;
*/

/**
 * \brief Split kind for split-scope propagation.
 */
typedef enum
{
  CUDENSITYMAT_PROPAGATION_SCOPE_SPLIT_TDVP = 0, ///< TDVP-based split propagation (default)
} cudensitymatTimePropagationScopeSplitKind_t;

/**
 * \brief Time propagation approach (time integration / exponentiation method).
 */
typedef enum
{
  CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV = 0,      ///< Krylov subspace method
} cudensitymatTimePropagationApproachKind_t;

/**
 * \brief Time propagation configuration attributes.
 */
typedef enum
{
  CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_KIND = 0,       ///< int32_t (cudensitymatTimePropagationScopeSplitKind_t): Split kind
//CUDENSITYMAT_PROPAGATION_FULL_SCOPE_KIND = 1,        ///< int32_t (cudensitymatTimePropagationScopeFullKind_t): Full-scope kind
//CUDENSITYMAT_PROPAGATION_FULL_SCOPE_EXACT_CONFIG = 2, ///< cudensitymatTimePropagationScopeFullExactConfig_t: Full exact-scope configuration
  CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_TDVP_CONFIG = 3, ///< cudensitymatTimePropagationScopeSplitTDVPConfig_t: TDVP split configuration
  CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_CONFIG = 10, ///< cudensitymatTimePropagationApproachKrylovConfig_t: Krylov approach configuration
} cudensitymatTimePropagationAttribute_t;

/*
 * \brief Configuration attributes for full exact-scope propagation.
 *
typedef enum
{
} cudensitymatTimePropagationScopeFullExactConfigAttribute_t;
*/

/**
 * \brief Configuration attributes for Krylov-subspace method.
 */
typedef enum
{
  CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_TOLERANCE = 0,         ///< double: Convergence tolerance (default: 0, resolved to machine epsilon of the compute precision)
  CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_MAX_DIM = 1,         ///< int32_t: Maximum Krylov subspace dimension (default: 30)
  CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_MIN_BETA = 2,          ///< double: Minimum threshold for off-diagonal Hessenberg element h_{m+1,m} to proceed with Krylov expansion; below this indicates linear dependence or numerical breakdown (default: 0, resolved to machine epsilon of the compute precision)
  CUDENSITYMAT_PROPAGATION_APPROACH_KRYLOV_ADAPTIVE_STEP_SIZE = 3 ///< int32_t: Enable adaptive step size control (0=disabled, 1=enabled, default: 1)
} cudensitymatTimePropagationApproachKrylovConfigAttribute_t;

/**
 * \brief Configuration attributes for TDVP time propagation method.
 */
typedef enum
{
  CUDENSITYMAT_PROPAGATION_SPLIT_SCOPE_TDVP_ORDER = 0,                    ///< int32_t: Order of TDVP sweeps (2 (default), 4 (not implemented))
} cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t;

/** 
 * \brief Memory spaces for workspace buffer allocation.
 */
typedef enum
{
  CUDENSITYMAT_MEMSPACE_DEVICE = 0,  ///< Device memory space (GPU)
  CUDENSITYMAT_MEMSPACE_HOST   = 1,  ///< Host memory space (CPU)
} cudensitymatMemspace_t;

/**
 * \brief Kinds of workspace memory buffers.
 */
typedef enum
{
  CUDENSITYMAT_WORKSPACE_SCRATCH = 0,  ///< Scratch workspace memory
//CUDENSITYMAT_WORKSPACE_CACHE   = 1,  ///< Cache workspace memory (must stay valid with unmodified content until all referencing operations are completed)
} cudensitymatWorkspaceKind_t;

/** \} end constenums */


// TYPES AND STRUCTURES

/**
 * \defgroup typestructs Types and Data Structures
 * \{
 */

/**
 * \brief Opaque data structure holding the library context (context handle).
 *
 * \details Context handle holds the library context (device properties, system information, etc.).
 * A context handle must be initialized prior and destroyed after any other library API call
 * using the `cudensitymatCreate` and `cudensitymatDestroy` API functions, respectively.
 */
typedef void * cudensitymatHandle_t;

/**
 * \brief Opaque data structure holding the quantum state representation.
 *
 * \details A quantum state is defined by its purity (pure or mixed),
 * shape (specification of all quantum degrees of freedom), numerical
 * representation (dense tensor or sparse tensor), structural
 * compression (full or factorized tensor), forced symmetries (if any), etc.
 *
 * \note Each quantum degree of freedom is represented by a vector space
 * of some dimension. The full quantum state lives in a tensor product space
 * constructed from the vector spaces associated with the given quantum
 * degrees of freedom.
 */
typedef void * cudensitymatState_t;

/**
 * \brief Opaque data structure representing an elementary tensor operator
 * (or their batch) acting on a single or multiple quantum degrees of freedom.
 */
typedef void * cudensitymatElementaryOperator_t;

/**
 * \brief Opaque data structure representing a full operator matrix
 * (or their batch) acting on all quantum degrees of freedom.
 */
typedef void * cudensitymatMatrixOperator_t;

/**
 * \brief Opaque data structure representing a matrix product operator (MPO)
 * (or their batch) acting on a subset of quantum degrees of freedom.
 */
typedef void * cudensitymatMatrixProductOperator_t;

/**
 * \brief Opaque data structure representing an operator term acting on
 * quantum degrees of freedom from either side (for a mixed quantum state
 * it can act either from the ket side or from the bra side whereas
 * for a pure quantum state it can only act from one side).
 *
 * \details Generally, an operator term is defined as a sum of components,
 * where each component is either a product of elementary tensor operators
 * or a full matrix operator, with some scalar coefficient. The sum may
 * contain one or more such components. Each product of elementary tensor
 * operators may consist of one or more elementary tensor operators, or
 * contain no elementary tensor operators at all, representing an identity.
 */
typedef void * cudensitymatOperatorTerm_t;

/**
 * \brief Opaque data structure representing a composite operator (collection
 * of operator terms with some scalar coefficients) acting on quantum degrees
 * of freedom from either side.
 */
typedef void * cudensitymatOperator_t;

/**
 * \brief Opaque data structure representing a collective r.h.s. action of a given
 * number of composite operators on the corresponding number of input quantum states,
 * producing an update to a single output quantum state (when the actual computation
 * is performed).
 *
 * \details This data structure allows specification of a collective operator action
 * where both the operators and the input quantum states may be different in each
 * operator action pair. The action pair is defined as an operator acting on an input
 * quantum state, producing an update to a given output quantum state for all action pairs.
 */
typedef void * cudensitymatOperatorAction_t;

/**
 * \brief Opaque data structure specifying the operator expectation value computation.
 *
 * \details This data structure encapsulates the desired operator for which it
 * will be able to compute the expectation value over a specified quantum state.
 */
typedef void * cudensitymatExpectation_t;

/**
 * \brief Opaque data structure specifying the operator eigen-spectrum computation.
 *
 * \details This data structure encapsulates the desired operator for which it
 * will be able to compute its extreme eigen-spectrum.
 */
typedef void * cudensitymatOperatorSpectrum_t;

/**
 * \brief Opaque data structure describing a workspace memory buffer.
 */
typedef void * cudensitymatWorkspaceDescriptor_t;

/**
 * \brief Opaque data structure representing time propagation
 * of a quantum state under the action of a super-linear map.
 */
typedef void * cudensitymatTimePropagation_t;

/**
 * \brief Opaque data structure holding Krylov-subspace propagation configuration.
 *
 * \details This configuration object controls the Krylov subspace expansion
 * used for computing matrix exponential action during local time propagation.
 */
typedef void * cudensitymatTimePropagationApproachKrylovConfig_t;

/*
 * \brief Opaque data structure holding Full-scope propagation configuration.
 *
typedef void * cudensitymatTimePropagationScopeFullExactConfig_t;
*/

/**
 * \brief Opaque data structure holding TDVP propagation configuration.
 *
 * \details This configuration object controls the TDVP time propagation
 * method and its associated configuration.
 */
typedef void * cudensitymatTimePropagationScopeSplitTDVPConfig_t;

/**
 * \brief Explicit data structure specifying a given time interval or time range.
 *
 * \details The time interval bounds are specified by two doubles,
 * timeStart and timeFinish. The explicit time range, that is, the
 * sequence of time points within the requested time interval can be
 * generated by either setting a time step or providing explicit
 * time points manually via a C array of doubles.
 *
 * \note Providing explicit time points (numPoints > 0)
 * overrides the timeStep value.
 */
typedef struct
{
  double timeStart;       ///< Start time
  double timeFinish;      ///< Finish time
  double timeStep;        ///< Time step (zero value means undefined)
  int64_t numPoints;      ///< (Optional) Number of explicit time points inside the [timeStart:timeFinish] interval, 0 otherwise
  const double * points;  ///< (Optional) Ordered array with explicit time points inside the [timeStart:timeFinish] interval, NULL otherwise
} cudensitymatTimeRange_t;

/**
 * \brief Explicit data structure for storing an inter-process communicator in a type-erased form.
 */
typedef struct {
  void * commPtr;   ///< pointer to the MPI_Comm data structure
  size_t commSize;  ///< size of the MPI_Comm data structure
} cudensitymatDistributedCommunicator_t;

/* INTERNAL: Opaque data structure storing a distributed communication request. */
typedef void * cudensitymatDistributedRequest_t;

/** \} end typestructs */


// USER-DEFINED FUNCTION SIGNATURES

/**
 * \brief External CPU/GPU callback function returning a batch of scalar
 * coefficients, including a special case of batch size one (single scalar).
 *
 * \details An external user-provided scalar callback function can be
 * registered with the library for deferred invocation at a given point
 * of time: Given the time value and a (batched) array of real parameter values,
 * which parameterize a (batched) quantum operator, it will fill in a batch
 * of scalar coefficients used for defining the (batched) quantum operator.
 * Batch size one corresponds to a single instance (no batching).
 *
 * \param[in] time Time value.
 * \param[in] batchSize User-defined batch size (>=1).
 * \param[in] numParams Number of external real parameters the full operator depends on.
 * \param[in] params GPU-accessible pointer to an F-ordered 2d-array
 * of user-defined real parameter values: `params[numParams, batchSize]`.
 * \param[in] dataType Data type of the returned scalar(s).
 * \param[inout] scalarStorage Pointer to the array storage in a CPU-
 * or GPU-accessible memory buffer of length `batchSize`.
 * \param[in] stream CUDA stream.
 * \return int32_t Error code.
 */
typedef int32_t (*cudensitymatScalarCallback_t) (double time,
                                                 int64_t batchSize,
                                                 int32_t numParams,
                                                 const double * params,
                                                 cudaDataType_t dataType,
                                                 void * scalarStorage,
                                                 cudaStream_t stream);

/**
 * \brief External CPU/GPU callback function returning a batch of tensors,
 * including a special case of batch size one (single tensor).
 *
 * \details An external user-provided tensor callback function can be
 * registered with the library for deferred invocation at a given point
 * of time: Given the time value and a (batched) array of real parameter values,
 * which parameterize a (batched) quantum operator, it will fill in a batch
 * of tensors used for representing the (batched) quantum operator.
 * Batch size one corresponds to a single instance (no batching).
 *
 * \note A tensor callback function fills in a (batched) dense array
 * which represents a (batched) elementary tensor operator with its
 * specific sparsity kind:
 *  - CUDENSITYMAT_OPERATOR_SPARSITY_NONE:
 *      The returned dense array has exactly the same shape
 *      as the elementary tensor operator itself;
 *  - CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL:
 *      The returned dense array has shape [N,M],
 *      where N is the dimension of the matrix of the elementary tensor operator,
 *      while M is the number of non-zero diagonals of that matrix,
 *      padded with trailing zeros to the full matrix dimension.
 *  In both cases, the batch mode is appended at the end (most senior mode).
 *
 * \param[in] sparsity Elementary tensor operator sparsity kind.
 * \param[in] numModes Number of modes in the returned tensor (without the batch mode).
 * Only even-rank tensors are supported, where the first half of the modes
 * are the ket modes, and the second half are the bra modes.
 * \param[in] modeExtents Mode extents of the returned tensor (without the batch mode).
 * \param[in] diagonalOffsets For CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL,
 * offsets of the stored non-zero diagonals of the elementary tensor operator
 * matrix: The main diagonal has offset zero, the below-main diagonals have
 * negative offsets, the above-main diagonals have positive offsets.
 * For CUDENSITYMAT_OPERATOR_SPARSITY_NONE, this argument has no meaning
 * and can be set to NULL.
 * \param[in] time Time value.
 * \param[in] batchSize User-defined batch size (>=1).
 * \param[in] numParams Number of external real parameters the full operator depends on.
 * \param[in] params GPU-accessible pointer to an F-ordered 2d-array
 * of user-defined real parameter values: `params[numParams, batchSize]`.
 * \param[in] dataType Data type of the returned (potentially batched) tensor.
 * \param[inout] tensorStorage Pointer to the tensor elements storage (array data)
 * in a CPU- or GPU-accessible memory buffer of length `batchSize` (times the tensor shape).
 * \param[in] stream CUDA stream.
 * \return int32_t Error code.
 */
typedef int32_t (*cudensitymatTensorCallback_t) (cudensitymatElementaryOperatorSparsity_t sparsity,
                                                 int32_t numModes,
                                                 const int64_t modeExtents[],
                                                 const int32_t diagonalOffsets[],
                                                 double time,
                                                 int64_t batchSize,
                                                 int32_t numParams,
                                                 const double * params,
                                                 cudaDataType_t dataType,
                                                 void * tensorStorage,
                                                 cudaStream_t stream);

/**
 * \brief Scalar callback wrapper data structure.
 *
 * \details A wrapped struct containing the scalar callback and the scalar callback wrapper.
 * The callback wrapper is only used when the library is used from Python and stores a Python
 * callback function. It should be set to `NULL` when the library is used from C and stores
 * a C callback function.
 */
typedef struct
{
  cudensitymatScalarCallback_t callback;  ///< Scalar callback function.
  cudensitymatCallbackDevice_t device;    ///< Device type for the stored callback function (CPU or GPU).
  void * wrapper;                         ///< Wrapper for a scalar callback, should be set to `NULL` when used from C.
} cudensitymatWrappedScalarCallback_t;

const cudensitymatWrappedScalarCallback_t cudensitymatScalarCallbackNone = {NULL};

/** 
 * \brief Tensor callback wrapper data structure.
 *
 * \details A wrapped struct containing the tensor callback and the tensor callback wrapper.
 * The callback wrapper is only used when the library is used from Python and stores a Python
 * callback function. It should be set to `NULL` when the library is used from C and stores
 * a C callback function.
 */
typedef struct
{
  cudensitymatTensorCallback_t callback;  ///< Tensor callback function.
  cudensitymatCallbackDevice_t device;    ///< Device type for the stored callback function (CPU or GPU).
  void * wrapper;                         ///< Wrapper for a tensor callback, should be set to `NULL` when used from C.
} cudensitymatWrappedTensorCallback_t;

const cudensitymatWrappedTensorCallback_t cudensitymatTensorCallbackNone = {NULL};

/**
 * \brief External scalar gradient CPU/GPU callback function returning
 * gradients with respect to the passed user-provided real parameter values.
 *
 * \details This callback function signature supports differentiation
 * in both directions:
 *  - Forward differentiation (not supported yet):
 *      The callback function takes a seed of the parameters gradient
 *      and returns the gradient of the (potentially batched) scalar
 *      coefficient with respect to the chosen parameters gradient seed.
 *  - Backward differentiation:
 *      The callback function takes the adjoint of the (potentailly batched)
 *      scalar coefficient and returns the contribution to the gradient of
 *      the total cost function with respect to the user-provided real parameter
 *      values that originates from that scalar coefficient.
 *
 * \note In case of backward differentiation, argument `scalarGrad[batchSize]`,
 * which is the adjoint of the (potentially batched) complex scalar coefficient
 * parameterizing a user-defined operator, is in general a complex vector of
 * size `batchSize`. Since we always assume a real total cost function which
 * we want to differentiate with respect to the user-provided real parameter values,
 * the scalar gradient callback function is supposed to implement the following:
 *   1. Compute the vector-Jacobian product (VJP) of the passed adjoint of the
 *      (potentially batched) scalar coefficient with the Jacobian of that
 *      (potentially batched) scalar coeffient with respect to all user-provided
 *      real parameter values, which will produce a generally complex tensor
 *      of shape `[numParams, batchSize]`, specifically:
 *        `Gradient[i,j] = AdjointCoefficient[j] * dAdjointCoefficient[j]/dParams[i,j]`,
 *        for all `i` in `[0, numParams-1]` and `j` in `[0, batchSize-1]`.
 *   2. Take the real part of the resulting complex tensor, multiply it by 2,
 *      and accumulate it into the `paramsGrad[numParams, batchSize]` array.
 *      Thus, `paramsGrad` must be initialized to zero externally by the user
 *      before the backward differentiation is started. During the backward
 *      differentiation pass, `paramsGrad` will accumulate all partial derivatives
 *      of all explicitly differentiated quantities with respect to the user-provided
 *      real parameter values (for all instances of the batch).
 *
 * \param[in] time Time value.
 * \param[in] batchSize User-defined batch size (>=1).
 * \param[in] numParams Number of external real parameters the full operator depends on.
 * \param[in] params GPU-accessible pointer to an F-ordered 2d-array
 * of user-defined real parameter values: `params[numParams, batchSize]`.
 * \param[in] dataType Data type of the passed (potentially batched) scalar.
 * \param scalarGrad CPU- or GPU-accessible pointer to either the (potentially batched)
 * scalar adjoint (IN) or scalar gradient (OUT) array of length `batchSize`,
 * depending on the differentiation direction.
 * \param paramsGrad GPU-accessible pointer to an F-ordered 2d-array of either
 * the parameters gradient seed (IN) or parameters gradients (OUT) of shape
 * `paramsGrad[numParams, batchSize]`, depending on the differentiation direction.
 * \param[in] stream CUDA stream.
 * \return int32_t Error code.
 */
typedef int32_t (*cudensitymatScalarGradientCallback_t) (double time,
                                                         int64_t batchSize,
                                                         int32_t numParams,
                                                         const double * params,
                                                         cudaDataType_t dataType,
                                                         void * scalarGrad,
                                                         double * paramsGrad,
                                                         cudaStream_t stream);

/**
 * \brief External tensor gradient CPU/GPU callback function returning
 * gradients with respect to the passed user-provided real parameter values.
 *
 * \details This callback function signature supports differentiation
 * in both directions:
 *  - Forward differentiation (not supported yet):
 *      The callback function takes a seed of the parameters gradient
 *      and returns the gradient of the (potentially batched) tensor
 *      with respect to the chosen parameters gradient seed.
 *  - Backward differentiation:
 *      The callback function takes the adjoint of the (potentailly batched)
 *      tensor and returns the contribution to the gradient of the total cost function
 *      with respect to the user-provided real parameter values that originates from
 *      that tensor.
 *
 * \note In case of backward differentiation, argument `tensorGrad[{tensorShape}, batchSize]`
 * is the adjoint of the (potentially batched) elementary operator tensor inside a user-defined
 * operator. Since we always assume a real total cost function which we want to differentiate
 * with respect to the user-provided real parameter values, the tensor gradient callback function
 * is supposed to implement the following:
 *   1. Compute the vector-Jacobian product (VJP) of the passed adjoint of the
 *      (potentially batched) elementary operator tensor with the Jacobian of that
 *      (potentially batched) elementary operator tensor with respect to all user-provided
 *      real parameter values, which will produce a generally complex tensor
 *      of shape `[numParams, batchSize]`, specifically:
 *        `Gradient[i,j] = Sum_{shape} AdjointTensor[{shape},j] * dAdjointTensor[{shape},j]/dParams[i,j]`,
 *        for all `i` in `[0, numParams-1]` and `j` in `[0, batchSize-1]`,
 *        where `{shape}` includes all tensor indices (except the batch index)
 *        that we are summing over in the summation.
 *   2. Take the real part of the resulting complex tensor, multiply it by 2,
 *      and accumulate it into the `paramsGrad[numParams, batchSize]` array.
 *      Thus, `paramsGrad` must be initialized to zero externally by the user
 *      before the backward differentiation is started. During the backward
 *      differentiation pass, `paramsGrad` will accumulate all partial derivatives
 *      of all explicitly differentiated quantities with respect to the user-provided
 *      real parameter values (for all instances of the batch).
 *
 * \param[in] sparsity Elementary tensor operator sparsity kind.
 * \param[in] numModes Number of modes in the returned tensor (without the batch mode).
 * Only even-rank tensors are supported, where the first half of the modes
 * are the ket modes, and the second half are the bra modes.
 * \param[in] modeExtents Mode extents of the returned tensor (without the batch mode).
 * \param[in] diagonalOffsets For CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL,
 * offsets of the stored non-zero diagonals of the elementary tensor operator
 * matrix: The main diagonal has offset zero, the below-main diagonals have
 * negative offsets, the above-main diagonals have positive offsets.
 * For CUDENSITYMAT_OPERATOR_SPARSITY_NONE, this argument has no meaning
 * and can be set to NULL.
 * \param[in] time Time value.
 * \param[in] batchSize User-defined batch size (>=1).
 * \param[in] numParams Number of external real parameters the full operator depends on.
 * \param[in] params GPU-accessible pointer to an F-ordered 2d-array
 * of user-defined real parameter values: `params[numParams, batchSize]`.
 * \param[in] dataType Data type of the passed (potentially batched) tensor.
 * \param tensorGrad CPU- or GPU-accessible pointer to either the (potentially batched)
 * tensor adjoint (IN) or tensor gradient (OUT) array of length `batchSize` times
 * the tensor shape, depending on the differentiation direction.
 * \param paramsGrad GPU-accessible pointer to an F-ordered 2d-array of either
 * the parameters gradient seed (IN) or parameters gradients (OUT) of shape
 * `paramsGrad[numParams, batchSize]`, depending on the differentiation direction.
 * \param[in] stream CUDA stream.
 * \return int32_t Error code.
 */
typedef int32_t (*cudensitymatTensorGradientCallback_t) (cudensitymatElementaryOperatorSparsity_t sparsity,
                                                         int32_t numModes,
                                                         const int64_t modeExtents[],
                                                         const int32_t diagonalOffsets[],
                                                         double time,
                                                         int64_t batchSize,
                                                         int32_t numParams,
                                                         const double * params,
                                                         cudaDataType_t dataType,
                                                         void * tensorGrad,
                                                         double * paramsGrad,
                                                         cudaStream_t stream);

/**
* \brief Scalar gradient callback wrapper data structure.
*
* \details A wrapped struct containing the scalar gradient callback and the scalar gradient callback wrapper.
* The callback `wrapper` is only used when the library is used from Python and stores a Python
* callback function. It should be set to `NULL` when the library is used from C and stores
* a C callback function.
*/
typedef struct
{
  cudensitymatScalarGradientCallback_t callback;  ///< Scalar gradient callback function.
  cudensitymatCallbackDevice_t device;            ///< Device type for the stored callback function (CPU or GPU).
  void * wrapper;                                 ///< Wrapper for a scalar gradient callback, should be set to `NULL` when used from C.
  cudensitymatDifferentiationDir_t direction;     ///< Differentiation direction.
} cudensitymatWrappedScalarGradientCallback_t;

const cudensitymatWrappedScalarGradientCallback_t cudensitymatScalarGradientCallbackNone = {NULL};

/** 
* \brief Tensor gradient callback wrapper data structure.
*
* \details A wrapped struct containing the tensor gradient callback and the tensor gradient callback wrapper.
* The callback `wrapper` is only used when the library is used from Python and stores a Python
* callback function. It should be set to `NULL` when the library is used from C and stores
* a C callback function.
*/
typedef struct
{
  cudensitymatTensorGradientCallback_t callback;  ///< Tensor gradient callback function.
  cudensitymatCallbackDevice_t device;            ///< Device type for the stored callback function (CPU or GPU).
  void * wrapper;                                 ///< Wrapper for a tensor gradient callback, should be set to `NULL` when used from C.
  cudensitymatDifferentiationDir_t direction;     ///< Differentiation direction.
} cudensitymatWrappedTensorGradientCallback_t;

const cudensitymatWrappedTensorGradientCallback_t cudensitymatTensorGradientCallbackNone = {NULL};

/**
 * \typedef cudensitymatLoggerCallback_t
 * \brief A callback function pointer type for logging APIs. Use `cudensitymatLoggerSetCallback` to set the callback function.
 * \param[in] logLevel Logging level.
 * \param[in] functionName Name of the API that logged this message.
 * \param[in] message Log message.
 */
typedef void (*cudensitymatLoggerCallback_t)(int32_t logLevel,
                                             const char * functionName,
                                             const char * message);

/**
 * \typedef cudensitymatLoggerCallbackData_t
 * \brief A callback function pointer type for logging APIs. Use `cudensitymatLoggerSetCallbackData` to set the callback function and user data.
 * \param[in] logLevel Loggin level.
 * \param[in] functionName Name of the API that logged this message.
 * \param[in] message Log message.
 * \param[in] userData User's data to be used by the callback.
 */
typedef void (*cudensitymatLoggerCallbackData_t)(int32_t logLevel,
                                                 const char * functionName,
                                                 const char * message,
                                                 void * userData);


// API FUNCTIONS

/**
 * \defgroup contextAPI Library Initialization and Management API
 * \{
 */

/**
 * \brief Returns the semantic version number of the cuDensityMat library.
 */
size_t cudensitymatGetVersion();

/**
 * \brief Creates and initializes the library context.
 * 
 * \param[out] handle Library handle.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreate(cudensitymatHandle_t * handle);

/**
 * \brief Destroys the library context.
 * 
 * \param[in] handle Library handle.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroy(cudensitymatHandle_t handle);

/**
 * \brief Resets the current distributed execution configuration
 * associated with the given library context by importing a user-provided
 * inter-process communicator (e.g., MPI_Comm).
 *
 * \details Accepts and stores a copy of the provided inter-process communicator
 * which will be used for distributing numerical operations across all involved
 * distributed processes.
 *
 * \param[inout] handle Library handle.
 * \param[in] provider Communication service provider.
 * \param[in] commPtr Pointer to the communicator in a type-erased form.
 * \param[in] commSize Size of the communicator in bytes.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatResetDistributedConfiguration(
                    cudensitymatHandle_t handle,
                    cudensitymatDistributedProvider_t provider,
                    const void * commPtr,
                    size_t commSize);

/**
 * \brief Returns the total number of distributed processes
 * associated with the given library context in its current
 * distributed execution configuration.
 * 
 * \param[in] handle Library handle.
 * \param[out] numRanks Number of distributed processes.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatGetNumRanks(
                    const cudensitymatHandle_t handle,
                    int32_t * numRanks);

/**
 * \brief Returns the rank of the current process in the distributed
 * execution configuration associated with the given library context.
 * 
 * \param[in] handle Library handle.
 * \param[out] procRank Rank of the current distributed process.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatGetProcRank(
                    const cudensitymatHandle_t handle,
                    int32_t * procRank);

/**
 * \brief Resets the context-level random seed used by the random
 * number generator inside the library context.
 * 
 * \param[inout] handle Library handle.
 * \param[in] randomSeed Random seed value.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatResetRandomSeed(
                    cudensitymatHandle_t handle,
                    int32_t randomSeed);

/** \} end contextAPI */

/**
 * \defgroup stateAPI Quantum State Definition API
 * \{
 */

/**
 * \brief Defines an empty dense quantum state of a given purity and shape,
 * or a batch of such dense quantum states.
 *
 * \details The number of space modes defining the state space is always the
 * number of quantum degrees of freedom used to define the corresponding
 * composite tensor-product space. With that, the number of modes in a
 * pure-state tensor equals the number of the space modes (quantum degrees of freedom).
 * The number of modes in a mixed-state tensor equals twice the number of the space modes,
 * consisting of a set of the ket modes and a set of the bra modes, which are identical in terms
 * of their extents between the two sets, with the ket modes preceding the bra modes, for example:
 * S[i0, i1, j0, j1] tensor represents a mixed quantum state with two degrees of freedom,
 * where modes {i0, i1} form the ket set, and modes {j0, j1} form the bra set such that
 * ket mode i0 corresponds to the bra mode j0, and ket mode i1 corresponds to the bra mode j1.
 * In contrast, a pure quantum state with two degrees of freedom is represented by the tensor
 * S[i0, i1] with only the ket modes (no bra modes). Furthermore, batched pure/mixed states
 * introduce one additional (batch) mode to their dense tensor representation, namely:
 * S[i0, i1, b] for the batched pure state, and S[i0, i1, j0, j1, b] for the batched
 * mixed state, where b is the size of the batch (batch size or extent of the batch mode).
 *
 * \param[in] handle Library handle.
 * \param[in] purity Desired quantum state purity.
 * \param[in] numSpaceModes Number of space modes (number of quantum degrees of freedom).
 * \param[in] spaceModeExtents Extents of the space modes (dimensions of the quantum degrees of freedom).
 * \param[in] batchSize Batch size (number of equally-shaped quantum states in the batch).
 * Note that setting the batch size to zero is the same as setting it to 1 (no batching).
 * \param[in] dataType Numerical representation data type (type of tensor elements).
 * \param[out] state Empty dense quantum state (or a batch of such quantum states).
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateState(
                    const cudensitymatHandle_t handle,
                    cudensitymatStatePurity_t purity,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    int64_t batchSize,
                    cudaDataType_t dataType,
                    cudensitymatState_t * state);

/**
 * \brief Defines an empty quantum state of a given purity and shape
 * in the Matrix Product State (MPS) factorized form,
 * or a batch of such MPS factorized quantum states.
 *
 * The MPS consists of `numSpaceModes` site tensors (components), one per site,
 * indexed from 0 to `numSpaceModes - 1`. For open boundary conditions the site
 * tensor mode ordering is:
 *   - Site 0 (leftmost):           [physical, right_bond]
 *   - Sites 1 .. N-2 (interior):   [left_bond, physical, right_bond]
 *   - Site N-1 (rightmost):        [left_bond, physical]
 *
 * where `physical` has extent `spaceModeExtents[i]`, `left_bond` has extent
 * `bondExtents[i-1]`, and `right_bond` has extent `bondExtents[i]`.
 *
 * \param[in] handle Library handle.
 * \param[in] purity Desired quantum state purity.
 * \param[in] numSpaceModes Number of space modes (number of quantum degrees of freedom).
 * \param[in] spaceModeExtents Extents of the space modes (dimensions of the quantum degrees of freedom).
 * \param[in] boundaryCondition Boundary condition.
 * \param[in] bondExtents Extents of the bond modes. For open boundary condition,
 * the length of the array must be equal to the number of space modes minus one,
 * where `bondExtents[i]` is the bond dimension between site `i` and site `i+1`.
 * For periodic boundary condition, the length of the array must be equal to the number of space modes.
 * \param[in] dataType Quantum state data type.
 * \param[in] batchSize Batch size (number of equally-shaped quantum states in the batch).
 * \param[out] state Empty quantum state (or a batch of quantum states) in the MPS factorized form.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatCreateStateMPS(
                    const cudensitymatHandle_t handle,
                    cudensitymatStatePurity_t purity,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    cudensitymatBoundaryCondition_t boundaryCondition,
                    const int64_t bondExtents[],
                    cudaDataType_t dataType,
                    int64_t batchSize,
                    cudensitymatState_t * state);

/**
 * \brief Destroys the quantum state.
 * 
 * \param[in] state Quantum state (or a batch of quantum states).
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyState(cudensitymatState_t state);

/**
 * \brief Queries the number of components (tensors) constituting
 * the chosen quantum state representation (on the current process
 * in multi-process runs).
 *
 * \details Quantum state representation may include one or more
 * components (tensors) distributed over one or more parallel
 * processes (in distributed multi-GPU runs). The plain state vector
 * or density matrix representations consist of only one component,
 * the full state tensor, which can be sliced and distributed over
 * all parallel processes (in distributed multi-GPU runs).
 * Factorized quantum state representations include more than one
 * component, and these components (tensors) are generally distributed
 * over all parallel processes (in distributed multi-GPU runs).
 *
 * \note In multi-process runs, this function returns the number
 * of locally stored components which, in general, can be smaller
 * than the total number of components stored across all parallel
 * processes. One can use API function `cudensitymatStateGetComponentInfo`
 * to obtain more information on a given local component by providing
 * its local id.
 *
 * \note Batching does not add new components to the quantum state
 * representation, it just makes all existing components batched.
 * The corresponding tensors acquire one additional (most significant)
 * mode which represents the batch dimension. Note, however, that
 * the batch dimension of a locally stored component may have a
 * smaller extent than the total batch size due to potential slicing.
 *
 * \param[in] handle Library handle.
 * \param[in] state Quantum state (or a batch of quantum states).
 * \param[out] numStateComponents Number of components (tensors)
 * in the quantum state representation (on the current process).
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateGetNumComponents(
                    const cudensitymatHandle_t handle,
                    const cudensitymatState_t state,
                    int32_t * numStateComponents);

/**
 * \brief Queries the storage size (in bytes) for each
 * component (tensor) constituting the quantum state representation
 * (on the current process in multi-process runs).
 *
 * \param[in] handle Library handle.
 * \param[in] state Quantum state (or a batch of quantum states).
 * \param[in] numStateComponents Number of components (tensors)
 * in the quantum state representation (on the current process).
 * \param[out] componentBufferSize Storage size (bytes) for each
 * component (tensor) consituting the quantum state representation
 * (on the current process).
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateGetComponentStorageSize(
                    const cudensitymatHandle_t handle,
                    const cudensitymatState_t state,
                    int32_t numStateComponents,
                    size_t componentBufferSize[]);

/**
 * \brief Attaches a user-owned GPU-accessible storage buffer for each
 * component (tensor) constituting the quantum state representation
 * (on the current process in multi-process runs).
 *
 * \details The provided user-owned GPU-accessible storage buffers
 * will be used for storing components (tensors) constituting
 * the quantum state representation (on the current process
 * in multi-process runs). The initial value of the provided
 * storage buffers will be respected by the library,
 * thus providing a mechanism for specifing any initial value
 * of the quantum state in its chosen representation form.
 * In multi-process runs, API function `cudensitymatGetComponentInfo`
 * can be used for retrieving the information on which slice of the
 * requested component (tensor) is stored on the current process.
 *
 * \param[in] handle Library handle.
 * \param[inout] state Quantum state (or a batch of quantum states)
 * \param[in] numStateComponents Number of components (tensors)
 * in the quantum state representation (on the current process).
 * The number of components can be retrived by calling the API
 * function `cudensitymatStateGetNumComponents`.
 * \param[in] componentBuffer Pointers to user-owned GPU-accessible
 * storage buffers for all components (tensors) constituting
 * the quantum state representation (on the current process).
 * \param[in] componentBufferSize Sizes of the provded storage
 * buffers for all components (tensors) constituting the quantum
 * state representation (on the current process).
 * \return cudensitymatStatus_t 
 *
 * \note The sizes of the provided storage buffers must be equal or larger
 * than the required sizes retrieved via `cudensitymatStateGetComponentStorageSize`.
 */
cudensitymatStatus_t cudensitymatStateAttachComponentStorage(
                    const cudensitymatHandle_t handle,
                    cudensitymatState_t state,
                    int32_t numStateComponents,
                    void * componentBuffer[],
                    const size_t componentBufferSize[]);

/**
 * \brief Queries the number of modes in a local component tensor
 * (on the current process in multi-process runs).
 *
 * \param[in] handle Library handle.
 * \param[in] state Quantum state (or a batch of quantum states).
 * \param[in] stateComponentLocalId Component local id (on the current parallel process).
 * \param[out] stateComponentGlobalId Component global id (across all parallel processes).
 * \param[out] stateComponentNumModes Number of modes in the queried component tensor.
 * \param[out] batchModeLocation Location of the batch mode (or -1 if the batch mode is absent).
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateGetComponentNumModes(
                    const cudensitymatHandle_t handle,
                    cudensitymatState_t state,
                    int32_t stateComponentLocalId,
                    int32_t * stateComponentGlobalId,
                    int32_t * stateComponentNumModes,
                    int32_t * batchModeLocation);

/**
 * \brief Queries information for a locally stored component
 * tensor which represents either the full component or its slice
 * (on the current process in multi-process runs).
 *
 * \details This API function queries the global component id
 * (across all parallel processes), the number of tensor modes
 * (including the batch mode, if present), the extents of all modes,
 * and the base offsets for all modes which can be different from
 * zero if the locally stored component tensor represents a slice
 * of the full component tensor (a base offset of a sliced mode is
 * the starting index value of that mode inside the full tensor mode).
 *
 * \param[in] handle Library handle.
 * \param[in] state Quantum state (or a batch of quantum states).
 * \param[in] stateComponentLocalId Component local id (on the current parallel process).
 * \param[out] stateComponentGlobalId Component global id (across all parallel processes).
 * \param[out] stateComponentNumModes Number of modes in the queried component tensor.
 * \param[out] stateComponentModeExtents Component tensor mode extents
 * (the size of the array must be sufficient, see `cudensitymatStateGetComponentNumModes`)
 * \param[out] stateComponentModeOffsets Component tensor mode offsets defining the locally
 * stored slice (the size of the array must be sufficient, see `cudensitymatStateGetComponentNumModes`)
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateGetComponentInfo(
                    const cudensitymatHandle_t handle,
                    cudensitymatState_t state,
                    int32_t stateComponentLocalId,
                    int32_t * stateComponentGlobalId,
                    int32_t * stateComponentNumModes,
                    int64_t stateComponentModeExtents[],
                    int64_t stateComponentModeOffsets[]);

/**
 * \brief Initializes the quantum state to zero (null state).
 *
 * \note This API function is typically used for initializing
 * the output quantum state to true zero before accumulating
 * in it the action of an operator on some input quantum state.
 *
 * \param[in] handle Library handle.
 * \param[inout] state Quantum state (or a batch of quantum states).
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateInitializeZero(
                    const cudensitymatHandle_t handle,
                    cudensitymatState_t state,
                    cudaStream_t stream);

/*
 * \brief Initializes the quantum state to a random value.
 * 
 * \param[in] handle Library handle.
 * \param[inout] state Quantum state (or a batch of quantum states).
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
/*cudensitymatStatus_t cudensitymatStateInitializeRandom(
                    const cudensitymatHandle_t handle,
                    cudensitymatState_t state,
                    cudaStream_t stream);
*/

/**
 * \brief Multiplies the quantum state(s) by a scalar factor(s).
 *
 * \param[in] handle Library handle.
 * \param[inout] state Quantum state (or a batch of quantum states).
 * \param[in] scalingFactors Array of scaling factor(s) of dimension
 * equal to the batch size in the GPU-accessible RAM (same data type
 * as used by the quantum state).
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateComputeScaling(
                    const cudensitymatHandle_t handle,
                    cudensitymatState_t state,
                    const void * scalingFactors,
                    cudaStream_t stream);

/**
 * \brief Computes the squared Frobenius norm(s) of the quantum state(s).
 *
 * \details The result is generally a vector of dimension
 * equal to the quantum state batch size.
 *
 * \note For quantum states represented by complex data types,
 * the actual data type of the returned norm is float for cuFloatComplex
 * and double for cuDoubleComplex, respectively.
 *
 * \param[in] handle Library handle.
 * \param[in] state Quantum state (or a batch of quantum states).
 * \param[out] norm Pointer to the squared Frobenius norm(s) vector storage
 * in the GPU-accessible RAM (float or double real data type).
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 *
 * \note For quantum states represented by FP32 complex numbers
 * the norm type is float; For quantum states represented by
 * FP64 complex numbers the norm type is double.
 */
cudensitymatStatus_t cudensitymatStateComputeNorm(
                    const cudensitymatHandle_t handle,
                    const cudensitymatState_t state,
                    void * norm,
                    cudaStream_t stream);

/**
 * \brief Computes the trace(s) of the quantum state(s).
 *
 * \details Trace of a pure quantum state is defined to be its squared norm.
 * Trace of a mixed quantum state is equal to the trace of its density matrix.
 * The result is generally a vector of dimension equal to the quantum state
 * batch size.
 *
 * \param[in] handle Library handle.
 * \param[in] state Quantum state (or a batch of quantum states).
 * \param[out] trace Pointer to the trace(s) vector storage
 * in the GPU-accessible RAM (same data type as used by the quantum state).
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateComputeTrace(
                    const cudensitymatHandle_t handle,
                    const cudensitymatState_t state,
                    void * trace,
                    cudaStream_t stream);

/**
 * \brief Computes accumulation of a quantum state(s)
 * into another quantum state(s) of compatible shape.
 * 
 * \param[in] handle Library handle.
 * \param[in] stateIn Accumulated quantum state (or a batch of quantum states).
 * \param[inout] stateOut Accumulating quantum state (or a batch of quantum states).
 * \param[in] scalingFactors Array of scaling factor(s) of dimension
 * equal to the batch size in the GPU-accessible RAM (same data type
 * as used by the quantum state).
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateComputeAccumulation(
                    const cudensitymatHandle_t handle,
                    const cudensitymatState_t stateIn,
                    cudensitymatState_t stateOut,
                    const void * scalingFactors,
                    cudaStream_t stream);

/**
 * \brief Computes the inner product(s) between the left quantum state(s)
 * and the right quantum state(s): < state(s)Left | state(s)Right >
 *
 * \details For pure quantum states, this function computes the regular
 * Hilbert-space inner product. For mixed quantum states, it computes
 * the matrix inner product induced by the Frobenius matrix norm:
 * The sum of regular Hilbert-space inner products for all columns
 * of two density matrices.
 *
 * \details The result is generally a vector of dimension
 * equal to the batch size of both quantum states, which must be the same.
 * The participating quantum states must have compatible shapes.
 *
 * \param[in] handle Library handle.
 * \param[in] stateLeft Left quantum state (or a batch of quantum states).
 * \param[in] stateRight Right quantum state (or a batch of quantum states).
 * \param[out] innerProduct Pointer to the inner product(s) vector storage
 * in the GPU-accessible RAM (same data type as the one used by the quantum states).
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatStateComputeInnerProduct(
                    const cudensitymatHandle_t handle,
                    const cudensitymatState_t stateLeft,
                    const cudensitymatState_t stateRight,
                    void * innerProduct,
                    cudaStream_t stream);

/** \} end stateAPI */

/**
 * \defgroup operatorAPI Quantum Many-Body Operator API
 * \{
 */

/**
 * \brief Creates an elementary tensor operator acting on
 * a given number of quantum state modes (aka space modes).
 *
 * \details An elementary tensor operator is a single tensor operator
 * acting on a specific set of space modes (quantum degrees of freedom).
 * A tensor operator is composed of a set of the ket modes and a set of the
 * corresponding bra modes of matching extents, where both sets have the same
 * number of modes and the ket modes precede the bra modes, both in the same order.
 * For example, T[i0, i1, j0, j1] is a 2-body tensor operator in which modes {i0, i1}
 * form a set of the ket modes while modes {j0, j1} form the corresponding set of
 * the bra modes, where the ket mode i0 corresponds to the bra mode j0, and the ket mode
 * i1 corresponds to the bra mode j1 (the modes are always paired this way). Only one mode
 * in each pair of the corresponding modes is contracted with the quantum state tensor,
 * either from the left or from the right. For example, either the bra mode j0 is
 * contracted with a specific ket mode of the quantum state, representing an operator
 * action from the left, or the ket mode i0 is contracted with a specific bra mode
 * of the quantum state, representing an operator action from the right. Then,
 * the corresponding remaining uncontracted mode replaces the contracted mode
 * of the quantum state.
 *
 * \details Storage of tensor elements in memory:
 * - CUDENSITYMAT_OPERATOR_SPARSITY_NONE:
 *     Dense tensor stored using the generalized column-wise layout.
 * - CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL:
 *     The full non-zero diagonals are stored in a concatenated form,
 *     following the order how they appear in the `diagonalOffsets` argument.
 *     The length of each stored diagonal is equal to the full matrix dimension,
 *     padded with trailing zeros for non-main diagonals.
 *
 * \note Currently the multi-diagonal storage format is only supported
 * by 1-body elementary tensor operators (restriction subject to lifting in future).
 *
 * \warning Different elementary tensor operators must not use the same
 * or overlapping GPU storage buffers, otherwise it will cause an undefined behavior.
 *
 * \param[in] handle Library handle.
 * \param[in] numSpaceModes Number of the (state) space modes acted on.
 * \param[in] spaceModeExtents Extents of the (state) space modes acted on.
 * \param[in] sparsity Tensor operator sparsity defining the storage scheme.
 * \param[in] numDiagonals For multi-diagonal tensor operator matrices,
 * specifies the total number of non-zero diagonals (>= 1), otherwise ignored.
 * \param[in] diagonalOffsets For multi-diagonal tensor operator matrices, these are
 * the offsets of the non-zero diagonals (for example, the main diagonal has offset 0,
 * the diagonal right above the main diagonal has offset +1, the diagonal right below
 * the main diagonal has offset -1, and so on).
 * \param[in] dataType Tensor operator data type.
 * \param[in] tensorData GPU-accessible pointer to the tensor operator elements storage.
 * \param[in] tensorCallback Optional user-defined tensor callback function
 * which can be called later to fill in the tensor elements in the provided storage, or NULL.
 * \param[in] tensorGradientCallback Optional user-defined tensor gradient callback function
 * which can be called later to compute the Vector-Jacobian Product (VJP) for the tensor operator,
 * to produce gradients with respect to the user-defined real parameters, or NULL.
 * \param[out] elemOperator Elementary tensor operator.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateElementaryOperator(
                    const cudensitymatHandle_t handle,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    cudensitymatElementaryOperatorSparsity_t sparsity,
                    int32_t numDiagonals,
                    const int32_t diagonalOffsets[],
                    cudaDataType_t dataType,
                    void * tensorData,
                    cudensitymatWrappedTensorCallback_t tensorCallback,
                    cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback,
                    cudensitymatElementaryOperator_t * elemOperator);

/**
 * \brief Creates a batch of elementary tensor operators acting on
 * a given number of quantum state modes (aka space modes). This is
 * a batched version of the `cudensitymatCreateElementaryOperator` API function.
 *
 * \details This API function is used to create a batch of elementary tensor operators
 * of the same shape stored contiguously in memory, where the batch size must match
 * the batch size of the quantum state acted on. An elementary tensor operator is a single
 * tensor operator acting on a specific set of space modes (quantum degrees of freedom).
 * A tensor operator is composed of a set of the ket modes and a set of the
 * corresponding bra modes of matching extents, where both sets have the same
 * number of modes and the ket modes precede the bra modes, both in the same order.
 * For example, T[i0, i1, j0, j1] is a 2-body tensor operator in which modes {i0, i1}
 * form a set of the ket modes while modes {j0, j1} form the corresponding set of
 * the bra modes, where the ket mode i0 corresponds to the bra mode j0, and the ket mode
 * i1 corresponds to the bra mode j1 (the modes are always paired this way). Only one mode
 * in each pair of the corresponding modes is contracted with the quantum state tensor,
 * either from the left or from the right. For example, either the bra mode j0 is
 * contracted with a specific ket mode of the quantum state, representing an operator
 * action from the left, or the ket mode i0 is contracted with a specific bra mode
 * of the quantum state, representing an operator action from the right. Then,
 * the corresponding remaining uncontracted mode replaces the contracted mode
 * of the quantum state.
 *
 * \details Storage of tensor elements in memory:
 * - CUDENSITYMAT_OPERATOR_SPARSITY_NONE:
 *     Dense tensor stored using the generalized column-wise layout.
 * - CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL:
 *     The full non-zero diagonals are stored in a concatenated form,
 *     following the order how they appear in the `diagonalOffsets` argument.
 *     The length of each stored diagonal is equal to the full matrix dimension,
 *     padded with trailing zeros for non-main diagonals.
 *
 * \note Currently the multi-diagonal storage format is only supported
 * by 1-body elementary tensor operators (restriction subject to lifting in future).
 * Furthermore, all elementary tensor operators within the batch are assumed to
 * have the same sparse structure in terms of which matrix diagonals are stored,
 * that is, the same `diagonalOffsets`.
 *
 * \warning Different elementary tensor operators must not use the same
 * or overlapping GPU storage buffers, otherwise it will cause an undefined behavior.
 *
 * \param[in] handle Library handle.
 * \param[in] numSpaceModes Number of the (state) space modes acted on.
 * \param[in] spaceModeExtents Extents of the (state) space modes acted on.
 * \param[in] batchSize Batch size (>= 1).
 * \param[in] sparsity Tensor operator sparsity defining the storage scheme.
 * \param[in] numDiagonals For multi-diagonal tensor operator matrices,
 * specifies the total number of non-zero diagonals (>= 1).
 * \param[in] diagonalOffsets Offsets of the non-zero diagonals (for example,
 * the main diagonal has offset 0, the diagonal right above the main diagonal
 * has offset +1, the diagonal right below the main diagonal has offset -1, and so on).
 * \param[in] dataType Tensor operator data type.
 * \param[in] tensorData GPU-accessible pointer to the tensor operator elements storage,
 * where all elementary tensor operators within the batch are stored contiguously in memory.
 * \param[in] tensorCallback Optional user-defined batched tensor callback function which can
 * be called later to fill in the tensor elements in the provided batched storage, or NULL.
 * Note that the provided batched tensor callback function is expected to fill in all tensor
 * instances within the batch in one call.
 * \param[in] tensorGradientCallback Optional user-defined batched tensor gradient callback function
 * which can be called later to compute the Vector-Jacobian Product (VJP) for the batched tensor operator,
 * to produce gradients with respect to the batched user-defined real parameters, or NULL.
 * \param[out] elemOperator Batched elementary tensor operator (a batch of individual
 * elementary tensor operators stored contiguously in memory).
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateElementaryOperatorBatch(
                    const cudensitymatHandle_t handle,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    int64_t batchSize,
                    cudensitymatElementaryOperatorSparsity_t sparsity,
                    int32_t numDiagonals,
                    const int32_t diagonalOffsets[],
                    cudaDataType_t dataType,
                    void * tensorData,
                    cudensitymatWrappedTensorCallback_t tensorCallback,
                    cudensitymatWrappedTensorGradientCallback_t tensorGradientCallback,
                    cudensitymatElementaryOperator_t * elemOperator);

/**
 * \brief Destroys an elementary tensor operator (simple or batched).
 * 
 * \param[in] elemOperator Elementary tensor operator.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyElementaryOperator(cudensitymatElementaryOperator_t elemOperator);

/**
 * \brief Creates a full matrix operator acting on all quantum state
 * modes (aka space modes) from a dense matrix stored (replicated)
 * locally on each process.
 *
 * \details A full matrix operator is a single tensor operator
 * represented by its square matrix acting in the full Hilbert space.
 * The dimension of the matrix is equal to the product of the extents
 * of all space modes. The elements of the provided dense matrix are
 * expected to be stored exactly in the same way as if the matrix operator
 * was defined as an elementary tensor operator acting in the full space
 * (acting on all space modes), please see description of the
 * `cudensitymatCreateElementaryOperator` API function.
 *
 * \param[in] handle Library handle.
 * \param[in] numSpaceModes Number of the (state) space modes acted on.
 * It must coincide with the total number of space modes in the Hilbert space.
 * \param[in] spaceModeExtents Extents of the (state) space modes acted on.
 * \param[in] dataType Matrix operator data type.
 * \param[in] matrixData GPU-accessible pointer to the matrix operator elements storage.
 * \param[in] matrixCallback Optional user-defined tensor callback function which can be
 * called later to fill in the matrix elements in the provided storage, or NULL.
 * \param[in] matrixGradientCallback Optional user-defined tensor gradient callback function
 * which can be called later to compute the Vector-Jacobian Product (VJP) for the matrix operator,
 * to produce gradients with respect to the user-defined real parameters, or NULL.
 * \param[out] matrixOperator Full matrix operator.
 * \return cudensitymatStatus_t 
 *
 * \note The optional matrix callback function is still expected to take the actual
 * tensor shape of the defined full matrix operator, namely, the tensor shape induced
 * by the provided array of the space mode extents.
 */
cudensitymatStatus_t cudensitymatCreateMatrixOperatorDenseLocal(
                    const cudensitymatHandle_t handle,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    cudaDataType_t dataType,
                    void * matrixData,
                    cudensitymatWrappedTensorCallback_t matrixCallback,
                    cudensitymatWrappedTensorGradientCallback_t matrixGradientCallback,
                    cudensitymatMatrixOperator_t * matrixOperator);

/**
 * \brief Creates a batch of full matrix operators acting on all quantum
 * state modes (aka space modes) from an array of dense matrices stored
 * (replicated) locally on each process. This is a batched version of the
 * `cudensitymatCreateMatrixOperatorDenseLocal` API function.
 *
 * \details This API function is used to create a batch of full matrix operators
 * stored contiguously in memory, where the batch size must match the batch size
 * of the quantum state acted on. A full matrix operator is a single tensor operator
 * represented by its square matrix acting in the full Hilbert space.
 * The dimension of the matrix is equal to the product of the extents
 * of all space modes. The elements of the provided dense matrix are
 * expected to be stored exactly in the same way as if the matrix operator
 * was defined as an elementary tensor operator acting in the full space
 * (acting on all space modes), please see description of the
 * `cudensitymatCreateElementaryOperatorBatch` API function.
 *
 * \param[in] handle Library handle.
 * \param[in] numSpaceModes Number of the (state) space modes acted on.
 * It must coincide with the total number of space modes in the Hilbert space.
 * \param[in] spaceModeExtents Extents of the (state) space modes acted on.
 * \param[in] batchSize Batch size (>= 1).
 * \param[in] dataType Matrix operator data type.
 * \param[in] matrixData GPU-accessible pointer to the matrix operator elements storage
 * where all matrix operator instances within the batch are stored contiguously in memory.
 * \param[in] matrixCallback Optional user-defined batched tensor callback function which can
 * be called later to fill in the matrix elements in the provided batched storage, or NULL.
 * Note that the provided batched tensor callback function is expected to fill in all matrix
 * instances within the batch in one call.
 * \param[in] matrixGradientCallback Optional user-defined batched tensor gradient callback function
 * which can be called later to compute the Vector-Jacobian Product (VJP) for the batched matrix operator,
 * to produce gradients with respect to the batched user-defined real parameters, or NULL.
 * \param[out] matrixOperator Batched full matrix operator (a batch of full matrix operators).
 * \return cudensitymatStatus_t 
 *
 * \note The optional batched matrix callback function is still expected to take the actual
 * tensor shape of the defined full matrix operator, namely, the tensor shape induced
 * by the provided array of the space mode extents and the corresponding batch size (last mode).
 */
cudensitymatStatus_t cudensitymatCreateMatrixOperatorDenseLocalBatch(
                    const cudensitymatHandle_t handle,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    int64_t batchSize,
                    cudaDataType_t dataType,
                    void * matrixData,
                    cudensitymatWrappedTensorCallback_t matrixCallback,
                    cudensitymatWrappedTensorGradientCallback_t matrixGradientCallback,
                    cudensitymatMatrixOperator_t * matrixOperator);

/**
 * \brief Destroys a full matrix operator (simple or batched).
 * 
 * \param[in] matrixOperator Full matrix operator.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyMatrixOperator(cudensitymatMatrixOperator_t matrixOperator);

/**
 * \brief Creates a matrix product operator (MPO) acting on a subset of quantum
 * state modes (aka space modes).
 *
 * The MPO consists of `numSpaceModes` site tensors, one per site, indexed from
 * 0 to `numSpaceModes - 1`. For open boundary conditions the site tensor mode
 * ordering is:
 *   - Site 0 (leftmost):           [phys_ket, right_bond, phys_bra]
 *   - Sites 1 .. N-2 (interior):   [left_bond, phys_ket, right_bond, phys_bra]
 *   - Site N-1 (rightmost):        [left_bond, phys_ket, phys_bra]
 *
 * where `phys_ket` and `phys_bra` both have extent `spaceModeExtents[i]`,
 * `left_bond` has extent `bondExtents[i-1]`, and `right_bond` has extent
 * `bondExtents[i]`.
 *
 * \param[in] handle Library handle.
 * \param[in] numSpaceModes Number of the (state) space modes acted on.
 * \param[in] spaceModeExtents Extents of the (state) space modes acted on.
 * \param[in] boundaryCondition Boundary condition.
 * \param[in] bondExtents Extents of the bond modes. For open boundary condition,
 * the length of the array must be equal to the number of space modes minus one,
 * where `bondExtents[i]` is the bond dimension between site `i` and site `i+1`.
 * For periodic boundary condition, the length of the array must be equal to the number of space modes.
 * \param[in] dataType Matrix product operator data type.
 * \param[in] tensorData GPU-accessible pointers to the elements of each site tensor
 * constituting the matrix product operator (`tensorData[i]` points to site `i`).
 * \param[in] tensorCallbacks Optional user-defined tensor callback functions (for each MPO tensor)
 * which can be called later to fill in the matrix product operator elements in the provided storage, or NULL.
 * \param[in] tensorGradientCallbacks Optional user-defined tensor gradient callback functions (for each MPO tensor)
 * which can be called later to compute the Vector-Jacobian Product (VJP) for the matrix product operator,
 * to produce gradients with respect to the user-defined real parameters, or NULL.
 * \param[out] matrixProductOperator Matrix product operator.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateMatrixProductOperator(
                    const cudensitymatHandle_t handle,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    cudensitymatBoundaryCondition_t boundaryCondition,
                    const int64_t bondExtents[],
                    cudaDataType_t dataType,
                    void * tensorData[],
                    cudensitymatWrappedTensorCallback_t tensorCallbacks[],
                    cudensitymatWrappedTensorGradientCallback_t tensorGradientCallbacks[],
                    cudensitymatMatrixProductOperator_t * matrixProductOperator);

/**
 * \brief Destroys a matrix product operator (MPO).
 * 
 * \param[in] matrixProductOperator Matrix product operator.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyMatrixProductOperator(cudensitymatMatrixProductOperator_t matrixProductOperator);

/**
 * \brief Creates an empty operator term which is going to be a sum
 * of products of either elementary tensor operators or full matrix
 * operators. Each individual elementary tensor operator within a product
 * acts on a subset of space modes, either from the left or from the right
 * (for each mode). Each full matrix operator within a product acts on all
 * space modes, either from the left or from the right (for all modes).
 *
 * \note The created operator term will only be able to act on the quantum
 * states which reside in the same space where the operator term is set to act.
 *
 * \param[in] handle Library handle.
 * \param[in] numSpaceModes Number of modes (quantum degrees of freedom) defining
 * the primary/dual tensor product space in which the operator term will act.
 * \param[in] spaceModeExtents Extents of the modes (quantum degrees of freedom)
 * defining the primary/dual tensor product space in which the operator term will act.
 * \param[out] operatorTerm Operator term.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateOperatorTerm(
                    const cudensitymatHandle_t handle,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    cudensitymatOperatorTerm_t * operatorTerm);

/**
 * \brief Destroys an operator term.
 * 
 * \param[in] operatorTerm Operator term.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyOperatorTerm(cudensitymatOperatorTerm_t operatorTerm);

/**
 * \brief Appends a product of elementary tensor operators
 * acting on quantum state modes to the operator term.
 *
 * \details The elementary tensor operators constituting the provided
 * tensor operator product are applied to the quantum state in-order,
 * each elementary tensor operator acting on a quantum state mode
 * either on the left or on the right, or both.
 *
 * \param[in] handle Library handle.
 * \param[inout] operatorTerm Operator term.
 * \param[in] numElemOperators Number of elementary tensor operators
 * in the tensor operator product.
 * \param[in] elemOperators Elementary tensor operators constituting
 * the tensor operator product.
 * \param[in] stateModesActedOn State modes acted on by the tensor operator product.
 * This is a concatenated list of the state modes acted on by all constituting elementary
 * tensor operators in the same order how they appear in the elemOperators argument.
 * \param[in] modeActionDuality Duality status of each mode action, that is,
 * whether the action applies to a ket mode of the quantum state (value zero)
 * or a bra mode of the quantum state (positive value).
 * \param[in] coefficient Constant (static) complex scalar coefficient
 * associated with the appended tensor operator product.
 * \param[in] coefficientCallback Optional user-defined complex scalar callback function
 * which can be called later to update the scalar coefficient associated with
 * the tensor operator product, or NULL. The total coefficient associated with
 * the tensor operator product is a product of the constant coefficient and
 * the result of the scalar callback function, if defined.
 * \param[in] coefficientGradientCallback Optional user-defined scalar gradient callback
 * function which can be called later to compute the gradients of the complex scalar coefficient
 * with respect to the user-defined real parameters, or NULL.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorTermAppendElementaryProduct(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperatorTerm_t operatorTerm,
                    int32_t numElemOperators,
                    const cudensitymatElementaryOperator_t elemOperators[],
                    const int32_t stateModesActedOn[],
                    const int32_t modeActionDuality[],
                    cuDoubleComplex coefficient,
                    cudensitymatWrappedScalarCallback_t coefficientCallback,
                    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);

/**
 * \brief Appends a batch of elementary tensor operator products
 * acting on quantum state modes to the operator term.
 *
 * \details This is a batched version of the `cudensitymatOperatorTermAppendElementaryProduct`
 * API function in which the provided static coefficients form an array of some length given
 * by the `batchSize` argument. The provided elementary tensor operators themselves may or
 * may not be batched. If any of them is batched itself, its batch size must match the
 * `batchSize` argument. Furthermore, all non-unity batch sizes must match the batch size
 * of the quantum state acted on.
 *
 * \param[in] handle Library handle.
 * \param[inout] operatorTerm Operator term.
 * \param[in] numElemOperators Number of elementary tensor operators
 * in the tensor operator product.
 * \param[in] elemOperators Elementary tensor operators constituting
 * the tensor operator product (each elementary tensor operator may or may not be batched).
 * \param[in] stateModesActedOn State modes acted on by the tensor operator product.
 * This is a concatenated list of the state modes acted on by all constituting elementary
 * tensor operators in the same order how they appear in the elemOperators argument.
 * \param[in] modeActionDuality Duality status of each mode action, that is,
 * whether the action applies to a ket mode of the quantum state (value zero)
 * or a bra mode of the quantum state (positive value).
 * \param[in] batchSize Batch size (>= 1).
 * \param[in] staticCoefficients GPU-accessible array of constant (static) complex scalar coefficients
 * associated with the appended batch of elementary tensor operator products (of length `batchSize`).
 * \param[in] totalCoefficients GPU-accessible storage for the array of total complex scalar coefficients
 * associated with the appended batch of elementary tensor operator products (of length `batchSize`).
 * Each coefficient will be a product of a static coefficient and a dynamic coefficient generated
 * by the provided scalar callback during the computation phase. If the scalar callback is not
 * supplied here (NULL), this argument can also be set to NULL.
 * \param[in] coefficientCallback Optional user-defined batched complex scalar callback function
 * which can be called later to update the array of dynamic scalar coefficients associated with
 * the defined batch of elementary tensor operator products, or NULL. The total coefficient
 * associated with an elementary tensor operator product is a product of the constant (static)
 * coefficient and the result of the scalar callback function, if defined.
 * \param[in] coefficientGradientCallback Optional user-defined batched scalar gradient callback function
 * which can be called later to compute the gradients of the batched complex scalar coefficients
 * with respect to the batched user-defined real parameters, or NULL.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorTermAppendElementaryProductBatch(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperatorTerm_t operatorTerm,
                    int32_t numElemOperators,
                    const cudensitymatElementaryOperator_t elemOperators[],
                    const int32_t stateModesActedOn[],
                    const int32_t modeActionDuality[],
                    int64_t batchSize,
                    const cuDoubleComplex staticCoefficients[],
                    cuDoubleComplex totalCoefficients[],
                    cudensitymatWrappedScalarCallback_t coefficientCallback,
                    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);

/**
 * \brief Appends a product of full matrix operators, each
 * acting on all quantum state modes, to the operator term.
 *
 * \details The full matrix operators constituting the provided
 * matrix operator product are applied to the quantum state in-order,
 * either on the left or on the rigt, either normal or in the
 * conjugate-transposed form.
 *
 * \param[in] handle Library handle.
 * \param[inout] operatorTerm Operator term.
 * \param[in] numMatrixOperators Number of full matrix operators
 * in the matrix operator product.
 * \param[in] matrixOperators Full matrix operators constituting
 * the matrix operator product.
 * \param[in] matrixConjugation Hermitean conjugation status of each matrix
 * in the matrix operator product (zero means normal, positive integer means
 * conjugate-transposed). For real matrices, hermitean conjugation reduces
 * to a mere matrix transpose since there is no complex conjugation involved.
 * \param[in] actionDuality Duality status of each matrix operator action,
 * that is, whether it acts on all ket modes of the quantum state (value zero)
 * or on all bra modes of the quantum state (positive integer value).
 * \param[in] coefficient Constant (static) complex scalar coefficient associated
 * with the matrix operator product.
 * \param[in] coefficientCallback Optional user-defined complex scalar callback function
 * which can be called later to update the scalar coefficient associated with
 * the matrix operator product, or NULL. The total coefficient associated with
 * the matrix operator product is a product of the constant coefficient and
 * the result of the scalar callback function, if defined.
 * \param[in] coefficientGradientCallback Optional user-defined scalar gradient callback
 * function which can be called later to compute the gradients of the complex scalar coefficient
 * with respect to the user-defined real parameters, or NULL.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorTermAppendMatrixProduct(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperatorTerm_t operatorTerm,
                    int32_t numMatrixOperators,
                    const cudensitymatMatrixOperator_t matrixOperators[],
                    const int32_t matrixConjugation[],
                    const int32_t actionDuality[],
                    cuDoubleComplex coefficient,
                    cudensitymatWrappedScalarCallback_t coefficientCallback,
                    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);

/**
 * \brief Appends a batch of full matrix operators to the operator term,
 * each full matrix operator acting on all quantum state modes. This is
 * a batched version of the `cudensitymatOperatorTermAppendMatrixProduct` API function.
 *
 * \details This is a batched version of the `cudensitymatOperatorTermAppendMatrixProduct`
 * API function in which the provided coefficients form an array of some length given by
 * the `batchSize` argument. The provided full matrix operator itself may or may not be batched.
 * If it is batched, its batch size must match the one the `batchSize` argument. If the full
 * matrix operator itlsef is not batched, the same matrix will be used for all batch instances
 * defined here. Furthermore, all non-unity batch sizes must match the batch size of the quantum
 * state acted on.
 *
 * \param[in] handle Library handle.
 * \param[inout] operatorTerm Operator term.
 * \param[in] numMatrixOperators Number of full matrix operators
 * in the matrix operator product.
 * \param[in] matrixOperators Full matrix operators constituting
 * the matrix operator product (each full matrix operator may or may not be batched).
 * \param[in] matrixConjugation Hermitean conjugation status of each matrix
 * in the matrix operator product (zero means normal, positive integer means
 * conjugate-transposed). For real matrices, hermitean conjugation reduces
 * to a mere matrix transpose since there is no complex conjugation involved.
 * \param[in] actionDuality Duality status of each matrix operator action,
 * that is, whether it acts on all ket modes of the quantum state (value zero)
 * or on all bra modes of the quantum state (positive integer value).
 * \param[in] batchSize Batch size (>= 1).
 * \param[in] staticCoefficients GPU-accessible array of constant (static) complex scalar coefficients
 * associated with the appended batch of full matrix operator products (of length `batchSize`).
 * \param[in] totalCoefficients GPU-accessible storage for the array of total complex scalar coefficients
 * associated with the appended batch of full matrix operator products (of length `batchSize`).
 * Each coefficient will be a product of a static coefficient and a dynamic coefficient generated
 * by the provided scalar callback during the computation phase. If the scalar callback is not
 * supplied here (NULL), this argument can also be set to NULL.
 * \param[in] coefficientCallback Optional user-defined batched complex scalar callback function
 * which can be called later to update the array of dynamic scalar coefficients associated with
 * the defined batch of full matrix operator products, or NULL. The total coefficient
 * associated with an elementary tensor operator product is a product of the constant
 * (static) coefficient and the result of the scalar callback function, if defined.
 * \param[in] coefficientGradientCallback Optional user-defined batched scalar gradient callback function
 * which can be called later to compute the gradients of the batched complex scalar coefficients
 * with respect to the batched user-defined real parameters, or NULL.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorTermAppendMatrixProductBatch(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperatorTerm_t operatorTerm,
                    int32_t numMatrixOperators,
                    const cudensitymatMatrixOperator_t matrixOperators[],
                    const int32_t matrixConjugation[],
                    const int32_t actionDuality[],
                    int64_t batchSize,
                    const cuDoubleComplex staticCoefficients[],
                    cuDoubleComplex totalCoefficients[],
                    cudensitymatWrappedScalarCallback_t coefficientCallback,
                    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);

/**
 * \brief Appends a product of matrix product operators (MPOs) to the operator term.
 *
 * \param[in] handle Library handle.
 * \param[inout] operatorTerm Operator term.
 * \param[in] numMPOOperators Number of MPO operators in the MPO product.
 * \param[in] mpoOperators MPO operators constituting the MPO product.
 * \param[in] mpoConjugation Hermitean conjugation status of each MPO in the MPO product
 * (zero means normal, positive integer means conjugate-transposed).
 * \param[in] stateModesActedOn State modes acted on by the product of matrix product operators.
 * This is a concatenated list of the state modes acted on by all constituting MPO operators
 * in the same order how they appear in the mpoOperators argument.
 * \param[in] modeActionDuality Duality status of each mode action, that is,
 * whether the action applies to a ket mode of the quantum state (value zero)
 * or a bra mode of the quantum state (positive value).
 * \param[in] coefficient Constant (static) complex scalar coefficient associated
 * with the MPO product.
 * \param[in] coefficientCallback Optional user-defined complex scalar callback function
 * which can be called later to update the scalar coefficient associated with
 * the MPO product, or NULL. The total coefficient associated with the MPO product
 * is a product of the constant coefficient and the result of the scalar callback function, if defined.
 * \param[in] coefficientGradientCallback Optional user-defined scalar gradient callback function
 * function which can be called later to compute the gradients of the complex scalar coefficient
 * with respect to the user-defined real parameters, or NULL.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorTermAppendMPOProduct(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperatorTerm_t operatorTerm,
                    int32_t numMPOOperators,
                    const cudensitymatMatrixProductOperator_t mpoOperators[],
                    const int32_t mpoConjugation[],
                    const int32_t stateModesActedOn[],
                    const int32_t modeActionDuality[],
                    cuDoubleComplex coefficient,
                    cudensitymatWrappedScalarCallback_t coefficientCallback,
                    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);

/**
 * \brief Creates an empty operator which is going to be
 * a collection of operator terms with some coefficients.
 *
 * \details If the operator is expected to act on a pure quantum state,
 * it is just a regular operator that will act on the pure state vector
 * from one side. If the operator is expected to act on a mixed quantum state,
 * its action can become more complicated where it may act on both sides
 * of the density matrix representing the mixed quantum state. In this case,
 * the operator is specifically called the super-operator. However, one
 * should note that in both cases this is still a mathematical operator,
 * just acting on a different kind of mathematical vector.
 *
 * \note The created operator will only be able to act on the quantum states
 * which reside in the same space where the operator is set to act.
 *
 * \param[in] handle Library handle.
 * \param[in] numSpaceModes Number of modes (degrees of freedom) defining
 * the primary/dual tensor product space in which the operator term will act.
 * \param[in] spaceModeExtents Extents of the modes (degrees of freedom) defining
 * the primary/dual tensor product space in which the operator term will act.
 * \param[out] superoperator Operator.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateOperator(
                    const cudensitymatHandle_t handle,
                    int32_t numSpaceModes,
                    const int64_t spaceModeExtents[],
                    cudensitymatOperator_t * superoperator);

/**
 * \brief Destroys an operator.
 * 
 * \param[in] superoperator Operator.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyOperator(cudensitymatOperator_t superoperator);
                    
/**
 * \brief Appends an operator term to the operator.
 * 
 * \param[in] handle Library handle.
 * \param[inout] superoperator Operator.
 * \param[in] operatorTerm Operator term.
 * \param[in] duality Duality status of the operator term action as a whole.
 * If not zero, the duality status of each mode action inside the operator
 * term will be flipped, that is, action from the left will be replaced by
 * action from the right, and vice versa.
 * \param[in] coefficient Constant (static) complex scalar coefficient
 * associated with the operator term.
 * \param[in] coefficientCallback Optional user-defined complex scalar callback function
 * which can be called later to update the scalar coefficient associated with
 * the operator term, or NULL. The total coefficient associated with
 * the operator term is a product of the constant coefficient and
 * the result of the scalar callback function, if defined.
 * \param[in] coefficientGradientCallback Optional user-defined scalar gradient callback
 * function which can be called later to compute the gradients of the complex scalar coefficient
 * with respect to the user-defined real parameters, or NULL.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorAppendTerm(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    cudensitymatOperatorTerm_t operatorTerm,
                    int32_t duality,
                    cuDoubleComplex coefficient,
                    cudensitymatWrappedScalarCallback_t coefficientCallback,
                    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);

/**
 * \brief Appends a batch of operator terms to the operator.
 *
 * \details This is a batched version of the `cudensitymatOperatorAppendTerm`
 * API function in which the provided coefficients form an array of some length
 * given by the `batchSize` argument. The operator term itself may or may not be
 * batched, that is, it may or may not contain batched components (batched
 * elementary tensor operator products or batched full matrix operator products).
 * If any of its components is batched, the batch sizes must match. They also
 * must match the batch size of the quantum state acted on. If any of the operator
 * terms or elementary tensor operators or full matrix operators inside any operator
 * term is not batched, its value will be replicated along the batch dimension.
 *
 * \param[in] handle Library handle.
 * \param[inout] superoperator Operator.
 * \param[in] operatorTerm Operator term.
 * \param[in] duality Duality status of the operator term action as a whole.
 * If not zero, the duality status of each mode action inside the operator
 * term will be flipped, that is, action from the left will be replaced by
 * action from the right, and vice versa.
 * \param[in] batchSize Batch size (>= 1).
 * \param[in] staticCoefficients GPU-accessible array of constant (static) complex scalar
 * coefficients associated with the appended batch of operator terms (of length `batchSize`).
 * \param[in] totalCoefficients GPU-accessible storage for the array of total complex scalar
 * coefficients associated with the appended batch of operator terms (of length `batchSize`).
 * Each coefficient will be a product of a static coefficient and a dynamic coefficient
 * generated by the coefficient callback during the computation phase. If the scalar callback
 * is not supplied here (NULL), this argument can also be set to NULL.
 * \param[in] coefficientCallback Optional user-defined batched complex scalar callback
 * function which can be called later to update the array of scalar coefficients
 * associated with the defined batch of operator terms, or NULL. The total coefficient
 * associated with an operator term is a product of the constant (static) coefficient
 * and the result of the scalar callback function, if defined.
 * \param[in] coefficientGradientCallback Optional user-defined batched scalar gradient callback
 * function which can be called later to compute the gradients of the batched complex scalar coefficients
 * with respect to the batched user-defined real parameters, or NULL.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorAppendTermBatch(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    cudensitymatOperatorTerm_t operatorTerm,
                    int32_t duality,
                    int64_t batchSize,
                    const cuDoubleComplex staticCoefficients[],
                    cuDoubleComplex totalCoefficients[],
                    cudensitymatWrappedScalarCallback_t coefficientCallback,
                    cudensitymatWrappedScalarGradientCallback_t coefficientGradientCallback);

/**
 * \brief Attaches batched coefficients to the operator's term and product coefficients.
 *
 * \details This function is used to attach batched coefficients to the operator's term and product coefficients.
 * This is used when the batched coefficients are unavailable during the preparation phase, but are available during the action phase.
 * A temporary buffer is used as a placeholder for the batched coefficients until the operator is prepared for action.
 * This API function maps the temporary buffer to the actual batched coefficients buffer and will be used during the action phase.
 * \param[in] handle Library handle.
 * \param[inout] superoperator Operator.
 * \param[in] numOperatorTermBatchedCoeffs Number of batched coefficients in the operator term. 
 * \param[in] operatorTermBatchedCoeffsTmp Temporary buffer for the batched coefficients in the operator term.
 * \param[in] operatorTermBatchedCoeffs Actual buffer for the batched coefficients in the operator term.
 * \param[in] numOperatorProductBatchedCoeffs Number of batched coefficients in the operator product. 
 * \param[in] operatorProductBatchedCoeffsTmp Temporary buffer for the batched coefficients in the operator product.
 * \param[in] operatorProductBatchedCoeffs Actual buffer for the batched coefficients in the operator product.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatAttachBatchedCoefficients(
                      const cudensitymatHandle_t handle,
                      cudensitymatOperator_t superoperator,
                      int32_t numOperatorTermBatchedCoeffs,
                      void * operatorTermBatchedCoeffsTmp[],
                      void * operatorTermBatchedCoeffs[],
                      int32_t numOperatorProductBatchedCoeffs,
                      void * operatorProductBatchedCoeffsTmp[],
                      void * operatorProductBatchedCoeffs[]);

/**
 * \brief Configures the operator action on a quantum state.
 *
 * \param[in] handle Library handle.
 * \param[inout] superoperator Operator.
 * \param[in] stateIn Representative input quantum state on which the operator
 * is supposed to act. The actual quantum state acted on during computation
 * may be different, but it has to be of the same shape, kind,
 * and factorization structure (topology, bond dimensions, etc).
 * \param[in] stateOut Representative output quantum state produced by the action
 * of the operator on the input quantum state. The actual quantum state acted on
 * during computation may be different, but it has to be of the same shape,
 * kind, and factorization structure (topology, bond dimensions, etc).
 * \param[in] attribute Configuration attribute.
 * \param[in] attributeValue Pointer to the configuration attribute value (type-erased).
 * \param[in] attributeSize The size of the configuration attribute value.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorConfigureAction(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    const cudensitymatState_t stateIn,
                    const cudensitymatState_t stateOut,
                    //cudensitymatOperatorActionAttributes_t attribute, //`FIXME
                    const void * attributeValue,
                    size_t attributeSize);

/**
 * \brief Prepares the operator for an action on a quantum state.
 *
 * \details In general, before the operator action on a specific
 * quantum state(s) can be computed, it needs to be prepared
 * for computation first, which is the purpose of this API function.
 * 
 * \param[in] handle Library handle.
 * \param[inout] superoperator Operator.
 * \param[in] stateIn Representative input quantum state on which the operator
 * is supposed to act. The actual quantum state acted on during computation
 * may be different, but it has to be of the same shape, kind,
 * and factorization structure (topology, bond dimensions, etc).
 * \param[in] stateOut Representative output quantum state produced by the action
 * of the operator on the input quantum state. The actual quantum state acted on
 * during computation may be different, but it has to be of the same shape,
 * kind, and factorization structure (topology, bond dimensions, etc).
 * \param[in] computeType Desired compute type.
 * \param[in] workspaceSizeLimit Workspace buffer size limit (bytes).
 * \param[inout] workspace Empty workspace descriptor on entrance.
 * The workspace size required for the computation will be set on exit.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 *
 * \note The required size of the workspace buffer returned inside
 * the workspace descriptor may sometimes be zero, in which case
 * there is no need to allocate a workspace buffer.
 */
cudensitymatStatus_t cudensitymatOperatorPrepareAction(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    const cudensitymatState_t stateIn,
                    const cudensitymatState_t stateOut,
                    cudensitymatComputeType_t computeType,
                    size_t workspaceSizeLimit,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/**
 * \brief Computes the action of the operator on a given input quantum state,
 * accumulating the result in the output quantum state (accumulative action).
 *
 * \note The provided input and output quantum states must be of the same
 * kind, shape, and structure as the quantum states provided during
 * the preceding preparation phase.
 * 
 * \param[in] handle Library handle.
 * \param[inout] superoperator Operator.
 * \param[in] time Time value.
 * \param[in] batchSize Batch size (>=1).
 * \param[in] numParams Number of variable parameters defined by the user.
 * \param[in] params GPU-accessible pointer to an F-order 2d-array
 * of user-defined real parameter values: params[numParams, batchSize].
 * \param[in] stateIn Input quantum state (or a batch of input quantum states).
 * \param[inout] stateOut Updated resulting quantum state which
 * accumulates operator action on the input quantum state.
 * \param[in] workspace Allocated workspace descriptor.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 *
 * \warning The output quantum state cannot coincide with the input quantum state.
 */
cudensitymatStatus_t cudensitymatOperatorComputeAction(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    double time,
                    int64_t batchSize,
                    int32_t numParams,
                    const double * params,
                    const cudensitymatState_t stateIn,
                    cudensitymatState_t stateOut,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/**
 * \brief Prepares backward differentiation of the operator action on a quantum state.
 *
 * \details In general, before the backward differentiation of the operator
 * action on a specific quantum state(s) can be computed, it needs to be prepared
 * for computation first, which is the purpose of this API function.
 *
 * \param[in] handle Library handle.
 * \param[inout] superoperator Operator.
 * \param[in] stateIn Representative input quantum state on which the operator
 * is supposed to act. The actual quantum state acted on during computation
 * may be different, but it has to be of the same shape, kind,
 * and factorization structure (topology, bond dimensions, etc).
 * \param[in] stateOutAdj Representative adjoint of the output quantum state
 * produced by the action of the operator on the input quantum state. The actual
 * output quantum state acted on during computation may be different, but it has to be
 * of the same shape, kind, and factorization structure (topology, bond dimensions, etc).
 * \param[in] computeType Desired compute type.
 * \param[in] workspaceSizeLimit Workspace buffer size limit (bytes).
 * \param[inout] workspace Empty workspace descriptor on entrance.
 * The workspace size required for the computation will be set on exit.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 *
 * \note The required size of the workspace buffer returned inside
 * the workspace descriptor may sometimes be zero, in which case
 * there is no need to allocate a workspace buffer.
 */
cudensitymatStatus_t cudensitymatOperatorPrepareActionBackwardDiff(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    const cudensitymatState_t stateIn,
                    const cudensitymatState_t stateOutAdj,
                    cudensitymatComputeType_t computeType,
                    size_t workspaceSizeLimit,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/**
 * \brief Computes backward differentiation of the operator action on a given quantum state.
 *
 * \details In general, an operator can be parameterized by a set of user-defined real parameters
 * which parameterize elementary tensor operators, matrix operators, and coefficients
 * which are used to define the operator. This function computes the partial derivatives
 * of the operator action on a quantum state with respect to the user-defined real parameters.
 * Specifically, the function computes the Vector-Jacobian Product (VJP) of the operator action
 * on the quantum state with respect to the user-defined real parameters. That is, given the adjoint
 * of the output quantum state, the function computes the following quantities:
 *   - partial derivatives of the cost function with respect to the user-defined real parameters
 *     which parameterize the operator (explicit derivatives).
 *   - adjoint of the input quantum state (implicit derivatives).
 * Symbolically, this can be expressed as follows:
 *
 *   Operator action:
 *     y[i] += A(p)[i,j] * x[j]
 *   where A(p) is the operator parameterized by N user-defined real parameters p[n], 1 <= n <= N,
 *   y is the output quantum state, and x is the input quantum state.
 *
 *   Forward derivatives:
 *     dy[i] = A[i,j] * dx[j] + dA[i,k] * x[k]
 *
 *   Backward derivatives (these two are computed by the current API function):
 *     Input quantum state adjoint: dc/dx[j] = dc/dy[i] * A[i,j]
 *     Partial operator derivatives: dc/dp[n] = dc/dy[i] * (dA[i,k] * x[k])/dp[n]
 *
 *     Given the above two quantities, the full operator action derivatives can be constructed as:
 *     Full operator action derivatives: dc/dp[n] = dc/dx[j] * dx[j]/dp[n] +
 *                                                  (dc/dy[i] * (dA[i,k] * x[k])/dp[n])
 *
 * When computing the partial operator derivatives, the term (dA[i,k] * x[k])/dp[n]
 * is computed as a sum of contributions over all differentiable elementary tensor
 * operators, matrix operators, and complex scalar coefficients which parameterize
 * the operator, as follows:
 *   dc/dp[n] = (dA[i,k] * x[k])/dp[n] = sum_j (dA[i,k] * x[k])/qQ[j] * dQ[j]/dp[n]
 * where Q[j] is a specific differentiable quantity (elementary tensor operator, matrix operator,
 * or complex scalar coefficient) which parameterizes the operator, and dQ[j]/dp[n]
 * is the partial derivative of Q[j] with respect to the n-th user-defined real parameter p[n].
 * Each dQ[j]/dp[n] is specified by a user-provided scalar/tensor gradient callback function
 * which takes the adjoint of the corresponding quantity Q[j] as an input argument (`scalarGrad`
 * or `tensorGrad`), multiplies it by the dQ[j]/dp[n] Jacobian, and accumulates the REAL part
 * of the result multipled by TWO (since the total cost function is always real) in the output array
 * of partial derivatives with respect to the user-defined real parameters (argument `paramsGrad`).
 * If the state/operator is batched, the partial derivatives are computed for the entire batch,
 * in which case the `paramsGrad` array must have shape [numParams, batchSize] in the F-order,
 * while the `scalarGrad` and `tensorGrad` arrays can be either batched or non-batched, based
 * on the corresponding quantity Q[j] being batched or non-batched. In case the quantity Q[j]
 * is not batched while the quantum states are batched, the `scalarGrad` and `tensorGrad` arrays
 * passed to the gradient callback function will not be batched, but the `params` and `paramsGrad`
 * arrays will be batched, but only their first batch instance will actually be used (argument
 * batchSize will be set to 1 in this case), specifically, only `params[0:N-1, 0]` and
 * `paramsGrad[0:N-1, 0]` will be used.
 * 
 * \param[in] handle Library handle.
 * \param[inout] superoperator Operator.
 * \param[in] time Time value.
 * \param[in] batchSize Batch size (>=1).
 * \param[in] numParams Number of variable real parameters defined by the user.
 * \param[in] params GPU-accessible pointer to an F-order 2d-array
 * of user-defined real parameter values: params[numParams, batchSize].
 * \param[in] stateIn Input quantum state (or a batch).
 * \param[in] stateOutAdj Adjoint of the output quantum state (or a batch).
 * \param[inout] stateInAdj Adjoint of the input quantum state (or a batch).
 * Note that this array will not be zeroed out on entrance, it will be accumulated into.
 * \param[inout] paramsGrad GPU-accessible pointer where the partial derivatives with respect
 * to the user-defined real parameters will be accumulated (same shape as params).
 * Note that this array will not be zeroed out on entrance, it will be accumulated into.
 * \param[in] workspace Allocated workspace descriptor.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorComputeActionBackwardDiff(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    double time,
                    int64_t batchSize,
                    int32_t numParams,
                    const double * params,
                    const cudensitymatState_t stateIn,
                    const cudensitymatState_t stateOutAdj,
                    cudensitymatState_t stateInAdj,
                    double * paramsGrad,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/**
 * \brief Creates an action descriptor for one or more operators,
 * thus defining an aggregate action of the operator(s) on a set
 * of input quantum states compliant with the operator domains,
 * where all input quantum states can also be batched.
 *
 * \details Specification of an operator itself is generally insufficient
 * for specifying the r.h.s. of the desired ordinary differential equation (ODE)
 * defining the evolution of the quantum state in time. In general,
 * the ODE r.h.s. specification requires specifying the action of one or more
 * operators on one or more (batched) quantum states (normally, density matrices).
 * The abstraction of the `OperatorAction` serves exactly this purpose.
 * When the aggregate operator action is computed, each provided operator
 * will act on its own input quantum state producing a contribution
 * to the same output quantum state.
 *
 * \note Sometimes one needs to solve a coupled system of ordinary
 * differential equations where a number of quantum states are
 * simultaneously evolved in time. In such a case, not all quantum
 * states have to affect the evolution of a given one of them.
 * To handle such cases, some of the operator-state products,
 * which do not contribute, can be set to zero by setting the
 * corresponding entry of the operators[] argument to NULL.
 * 
 * \param[in] handle Library handle.
 * \param[in] numOperators Number of operators involved (number of operator-state products).
 * \param[in] operators Constituting operator(s) with the same domain of action.
 * Some of the operators may be set to NULL to represent zero action on a specific
 * input quantum state.
 * \param[out] operatorAction Operator action.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateOperatorAction(
                    const cudensitymatHandle_t handle,
                    int32_t numOperators,
                    cudensitymatOperator_t operators[],
                    cudensitymatOperatorAction_t * operatorAction);

/**
 * \brief Destroys the operator action descriptor.
 * 
 * \param[in] operatorAction Operator action.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyOperatorAction(cudensitymatOperatorAction_t operatorAction);

/**
 * \brief Prepares the (aggregate) operator(s) action for computation.
 *
 * \details In general, before the (aggregate) operator(s) action
 * on specific quantum states can be computed, it needs to be prepared
 * for computation first, which is the purpose of this API function.
 *
 * \param[in] handle Library handle.
 * \param[inout] operatorAction Operator(s) action specification.
 * \param[in] stateIn Input quantum state(s) for all operator(s)
 * defining the current Operator Action. Each input quantum state
 * can be a batch of quantum states itself (with the same batch size).
 * \param[in] stateOut Updated output quantum state (or a batch) which
 * accumulates the (aggregate) operator(s) action on all input quantum state(s).
 * \param[in] computeType Desired compute type.
 * \param[in] workspaceSizeLimit Workspace buffer size limit (bytes).
 * \param[inout] workspace Empty workspace descriptor on entrance.
 * The workspace size required for the computation will be set on exit.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 *
 * \note The required size of the workspace buffer returned inside
 * the workspace descriptor may sometimes be zero, in which case
 * there is no need to allocate a workspace buffer.
 */
cudensitymatStatus_t cudensitymatOperatorActionPrepare(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperatorAction_t operatorAction,
                    const cudensitymatState_t stateIn[],
                    const cudensitymatState_t stateOut,
                    cudensitymatComputeType_t computeType,
                    size_t workspaceSizeLimit,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/**
 * \brief Executes the action of one or more operators constituting
 * the aggreggate operator(s) action on the same number of input
 * quantum states, accumulating the results into a single output
 * quantum state.
 * 
 * \param[in] handle Library handle.
 * \param[inout] operatorAction Operator(s) action.
 * \param[in] time Time value.
 * \param[in] batchSize Batch size (>=1).
 * \param[in] numParams Number of variable parameters defined by the user.
 * \param[in] params GPU-accessible pointer to an F-order 2d-array
 * of user-defined real parameter values: params[numParams, batchSize].
 * \param[in] stateIn Input quantum state(s). Each input quantum state
 * can be a batch of quantum states, in general.
 * \param[inout] stateOut Updated output quantum state which
 * accumulates operator action(s) on all input quantum state(s).
 * \param[in] workspace Allocated workspace descriptor.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t
 *
 * \note The output quantum state cannot be one of the input quantum states. 
 */
cudensitymatStatus_t cudensitymatOperatorActionCompute(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperatorAction_t operatorAction,
                    double time,
                    int64_t batchSize,
                    int32_t numParams,
                    const double * params,
                    const cudensitymatState_t stateIn[],
                    cudensitymatState_t stateOut,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/** \} end operatorAPI */

/**
 * \defgroup expectationAPI Operator expectation value API
 * \{
 */

/**
 * \brief Creates the operator expectation value computation object.
 *
 * \note The unnormalized expectation value will be produced
 * during the computation. If the quantum state is not normalized,
 * one will need to additionally compute the state norm or trace
 * in order to obtain the normalized operator expectation value.
 *
 * \param[in] handle Library handle.
 * \param[in] superoperator Operator.
 * \param[out] expectation Expectation value computation object.
 * \return cudensitymatStatus_t 
 * 
 * \note The operator must stay alive during the lifetime of the created expectation object.
 */
cudensitymatStatus_t cudensitymatCreateExpectation(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    cudensitymatExpectation_t * expectation);

/**
 * \brief Destroys an expectation value computation object.
 * 
 * \param[in] expectation Expectation value computation object.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyExpectation(cudensitymatExpectation_t expectation);

/**
 * \brief Prepares the expectation value object for computation.
 *
 * \details In general, before the expectation value can be computed,
 * it needs to be prepared for computation first, which is the purpose
 * of this API function.
 * 
 * \param[in] handle Library handle.
 * \param[inout] expectation Expectation value object.
 * \param[in] state Representative quantum state (or a batch of quantum states).
 * \param[in] computeType Desired compute type.
 * \param[in] workspaceSizeLimit Workspace buffer size limit (bytes).
 * \param[inout] workspace Empty workspace descriptor on entrance.
 * The workspace size required for the computation will be set on exit.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 *
 * \note The required size of the workspace buffer returned inside
 * the workspace descriptor may sometimes be zero, in which case
 * there is no need to allocate a workspace buffer.
 */
cudensitymatStatus_t cudensitymatExpectationPrepare(
                    const cudensitymatHandle_t handle,
                    cudensitymatExpectation_t expectation,
                    const cudensitymatState_t state,
                    cudensitymatComputeType_t computeType,
                    size_t workspaceSizeLimit,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/**
 * \brief Computes the operator expectation value(s) with respect to the given quantum state(s).
 * 
 * \details The result is generally a vector of dimension equal to the state batch size.
 * 
 * \param[in] handle Library handle.
 * \param[inout] expectation Expectation value object.
 * \param[in] time Specified time.
 * \param[in] batchSize Batch size (>=1).
 * \param[in] numParams Number of variable parameters defined by the user.
 * \param[in] params GPU-accessible pointer to an F-order 2d-array
 * of user-defined real parameter values: params[numParams, batchSize].
 * \param[in] state Quantum state (or a batch of quantum states).
 * \param[out] expectationValue Pointer to the expectation value(s) vector storage
 * in GPU-accessible RAM of the same data type as used by the state and operator.
 * \param[in] workspace Allocated workspace descriptor.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatExpectationCompute(
                    const cudensitymatHandle_t handle,
                    cudensitymatExpectation_t expectation,
                    double time,
                    int64_t batchSize,
                    int32_t numParams,
                    const double * params,
                    const cudensitymatState_t state,
                    void * expectationValue,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/** \} end expectationAPI */

/**
 * \defgroup spectrumAPI Operator eigen-spectrum API
 * \{
 */

/**
 * \brief Creates the eigen-spectrum computation object for a given operator.
 *
 * \param[in] handle Library handle.
 * \param[in] superoperator Operator (cannot be batched).
 * \param[in] isHermitian Specifies whether the operator is Hermitian (!=0) or not (0).
 * \param[in] spectrumKind Requested kind of the eigen-spectrum computation.
 * \param[out] spectrum Eigen-spectrum computation object.
 * \return cudensitymatStatus_t 
 * 
 * \note The operator must stay alive during the lifetime of the created eigen-spectrum object.
 */
cudensitymatStatus_t cudensitymatCreateOperatorSpectrum(
                    const cudensitymatHandle_t handle,
                    const cudensitymatOperator_t superoperator,
                    int32_t isHermitian,
                    cudensitymatOperatorSpectrumKind_t spectrumKind,
                    cudensitymatOperatorSpectrum_t * spectrum);

/**
* \brief Destroys an eigen-spectrum computation object.
* 
* \param[in] spectrum Eigen-spectrum computation object.
* \return cudensitymatStatus_t 
*/
cudensitymatStatus_t cudensitymatDestroyOperatorSpectrum(cudensitymatOperatorSpectrum_t spectrum);

/**
 * \brief Configures the eigen-spectrum computation object.
 * 
 * \param[in] handle Library handle.
 * \param[inout] spectrum Eigen-spectrum computation object.
 * \param[in] attribute Attribute to configure.
 * \param[in] attributeValue CPU-accessible pointer to the attribute value.
 * \param[in] attributeValueSize Size of the attribute value in bytes.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatOperatorSpectrumConfigure(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperatorSpectrum_t spectrum,
                    cudensitymatOperatorSpectrumConfig_t attribute,
                    const void * attributeValue,
                    size_t attributeValueSize);

/**
* \brief Prepares the eigen-spectrum object for computation.
*
* \details In general, before the eigen-spectrum can be computed,
* it needs to be prepared for computation first (once), which is
* the purpose of this API function.
* 
* \param[in] handle Library handle.
* \param[inout] spectrum Eigen-spectrum computation object.
* \param[in] maxEigenStates Maximum number of eigen-pairs to compute.
* \param[in] state Representative quantum state (cannot be batched).
* \param[in] computeType Desired compute type.
* \param[in] workspaceSizeLimit Workspace buffer size limit (bytes).
* \param[inout] workspace Empty workspace descriptor on entrance.
* The workspace buffer sizes required for the computation will be set on return.
* \param[in] stream CUDA stream.
* \return cudensitymatStatus_t 
*
*/
cudensitymatStatus_t cudensitymatOperatorSpectrumPrepare(
  const cudensitymatHandle_t handle,
  cudensitymatOperatorSpectrum_t spectrum,
  int32_t maxEigenStates,
  const cudensitymatState_t state,
  cudensitymatComputeType_t computeType,
  size_t workspaceSizeLimit,
  cudensitymatWorkspaceDescriptor_t workspace,
  cudaStream_t stream);

/**
* \brief Computes the eigen-spectrum of an operator.
*
* \details Computes a requested number of eigen-pairs of the operator
* encapsulated inside the eigen-spectrum computation object.
*
* \param[in] handle Library handle.
* \param[inout] spectrum Eigen-spectrum computation object.
* \param[in] time Specified time.
* \param[in] batchSize Batch size (==1).
* \param[in] numParams Number of variable parameters defined by the user.
* \param[in] params GPU-accessible pointer to an F-order 2d-array
* of user-defined real parameter values: params[numParams, batchSize].
* \param[in] numEigenStates Actual number of eigenstates to compute,
* which must not exceed the value of the `maxEigenStates` parameter
* provided during the preparation of the eigen-spectrum computation object.
* \param[inout] eigenstates Quantum eigenstates (cannot be batched).
* The initial values of the provided quantum states will be used
* as the initial guesses for the first Krylov subspace block (if the block
* size is smaller than the number of requested eigenstates, only the leading
* quantum states will be used).
* \param[out] eigenvalues Pointer to the eigenvalues storage (F-order array
* of shape [numEigenStates, batchSize]) in GPU-accessible RAM (same data type
* as used by the quantum state and operator).
* \param[inout] tolerances Pointer to an F-order array of shape [numEigenStates, batchSize]
* in CPU-accessible RAM. The initial values represent the desirable convergence tolerances
* for all eigen-states. The returned values represent the actually achieved residual norms
* for all eigen-states.
* \param[in] workspace Allocated workspace descriptor.
* \param[in] stream CUDA stream.
* \return cudensitymatStatus_t
*
* \note The initial quantum states passed via the `eigenstates` parameter
* must form a linearly independent set (but it does not have to be orthonormal).
*/
cudensitymatStatus_t cudensitymatOperatorSpectrumCompute(
  const cudensitymatHandle_t handle,
  cudensitymatOperatorSpectrum_t spectrum,
  double time,
  int64_t batchSize,
  int32_t numParams,
  const double * params,
  int32_t numEigenStates,
  cudensitymatState_t eigenstates[],
  void * eigenvalues,
  double * tolerances,
  cudensitymatWorkspaceDescriptor_t workspace,
  cudaStream_t stream);

/** \} end spectrumAPI */

/**
 * \defgroup propagatorAPI Time propagation API
 * \{
 */

/*
// ============================================================================
// Full Exact Configuration API
// ============================================================================

cudensitymatStatus_t cudensitymatCreateTimePropagationScopeFullExactConfig(
                    const cudensitymatHandle_t handle,
                    cudensitymatTimePropagationScopeFullExactConfig_t * config);

cudensitymatStatus_t cudensitymatDestroyTimePropagationScopeFullExactConfig(cudensitymatTimePropagationScopeFullExactConfig_t config);

cudensitymatStatus_t cudensitymatTimePropagationScopeFullExactConfigSetAttribute(
                    const cudensitymatHandle_t handle,
                    cudensitymatTimePropagationScopeFullExactConfig_t config,
                    cudensitymatTimePropagationScopeFullExactConfigAttribute_t attribute,
                    const void * attributeValue,
                    size_t attributeSize);

cudensitymatStatus_t cudensitymatTimePropagationScopeFullExactConfigGetAttribute(
                    const cudensitymatHandle_t handle,
                    const cudensitymatTimePropagationScopeFullExactConfig_t config,
                    cudensitymatTimePropagationScopeFullExactConfigAttribute_t attribute,
                    void * attributeValue,
                    size_t attributeSize);
*/

// ============================================================================
// TDVP Configuration API
// ============================================================================

/**
 * \brief Creates a TDVP configuration object with default settings.
 *
 * \param[in] handle Library handle.
 * \param[out] config TDVP configuration object.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatCreateTimePropagationScopeSplitTDVPConfig(
                    const cudensitymatHandle_t handle,
                    cudensitymatTimePropagationScopeSplitTDVPConfig_t * config);

/**
 * \brief Destroys a TDVP configuration object.
 *
 * \param[in] config TDVP configuration object.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatDestroyTimePropagationScopeSplitTDVPConfig(cudensitymatTimePropagationScopeSplitTDVPConfig_t config);

/**
 * \brief Sets an attribute of the TDVP configuration.
 *
 * \param[in] handle Library handle.
 * \param[in] config TDVP configuration object.
 * \param[in] attribute Attribute to set.
 * \param[in] attributeValue Pointer to the attribute value.
 * \param[in] attributeSize Size of the attribute value in bytes.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatTimePropagationScopeSplitTDVPConfigSetAttribute(
                    const cudensitymatHandle_t handle,
                    cudensitymatTimePropagationScopeSplitTDVPConfig_t config,
                    cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t attribute,
                    const void * attributeValue,
                    size_t attributeSize);

/**
 * \brief Gets an attribute of the TDVP configuration.
 *
 * \param[in] handle Library handle.
 * \param[in] config TDVP configuration object.
 * \param[in] attribute Attribute to get.
 * \param[out] attributeValue Pointer to store the attribute value.
 * \param[in] attributeSize Size of the buffer in bytes.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatTimePropagationScopeSplitTDVPConfigGetAttribute(
                    const cudensitymatHandle_t handle,
                    const cudensitymatTimePropagationScopeSplitTDVPConfig_t config,
                    cudensitymatTimePropagationScopeSplitTDVPConfigAttribute_t attribute,
                    void * attributeValue,
                    size_t attributeSize);

// ============================================================================
// Krylov Configuration API
// ============================================================================

/**
 * \brief Creates a Krylov configuration object with default settings.
 *
 * \param[in] handle Library handle.
 * \param[out] config Krylov configuration object.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatCreateTimePropagationApproachKrylovConfig(
                    const cudensitymatHandle_t handle,
                    cudensitymatTimePropagationApproachKrylovConfig_t * config);

/**
 * \brief Destroys a Krylov configuration object.
 *
 * \param[in] config Krylov configuration object.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatDestroyTimePropagationApproachKrylovConfig(cudensitymatTimePropagationApproachKrylovConfig_t config);

/**
 * \brief Sets an attribute of the Krylov configuration.
 *
 * \param[in] handle Library handle.
 * \param[in] config Krylov configuration object.
 * \param[in] attribute Attribute to set.
 * \param[in] attributeValue Pointer to the attribute value.
 * \param[in] attributeSize Size of the attribute value in bytes.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatTimePropagationApproachKrylovConfigSetAttribute(
                    const cudensitymatHandle_t handle,
                    cudensitymatTimePropagationApproachKrylovConfig_t config,
                    cudensitymatTimePropagationApproachKrylovConfigAttribute_t attribute,
                    const void * attributeValue,
                    size_t attributeSize);

/**
 * \brief Gets an attribute of the Krylov configuration.
 *
 * \param[in] handle Library handle.
 * \param[in] config Krylov configuration object.
 * \param[in] attribute Attribute to get.
 * \param[out] attributeValue Pointer to store the attribute value.
 * \param[in] attributeSize Size of the buffer in bytes.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatTimePropagationApproachKrylovConfigGetAttribute(
                    const cudensitymatHandle_t handle,
                    const cudensitymatTimePropagationApproachKrylovConfig_t config,
                    cudensitymatTimePropagationApproachKrylovConfigAttribute_t attribute,
                    void * attributeValue,
                    size_t attributeSize);

// ============================================================================
// Time Propagation API
// ============================================================================

/**
 * \brief Creates a time propagation object for a given operator.
 *
 * \param[in] handle Library handle.
 * \param[in] superoperator Operator.
 * \param[in] isHermitian Specifies whether the operator is Hermitian (!=0) or not (0).
 * \param[in] scopeKind Requested propagation scope.
 * \param[in] approachKind Requested propagation approach.
 * \param[out] timePropagation Time propagation object.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateTimePropagation(
                    const cudensitymatHandle_t handle,
                    cudensitymatOperator_t superoperator,
                    int32_t isHermitian,
                    cudensitymatTimePropagationScopeKind_t scopeKind,
                    cudensitymatTimePropagationApproachKind_t approachKind,
                    cudensitymatTimePropagation_t * timePropagation);

/**
 * \brief Destroys a time propagation object.
 *
 * \param[in] timePropagation Time propagation object.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyTimePropagation(cudensitymatTimePropagation_t timePropagation);

/**
 * \brief Configures the time propagation object with a configuration attribute.
 *
 * \param[in] handle Library handle.
 * \param[inout] timePropagation Time propagation object.
 * \param[in] attribute Attribute to set.
 * \param[in] attributeValue Pointer to the attribute value.
 * \param[in] attributeSize Size of the attribute value in bytes.
 * \return cudensitymatStatus_t
 */
cudensitymatStatus_t cudensitymatTimePropagationConfigure(
                    const cudensitymatHandle_t handle,
                    cudensitymatTimePropagation_t timePropagation,
                    cudensitymatTimePropagationAttribute_t attribute,
                    const void * attributeValue,
                    size_t attributeSize);

/**
 * \brief Prepares the time propagation object for computation.
 *
 * \param[in] handle Library handle.
 * \param[inout] timePropagation Time propagation object.
 * \param[in] stateIn Representative input quantum state for the time propagation.
 * \param[in] stateOut Representative output quantum state for the time propagation.
 * \param[in] computeType Desired compute type.
 * \param[in] workspaceSizeLimit Workspace buffer size limit (bytes).
 * \param[inout] workspace Empty workspace descriptor on entrance.
 * The workspace size required for the computation will be set on exit.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
 cudensitymatStatus_t cudensitymatTimePropagationPrepare(
  const cudensitymatHandle_t handle,
  cudensitymatTimePropagation_t timePropagation,
  const cudensitymatState_t stateIn,
  const cudensitymatState_t stateOut,
  cudensitymatComputeType_t computeType,
  size_t workspaceSizeLimit,
  cudensitymatWorkspaceDescriptor_t workspace,
  cudaStream_t stream);

/**
 * \brief Computes the time propagation of a quantum state under the action of the operator.
 *
 * \details The propagation advances mixed states according to the Liouvillian
 * equation \f$ \frac{d}{dt} \rho = \mathcal{L}\rho \f$, producing
 * \f$ \rho(t + \Delta t) \f$ from \f$ \rho(t) \f$, where
 * \f$ \Delta t = \Delta t_r + i\,\Delta t_i \f$ with
 * \f$ \Delta t_r \f$ and \f$ \Delta t_i \f$ given by
 * `timeStepReal` and `timeStepImag`.
 * The propagation advances pure states according to the Schrödinger
 * equation \f$ \frac{d}{dt} \psi = -i\,H\,\psi \f$, producing
 * \f$ \psi(t + \Delta t) \f$ from \f$ \psi(t) \f$, where
 * \f$ \Delta t = \Delta t_r + i\,\Delta t_i \f$ with
 * \f$ \Delta t_r \f$ and \f$ \Delta t_i \f$ given by
 * `timeStepReal` and `timeStepImag`.
 *
 * \param[in] handle Library handle.
 * \param[inout] timePropagation Time propagation object.
 * \param[in] timeStepReal Real part of time step for propagation.
 * \param[in] timeStepImag Imaginary part of time step for propagation.
 * \param[in] time Time value.
 * \param[in] batchSize Batch size (>=1).
 * \param[in] numParams Number of variable parameters defined by the user.
 * \param[in] params GPU-accessible pointer to an F-order 2d-array
 * of user-defined real parameter values: params[numParams, batchSize].
 * \param[in] stateIn Input quantum state (can be batched).
 * \param[inout] stateOut Time propagated output quantum state (can be batched).
 * \param[in] workspace Allocated workspace descriptor.
 * \param[in] stream CUDA stream.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatTimePropagationCompute(
                    const cudensitymatHandle_t handle,
                    cudensitymatTimePropagation_t timePropagation,
                    double timeStepReal,
                    double timeStepImag,
                    double time,
                    int64_t batchSize,
                    int32_t numParams,
                    const double * params,
                    const cudensitymatState_t stateIn,
                    cudensitymatState_t stateOut,
                    cudensitymatWorkspaceDescriptor_t workspace,
                    cudaStream_t stream);

/** \} end propagatorAPI */

/**
 * \defgroup workspaceAPI Workspace API
 * \{
 */

/**
 * \brief Creates a workspace descriptor.
 * 
 * \param[in] handle Library handle.
 * \param[out] workspaceDescr Workspace descriptor.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatCreateWorkspace(
                    const cudensitymatHandle_t handle,
                    cudensitymatWorkspaceDescriptor_t * workspaceDescr);

/**
 * \brief Destroys a workspace descriptor.
 * 
 * \param[inout] workspaceDescr Workspace descriptor.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatDestroyWorkspace(
                    cudensitymatWorkspaceDescriptor_t workspaceDescr);

/**
 * \brief Queries the required workspace buffer size.
 * 
 * \param[in] handle Library handle.
 * \param[in] workspaceDescr Workspace descriptor.
 * \param[in] memSpace Memory space.
 * \param[in] workspaceKind Workspace kind.
 * \param[out] memoryBufferSize Required workspace buffer size in bytes.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatWorkspaceGetMemorySize(
                    const cudensitymatHandle_t handle,
                    const cudensitymatWorkspaceDescriptor_t workspaceDescr,
                    cudensitymatMemspace_t memSpace,
                    cudensitymatWorkspaceKind_t workspaceKind,
                    size_t * memoryBufferSize);

/**
 * \brief Attaches memory to a workspace buffer.
 * 
 * \param[in] handle Library handle.
 * \param[inout] workspaceDescr Workspace descriptor.
 * \param[in] memSpace Memory space.
 * \param[in] workspaceKind Workspace kind.
 * \param[in] memoryBuffer Pointer to a user-owned memory buffer
 * to be used by the specified workspace.
 * \param[in] memoryBufferSize Size of the provided memory buffer in bytes.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatWorkspaceSetMemory(
                    const cudensitymatHandle_t handle,
                    cudensitymatWorkspaceDescriptor_t workspaceDescr,
                    cudensitymatMemspace_t memSpace,
                    cudensitymatWorkspaceKind_t workspaceKind,
                    void * memoryBuffer,
                    size_t memoryBufferSize);

/**
 * \brief Retrieves a workspace buffer.
 * 
 * \param[in] handle Library handle.
 * \param[in] workspaceDescr Workspace descriptor.
 * \param[in] memSpace Memory space.
 * \param[in] workspaceKind Workspace kind.
 * \param[out] memoryBuffer Pointer to a user-owned memory buffer
 * used by the specified workspace.
 * \param[out] memoryBufferSize Size of the memory buffer in bytes.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatWorkspaceGetMemory(
                    const cudensitymatHandle_t handle,
                    const cudensitymatWorkspaceDescriptor_t workspaceDescr,
                    cudensitymatMemspace_t memSpace,
                    cudensitymatWorkspaceKind_t workspaceKind,
                    void ** memoryBuffer,
                    size_t * memoryBufferSize);

/**
 * \brief Attaches a buffer to the elementary tensor operator (either batched or non-batched).
 * 
 * \param[in] handle Library handle.
 * \param[in] elemOperator Elementary tensor operator (either batched or non-batched).
 * \param[in] buffer GPU-accessible pointer to the tensor operator elements storage.
 * \param[in] bufferSize Size of the memory buffer in bytes.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatElementaryOperatorAttachBuffer(
                    const cudensitymatHandle_t handle, 
                    cudensitymatElementaryOperator_t elemOperator, 
                    void * buffer, 
                    size_t bufferSize);

/**
 * \brief Attaches a buffer to the full dense local matrix operator (either batched or non-batched).
 * 
 * \param[in] handle Library handle.
 * \param[in] matrixOperator Full dense local matrix operator (either batched or non-batched).
 * \param[in] buffer GPU-accessible pointer to the matrix operator elements storage.
 * \param[in] bufferSize Size of the memory buffer in bytes.
 * \return cudensitymatStatus_t 
 */
cudensitymatStatus_t cudensitymatMatrixOperatorDenseLocalAttachBuffer(
                    const cudensitymatHandle_t handle, 
                    cudensitymatMatrixOperator_t matrixOperator, 
                    void * buffer, 
                    size_t bufferSize);
/** \} end workspaceAPI */

/**
 * \defgroup loggerAPI Logger API
 * \{
 */

/**
 * \brief This function sets the logging callback routine.
 * \param[in] callback Pointer to a callback function. Check ::cudensitymatLoggerCallback_t.
 */
/*cudensitymatStatus_t cudensitymatLoggerSetCallback(cudensitymatLoggerCallback_t callback);*/

/**
 * \brief This function sets the logging callback routine, along with user data.
 * \param[in] callback Pointer to a callback function. Check ::cudensitymatLoggerCallbackData_t.
 * \param[in] userData Pointer to user-provided data to be used by the callback.
 */
/*cudensitymatStatus_t cudensitymatLoggerSetCallbackData(cudensitymatLoggerCallbackData_t callback,
                                                 void *userData);*/

/**
 * \brief This function sets the logging output file.
 * \param[in] file An open file with write permission.
 */
/*cudensitymatStatus_t cudensitymatLoggerSetFile(FILE *file);*/

/**
 * \brief This function opens a logging output file in the given path.
 * \param[in] logFile Path to the logging output file.
 */
/*cudensitymatStatus_t cudensitymatLoggerOpenFile(const char *logFile);*/

/**
 * \brief This function sets the value of the logging level.
 * \param[in] level Log level, should be one of the following:
 * Level| Summary           | Long Description
 * -----|-------------------|-----------------
 *  "0" | Off               | logging is disabled (default)
 *  "1" | Errors            | only errors will be logged
 *  "2" | Performance Trace | API calls that launch CUDA kernels will log their parameters and important information
 *  "3" | Performance Hints | hints that can potentially improve the application's performance
 *  "4" | Heuristics Trace  | provides general information about the library execution, may contain details about heuristic status
 *  "5" | API Trace         | API Trace - API calls will log their parameter and important information
 */
/*cudensitymatStatus_t cudensitymatLoggerSetLevel(int32_t level);*/

/**
 * \brief This function sets the value of the log mask.
 *
 * \param[in]  mask  Value of the logging mask.
 * Masks are defined as a combination (bitwise OR) of the following masks:
 * Level| Description       |
 * -----|-------------------|
 *  "0" | Off               |
 *  "1" | Errors            |
 *  "2" | Performance Trace |
 *  "4" | Performance Hints |
 *  "8" | Heuristics Trace  |
 *  "16"| API Trace         |
 *
 * Refer to cudensitymatLoggerSetLevel() for details.
 */
/*cudensitymatStatus_t cudensitymatLoggerSetMask(int32_t mask);*/

/**
 * \brief This function disables logging for the entire run.
 */
/*cudensitymatStatus_t cudensitymatLoggerForceDisable();*/

/** \} end loggerAPI */

/**
 * \defgroup distrBindings Distributed Interface Bindings
 * \{
 */

#define CUDENSITYMAT_DISTRIBUTED_INTERFACE_VERSION 260110

/**
 * \brief (Internal): Dynamic API wrapper runtime binding table for the distributed communication service.
 *
 * Caller must ensure stream ordering before calling distributed operations
 * (e.g., via ensureStreamOrderingForDistributedBackend which synchronizes the stream).
 *
 * The barrier operation additionally takes a barrierBuffer (device pointer):
 *  - For MPI: the buffer is ignored (can be NULL)
 *  - For NCCL: used for an allreduce-based barrier implementation
 * 
 * \note The NCCL interface is currently in an experimental state and we do not guarantee stability or performance.
*/
typedef struct {
  int version;
  int (*getNumRanks)(const cudensitymatDistributedCommunicator_t*, int32_t*);
  int (*getNumRanksShared)(const cudensitymatDistributedCommunicator_t*, int32_t*);
  int (*getProcRank)(const cudensitymatDistributedCommunicator_t*, int32_t*);
  int (*barrier)(const cudensitymatDistributedCommunicator_t*, void* barrierBuffer);
  int (*createRequest)(cudensitymatDistributedRequest_t*);
  int (*destroyRequest)(cudensitymatDistributedRequest_t);
  int (*waitRequest)(cudensitymatDistributedRequest_t);
  int (*testRequest)(cudensitymatDistributedRequest_t, int32_t*);
  int (*groupStart)(void);  /**< Begin a group of communication operations (for NCCL) */
  int (*groupEnd)(void);    /**< End a group of communication operations (for NCCL) */
  int (*send)(const cudensitymatDistributedCommunicator_t*,
              const void*, int32_t, cudaDataType_t, int32_t, int32_t);
  int (*sendAsync)(const cudensitymatDistributedCommunicator_t*,
                   const void*, int32_t, cudaDataType_t, int32_t, int32_t,
                   cudensitymatDistributedRequest_t);
  int (*receive)(const cudensitymatDistributedCommunicator_t*,
                 void*, int32_t, cudaDataType_t, int32_t, int32_t);
  int (*receiveAsync)(const cudensitymatDistributedCommunicator_t*,
                      void*, int32_t, cudaDataType_t, int32_t, int32_t,
                      cudensitymatDistributedRequest_t);
  int (*bcast)(const cudensitymatDistributedCommunicator_t*,
               void*, int32_t, cudaDataType_t, int32_t);
  int (*allreduce)(const cudensitymatDistributedCommunicator_t*,
                   const void*, void*, int32_t, cudaDataType_t);
  int (*allreduceInPlace)(const cudensitymatDistributedCommunicator_t*,
                          void*, int32_t, cudaDataType_t);
  int (*allreduceInPlaceMin)(const cudensitymatDistributedCommunicator_t*,
                             void*, int32_t, cudaDataType_t);
  int (*allreduceDoubleIntMinloc)(const cudensitymatDistributedCommunicator_t*,
                                  const void*, void*);
  int (*allgather)(const cudensitymatDistributedCommunicator_t*,
                   const void*, void*, int32_t, cudaDataType_t);
} cudensitymatDistributedInterface_t;

/** \} end distrBindings */

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)
