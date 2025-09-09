/* Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cudensitymat.h> // cuDensityMat library header
#include "helpers.h"      // GPU helper functions

#include <cmath>
#include <complex>
#include <vector>
#include <iostream>
#include <cassert>


/* DESCRIPTION:
   Transverse-field Ising Hamiltonian operator with ordered and fused ZZ terms:
    H = sum_{i} {h_i * X_i}         // transverse field sum of X_i operators with static h_i coefficients 
      + sum_{i < j} {g_ij * ZZ_ij}  // sum of the fused ordered {Z_i * Z_j} terms with static g_ij coefficients
*/

/** Define the numerical type and data type for the GPU computations (same) */
using NumericalType = std::complex<double>;      // do not change
constexpr cudaDataType_t dataType = CUDA_C_64F;  // do not change


/** Convenience class which encapsulates a user-defined Liouvillian operator (system Hamiltonian + dissipation terms):
 *  - Constructor constructs the desired Liouvillian operator (`cudensitymatOperator_t`)
 *  - Method `get()` returns a reference to the constructed Liouvillian operator
 *  - Destructor releases all resources used by the Liouvillian operator
 */
class UserDefinedLiouvillian final
{
private:
  // Data members
  cudensitymatHandle_t handle;             // library context handle
  int64_t stateBatchSize;                  // quantum state batch size
  const std::vector<int64_t> spaceShape;   // Hilbert space shape (extents of the modes of the composite Hilbert space)
  void * spinXelems {nullptr};             // elements of the X spin operator in GPU RAM (F-order storage)
  void * spinZZelems {nullptr};            // elements of the fused ZZ two-spin operator in GPU RAM (F-order storage)
  cudensitymatElementaryOperator_t spinX;  // X spin operator (elementary tensor operator)
  cudensitymatElementaryOperator_t spinZZ; // fused ZZ two-spin operator (elementary tensor operator)
  cudensitymatOperatorTerm_t oneBodyTerm;  // operator term: H1 = sum_{i} {h_i * X_i} (one-body term)
  cudensitymatOperatorTerm_t twoBodyTerm;  // operator term: H2 = sum_{i < j} {g_ij * ZZ_ij} (two-body term)
  cudensitymatOperator_t liouvillian;      // full operator: H = H1 + H2

public:

  // Constructor constructs a user-defined Liouvillian operator
  UserDefinedLiouvillian(cudensitymatHandle_t contextHandle,             // library context handle
                         const std::vector<int64_t> & hilbertSpaceShape, // Hilbert space shape
                         int64_t batchSize):                             // batch size for the quantum state
    handle(contextHandle), stateBatchSize(batchSize), spaceShape(hilbertSpaceShape)
  {
    // Define the necessary operator tensors in GPU memory (F-order storage!)
    spinXelems = createInitializeArrayGPU<NumericalType>(  // X[i0; j0]
                  {{0.0, 0.0}, {1.0, 0.0},   // 1st column of matrix X
                   {1.0, 0.0}, {0.0, 0.0}}); // 2nd column of matrix X

    spinZZelems = createInitializeArrayGPU<NumericalType>(  // ZZ[i0, i1; j0, j1] := Z[i0; j0] * Z[i1; j1]
                    {{1.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},  {0.0, 0.0},   // 1st column of matrix ZZ
                     {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},   // 2nd column of matrix ZZ
                     {0.0, 0.0}, {0.0, 0.0},  {-1.0, 0.0}, {0.0, 0.0},   // 3rd column of matrix ZZ
                     {0.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},  {1.0, 0.0}}); // 4th column of matrix ZZ

    // Construct the necessary Elementary Tensor Operators
    //  X_i operator
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        1,                                   // one-body operator
                        std::vector<int64_t>({2}).data(),    // acts in tensor space of shape {2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        dataType,                            // data type
                        spinXelems,                          // tensor elements in GPU memory
                        cudensitymatTensorCallbackNone,      // no tensor callback function (tensor is not time-dependent)
                        cudensitymatTensorGradientCallbackNone, // no tensor gradient callback function
                        &spinX));                            // the created elementary tensor operator
    //  ZZ_ij = Z_i * Z_j fused operator
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        2,                                   // two-body operator
                        std::vector<int64_t>({2,2}).data(),  // acts in tensor space of shape {2,2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        dataType,                            // data type
                        spinZZelems,                         // tensor elements in GPU memory
                        cudensitymatTensorCallbackNone,      // no tensor callback function (tensor is not time-dependent)
                        cudensitymatTensorGradientCallbackNone, // no tensor gradient callback function
                        &spinZZ));                           // the created elementary tensor operator

    // Construct the necessary Operator Terms from tensor products of Elementary Tensor Operators
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of modes)
                        spaceShape.data(),                   // Hilbert space shape (mode extents)
                        &oneBodyTerm));                      // the created empty operator term
    //  Define the operator term: H1 = sum_{i} {h_i * X_i}
    for (int32_t i = 0; i < spaceShape.size(); ++i) {
      const double h_i = 1.0 / static_cast<double>(i+1); // assign some value to the time-independent h_i coefficient
      HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(handle,
                          oneBodyTerm,
                          1,                                                             // number of elementary tensor operators in the product
                          std::vector<cudensitymatElementaryOperator_t>({spinX}).data(), // elementary tensor operators forming the product
                          std::vector<int32_t>({i}).data(),                              // space modes acted on by the operator product
                          std::vector<int32_t>({0}).data(),                              // space mode action duality (0: from the left; 1: from the right)
                          make_cuDoubleComplex(h_i, 0.0),                                // h_i constant coefficient: Always 64-bit-precision complex number
                          cudensitymatScalarCallbackNone,                                // no time-dependent coefficient associated with this operator product
                          cudensitymatScalarGradientCallbackNone));                      // no coefficient gradient associated with this operator product
    }
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of modes)
                        spaceShape.data(),                   // Hilbert space shape (mode extents)
                        &twoBodyTerm));                      // the created empty operator term
    //  Define the operator term: H2 = sum_{i < j} {g_ij * ZZ_ij}
    for (int32_t i = 0; i < spaceShape.size() - 1; ++i) {
      for (int32_t j = (i + 1); j < spaceShape.size(); ++j) {
        const double g_ij = -1.0 / static_cast<double>(i + j + 1); // assign some value to the time-independent g_ij coefficient
        HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(handle,
                            twoBodyTerm,
                            1,                                                              // number of elementary tensor operators in the product
                            std::vector<cudensitymatElementaryOperator_t>({spinZZ}).data(), // elementary tensor operators forming the product
                            std::vector<int32_t>({i, j}).data(),                            // space modes acted on by the operator product
                            std::vector<int32_t>({0, 0}).data(),                            // space mode action duality (0: from the left; 1: from the right)
                            make_cuDoubleComplex(g_ij, 0.0),                                // g_ij constant coefficient: Always 64-bit-precision complex number
                            cudensitymatScalarCallbackNone,                                 // no time-dependent coefficient associated with this operator product
                            cudensitymatScalarGradientCallbackNone));                       // no coefficient gradient associated with this operator product
      }
    }

    // Construct the full Liouvillian operator as a sum of the operator terms
    //  Create an empty operator
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle,
                        spaceShape.size(),                // Hilbert space rank (number of modes)
                        spaceShape.data(),                // Hilbert space shape (modes extents)
                        &liouvillian));                   // the created empty operator
    //  Append an operator term to the operator
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        oneBodyTerm,                      // appended operator term
                        0,                                // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(1.0, 0.0),   // constant coefficient associated with the operator term as a whole
                        cudensitymatScalarCallbackNone,   // no time-dependent coefficient associated with the operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with the operator term as a whole
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        twoBodyTerm,                      // appended operator term
                        0,                                // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(1.0, 0.0),   // constant coefficient associated with the operator term as a whole
                        cudensitymatScalarCallbackNone,   // no time-dependent coefficient associated with the operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with this operator term as a whole
  }

  // Destructor destructs the user-defined Liouvillian operator
  ~UserDefinedLiouvillian()
  {
    // Destroy the Liouvillian operator
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian));

    // Destroy operator terms
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(twoBodyTerm));
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(oneBodyTerm));

    // Destroy elementary tensor operators
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(spinZZ));
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(spinX));

    // Destroy operator tensors
    destroyArrayGPU(spinZZelems);
    destroyArrayGPU(spinXelems);
  }

  // Disable copy constructor/assignment (GPU resources are private, no deep copy)
  UserDefinedLiouvillian(const UserDefinedLiouvillian &) = delete;
  UserDefinedLiouvillian & operator=(const UserDefinedLiouvillian &) = delete;
  UserDefinedLiouvillian(UserDefinedLiouvillian &&) = delete;
  UserDefinedLiouvillian & operator=(UserDefinedLiouvillian &&) = delete;

  /** Returns the number of externally provided Hamiltonian parameters. */
  int32_t getNumParameters() const
  {
    return 0; // no free parameters
  }

  /** Get access to the constructed Liouvillian operator. */
  cudensitymatOperator_t & get()
  {
    return liouvillian;
  }

};
