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
   Time-dependent transverse-field Ising Hamiltonian operator
   with ordered and fused ZZ terms, plus fused unitary dissipation terms:
    H = sum_{i} {h_i * X_i}                // transverse field sum of X_i operators with static h_i coefficients 
      + f(t) * sum_{i < j} {g_ij * ZZ_ij}  // modulated sum of the fused ordered {Z_i * Z_j} terms with static g_ij coefficients
      + d * sum_{i} {Y_i * {..} * Y_i}     // scaled sum of the dissipation terms {Y_i * {..} * Y_i} fused into the YY_ii super-operators
   where {..} is the placeholder for the density matrix to show that the Y_i operators act from different sides.
*/

/** Define the numerical type and data type for the GPU computations (same) */
using NumericalType = std::complex<double>;  // do not change
constexpr cudaDataType_t dataType = CUDA_C_64F;        // do not change


/** Example of a user-provided scalar CPU callback C function
 *  defining a time-dependent coefficient inside the Hamiltonian:
 *  f(t) = exp(i * Omega * t) = cos(Omega * t) + i * sin(Omega * t)
 */
extern "C"
int32_t fCoefComplex64(
  double time,             //in: time point
  int64_t batchSize,       //in: user-defined batch size (number of coefficients in the batch)
  int32_t numParams,       //in: number of external user-provided Hamiltonian parameters (this function expects one parameter, Omega)
  const double * params,   //in: params[0:numParams-1][0:batchSize-1]: GPU-accessible F-ordered array of user-provided Hamiltonian parameters for all instances of the batch
  cudaDataType_t dataType, //in: data type (expecting CUDA_C_64F in this specific callback function)
  void * scalarStorage,    //inout: CPU-accessible storage for the returned coefficient value(s) of shape [0:batchSize-1]
  cudaStream_t stream)     //in: CUDA stream (default is 0x0)
{
  if (dataType == CUDA_C_64F) {
    auto * tdCoef = static_cast<cuDoubleComplex*>(scalarStorage); // casting to cuDoubleComplex because this callback function expects CUDA_C_64F data type
    for (int64_t i = 0; i < batchSize; ++i) {
      const auto omega = params[i * numParams + 0]; // params[0][i]: 0-th parameter for i-th instance of the batch
      tdCoef[i] = make_cuDoubleComplex(std::cos(omega * time), std::sin(omega * time)); // value of the i-th instance of the coefficients batch
    }
  } else {
    return 1; // error code (1: Error)
  }
  return 0; // error code (0: Success)
}


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
  void * spinYYelems {nullptr};            // elements of the fused YY two-spin operator in GPU RAM (F-order storage)
  void * spinZZelems {nullptr};            // elements of the fused ZZ two-spin operator in GPU RAM (F-order storage)
  cudensitymatElementaryOperator_t spinX;  // X spin operator (elementary tensor operator)
  cudensitymatElementaryOperator_t spinYY; // fused YY two-spin operator (elementary tensor operator)
  cudensitymatElementaryOperator_t spinZZ; // fused ZZ two-spin operator (elementary tensor operator)
  cudensitymatOperatorTerm_t oneBodyTerm;  // operator term: H1 = sum_{i} {h_i * X_i} (one-body term)
  cudensitymatOperatorTerm_t twoBodyTerm;  // operator term: H2 = f(t) * sum_{i < j} {g_ij * ZZ_ij} (two-body term)
  cudensitymatOperatorTerm_t noiseTerm;    // operator term: D1 = d * sum_{i} {YY_ii}  // Y_i operators act from different sides on the density matrix (two-body mixed term)
  cudensitymatOperator_t liouvillian;      // full operator: (-i * (H1 + H2) * {..}) + (i * {..} * (H1 + H2)) + D1{..} (super-operator)

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

    spinYYelems = createInitializeArrayGPU<NumericalType>(  // YY[i0, i1; j0, j1] := Y[i0; j0] * Y[i1; j1]
                    {{0.0, 0.0},  {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0},  // 1st column of matrix YY
                     {0.0, 0.0},  {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},   // 2nd column of matrix YY
                     {0.0, 0.0},  {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},   // 3rd column of matrix YY
                     {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}); // 4th column of matrix YY

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
    //  YY_ii = Y_i * {..} * Y_i fused operator (note action from different sides)
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        2,                                   // two-body operator
                        std::vector<int64_t>({2,2}).data(),  // acts in tensor space of shape {2,2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        dataType,                            // data type
                        spinYYelems,                         // tensor elements in GPU memory
                        cudensitymatTensorCallbackNone,      // no tensor callback function (tensor is not time-dependent)
                        cudensitymatTensorGradientCallbackNone, // no tensor gradient callback function
                        &spinYY));                           // the created elementary tensor operator

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
    //  Define the operator term: H2 = f(t) * sum_{i < j} {g_ij * ZZ_ij}
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
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of modes)
                        spaceShape.data(),                   // Hilbert space shape (mode extents)
                        &noiseTerm));                        // the created empty operator term
    //  Define the operator term: D1 = d * sum_{i} {YY_ii}
    for (int32_t i = 0; i < spaceShape.size(); ++i) {
      HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(handle,
                          noiseTerm,
                          1,                                                              // number of elementary tensor operators in the product
                          std::vector<cudensitymatElementaryOperator_t>({spinYY}).data(), // elementary tensor operators forming the product
                          std::vector<int32_t>({i, i}).data(),                            // space modes acted on by the operator product (from different sides)
                          std::vector<int32_t>({0, 1}).data(),                            // space mode action duality (0: from the left; 1: from the right)
                          make_cuDoubleComplex(1.0, 0.0),                                 // default coefficient: Always 64-bit-precision complex number
                          cudensitymatScalarCallbackNone,                                 // no time-dependent coefficient associated with this operator product
                          cudensitymatScalarGradientCallbackNone));                       // no coefficient gradient associated with this operator product
    }

    // Construct the full Liouvillian operator as a sum of the operator terms
    //  Create an empty operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle,
                        spaceShape.size(),                // Hilbert space rank (number of modes)
                        spaceShape.data(),                // Hilbert space shape (modes extents)
                        &liouvillian));                   // the created empty operator (super-operator)
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        oneBodyTerm,                      // appended operator term
                        0,                                // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, -1.0),  // -i constant
                        cudensitymatScalarCallbackNone,   // no time-dependent coefficient associated with the operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with the operator term as a whole
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        twoBodyTerm,                     // appended operator term
                        0,                               // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, -1.0), // -i constant
                        {fCoefComplex64, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr}, // CPU scalar callback function defining the time-dependent coefficient associated with this operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with this operator term as a whole
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        oneBodyTerm,                      // appended operator term
                        1,                                // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, 1.0),   // i constant
                        cudensitymatScalarCallbackNone,   // no time-dependent coefficient associated with the operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with the operator term as a whole
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        twoBodyTerm,                     // appended operator term
                        1,                               // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, 1.0),  // i constant
                        {fCoefComplex64, CUDENSITYMAT_CALLBACK_DEVICE_CPU, nullptr}, // CPU scalar callback function defining the time-dependent coefficient associated with this operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with this operator term as a whole
    //  Append an operator term to the operator (super-operator)
    const double d = 0.42; // assign some value to the time-independent coefficient
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        noiseTerm,                        // appended operator term
                        0,                                // operator term action duality as a whole (no duality reversing in this case)
                        make_cuDoubleComplex(d, 0.0),     // constant coefficient associated with the operator term as a whole
                        cudensitymatScalarCallbackNone,   // no time-dependent coefficient associated with the operator term as a whole
                        cudensitymatScalarGradientCallbackNone)); // no coefficient gradient associated with the operator term as a whole
  }

  // Destructor destructs the user-defined Liouvillian operator
  ~UserDefinedLiouvillian()
  {
    // Destroy the Liouvillian operator
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian));

    // Destroy operator terms
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(noiseTerm));
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(twoBodyTerm));
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(oneBodyTerm));

    // Destroy elementary tensor operators
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(spinYY));
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(spinZZ));
    HANDLE_CUDM_ERROR(cudensitymatDestroyElementaryOperator(spinX));

    // Destroy operator tensors
    destroyArrayGPU(spinYYelems);
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
    return 1; // one parameter Omega
  }

  /** Get access to the constructed Liouvillian operator. */
  cudensitymatOperator_t & get()
  {
    return liouvillian;
  }

};
