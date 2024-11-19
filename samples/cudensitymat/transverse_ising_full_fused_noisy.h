/* Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cudensitymat.h> // cuDensityMat library header
#include "helpers.h"   // helper functions

#include <cmath>
#include <complex>
#include <vector>
#include <iostream>
#include <cassert>


/* Time-dependent transverse-field Ising Hamiltonian operator
   with ordered and fused ZZ terms, plus fused unitary dissipation terms:
    H = sum_{i} {h_i * X_i}                // transverse field sum of X_i
      + f(t) * sum_{i < j} {g_ij * ZZ_ij}  // modulated sum of fused {Z_i * Z_j} terms
      + d * sum_{i} {Y_i * {..} * Y_i}     // dissipation terms {Y_i * {..} * Y_i} will be fused into the YY_ii super-operator
   where {..} is the placeholder for the density matrix to show that the operators act from a different side
*/


// User-defined C++ callback function defining a time-dependent coefficient inside the Hamiltonian:
// f(t) = cos(omega * t) + i * sin(omega * t)
extern "C"
int32_t tdCoefComplex64(double time,             // time point
                        int32_t numParams,       // number of external user-defined Liouvillian parameters (= 1 here)
                        const double params[],   // params[0] is omega (user-defined Liouvillian parameter)
                        cudaDataType_t dataType, // data type (CUDA_C_64F here)
                        void * scalarStorage)    // CPU storage for the returned function value
{
  const auto omega = params[0];
  auto * tdCoef = static_cast<std::complex<double>*>(scalarStorage); // casting to complex<double> because it returns CUDA_C_64F data type
  *tdCoef = {std::cos(omega * time), std::sin(omega * time)};
  return 0; // error code (0: Success)
}


/** Convenience class to encapsulate the Liouvillian operator:
 *   - Constructor constructs the desired Liouvillian operator (`cudensitymatOperator_t`)
 *   - get() method returns a reference to the constructed Liouvillian operator
 *   - Destructor releases all resources used by the Liouvillian operator
 */
class UserDefinedLiouvillian final
{
private:
  // Data members
  cudensitymatHandle_t handle;              // library context handle
  const std::vector<int64_t> spaceShape;    // Hilbert space shape
  void * spinXelems {nullptr};              // elements of the X spin operator in GPU RAM
  void * spinYYelems {nullptr};             // elements of the YY two-spin operator in GPU RAM
  void * spinZZelems {nullptr};             // elements of the ZZ two-spin operator in GPU RAM
  cudensitymatElementaryOperator_t spinX;   // X spin operator
  cudensitymatElementaryOperator_t spinYY;  // YY two-spin operator
  cudensitymatElementaryOperator_t spinZZ;  // ZZ two-spin operator
  cudensitymatOperatorTerm_t oneBodyTerm;   // operator term: H1 = sum_{i} {h_i * X_i}
  cudensitymatOperatorTerm_t twoBodyTerm;   // operator term: H2 = f(t) * sum_{i < j} {g_ij * ZZ_ij}
  cudensitymatOperatorTerm_t noiseTerm;     // operator term: D1 = d * sum_{i} {YY_ii}  // Y_i operators act from different sides on the density matrix
  cudensitymatOperator_t liouvillian;       // (-i * (H1 + f(t) * H2) * rho) + (i * rho * (H1 + f(t) * H2)) + D1

public:

  // Constructor constructs a user-defined Liouvillian operator
  UserDefinedLiouvillian(cudensitymatHandle_t contextHandle,              // library context handle
                         const std::vector<int64_t> & hilbertSpaceShape): // Hilbert space shape
    handle(contextHandle), spaceShape(hilbertSpaceShape)
  {
    // Define the necessary elementary tensors in GPU memory (F-order storage!)
    spinXelems = createArrayGPU<std::complex<double>>(
                  {{0.0, 0.0}, {1.0, 0.0},   // 1st column of matrix X
                   {1.0, 0.0}, {0.0, 0.0}}); // 2nd column of matrix X

    spinYYelems = createArrayGPU<std::complex<double>>(  // YY[i0, i1; j0, j1] := Y[i0; j0] * Y[i1; j1]
                    {{0.0, 0.0},  {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0},  // 1st column of matrix YY
                     {0.0, 0.0},  {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},   // 2nd column of matrix YY
                     {0.0, 0.0},  {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},   // 3rd column of matrix YY
                     {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}); // 4th column of matrix YY

    spinZZelems = createArrayGPU<std::complex<double>>(  // ZZ[i0, i1; j0, j1] := Z[i0; j0] * Z[i1; j1]
                    {{1.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},  {0.0, 0.0},   // 1st column of matrix ZZ
                     {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},   // 2nd column of matrix ZZ
                     {0.0, 0.0}, {0.0, 0.0},  {-1.0, 0.0}, {0.0, 0.0},   // 3rd column of matrix ZZ
                     {0.0, 0.0}, {0.0, 0.0},  {0.0, 0.0},  {1.0, 0.0}}); // 4th column of matrix ZZ

    // Construct the necessary Elementary Tensor Operators
    //   X_i operator
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        1,                                   // one-body operator
                        std::vector<int64_t>({2}).data(),    // acts in tensor space of shape {2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        CUDA_C_64F,                          // data type
                        spinXelems,                          // tensor elements in GPU memory
                        {nullptr, nullptr},                  // no tensor callback function (tensor is not time-dependent)
                        &spinX));                            // the created elementary tensor operator
    //  ZZ_ij = Z_i * Z_j fused operator
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        2,                                   // two-body operator
                        std::vector<int64_t>({2,2}).data(),  // acts in tensor space of shape {2,2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        CUDA_C_64F,                          // data type
                        spinZZelems,                         // tensor elements in GPU memory
                        {nullptr, nullptr},                  // no tensor callback function (tensor is not time-dependent)
                        &spinZZ));                           // the created elementary tensor operator
    //  YY_ii = Y_i * {..} * Y_i fused operator (note action from different sides)
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(handle,
                        2,                                   // two-body operator
                        std::vector<int64_t>({2,2}).data(),  // acts in tensor space of shape {2,2}
                        CUDENSITYMAT_OPERATOR_SPARSITY_NONE, // dense tensor storage
                        0,                                   // 0 for dense tensors
                        nullptr,                             // nullptr for dense tensors
                        CUDA_C_64F,                          // data type
                        spinYYelems,                         // tensor elements in GPU memory
                        {nullptr, nullptr},                  // no tensor callback function (tensor is not time-dependent)
                        &spinYY));                           // the created elementary tensor operator

    // Construct the necessary Operator Terms from direct products of Elementary Tensor Operators
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of dimensions)
                        spaceShape.data(),                   // Hilbert space shape
                        &oneBodyTerm));                      // the created empty operator term
    //  Define the operator term
    for (int32_t i = 0; i < spaceShape.size(); ++i) {
      const double h_i = 1.0 / static_cast<double>(i+1);  // just some value (time-independent h_i coefficient)
      HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(handle,
                          oneBodyTerm,
                          1,                                                             // number of elementary tensor operators in the product
                          std::vector<cudensitymatElementaryOperator_t>({spinX}).data(), // elementary tensor operators forming the product
                          std::vector<int32_t>({i}).data(),                              // space modes acted on by the operator product
                          std::vector<int32_t>({0}).data(),                              // space mode action duality (0: from the left; 1: from the right)
                          make_cuDoubleComplex(h_i, 0.0),                                // h_i constant coefficient: Always 64-bit-precision complex number
                          {nullptr, nullptr}));                                          // no time-dependent coefficient associated with the operator product
    }
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of dimensions)
                        spaceShape.data(),                   // Hilbert space shape
                        &twoBodyTerm));                      // the created empty operator term
    //  Define the operator term
    for (int32_t i = 0; i < spaceShape.size() - 1; ++i) {
      for (int32_t j = (i + 1); j < spaceShape.size(); ++j) {
        const double g_ij = -1.0 / static_cast<double>(i + j + 1);  // just some value (time-independent g_ij coefficient)
        HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(handle,
                            twoBodyTerm,
                            1,                                                              // number of elementary tensor operators in the product
                            std::vector<cudensitymatElementaryOperator_t>({spinZZ}).data(), // elementary tensor operators forming the product
                            std::vector<int32_t>({i, j}).data(),                            // space modes acted on by the operator product
                            std::vector<int32_t>({0, 0}).data(),                            // space mode action duality (0: from the left; 1: from the right)
                            make_cuDoubleComplex(g_ij, 0.0),                                // g_ij constant coefficient: Always 64-bit-precision complex number
                            {nullptr, nullptr}));                                           // no time-dependent coefficient associated with the operator product
      }
    }
    //  Create an empty operator term
    HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle,
                        spaceShape.size(),                   // Hilbert space rank (number of dimensions)
                        spaceShape.data(),                   // Hilbert space shape
                        &noiseTerm));                        // the created empty operator term
    //  Define the operator term
    for (int32_t i = 0; i < spaceShape.size(); ++i) {
      HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(handle,
                          noiseTerm,
                          1,                                                              // number of elementary tensor operators in the product
                          std::vector<cudensitymatElementaryOperator_t>({spinYY}).data(), // elementary tensor operators forming the product
                          std::vector<int32_t>({i, i}).data(),                            // space modes acted on by the operator product (from different sides)
                          std::vector<int32_t>({0, 1}).data(),                            // space mode action duality (0: from the left; 1: from the right)
                          make_cuDoubleComplex(1.0, 0.0),                                 // default coefficient: Always 64-bit-precision complex number
                          {nullptr, nullptr}));                                           // no time-dependent coefficient associated with the operator product
    }

    // Construct the full Liouvillian operator as a sum of the operator terms
    //  Create an empty operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(handle,
                        spaceShape.size(),               // Hilbert space rank (number of dimensions)
                        spaceShape.data(),               // Hilbert space shape
                        &liouvillian));                  // the created empty operator (super-operator)
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        oneBodyTerm,                     // appended operator term
                        0,                               // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, -1.0), // -i constant
                        {nullptr, nullptr}));            // no time-dependent coefficient associated with the operator term as a whole
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        twoBodyTerm,                     // appended operator term
                        0,                               // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, -1.0), // -i constant
                        {tdCoefComplex64, nullptr}));    // function callback defining the time-dependent coefficient associated with this operator term as a whole
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        oneBodyTerm,                    // appended operator term
                        1,                              // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, 1.0), // i constant
                        {nullptr, nullptr}));           // no time-dependent coefficient associated with the operator term as a whole
    //  Append an operator term to the operator (super-operator)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        twoBodyTerm,                    // appended operator term
                        1,                              // operator term action duality as a whole (0: acting from the left; 1: acting from the right)
                        make_cuDoubleComplex(0.0, 1.0), // i constant
                        {tdCoefComplex64, nullptr}));   // function callback defining the time-dependent coefficient associated with this operator term as a whole
    //  Append an operator term to the operator (super-operator)
    const double d = 0.42; // just some value (time-independent coefficient)
    HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(handle,
                        liouvillian,
                        noiseTerm,                    // appended operator term
                        0,                            // operator term action duality as a whole (no duality reversing in this case)
                        make_cuDoubleComplex(d, 0.0), // constant coefficient associated with the operator term as a whole
                        {nullptr, nullptr}));         // no time-dependent coefficient associated with the operator term as a whole
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

    // Destroy elementary tensors
    destroyArrayGPU(spinYYelems);
    destroyArrayGPU(spinZZelems);
    destroyArrayGPU(spinXelems);
  }

  // Disable copy constructor/assignment (GPU resources are private, no deep copy)
  UserDefinedLiouvillian(const UserDefinedLiouvillian &) = delete;
  UserDefinedLiouvillian & operator=(const UserDefinedLiouvillian &) = delete;
  UserDefinedLiouvillian(UserDefinedLiouvillian &&) noexcept = default;
  UserDefinedLiouvillian & operator=(UserDefinedLiouvillian &&) noexcept = default;

  // Get access to the constructed Liouvillian
  cudensitymatOperator_t & get()
  {
    return liouvillian;
  }

};
