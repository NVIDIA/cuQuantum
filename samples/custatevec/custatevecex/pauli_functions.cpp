/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

//
// This example shows the usage of custatevecExApplyPauliRotation() and
// custatevecExComputeExpectationOnPauliBasis()
//
// 1. The state vector is initialized to |+> state on all wires.
// 2. Rotate state vector by 15 deg by using Z string.
// 3. Compute expectation with X and Z strings.
// 4. Repeat 2 and 3.
//

#include <custatevecEx.h> // custatevecEx API
#include <stdlib.h>       // exit()
#include <complex>        // std::complex<>
#include <vector>         // std::vector<>
#include <random>         // std::uniform_real_distribution<>
#include <numeric>        // std::iota
#include <cmath>          // std::abs, M_PI
#include <cstring>        // strcmp
#include <cstdarg>        // va_list, va_start, va_end
#include "common.hpp"     // errCheck.hpp

// typedef's for short names
typedef custatevecExStateVectorDescriptor_t ExStateVector;
typedef custatevecExDictionaryDescriptor_t ExDictionary;

// Matrices are expressed by c128 values.
typedef std::complex<double> DblComplex;

//
// create state vector
//
custatevecExStateVectorDescriptor_t createStateVector(cudaDataType_t svDataType, int numWires)
{
    ExDictionary svConfig{nullptr};
    custatevecExStateVectorDescriptor_t stateVector{nullptr};
    // numWires should be specified twice for device state vector.
    ERRCHK(custatevecExConfigureStateVectorSingleDevice(&svConfig, svDataType, numWires, numWires,
                                                        0, 0));
    // create state vector
    // stream is not specified here, then, stateVector instance will use the default stream.
    // Note: custom memory allocator is not supported in the current release.
    ERRCHK(custatevecExStateVectorCreateSingleProcess(&stateVector, svConfig, nullptr, 0, nullptr));
    // destroy dictionary
    ERRCHK(custatevecExDictionaryDestroy(svConfig));

    // Use custatevecExStateVectorGetProperty() to confirm
    // the data type and the number of wires.
    int numWiresProp = -1;
    cudaDataType_t svDataTypeProp{cudaDataType_t(0)};

    ERRCHK(custatevecExStateVectorGetProperty(stateVector, CUSTATEVEC_EX_SV_PROP_NUM_WIRES,
                                              &numWiresProp, sizeof(numWiresProp)));
    ERRCHK(custatevecExStateVectorGetProperty(stateVector, CUSTATEVEC_EX_SV_PROP_DATA_TYPE,
                                              &svDataTypeProp, sizeof(svDataTypeProp)));

    const char* dataTypeStr = "unknown";
    if (svDataTypeProp == CUDA_C_64F)
        dataTypeStr = "c128";
    else if (svDataTypeProp == CUDA_C_32F)
        dataTypeStr = "c64";
    output("dataType=%s, numWires=%d\n", dataTypeStr, numWires);

    return stateVector;
}

//
// Apply the circuit by using custatevecExApplyMatrix()
//
void applyH(custatevecExStateVectorDescriptor_t stateVector, int numWires)
{
    // 2x2 unitary matrix in row-major order.
    DblComplex H[] = {1, 1, 1, -1};
    for (auto& elm : H)
        elm *= 1. / std::sqrt(2.);
    const int adjoint = 0;

    for (int target = 0; target < numWires; ++target)
    {
        ERRCHK(custatevecExApplyMatrix(stateVector, H, CUDA_C_64F, CUSTATEVEC_EX_MATRIX_DENSE,
                                       CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, &target, 1, nullptr,
                                       nullptr, 0));
    }
}

void applyRotationByZ(ExStateVector stateVector, int numWires, double theta)
{
    std::vector<custatevecPauli_t> Zstr(numWires, CUSTATEVEC_PAULI_Z);
    std::vector<int> targets(numWires);
    std::iota(targets.begin(), targets.end(), 0);

    // Apply Zstr rotation
    ERRCHK(custatevecExApplyPauliRotation(stateVector, theta, Zstr.data(), targets.data(), numWires,
                                          nullptr, nullptr, 0));
}

void computeExpectation(ExStateVector stateVector, int numWires, double* expX, double* expZ)
{
    std::vector<custatevecPauli_t> Xstr(numWires, CUSTATEVEC_PAULI_X);
    std::vector<custatevecPauli_t> Zstr(numWires, CUSTATEVEC_PAULI_Z);
    std::vector<int> basisWires(numWires);
    std::iota(basisWires.begin(), basisWires.end(), 0);

    const custatevecPauli_t* pauliStrings[]{Xstr.data(), Zstr.data()};
    const int* basisWiresArray[]{basisWires.data(), basisWires.data()};
    int nBasisWiresArray[]{numWires, numWires};

    // Compute the expectation values.
    double expValues[2];
    ERRCHK(custatevecExComputeExpectationOnPauliBasis(stateVector, expValues, pauliStrings, 2,
                                                      basisWiresArray, nBasisWiresArray));
    // synchronize to complete device to host transfer.
    ERRCHK(custatevecExStateVectorSynchronize(stateVector));
    *expX = expValues[0];
    *expZ = expValues[1];
}

int main(int argc, char* argv[])
{
    // Check for quiet mode option
    if (argc >= 2 && strcmp(argv[1], "-q") == 0)
        setOutputEnabled(false);

    int numWires = 27;
    auto svDataType = CUDA_C_64F;

    // create state vector
    output("create state vector\n");
    auto stateVector = createStateVector(svDataType, numWires);
    // set |00..0>
    ERRCHK(custatevecExStateVectorSetZeroState(stateVector));

    // Apply H gates
    applyH(stateVector, numWires);

    const double thetaDelta = (15. / 180.) * M_PI;
    double theta = 0.;
    bool pass = true;

    constexpr double ep = 1.e-12;

    for (int idx = 0; idx < 6; ++idx)
    {
        // <+|exp(-i theta Z) X exp(i theta Z)|+> = cos(2 theta)
        // <+|exp(-i theta Z) Z exp(i theta Z)|+> = 0
        double expX, expZ;
        computeExpectation(stateVector, numWires, &expX, &expZ);
        auto delta = expX - std::cos(2. * theta);
        auto thetaInDeg = theta * 180. / M_PI;
        output("theta=%g, exp(X)=%g, err=%g, exp(Z)=%g\n", thetaInDeg, expX, delta, expZ);
        if (ep < std::abs(delta))
            pass = false;
        if (ep < std::abs(expZ))
            pass = false;

        // rotate by Z string
        applyRotationByZ(stateVector, numWires, thetaDelta);
        theta += thetaDelta;
    }
    ERRCHK(custatevecExStateVectorDestroy(stateVector));

    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
