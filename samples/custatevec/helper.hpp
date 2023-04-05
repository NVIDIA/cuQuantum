/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define HANDLE_ERROR(x)                                                        \
{   const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS ) {                                   \
        printf("Error: %s in line %d\n",                                       \
               custatevecGetErrorString(err), __LINE__); return err; }         \
};

#define HANDLE_CUDA_ERROR(x)                                                   \
{   const auto err = x;                                                        \
    if (err != cudaSuccess ) {                                                 \
        printf("Error: %s in line %d\n",                                       \
               cudaGetErrorString(err), __LINE__); return err; }               \
};

bool almost_equal(cuDoubleComplex x, cuDoubleComplex y) {
    const double eps = 1.0e-5;
    const cuDoubleComplex diff = cuCsub(x, y);
    return (cuCabs(diff) < eps);
}

bool almost_equal(double x, double y) {
    const double eps = 1.0e-5;
    const double diff = x - y;
    return (abs(diff) < eps);
}
