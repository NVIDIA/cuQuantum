#!/bin/bash

#
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Example build script for MPI communicator plugin
# Usage: ./build_mpi_communicator.sh [MPI_PREFIX]
# Example: ./build_mpi_communicator.sh /usr/lib/x86_64-linux-gnu/openmpi
# Environment variables: CUSTATEVEC_ROOT, CUQUANTUM_ROOT, CUDA_TOOLKIT, CC

set -e  # Exit on error

# Configuration
MPI_PREFIX="${1:-/usr}"

# Auto-detect CUDA toolkit from nvcc if not set
if [ -z "$CUDA_TOOLKIT" ]; then
    NVCC_PATH=$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
    CUDA_TOOLKIT=$(dirname $(dirname $NVCC_PATH))
fi

# Use CUQUANTUM_ROOT as fallback for CUSTATEVEC_ROOT
if [ -z "$CUSTATEVEC_ROOT" ]; then
    CUSTATEVEC_ROOT="$CUQUANTUM_ROOT"
fi

OUTPUT="libmpiCommunicator.so"

# Compiler and flags
CC="${CC:-gcc}"
CFLAGS="-fPIC -Wall -O2"
INCLUDES="-I$MPI_PREFIX/include -I$CUSTATEVEC_ROOT/include -I$CUDA_TOOLKIT/include"
LDFLAGS="-shared"
LIBS="-lmpi"

# Find MPI library directory
for LIBDIR in "$MPI_PREFIX/lib64" "$MPI_PREFIX/lib"; do
    [ -d "$LIBDIR" ] && MPI_LIBDIR="$LIBDIR" && break
done
[ -n "$MPI_LIBDIR" ] && LIBS="-L$MPI_LIBDIR $LIBS"

# Build
echo ""
echo "CUDA_TOOLKIT=$CUDA_TOOLKIT"
echo "CUSTATEVEC_ROOT=$CUSTATEVEC_ROOT"
echo "MPI_PREFIX=$MPI_PREFIX"
echo ""
echo "Building $OUTPUT..."
$CC $CFLAGS $INCLUDES $LDFLAGS mpiCommunicator.c $LIBS -o $OUTPUT
echo "✓ Success: $OUTPUT ($(ls -lh $OUTPUT | awk '{print $5}'))"
