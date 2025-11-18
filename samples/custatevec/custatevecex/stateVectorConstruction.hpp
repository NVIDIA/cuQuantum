/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file stateVectorConstruction.hpp
 * @brief State Vector Construction Factory for cuStateVec Ex API
 */

#pragma once

#include <custatevecEx.h>
#include <cuda_runtime.h>
#include "networkStructure.hpp"
#include "common.hpp"

//
// State Vector Construction API
//

/**
 * @brief Bootstrap multi-process environment
 *
 * @param argc Pointer to argument count from main()
 * @param argv Pointer to argument vector from main()
 * @return bool True if multi-process environment is available, false for single-process
 *
 * @details Initializes MPI communicator infrastructure if running in multi-process mode.
 * Handles MPI_Init and custatevecExCommunicatorInitialize automatically.
 * Returns true if MPI environment is detected and initialized successfully.
 * Returns false for single-process execution. Call this before configureStateVector().
 */
void bootstrapMultiProcessEnvironment(int* argc, char*** argv);

/**
 * @brief Finalize multi-process environment
 *
 * @details Cleans up MPI communicator infrastructure and resets module state.
 * Destroys the stored communicator, calls custatevecExCommunicatorFinalize(),
 * and resets internal flags. Call this at the end of your program after
 * destroying state vectors and configurations to ensure proper MPI cleanup.
 */
void finalizeMultiProcessEnvironment();

/**
 * @brief Configure state vector from command line arguments
 *
 * @param argc Argument count from main()
 * @param argv Argument vector from main()
 * @return custatevecExDictionaryDescriptor_t Configuration dictionary from cuStateVec Ex API
 *
 * @details Parses command line arguments and creates appropriate state vector configuration.
 * Automatically detects single-device, multi-device, or multi-process configuration based on
 * environment detected by bootstrapMultiProcessEnvironment().
 *
 * Command Line Options:
 *   -h          : Show help message and exit
 *   -q          : Quiet mode (suppress output)
 *   -d <num>    : Number of devices for multi-device (default: auto-detect)
 *   -t <type>   : Data type - 'f'/'float' or 'd'/'double' (default: float)
 *   -k <net>    : Network topology
 *                 For multi-device: 1=SWITCH (default), 2=FULLMESH
 *                 For multi-process: 3=SuperPOD (default), 4=GB200NVL, 5=SwitchTree
 */
custatevecExDictionaryDescriptor_t configureStateVector(int argc, char* argv[], int numWires);

/**
 * @brief Create state vector from configuration
 *
 * @param config Configuration dictionary from custatevecExConfigureStateVector*() APIs
 * @return custatevecExStateVectorDescriptor_t Created state vector instance
 */
custatevecExStateVectorDescriptor_t createStateVector(custatevecExDictionaryDescriptor_t config);

/**
 * @brief Get the multi-process communicator
 *
 * @return custatevecExCommunicatorDescriptor_t The communicator instance, or nullptr if not in
 * multi-process mode
 */
custatevecExCommunicatorDescriptor_t getMultiProcessCommunicator();

/**
 * @brief Get the configured state vector data type
 *
 * @return cudaDataType_t The data type used for state vector (CUDA_C_32F or CUDA_C_64F)
 *
 * @details Returns the data type that was configured via the -t command line option.
 * Default is CUDA_C_32F (float). Call this after configureStateVector() to get the
 * data type used for state vector operations.
 */
cudaDataType_t getStateVectorDataType();
