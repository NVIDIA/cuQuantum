/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file stateVectorConstruction.cpp
 * @brief State Vector Construction Factory for cuStateVec Ex API
 *
 * This module provides a unified factory interface for creating state vector
 * configurations across different distribution types:
 * - Single Device: One GPU
 * - Multi Device: Multiple GPUs with P2P
 * - Multi Process: Distributed across processes with MPI
 */

#include <custatevecEx.h>
#include <custatevecEx_ext.h>
#include "stateVectorConstruction.hpp"
#include "networkStructure.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>  // std::iota
#include <cstring>  // strcmp
#include <cstdarg>  // va_list, va_start, va_end
#include <unistd.h> // getopt

// Module-level flag to track multi-process environment state
static bool isMultiProcess_ = false;

// True when custatevecExCommunicatorInitialize() succeeded (Finalize must be called on shutdown)
static bool communicatorLibraryInitialized_ = false;

// Module-level communicator for multi-process operations (non-null when creation succeeded)
static custatevecExCommunicatorDescriptor_t exCommunicator_ = nullptr;

// Rank of module-level communicator, permanently bound after exCommunicator
// initialisation, persisting even after exCommunicator destruction
static int exCommunicatorRank_ = 0;

// Module-level data type for state vector
static cudaDataType_t svDataType_ = CUDA_C_32F;

//
// State Vector Configuration Factory
//

/**
 * @brief Internal function for silently aborting or finalizing communication
 */
void disableAndCleanUpMultiProcessEnvironment()
{
    // Destroy communicator first (required before Finalize per API contract)
    if (exCommunicator_ != nullptr)
    {
        custatevecExCommunicatorDestroy(exCommunicator_);
        exCommunicator_ = nullptr;
    }

    // Finalize IPC library if it was successfully initialized
    if (communicatorLibraryInitialized_)
    {
        custatevecExCommunicatorStatus_t status;
        custatevecExCommunicatorFinalize(&status);
        communicatorLibraryInitialized_ = false;
    }

    // Mark multi-process as disabled
    isMultiProcess_ = false;
}

/**
 * @brief Bootstrap multi-process environment with quiet mode handling
 */
void bootstrapMultiProcessEnvironment(int* argc, char*** argv)
{
    // Communicator configuration - choose one:
    // Option 1: Use built-in OPENMPI (default)
    custatevecCommunicatorType_t communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
    const char* libraryPath = nullptr;

    // Option 2: Use external communicator plugin (uncomment to enable)
    // custatevecCommunicatorType_t communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL;
    // const char* libraryPath = "./libmpiCommunicator.so";  // or nullptr to search in process

    // Begin by assuming multi-process and successful comm initialisation, later disabling if
    // impossible or initialisation fails, as done by disableAndCleanUpMultiProcessEnvironment()
    isMultiProcess_ = true;
    communicatorLibraryInitialized_ = true;

    // Try to initialize communicator
    custatevecExCommunicatorStatus_t commStatus;
    // clang-format off
    custatevecStatus_t status = custatevecExCommunicatorInitialize(
        communicatorType,       // communicatorType
        libraryPath,            // libraryPath
        argc,                   // argc
        argv,                   // argv
        &commStatus             // output status
    );
    // clang-format on

    // Disable multi-process if MPI was not successfully initialized
    if (status != CUSTATEVEC_STATUS_SUCCESS ||
        commStatus != CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS)
    {
        disableAndCleanUpMultiProcessEnvironment();
    }

    // Try create communicator, otherwise disable multi-process
    if (isMultiProcess_)
    {
        status = custatevecExCommunicatorCreate(&exCommunicator_);
        if (status != CUSTATEVEC_STATUS_SUCCESS)
        {
            output("Failed creating communicator; proceeding with single process.\n");
            disableAndCleanUpMultiProcessEnvironment();
        }
    }

    // Get and validate the number of processes
    int numProcesses = 1;
    if (isMultiProcess_)
    {
        ERRCHK_EXCOMM(exCommunicator_->intf->getSize(exCommunicator_, &numProcesses));

        // Clean up and exit if process config is invalid (do not fallback to single-process)
        if (numProcesses == 0 || (numProcesses & (numProcesses - 1)) != 0)
        {
            output("Number of processes must be positive and a power of 2\n");
            exit(EXIT_FAILURE);
        }
    }

    // Disable multi-process when only a single process was deployed
    if (numProcesses == 1)
    {
        disableAndCleanUpMultiProcessEnvironment();
    }

    // In multi-process mode, bind rank, and disable output for non-root processes
    if (isMultiProcess_)
    {
        ERRCHK_EXCOMM(exCommunicator_->intf->getRank(exCommunicator_, &exCommunicatorRank_));
        if (exCommunicatorRank_ != 0)
            setOutputEnabled(false);
    }

    if (isMultiProcess_)
        output("Multi-process (%d processes) environment initialized successfully\n", numProcesses);
    else
        output("Running in single-process mode\n");
}

/**
 * @brief Finalize multi-process environment
 */
void finalizeMultiProcessEnvironment()
{
    bool wasMultiProcess = isMultiProcess_;

    disableAndCleanUpMultiProcessEnvironment();

    if (wasMultiProcess)
    {
        output("Multi-process environment finalized\n");
    }
}

/**
 * @brief Create single-device state vector configuration (internal)
 *
 * @param svDataType State vector data type (CUDA_C_32F or CUDA_C_64F)
 * @param numWires Number of qubits
 * @param numMigrationWires Number of migration wires, informing portion of
 *   the state vector stored in host memory
 * @return Dictionary containing state vector configuration
 */
static custatevecExDictionaryDescriptor_t
createSingleDeviceConfig(cudaDataType_t svDataType, int32_t numWires, int32_t numMigrationWires)
{
    custatevecExDictionaryDescriptor_t svConfig{nullptr};

    // clang-format off
    ERRCHK(custatevecExConfigureStateVectorSingleDevice(
        &svConfig,
        svDataType,
        numWires,                     // numWires
        numWires - numMigrationWires, // numDeviceWires
        -1,                           // deviceId, -1 specifies the current device.
        0                             // capability
    ));
    // clang-format on

    return svConfig;
}

/**
 * @brief Create multi-device state vector configuration
 */
static custatevecExDictionaryDescriptor_t
createMultiDeviceConfig(cudaDataType_t svDataType, int numWires, int numDevices, int networkType)
{
    custatevecExDictionaryDescriptor_t svConfig;

    // Generate device IDs (0, 1, 2, ..., numDevices-1)
    std::vector<int32_t> deviceIds(numDevices);
    std::iota(deviceIds.begin(), deviceIds.end(), 0);

    // Calculate device wires: numWires = numGlobalBits + numDeviceWires
    // numGlobalBits = log2(numDevices) for inter-device communication
    int32_t numGlobalBits = 0;
    int32_t tempDevices = numDevices;
    while (tempDevices > 1)
    {
        numGlobalBits++;
        tempDevices >>= 1;
    }
    int32_t numDeviceWires = numWires - numGlobalBits;

    // Map network type: 1=SWITCH, 2=FULLMESH for multi-device
    custatevecDeviceNetworkType_t deviceNetworkType = CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH;
    switch (networkType)
    {
    case 0: // default
    case 1:
        deviceNetworkType = CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH;
        break;
    case 2:
        deviceNetworkType = CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH;
        break;
    default:
        output("Unknown networkType\n");
        exit(EXIT_FAILURE);
    }

    // clang-format off
    ERRCHK(custatevecExConfigureStateVectorMultiDevice(&svConfig,
        svDataType,                              // data type
        numWires,                                // total number of qubits
        numDeviceWires,                          // qubits per device
        deviceIds.data(),                        // device IDs
        numDevices,                              // number of devices
        deviceNetworkType,                       // network type (configurable)
        0                                        // capability flags
    ));
    // clang-format on

    return svConfig;
}

/**
 * @brief Create multi-process state vector configuration (internal)
 *
 * @param svDataType State vector data type (CUDA_C_32F or CUDA_C_64F)
 * @param numWires Total number of qubits
 * @param networkLayers Network topology configuration
 * @param exCommDesc Communicator descriptor
 * @return Dictionary containing state vector configuration
 */
static custatevecExDictionaryDescriptor_t createMultiProcessConfig(
    cudaDataType_t svDataType, int32_t numWires,
    const NetworkLayers& networkLayers,
    custatevecExCommunicatorDescriptor_t exCommDesc)
{
    custatevecExDictionaryDescriptor_t svConfig{nullptr};

    // Get rank and size from communicator
    int32_t rank, numProcesses;
    ERRCHK_EXCOMM(exCommDesc->intf->getRank(exCommDesc, &rank));
    ERRCHK_EXCOMM(exCommDesc->intf->getSize(exCommDesc, &numProcesses));

    // Find numInterProcWires = log2(numProcesses)
    int32_t numInterProcBits = 0;
    int32_t temp = numProcesses;
    while (temp > 1)
    {
        numInterProcBits++;
        temp >>= 1;
    }

    // We will collect and group global bits, spanned across processes
    std::vector<custatevecExGlobalIndexBitClass_t> globalIndexBitClasses;
    std::vector<int32_t> numGlobalIndexBitsPerLayer;

    // Collect non-empty inter-process global bit groups and count total assigned bits
    int32_t numAccumulatedGlobalIndexBits = 0;
    for (const auto& layer : networkLayers)
    {
        globalIndexBitClasses.push_back(layer.globalIndexBitClass);
        int numGlobalBitsPerLayer;
        if (layer.numGlobalIndexBits == 0)
        {
            numGlobalBitsPerLayer = numInterProcBits - numAccumulatedGlobalIndexBits;
            numGlobalIndexBitsPerLayer.push_back(numGlobalBitsPerLayer);
            numAccumulatedGlobalIndexBits = numInterProcBits;
            break;
        }
        auto maxNumGlobalIndexBitsPerLayer = numInterProcBits - numAccumulatedGlobalIndexBits;
        if (maxNumGlobalIndexBitsPerLayer <= layer.numGlobalIndexBits)
        {
            numGlobalIndexBitsPerLayer.push_back(maxNumGlobalIndexBitsPerLayer);
            numAccumulatedGlobalIndexBits += maxNumGlobalIndexBitsPerLayer;
            break;
        }
        numGlobalIndexBitsPerLayer.push_back(layer.numGlobalIndexBits);
        numAccumulatedGlobalIndexBits += layer.numGlobalIndexBits;
    }
    if (numAccumulatedGlobalIndexBits < numInterProcBits)
    {
        output("NetworkLayers is too thin to build numWires state vector.\n");
        exit(EXIT_FAILURE);
    }
    
    // All remaining wires are local, spanned within a single device
    int32_t numDeviceWires = numWires - numInterProcBits;
    int32_t deviceId = -1; // Dynamic device assignment on creating state vector

    // Specify only exchanges between P2P processes can leverage shared memory
    custatevecExMemorySharingMethod_t memorySharingMethod =
        CUSTATEVEC_EX_MEMORY_SHARING_METHOD_NONE;
    for (const auto& layer : networkLayers)
    {
        if (layer.globalIndexBitClass == CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P)
        {
            memorySharingMethod = CUSTATEVEC_EX_MEMORY_SHARING_METHOD_AUTODETECT;
            break;
        }
    }

    // Transfer workspace size (16 MB default)
    const size_t transferWorkspaceSizeInBytes = 16 * 1024 * 1024;
    // clang-format off
    ERRCHK(custatevecExConfigureStateVectorMultiProcess(
        &svConfig,
        svDataType,
        numWires,                                           // numWires
        numDeviceWires,                                     // numDeviceWires
        deviceId,                                           // deviceId
        memorySharingMethod,                                // memorySharingMethod (auto-detect if P2P used)
        globalIndexBitClasses.data(),                       // globalIndexBitClasses
        numGlobalIndexBitsPerLayer.data(),                  // numGlobalIndexBitsPerLayer
        static_cast<int32_t>(globalIndexBitClasses.size()), // numGlobalIndexBitLayers
        transferWorkspaceSizeInBytes,                       // transferWorkspaceSizeInBytes
        nullptr,                                            // auxConfig (reserved)
        0                                                   // capability (reserved)
    ));
    // clang-format on

    return svConfig;
}

/**
 * @brief Show usage information
 */
static void showUsage(const char* programName)
{
    output("Usage: %s [options]\n", programName);
    output("\n");
    output("Options:\n");
    output("  -h          Show this help message and exit\n");
    output("  -q          Quiet mode (suppress output)\n");
    output("  -d <num>    Number of devices for multi-device (default: auto-detect)\n");
    output("  -m <num>    Number of migration wires for host-memory usage (default: 0)\n");
    output("  -t <type>   Data type - 'f'/'float' or 'd'/'double' (default: float)\n");
    output("  -k <net>    Device network topology for multi-device: 1=SWITCH, 2=FULLMESH "
           "(default: SWITCH)\n");
    output("              Network structure for multi-process: 3=SuperPOD, 4=GB200NVL, "
           "5=SwitchTree, 6=Communicator (default: SuperPOD)\n");
    output("\n");
    output("Examples:\n");
    output("  %s                    # Use default settings\n", programName);
    output("  %s -d 2               # Use 2 GPUs in multi-device mode\n", programName);
    output("  %s -m 2               # Use 2 migration wires, storing 75% of state in host memory\n", programName);
    output("  %s -t double          # Use double precision\n", programName);
    output("  %s -k 4               # Use GB200NVL for multi-process mode\n", programName);
    output("\n");
}

/**
 * @brief Configure state vector from command line arguments
 */
custatevecExDictionaryDescriptor_t configureStateVector(int argc, char* argv[], int numWires)
{
    // Default parameters
    cudaDataType_t svDataType = CUDA_C_32F;
    int networkType = 0; // 0: Use the default according to the state vector configuration.
    int numDevices = -1;
    int numMigrationWires = 0;

    // Parse command line options with getopt
    int opt;

    // Reset getopt state for proper parsing
    optind = 1;

    while ((opt = getopt(argc, argv, "hqd:m:t:k:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            showUsage(argv[0]);
            exit(EXIT_SUCCESS);
            break;
        case 'q':
            setOutputEnabled(false);
            break;
        case 'd':
            numDevices = atoi(optarg);
            if (numDevices < 1)
            {
                output("Error: Number of devices must be positive\n");
                exit(EXIT_FAILURE);
            }
            if (numDevices & (numDevices - 1))
            {
                output("Error: Number of devices must be a power of 2\n");
                exit(EXIT_FAILURE);
            }
            if (numDevices > 1 && isMultiProcess_)
            {
                output("Error: Only one device can be utilised per-process in multi-process mode\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'm':
            numMigrationWires = atoi(optarg);
            if (numMigrationWires < 0)
            {
                output("Error: Number of migration wires must be positive or zero\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 't':
            // Data type: f=float (CUDA_C_32F), d=double (CUDA_C_64F)
            if (strcmp(optarg, "f") == 0 || strcmp(optarg, "float") == 0)
            {
                svDataType = CUDA_C_32F;
            }
            else if (strcmp(optarg, "d") == 0 || strcmp(optarg, "double") == 0)
            {
                svDataType = CUDA_C_64F;
            }
            else
            {
                output("Error: Data type must be 'f'/'float' or 'd'/'double'\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'k':
            // Network topology: 1-2 for multi-device, 3-5 for multi-process
            networkType = atoi(optarg);
            if (networkType < 1 || networkType > 6)
            {
                output("Error: Network type must be in range 1-6\n");
                exit(EXIT_FAILURE);
            }
            break;
        case '?':
            // getopt already printed error message for unknown option
            break;
        default:
            break;
        }
    }

    // Auto-detect devices if not specified (not consulted by multi-process; assumes =1)
    if (numDevices == -1 && (!isMultiProcess_))
        ERRCHK_CUDA(cudaGetDeviceCount(&numDevices));

    // Store the configured data type for later retrieval
    svDataType_ = svDataType;

    // Branch based on detected environment
    if ((numDevices == 1) && (!isMultiProcess_))
    {
        output("Configure state vector: Single-device, Qubits: %d, MigrationWires: %d, DataType: %s\n", 
               numWires, numMigrationWires, (svDataType == CUDA_C_32F) ? "float" : "double");
        return createSingleDeviceConfig(svDataType, numWires, numMigrationWires);
    }
    else if ((1 < numDevices) && (!isMultiProcess_))
    {
        if (numMigrationWires > 0)
        {
            output("Error: Cannot use host memory (via migration wires) in multi-device mode\n");
            exit(EXIT_FAILURE);
        }

        output("Configure state vector: Multi-device, Qubits: %d, DataType: %s, Devices: %d\n",
               numWires, (svDataType == CUDA_C_32F) ? "float" : "double", numDevices);
        return createMultiDeviceConfig(svDataType, numWires, numDevices, networkType);
    }
    else
    {
        if (numMigrationWires > 0)
        {
            output("Error: Cannot use host memory (via migration wires) in multi-process mode\n");
            exit(EXIT_FAILURE);
        }

        output("Configure state vector: Multi-process, Qubits: %d, DataType: %s, NetworkType: %d\n",
               numWires, (svDataType == CUDA_C_32F) ? "float" : "double", networkType);

        // Select network configuration based on user choice
        NetworkLayers networkConfig;
        switch (networkType)
        {
        case 0: // default
        case 3:
            networkConfig = createSuperPODNetworkConfig();
            break;
        case 4:
            networkConfig = createGB200NVLNetworkConfig();
            break;
        case 5:
            networkConfig = createSwitchTreeNetworkConfig();
            break;
        case 6:
            networkConfig = createCommunicatorNetwork();
            break;
        default:
            output("Error: Unknown networkType\n");
            exit(EXIT_FAILURE);
        }
        return createMultiProcessConfig(svDataType, numWires, networkConfig, exCommunicator_);
    }
}

/**
 * @brief Create state vector from configuration
 */
custatevecExStateVectorDescriptor_t createStateVector(custatevecExDictionaryDescriptor_t config)
{
    custatevecExStateVectorDescriptor_t stateVector;

    // Branch based on detected environment
    if (isMultiProcess_)
    {
        // Use the stored communicator from bootstrapMultiProcessEnvironment
        // clang-format off
        ERRCHK(custatevecExStateVectorCreateMultiProcess(&stateVector, config,
                                                         nullptr,           // stream
                                                         exCommunicator_,   // communicator
                                                         nullptr));         // resourceManager
        // clang-format on
    }
    else
    {
        ERRCHK(
            custatevecExStateVectorCreateSingleProcess(&stateVector, config, nullptr, 0, nullptr));
    }

    return stateVector;
}

/**
 * @brief Get the multi-process communicator
 */
custatevecExCommunicatorDescriptor_t getMultiProcessCommunicator()
{
    return exCommunicator_;
}

/**
 * @brief Get rank in the multi-process communicator
 */
int getMultiProcessRank() 
{
    // Can be non-zero even after exCommunicator_ destruction (isMultiProcess_=0)
    return exCommunicatorRank_;
}

/**
 * @brief Get the configured state vector data type
 */
cudaDataType_t getStateVectorDataType()
{
    return svDataType_;
}
