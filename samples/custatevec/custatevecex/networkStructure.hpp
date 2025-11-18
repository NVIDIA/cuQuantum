/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <vector>
#include <custatevecEx.h>

/*
 * Device network structure
 *
 * Multi-device device network topology
 *
 * 1. SWITCH TOPOLOGY (CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH)
 *    GPUs connected via central switching fabric (NVSwitch, PCIe switch)
 *
 *    Example: DGX A100 (8x A100 + NVSwitch)
 *
 *         GPU0 ──┐
 *         GPU1 ──┤
 *         GPU2 ──┤    NVSwitch
 *         GPU3 ──┤    Fabric
 *         GPU4 ──┤      or
 *         GPU5 ──┤   PCIe Switch
 *         GPU6 ──┤
 *         GPU7 ──┘
 *
 * 2. MESH TOPOLOGY (CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH)
 *    GPUs directly connected via point-to-point links (NVLink)
 *
 *    Example: 4-GPU Direct NVLink Mesh
 *
 *         GPU0 ────────── GPU1
 *          │ ╲        ╱ │
 *          │   ╲    ╱   │
 *          │     ╲╱     │
 *          │     ╱╲     │
 *          │   ╱    ╲   │
 *          │ ╱        ╲ │
 *         GPU2 ────────── GPU3
 */

/**
 * Multi-process network templates
 */

struct NetworkLayer
{
    custatevecExGlobalIndexBitClass_t globalIndexBitClass;
    int numGlobalIndexBits;
};

typedef std::vector<NetworkLayer> NetworkLayers;

// SuperPOD
// One GPU is assigned to each process
// Each hardware node has 8 B200 GPUs connected by NVSwitch
// Nodes are connected via full-fat tree IB network

inline NetworkLayers createSuperPODNetworkConfig()
{

    return {{
                // 8 GPUs connected by NVSwitch
                CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P, // NVSwitch
                3                                                   // = log(8 GPUs)
            },
            {
                CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR, // IB
                0 // Will be adjusted to fit numWires
            }};
}

// GB200 NVL cluster
// One GPU is assigned to each process
// All GPUs are connected via multi-node NVLink

inline NetworkLayers createGB200NVLNetworkConfig()
{

    return {{
        // 8 GPUs connected by NVSwitch
        CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P, // NVLink Switch
        0                                                   // Will be adjusted to fit numWires
    }};
}

// PCIe switch tree
// CPU has 1 PCIe root complex
// Two PCIe switches are connected to the root complex
// and have 4 GPUs for each.
inline NetworkLayers createSwitchTreeNetworkConfig()
{
    return {{
                // 4 GPUs connected by PCIe switch
                CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P, // PCIe Switch
                2                                                   // = log(4 GPUs)
            },
            {
                // 2 PCIe switches connected to root complex
                CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P, // PCIe Root Complex
                1                                                   // = log(2 PCIe switches)
            }};
}

// Network only with communicator
// GPUDirect P2P is not available in this network
inline NetworkLayers createCommunicatorNetwork()
{
    return {{CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR, // Use Communicactor
             0}};
}
