# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This code was automatically generated across versions from 25.11.0 to 26.02.0, generator version 0.3.1.dev1380+g6ceff55cb.d20260311. Do not modify it directly.

from libc.stdint cimport intptr_t

from .cycustabilizer cimport *


###############################################################################
# Types
###############################################################################

ctypedef custabilizerCircuit_t Circuit
ctypedef custabilizerFrameSimulator_t FrameSimulator
ctypedef custabilizerHandle_t Handle

ctypedef cudaStream_t Stream
ctypedef cudaEvent_t Event
ctypedef cudaDataType DataType


###############################################################################
# Enum
###############################################################################

ctypedef custabilizerStatus_t _Status


###############################################################################
# Functions
###############################################################################

cpdef int get_version() except? 0
cpdef str get_error_string(int status)
cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef int64_t circuit_size_from_string(intptr_t handle, circuit_string) except? 0
cpdef intptr_t create_circuit_from_string(intptr_t handle, circuit_string, buffer_device, int64_t buffer_size) except? 0
cpdef destroy_circuit(intptr_t circuit)
cpdef intptr_t create_frame_simulator(intptr_t handle, int64_t num_qubits, int64_t num_shots, int64_t num_measurements, int64_t table_stride_major) except? 0
cpdef destroy_frame_simulator(intptr_t frame_simulator)
cpdef frame_simulator_apply_circuit(intptr_t handle, intptr_t frame_simulator, intptr_t circuit, int randomize_frame_after_measurement, uint64_t seed, intptr_t x_table_device, intptr_t z_table_device, intptr_t m_table_device, intptr_t stream)
cpdef sample_prob_array(intptr_t handle, int64_t num_samples, int64_t num_probs, intptr_t probs, uint64_t seed, intptr_t samples, intptr_t stream)
cpdef size_t sample_prob_array_sparse_prepare(intptr_t handle, int64_t num_samples, int64_t num_probs) except? 0
cpdef sample_prob_array_sparse_compute(intptr_t handle, int64_t num_samples, int64_t num_probs, intptr_t probs, uint64_t seed, intptr_t nnz, intptr_t column_indices, intptr_t row_offsets, intptr_t workspace, size_t workspace_size, intptr_t stream)
cpdef gf2_sparse_dense_matrix_multiply(intptr_t handle, uint64_t m, uint64_t n, uint64_t k, uint64_t nnz, intptr_t column_indices, intptr_t row_offsets, intptr_t b, int32_t beta, intptr_t c, intptr_t stream)
cpdef gf2_sparse_sparse_matrix_multiply(intptr_t handle, uint64_t m, uint64_t n, uint64_t k, intptr_t a_column_indices, intptr_t a_row_offsets, uint64_t b_nnz, intptr_t b_column_indices, intptr_t b_row_offsets, int32_t beta, intptr_t c, intptr_t stream)
