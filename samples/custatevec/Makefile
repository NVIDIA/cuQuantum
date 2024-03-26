# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

SHELL           := /bin/bash
CUDA_TOOLKIT    := $(shell realpath $$(dirname $$(command -v nvcc))/..)
CUSTATEVEC_ROOT ?= ${CUQUANTUM_ROOT}

INCLUDE_DIRS    := -I$(CUSTATEVEC_ROOT)/include
LIBRARY_DIRS    := -L$(CUSTATEVEC_ROOT)/lib64 -L$(CUSTATEVEC_ROOT)/lib
LINKER_FLAGS    := -lcudart -lcustatevec

ARCH_FLAGS_SM70 = -gencode arch=compute_70,code=sm_70
ARCH_FLAGS_SM75 = -gencode arch=compute_75,code=sm_75
ARCH_FLAGS_SM80 = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
ARCH_FLAGS_SM90 = -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
ARCH_FLAGS      = $(ARCH_FLAGS_SM70) $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80) $(ARCH_FLAGS_SM90)

CXX_FLAGS       = -std=c++11 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(ARCH_FLAGS) $(LINKER_FLAGS)

all: check-env
	nvcc initialize_sv.cu            -o initialize_sv            ${CXX_FLAGS}
	nvcc gate_application.cu         -o gate_application         ${CXX_FLAGS}
	nvcc permutation_matrix.cu       -o permutation_matrix       ${CXX_FLAGS}
	nvcc diagonal_matrix.cu          -o diagonal_matrix          ${CXX_FLAGS}
	nvcc batched_gate_application.cu -o batched_gate_application ${CXX_FLAGS}
	nvcc exponential_pauli.cu        -o exponential_pauli        ${CXX_FLAGS}
	nvcc expectation.cu              -o expectation              ${CXX_FLAGS}
	nvcc batched_expectation.cu      -o batched_expectation      ${CXX_FLAGS}
	nvcc expectation_pauli.cu        -o expectation_pauli        ${CXX_FLAGS}
	nvcc sampler.cu                  -o sampler                  ${CXX_FLAGS}
	nvcc measure_zbasis.cu           -o measure_zbasis           ${CXX_FLAGS}
	nvcc batch_measure.cu            -o batch_measure            ${CXX_FLAGS}
	nvcc batched_collapse.cu         -o batched_collapse         ${CXX_FLAGS}
	nvcc batched_abs2sum.cu          -o batched_abs2sum          ${CXX_FLAGS}
	nvcc batched_measure.cu          -o batched_measure          ${CXX_FLAGS}
	nvcc accessor_get.cu             -o accessor_get             ${CXX_FLAGS}
	nvcc accessor_set.cu             -o accessor_set             ${CXX_FLAGS}
	nvcc test_matrix_type.cu         -o test_matrix_type         ${CXX_FLAGS}
	nvcc memory_handler.cu           -o memory_handler           ${CXX_FLAGS}
	nvcc swap_index_bits.cu          -o swap_index_bits          ${CXX_FLAGS}
	nvcc mgpu_swap_index_bits.cu     -o mgpu_swap_index_bits     ${CXX_FLAGS}
	nvcc mgpu_batch_measure.cu       -o mgpu_batch_measure       ${CXX_FLAGS}
	nvcc mgpu_sampler.cu             -o mgpu_sampler             ${CXX_FLAGS}
	nvcc subsv_migration.cu          -o subsv_migration          ${CXX_FLAGS}


check-env:
	@ echo "" && \
	echo "CUDA_TOOLKIT=${CUDA_TOOLKIT}"; \
	echo "CUSTATEVEC_ROOT=${CUSTATEVEC_ROOT}"; \
	echo ""; \
	if [[ -z "${CUSTATEVEC_ROOT}" ]]; \
	then \
		echo "" && \
		echo "Neither CUSTATEVEC_ROOT nor CUQUANTUM_ROOT is set." && \
		exit 1; \
	fi

clean:
	rm -f initialize_sv \
		gate_application \
		permutation_matrix \
		diagonal_matrix \
		batched_gate_application \
		exponential_pauli \
		expectation \
		batched_expectation \
		expectation_pauli \
		sampler \
		measure_zbasis \
		batch_measure \
		batched_collapse \
		batched_abs2sum \
		batched_measure \
		accessor_get \
		accessor_set \
		test_matrix_type \
		memory_handler \
		swap_index_bits \
		mgpu_swap_index_bits \
		mgpu_batch_measure \
		mgpu_sampler \
		subsv_migration
