# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

SHELL           := /bin/bash
CUDA_TOOLKIT    := $(shell realpath $$(dirname $$(command -v nvcc))/..)
CUSTATEVEC_ROOT ?= ${CUQUANTUM_ROOT}

CXX             := g++
INCLUDE_DIRS    := -I$(CUSTATEVEC_ROOT)/include -I$(CUDA_TOOLKIT)/include
LIBRARY_DIRS    := -L$(CUSTATEVEC_ROOT)/lib64 -L$(CUSTATEVEC_ROOT)/lib -L$(CUDA_TOOLKIT)/lib64
LINKER_FLAGS    := -lcudart -lcustatevec

CXX_FLAGS       = -std=c++11 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(LINKER_FLAGS)

all: check-env
	${CXX} estimate_pi.cpp            -o estimate_pi            ${CXX_FLAGS}
	${CXX} pauli_functions.cpp        -o pauli_functions        ${CXX_FLAGS}
	${CXX} noise_channel.cpp          -o noise_channel          ${CXX_FLAGS}
	${CXX} quantum_state_initialization.cpp -o quantum_state_initialization ${CXX_FLAGS}
	${CXX} index_bit_permutation.cpp  -o index_bit_permutation  ${CXX_FLAGS}

# Individual targets
estimate_pi: check-env
	${CXX} estimate_pi.cpp -o estimate_pi ${CXX_FLAGS}

pauli_functions: check-env
	${CXX} pauli_functions.cpp -o pauli_functions ${CXX_FLAGS}

noise_channel: check-env
	${CXX} noise_channel.cpp -o noise_channel ${CXX_FLAGS}

quantum_state_initialization: check-env
	${CXX} quantum_state_initialization.cpp -o quantum_state_initialization ${CXX_FLAGS}

index_bit_permutation: check-env
	${CXX} index_bit_permutation.cpp -o index_bit_permutation ${CXX_FLAGS}

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
	rm -f estimate_pi pauli_functions noise_channel quantum_state_initialization index_bit_permutation
