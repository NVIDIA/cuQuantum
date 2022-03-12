#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SHELL           := /bin/bash
CUDA_TOOLKIT    := $(shell realpath $$(dirname $$(command -v nvcc))/..)
CUSTATEVEC_ROOT ?= ${CUQUANTUM_ROOT}

INCLUDE_DIRS    := -I$(CUSTATEVEC_ROOT)/include
LIBRARY_DIRS    := -L$(CUSTATEVEC_ROOT)/lib64 -L$(CUSTATEVEC_ROOT)/lib
LINKER_FLAGS    := -lcudart -lcustatevec

ARCH_FLAGS_SM70 = -gencode arch=compute_70,code=sm_70
ARCH_FLAGS_SM75 = -gencode arch=compute_75,code=sm_75
ARCH_FLAGS_SM80 = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
ARCH_FLAGS      = $(ARCH_FLAGS_SM70) $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80)

CXX_FLAGS       = -std=c++11 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(ARCH_FLAGS) $(LINKER_FLAGS)

all: check-env
	nvcc gate_application.cu   -o gate_application   ${CXX_FLAGS}
	nvcc permutation_matrix.cu -o permutation_matrix ${CXX_FLAGS}
	nvcc diagonal_matrix.cu    -o diagonal_matrix    ${CXX_FLAGS}
	nvcc exponential_pauli.cu  -o exponential_pauli  ${CXX_FLAGS}
	nvcc expectation.cu        -o expectation        ${CXX_FLAGS}
	nvcc expectation_pauli.cu  -o expectation_pauli  ${CXX_FLAGS}
	nvcc sampler.cu            -o sampler            ${CXX_FLAGS}
	nvcc measure_zbasis.cu     -o measure_zbasis     ${CXX_FLAGS}
	nvcc batch_measure.cu      -o batch_measure      ${CXX_FLAGS}
	nvcc accessor_get.cu       -o accessor_get       ${CXX_FLAGS}
	nvcc accessor_set.cu       -o accessor_set       ${CXX_FLAGS}

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
	rm -f gate_application \
		permutation_matrix \
		diagonal_matrix \
		exponential_pauli \
		expectation \
		expectation_pauli \
		sampler \
		measure_zbasis \
		batch_measure \
		accessor_get \
		accessor_set
