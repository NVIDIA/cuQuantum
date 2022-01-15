ARCH_FLAGS_SM70 = -gencode arch=compute_70,code=sm_70
ARCH_FLAGS_SM75 = -gencode arch=compute_75,code=sm_75
ARCH_FLAGS_SM80 = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
ARCH_FLAGS = $(ARCH_FLAGS_SM70) $(ARCH_FLAGS_SM75) $(ARCH_FLAGS_SM80)
INCLUDE_DIRS = -I${CUTENSORNET_ROOT}/include -I${CUTENSOR_ROOT}/include
LIBRARY_DIRS = -L${CUTENSORNET_ROOT}/lib64 -L${CUTENSORNET_ROOT}/lib -L${CUTENSOR_ROOT}/lib/11
LINKER_FLAGS = -lcutensornet -lcutensor -lcudart -lstdc++

CXX_FLAGS=-std=c++11 $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(LINKER_FLAGS) $(ARCH_FLAGS)

all:
	nvcc tensornet_example.cu -o  tensornet_example ${CXX_FLAGS}

clean:
	rm -f tensornet_example tensornet_example.o
