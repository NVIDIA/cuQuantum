CXX_FLAGS=-std=c++11 -I${CUTENSORNET_ROOT}/include -L${CUTENSORNET_ROOT}/lib64 -L${CUTENSORNET_ROOT}/lib -lcutensornet -lcudart -lstdc++

all:
	nvcc tensornet_example.cu -o  tensornet_example ${CXX_FLAGS}

clean:
	rm -f tensornet_example tensornet_example.o
