# Distributed index bit swap - Samples

# Build

## Linux

Plase use make to compile the cuStateVec samples. Options for CUSTATEVEC_ROOT can be skipped if cuStateVec is in the CUDA installation folder.

With make

```
export CUSTATEVEC_ROOT=<path_to_custatevec_root>
make
```

## Run

```
mpirun -n 2 ./distributedIndexBitSwap
```

## Files

- distributedIndexBitSwap.cpp

  Example for the utilization of distributed index bit swap API

- mpicomm.c

  Example of the external communicator

