import cupy as cp


# This is future proof: In the future when CuPy enables cuQuantum Python
# as an optional backend, we don't want to create a circular dependency
# that ultimately tests against ourselves. Here we enable CUB as the only
# optinaly backend and exclude cuTENSOR/cuQuantum Python/etc, using CuPy's
# private API (for development/testing).
cp._core.set_reduction_accelerators(['cub'])
cp._core.set_routine_accelerators(['cub'])
