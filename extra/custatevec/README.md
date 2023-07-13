## MPI Comm Plugin Extension

The first version of [multi-node state vector simulator](https://docs.nvidia.com/cuda/cuquantum/latest/appliance/qiskit.html) has been released in cuQuantum Appliance 22.11.  It currently supports a limited set of versions of Open MPI and MPICH.  Other MPI libraries are supported by using an extension module (called External CommPlugin).
External CommPlugin is a small shared object that wraps MPI functions.  A customer needs to build its own external CommPlugin and link it to the MPI library of its preference to create a shared object.  Then, by specifying appropriate options to the simulator, the compiled shared object is dynamically loaded to use the MPI library for inter-process communications.

## Prerequisite

The MPI library should support GPUDirect RDMA for `MPI_Isend()` and `MPI_Irecv()`.

## Build

```
$ gcc -I <path to CUDA include> -I <path to MPI include> -L <Path to mpi library> -fPIC -shared mpiCommPlugin.c -lmpi -o <custom plugin name>.so
```
### Example

Assuming an MPI library is installed under `/opt/mpi`, and its include and lib directories are `include` and `lib`, respectively, the following command line compiles a custom Comm Plugin with a user-specified MPI library.

```
$ gcc -I /usr/local/cuda/include -I /opt/mpi/include -L/opt/mpi/lib -fPIC -shared  mpiCommPlugin.c -lmpi -o mpiCommmPlugin.so
$ ls -l
-rw-r--r-- 1 user group  9063 Dec 16 18:08 mpiCommPlugin.c
-rwxr-xr-x 1 user group 18088 Dec 16 18:38 mpiCommPlugin.so
```

## Simulator options

The custom Comm Plugin object is selected by [cusvaer options](https://docs.nvidia.com/cuda/cuquantum/latest/appliance/cusvaer.html#commplugin), `cusvaer_comm_plugin_type` and `cusvaer_comm_plugin_soname`.

- `cusvaer_comm_plugin_type`: The value is `cusvaer.CommPluginType.EXTERNAL`
- `cusvaer_comm_plugin_soname`  The name of the shared object of an external comm plugin

These two options should be set by using `set_options()` simulator method before executing simulation.  The following is a simplified example to use external CommPlugin.

Environmental variables such as `LD_LIBRARY_PATH` should be set for the simulator process to find out the shared object.

### Example

```
from qiskit import Aer
from cusvaer import CommPluginType

sim = Aer.get_backend('aer_simulator')
sim.set_options(cusvaer_comm_plugin_type=CommPluginType.EXTERNAL,
                cusvaer_comm_plugin_soname='mpiCommPlugin.so')

job = sim.run(circuit)
```
