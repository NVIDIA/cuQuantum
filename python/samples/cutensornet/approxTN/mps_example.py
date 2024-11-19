# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools

import cupy as cp
import numpy as np

import cuquantum
from cuquantum import cutensornet as cutn

class MPSHelper:
    
    """
    MPSHelper(num_sites, phys_extent, max_virtual_extent, initial_state, data_type, compute_type)

    Create an MPSHelper object for gate splitting algorithm.
           i              j       
        -------A-------B-------                      i        j        k
              p|       |q            ------->     -------A`-------B`-------
               GGGGGGGGG                                r|        |s
              r|       |s
    
    Args:
        num_sites: The number of sites in the MPS.
        phys_extents: The extent for the physical mode where the gate tensors are acted on.
        max_virtual_extent: The maximal extent allowed for the virtual mode shared between adjacent MPS tensors.
        initial_state: A sequence of :class:`cupy.ndarray` representing the initial state of the MPS.
        data_type (cuquantum.cudaDataType): The data type for all tensors and gates. 
        compute_type (cuquantum.ComputeType): The compute type for all gate splitting.
    
    """

    def __init__(self, num_sites, phys_extent, max_virtual_extent, initial_state, data_type, compute_type):
        self.num_sites = num_sites
        self.phys_extent = phys_extent
        self.data_type = data_type
        self.compute_type = compute_type
        
        self.phys_modes = []
        self.virtual_modes = []
        self.new_mode = itertools.count(start=0, step=1)
        
        for i in range(num_sites+1):
            self.virtual_modes.append(next(self.new_mode))
            if i != num_sites:
                self.phys_modes.append(next(self.new_mode))
        
        untruncated_max_extent = phys_extent ** (num_sites // 2)
        if max_virtual_extent == 0:
            self.max_virtual_extent = untruncated_max_extent
        else:
            self.max_virtual_extent = min(max_virtual_extent, untruncated_max_extent)
        
        self.handle = cutn.create()
        self.work_desc = cutn.create_workspace_descriptor(self.handle)
        self.svd_config = cutn.create_tensor_svd_config(self.handle)
        self.svd_info = cutn.create_tensor_svd_info(self.handle)
        self.gate_algo = cutn.GateSplitAlgo.DIRECT

        self.desc_tensors = []
        self.state_tensors = []

        # create tensor descriptors
        for i in range(self.num_sites):
            self.state_tensors.append(cp.asarray(initial_state[i], order="F"))
            extent = self.get_tensor_extent(i)
            modes = self.get_tensor_modes(i)
            desc_tensor = cutn.create_tensor_descriptor(self.handle, 3, extent, 0, modes, self.data_type)
            self.desc_tensors.append(desc_tensor)
    
    def get_tensor(self, site):
        """Get the tensor operands for a specific site."""
        return self.state_tensors[site]
    
    def get_tensor_extent(self, site):
        """Get the extent of the MPS tensor at a specific site."""
        return self.state_tensors[site].shape
    
    def get_tensor_modes(self, site):
        """Get the current modes of the MPS tensor at a specific site."""
        return (self.virtual_modes[site], self.phys_modes[site], self.virtual_modes[site+1])

    def set_svd_config(self, abs_cutoff, rel_cutoff, renorm, partition):
        """Update the SVD truncation setting.
        
        Args:
            abs_cutoff: The cutoff value for absolute singular value truncation.
            rel_cutoff: The cutoff value for relative singular value truncation.
            renorm (cuquantum.cutensornet.TensorSVDNormalization): The option for renormalization of the truncated singular values.
            partition (cuquantum.cutensornet.TensorSVDPartition): The option for partitioning of the singular values.
        """        
        
        if partition != cutn.TensorSVDPartition.UV_EQUAL:
            raise NotImplementedError("this basic example expects partition to be cutensornet.TensorSVDPartition.UV_EQUAL")

        svd_config_attributes = [cutn.TensorSVDConfigAttribute.ABS_CUTOFF, 
                                 cutn.TensorSVDConfigAttribute.REL_CUTOFF, 
                                 cutn.TensorSVDConfigAttribute.S_NORMALIZATION,
                                 cutn.TensorSVDConfigAttribute.S_PARTITION]
            
        for (attr, value) in zip(svd_config_attributes, [abs_cutoff, rel_cutoff, renorm, partition]):
            dtype = cutn.tensor_svd_config_get_attribute_dtype(attr)
            value = np.array([value], dtype=dtype)
            cutn.tensor_svd_config_set_attribute(self.handle, 
                self.svd_config, attr, value.ctypes.data, value.dtype.itemsize)
    
    def set_gate_algorithm(self, gate_algo):
        """Set the algorithm to use for all gate split operations.
        
        Args:
            gate_algo (cuquantum.cutensornet.GateSplitAlgo): The gate splitting algorithm to use.
        """

        self.gate_algo = gate_algo

    def compute_max_workspace_sizes(self):
        """Compute the maximal workspace needed for MPS gating algorithm."""
        modes_in_A = [ord(c) for c in ('i', 'p', 'j')]
        modes_in_B = [ord(c) for c in ('j', 'q', 'k')]
        modes_in_G = [ord(c) for c in ('p', 'q', 'r', 's')]
        modes_out_A = [ord(c) for c in ('i', 'r', 'j')]
        modes_out_B = [ord(c) for c in ('j', 's', 'k')]
        
        max_extents_AB = (self.max_virtual_extent, self.phys_extent, self.max_virtual_extent)
        extents_in_G = (self.phys_extent, self.phys_extent, self.phys_extent, self.phys_extent)
        
        desc_tensor_in_A = cutn.create_tensor_descriptor(self.handle, 3, max_extents_AB, 0, modes_in_A, self.data_type)
        desc_tensor_in_B = cutn.create_tensor_descriptor(self.handle, 3, max_extents_AB, 0, modes_in_B, self.data_type)
        desc_tensor_in_G = cutn.create_tensor_descriptor(self.handle, 4, extents_in_G, 0, modes_in_G, self.data_type)
        desc_tensor_out_A = cutn.create_tensor_descriptor(self.handle, 3, max_extents_AB, 0, modes_out_A, self.data_type)
        desc_tensor_out_B = cutn.create_tensor_descriptor(self.handle, 3, max_extents_AB, 0, modes_out_B, self.data_type)
        
        cutn.workspace_compute_gate_split_sizes(self.handle, 
            desc_tensor_in_A, desc_tensor_in_B, desc_tensor_in_G, 
            desc_tensor_out_A, desc_tensor_out_B, 
            self.gate_algo, self.svd_config, self.compute_type, self.work_desc)
        
        workspace_size = cutn.workspace_get_memory_size(self.handle, self.work_desc, cutn.WorksizePref.MIN, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)

        # free resources
        cutn.destroy_tensor_descriptor(desc_tensor_in_A)
        cutn.destroy_tensor_descriptor(desc_tensor_in_B)
        cutn.destroy_tensor_descriptor(desc_tensor_in_G)
        cutn.destroy_tensor_descriptor(desc_tensor_out_A)
        cutn.destroy_tensor_descriptor(desc_tensor_out_B)
        return workspace_size

    def set_workspace(self, work, workspace_size):
        """Compute the maximal workspace needed for MPS gating algorithm.
        
        Args:
            work: Pointer to the allocated workspace.
            workspace_size: The required workspace size on the device.
        """
        cutn.workspace_set_memory(self.handle, self.work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, work.ptr, workspace_size)

    def apply_gate(self, site_A, site_B, gate, verbose, stream):
        """Inplace execution of the apply gate algoritm on site A and site B.
        
        Args:
            site_A: The first site on which the gate is applied to.
            site_B: The second site on which the gate is applied to.
            gate (cupy.ndarray): The input data for the gate tensor.
            verbose: Whether to print out the runtime information during truncation.
            stream (cupy.cuda.Stream): The CUDA stream on which the computation is performed.
        """
        if site_B - site_A != 1:
            raise ValueError("Site B must be the right site of site A")
        if site_B >= self.num_sites:
            raise ValueError("Site index cannot exceed maximum number of sites")
        
        desc_tensor_in_A = self.desc_tensors[site_A]
        desc_tensor_in_B = self.desc_tensors[site_B]
        
        phys_mode_in_A = self.phys_modes[site_A]
        phys_mode_in_B = self.phys_modes[site_B]
        phys_mode_out_A = next(self.new_mode)
        phys_mode_out_B = next(self.new_mode)
        modes_G = (phys_mode_in_A, phys_mode_in_B, phys_mode_out_A, phys_mode_out_B)
        extent_G = (self.phys_extent, self.phys_extent, self.phys_extent, self.phys_extent)
        desc_tensor_in_G = cutn.create_tensor_descriptor(self.handle, 4, extent_G, 0, modes_G, self.data_type)

        # construct and initialize the expected output A and B
        tensor_in_A = self.state_tensors[site_A]
        tensor_in_B = self.state_tensors[site_B]
        left_extent_A = tensor_in_A.shape[0]
        extent_AB_in = tensor_in_A.shape[2]
        right_extent_B = tensor_in_B.shape[2]
        combined_extent_left = min(left_extent_A, extent_AB_in * self.phys_extent) * self.phys_extent
        combined_extent_right = min(right_extent_B, extent_AB_in * self.phys_extent) * self.phys_extent
        extent_Aout_B = min(combined_extent_left, combined_extent_right, self.max_virtual_extent)
        
        extent_out_A = (left_extent_A, self.phys_extent, extent_Aout_B)
        extent_out_B = (extent_Aout_B, self.phys_extent, right_extent_B)

        tensor_out_A = cp.zeros(extent_out_A, dtype=tensor_in_A.dtype, order="F")
        tensor_out_B = cp.zeros(extent_out_B, dtype=tensor_in_B.dtype, order="F")

        # create tensor descriptors for output A and B
        modes_out_A = (self.virtual_modes[site_A], phys_mode_out_A, self.virtual_modes[site_A+1])
        modes_out_B = (self.virtual_modes[site_B], phys_mode_out_B, self.virtual_modes[site_B+1])

        desc_tensor_out_A = cutn.create_tensor_descriptor(self.handle, 3, extent_out_A, 0, modes_out_A, self.data_type)
        desc_tensor_out_B = cutn.create_tensor_descriptor(self.handle, 3, extent_out_B, 0, modes_out_B, self.data_type)

        cutn.gate_split(self.handle, 
                        desc_tensor_in_A, tensor_in_A.data.ptr, 
                        desc_tensor_in_B, tensor_in_B.data.ptr, 
                        desc_tensor_in_G, gate.data.ptr, 
                        desc_tensor_out_A, tensor_out_A.data.ptr, 
                        0, # we factorize singular values equally onto output A and B.
                        desc_tensor_out_B, tensor_out_B.data.ptr, 
                        self.gate_algo, self.svd_config, self.compute_type, 
                        self.svd_info, self.work_desc, stream.ptr)
        
        if verbose:
            full_extent = np.array([0], dtype=cutn.tensor_svd_info_get_attribute_dtype(cutn.TensorSVDInfoAttribute.FULL_EXTENT))
            reduced_extent = np.array([0], dtype=cutn.tensor_svd_info_get_attribute_dtype(cutn.TensorSVDInfoAttribute.REDUCED_EXTENT))
            discarded_weight = np.array([0], dtype=cutn.tensor_svd_info_get_attribute_dtype(cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT))
            
            cutn.tensor_svd_info_get_attribute(
                self.handle, self.svd_info, cutn.TensorSVDInfoAttribute.FULL_EXTENT, 
                full_extent.ctypes.data, full_extent.dtype.itemsize)
            cutn.tensor_svd_info_get_attribute(
                self.handle, self.svd_info, cutn.TensorSVDInfoAttribute.REDUCED_EXTENT, 
                reduced_extent.ctypes.data, reduced_extent.dtype.itemsize)
            cutn.tensor_svd_info_get_attribute(
                self.handle, self.svd_info, cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT, 
                discarded_weight.ctypes.data, discarded_weight.dtype.itemsize)
            
            print("Virtual bond truncated from {0} to {1} with a discarded weight of {2:.6f}".format(full_extent[0], reduced_extent[0], discarded_weight[0]))

        self.phys_modes[site_A] = phys_mode_out_A
        self.phys_modes[site_B] = phys_mode_out_B
        self.desc_tensors[site_A] = desc_tensor_out_A
        self.desc_tensors[site_B] = desc_tensor_out_B
        
        extent_out_A = np.zeros((3,), dtype=np.int64)
        extent_out_B = np.zeros((3,), dtype=np.int64)
        extent_out_A, strides_out_A = cutn.get_tensor_details(self.handle, desc_tensor_out_A)[2:]
        extent_out_B, strides_out_B = cutn.get_tensor_details(self.handle, desc_tensor_out_B)[2:]

        # Recall that `cutensornet.gate_split` can potentially find reduced extent during SVD truncation when value-based truncation is used.
        # Therefore we here update the container for output tensor A and B.
        if extent_out_A[2] != extent_Aout_B:
            # note strides in cutensornet are in the unit of count and strides in cupy/numpy are in the unit of nbytes
            strides_out_A = [i * tensor_out_A.itemsize for i in strides_out_A]
            strides_out_B = [i * tensor_out_B.itemsize for i in strides_out_B]
            tensor_out_A = cp.ndarray(extent_out_A, dtype=tensor_out_A.dtype, memptr=tensor_out_A.data, strides=strides_out_A)
            tensor_out_B = cp.ndarray(extent_out_B, dtype=tensor_out_B.dtype, memptr=tensor_out_B.data, strides=strides_out_B)

        self.state_tensors[site_A] = tensor_out_A
        self.state_tensors[site_B] = tensor_out_B

        cutn.destroy_tensor_descriptor(desc_tensor_in_A)
        cutn.destroy_tensor_descriptor(desc_tensor_in_B)
        cutn.destroy_tensor_descriptor(desc_tensor_in_G)
    
    def __del__(self):
        """
        Calls `MPSHelper.free()`.

        An explicit call to `MPSHelper.free()` by the user of this class allows
        to free resources at a predictable moment in time. In some cases,
        relying on the garbage collection can cause resource over-utilization
        or other problems.

        It is advised to always call `MPSHelper.free()` when you no longer need
        the object.
        """
        self.free()

    def free(self):
        """Free all resources owned by the object, if not already freed."""
        if self.handle is None:
            return
        self.handle = cutn.destroy(self.handle) # free() should be idempotent
        for desc_tensor in self.desc_tensors:
            cutn.destroy_tensor_descriptor(desc_tensor)
        cutn.destroy_workspace_descriptor(self.work_desc)
        cutn.destroy_tensor_svd_config(self.svd_config)
        cutn.destroy_tensor_svd_info(self.svd_info)


def main():
    print("cuTensorNet-vers:", cutn.get_version())
    dev = cp.cuda.Device()  # get current device
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print("===== device info ======")
    print("GPU-local-id:", dev.id)
    print("GPU-name:", props["name"].decode())
    print("GPU-clock:", props["clockRate"])
    print("GPU-memoryClock:", props["memoryClockRate"])
    print("GPU-nSM:", props["multiProcessorCount"])
    print("GPU-major:", props["major"])
    print("GPU-minor:", props["minor"])
    print("========================")

    data_type = cuquantum.cudaDataType.CUDA_C_64F
    compute_type = cuquantum.ComputeType.COMPUTE_64F

    num_sites = 16
    phys_extent = 2
    max_virtual_extent = 12
    
    ## we initialize the MPS state as a product state |000...000>
    initial_state = []
    for i in range(num_sites):
        # we create dummpy indices for MPS tensors on the boundary for easier bookkeeping
        # we'll use Fortran layout throughout this example
        # all tensors have to have the same dtype
        tensor = cp.zeros((1,2,1), dtype=np.complex128, order="F")
        tensor[0,0,0] = 1.0
        initial_state.append(tensor)
    
    ##################################
    # Initialize an MPSHelper object
    ##################################

    mps_helper = MPSHelper(num_sites, phys_extent, max_virtual_extent, initial_state, data_type, compute_type)
    
    ##################################
    # Setup options for gate operation
    ##################################
    
    abs_cutoff = 1e-2
    rel_cutoff = 1e-2
    renorm = cutn.TensorSVDNormalization.L2
    partition = cutn.TensorSVDPartition.UV_EQUAL
    mps_helper.set_svd_config(abs_cutoff, rel_cutoff, renorm, partition)

    gate_algo = cutn.GateSplitAlgo.REDUCED
    mps_helper.set_gate_algorithm(gate_algo)

    #####################################
    # Workspace estimation and allocation
    #####################################

    free_mem, total_mem = dev.mem_info
    worksize = free_mem *.7
    required_workspace_size = mps_helper.compute_max_workspace_sizes()
    work = cp.cuda.alloc(worksize)
    print(f"Maximal workspace size requried: {required_workspace_size / 1024 ** 3:.3f} GB")
    mps_helper.set_workspace(work, required_workspace_size)

    ###########
    # Execution
    ###########
    
    stream = cp.cuda.Stream()
    cp.random.seed(0)
    num_layers = 10
    for i in range(num_layers):
        start_site = i % 2
        print(f"Cycle {i}:")
        verbose = (i == num_layers-1)
        for j in range(start_site, num_sites-1, 2):
            # initialize a random 2-qubit gate
            gate = cp.random.random([phys_extent,]*4) + 1.j * cp.random.random([phys_extent,]*4)
            gate = gate.astype(gate.dtype, order="F")
            mps_helper.apply_gate(j, j+1, gate, verbose, stream)
        
    stream.synchronize()
    print("========================")
    print("After gate application")
    for i in range(num_sites):
        tensor = mps_helper.get_tensor(i)
        modes = mps_helper.get_tensor_modes(i)
        print(f"Site {i}, extent: {tensor.shape}, modes: {modes}")

    mps_helper.free()

if __name__ == '__main__':
    main()


