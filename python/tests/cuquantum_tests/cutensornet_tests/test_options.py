# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import pytest

from cuquantum import cutensornet as cutn
from cuquantum import BaseCUDAMemoryManager
from cuquantum import ComputeType
from cuquantum import MemoryPointer
from cuquantum import NetworkOptions
from cuquantum import OptimizerInfo
from cuquantum import OptimizerOptions
from cuquantum import PathFinderOptions
from cuquantum import ReconfigOptions
from cuquantum import SlicerOptions


class _OptionsBase:

    __slots__ = ('options_type',)

    def create_options(self, options):
        return self.options_type(**options)


class TestNetworkOptions(_OptionsBase):

    options_type = NetworkOptions

    @pytest.mark.parametrize(
        'compute_type', [t for t in ComputeType]
    )
    def test_compute_type(self, compute_type):
        self.create_options({'compute_type': compute_type})

    def test_device_id(self):
        self.create_options({'device_id': 0})

    def test_handle(self):
        handle = 10000
        self.create_options({'handle': handle})

    def test_logger(self):
        self.create_options({"logger": logging.getLogger()})

    @pytest.mark.parametrize(
        'memory_limit', (int(1e8), "100 MiB", "80%")
    )
    def test_memory_limit(self, memory_limit):
        self.create_options({'memory_limit': memory_limit})

    # since BaseCUDAMemoryManager is a protocol, as long as the method
    # is there it doesn't matter if it's used as the base class or not
    @pytest.mark.parametrize(
        "base", (object, BaseCUDAMemoryManager)
    )
    def test_allocator(self, base):

        class MyAllocator(base):

            def memalloc(self, size):
                return MemoryPointer(0, size, None)

        allocator = MyAllocator()
        self.create_options({'allocator': allocator})


class TestOptimizerOptions(_OptionsBase):

    options_type = OptimizerOptions

    def test_samples(self):
        self.create_options({'samples': 100})

    def test_threads(self):
        self.create_options({'threads': 8})

    def test_path(self):
        self.create_options({'path': {"num_partitions": 100}})
        self.create_options({
            'path': PathFinderOptions(**{"num_partitions": 100}),
        })

    def test_slicing(self):
        self.create_options({'slicing': {"disable_slicing": 1}})
        self.create_options({
            'slicing': SlicerOptions(**{"disable_slicing": 1}),
        })

    def test_reconfiguration(self):
        self.create_options({'reconfiguration': {"num_leaves": 100}})
        self.create_options({
            'reconfiguration': ReconfigOptions(**{"num_leaves": 100}),
        })

    def test_seed(self):
        self.create_options({'seed': 100})


class TestOptimizerInfo(_OptionsBase):

    options_type = OptimizerInfo

    # All fields in OptimizerInfo are required, so we must test
    # them at once
    def test_optimizer_info(self):
        self.create_options({
            "largest_intermediate": 100.0,
            "opt_cost": 100.0,
            "path": [(0, 1), (0, 1)],
            "slices": [("a", 4), ("b", 3)],
            "num_slices": 10,
            "intermediate_modes": [(1, 3), (2, 4)],
        })


class TestPathFinderOptions(_OptionsBase):

    options_type = PathFinderOptions

    def test_num_partitions(self):
        self.create_options({"num_partitions": 100})

    def test_cutoff_size(self):
        self.create_options({"cutoff_size": 100})

    @pytest.mark.parametrize(
        "algorithm", [algo for algo in cutn.GraphAlgo]
    )
    def test_algorithm(self, algorithm):
        self.create_options({"algorithm": algorithm})

    def test_imbalance_factor(self):
        self.create_options({"imbalance_factor": 1000})

    def test_num_iterations(self):
        self.create_options({"num_iterations": 100})

    def test_num_cuts(self):
        self.create_options({"num_cuts": 100})


class TestReconfigOptions(_OptionsBase):

    options_type = ReconfigOptions

    def test_num_iterations(self):
        self.create_options({"num_iterations": 100})

    def test_num_leaves(self):
        self.create_options({"num_leaves": 100})


class TestSlicerOptions(_OptionsBase):

    options_type = SlicerOptions

    def test_disable_slicing(self):
        self.create_options({"disable_slicing": 1})

    @pytest.mark.parametrize(
        "memory_model", [m for m in cutn.MemoryModel]
    )
    def test_memory_model(self, memory_model):
        self.create_options({"memory_model": memory_model})

    def test_memory_factor(self):
        self.create_options({"memory_factor": 20})

    def test_min_slices(self):
        self.create_options({"min_slices": 10})

    def test_slice_factor(self):
        self.create_options({"slice_factor": 5})
