# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import pytest

from .cudensitymat_tests.test_cudensitymat_samples import sample_files as cudm_samples
from .custatevec_tests.test_custatevec_samples import sample_files as cusv_samples
from .cutensornet_tests.test_cutensornet_samples import sample_files as cutn_samples
from .cutensornet_tests.test_cutensornet_samples import notebook_files as cutn_notebooks

testing_python_samples = set(cudm_samples + cusv_samples + cutn_samples)
testing_notebook_samples = set(cutn_notebooks)

samples_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'samples'))
all_sample_files = glob.glob(samples_path+'/**/*.py', recursive=True)
all_notebook_files = glob.glob(samples_path+'/**/*.ipynb', recursive=True)

@pytest.mark.parametrize(
    'samples', (cudm_samples, cusv_samples, cutn_samples, cutn_notebooks)
)
def test_non_empty_testing_samples(samples):
    """make sure the specified sample path is not empty"""
    assert samples


@pytest.mark.parametrize(
    'sample', all_sample_files + all_notebook_files
)
def test_samples_included(sample):
    if sample.endswith('.py'):
        if sample not in testing_python_samples:
            # This must be either a pythonic mpi sample under 'samples/tensornet/' 
            # or an mpi sample using bindings under 'samples/bindings/cutensornet/' 
            # or an mpi sample using bindings under 'samples/bindings/custatevec/
            assert '_mpi' in sample
            assert 'bindings/custatevec' in sample or 'bindings/cutensornet' in sample or '/tensornet' in sample
    elif sample.endswith('.ipynb'):
        assert sample in testing_notebook_samples
    else:
        raise AssertionError(f"{sample} not recognized")



