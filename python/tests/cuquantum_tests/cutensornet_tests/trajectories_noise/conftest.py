# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Fixture configuration for tests in trajectories_noise directory

network_state_wrap.py defines `TrajectorySim` class which provides a basic quantum simulator API.
It is also used in cusvsim ubackend tests, so the same API can be used for future tests that overlap between ubackend and cuTN

- trajectory_sim fixture allows to use different TrajectorySim classes.
- dtype fixture parametrizes over different data types

If you want to specialize parameters for some particular test, redefine the fixtures in the corresponding test file.

The test_* files are identical to the corresponding test files in cusvsim ubackend
"""

import pytest
from .network_state_wrap import (
    TrajectoryNaive,
    TrajectoryApplyChannel,
)

# pytestmark doesn't work in conftest.py
# pytestmark = pytest.mark.parametrize("state_algo", ["mps", "tn"])


@pytest.fixture(params=["mps", "tn"])
def state_algo(request):
    return request.param


@pytest.fixture(params=[TrajectoryNaive, TrajectoryApplyChannel])
def trajectory_sim(request, n_qubits, state_algo):
    return request.param(n_qubits, algo=state_algo)
