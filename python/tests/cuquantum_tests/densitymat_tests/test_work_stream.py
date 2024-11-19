# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


import weakref

import cupy as cp
import pytest

from cuquantum.densitymat import WorkStream


class TestWorkStream:

    def test_default(self):
        ctx = WorkStream()
        assert ctx.device_id == 0
        assert ctx.blocking == True
        assert ctx._valid_state
        assert ctx._handle._valid_state

    def test_set_memory_limit(self):
        ctx = WorkStream(memory_limit="75%")
        assert ctx.memory_limit == "75%"

    def test_set_stream(self):
        ctx = WorkStream(stream=cp.cuda.Stream(4))
        assert ctx.stream == cp.cuda.Stream(4)

    @pytest.mark.skipif(cp.cuda.runtime.getDeviceCount() < 2, reason="not enough GPUs")
    def test_multiple_devices(self):
        with cp.cuda.Device(1):
            ctx_default = WorkStream()
            ctx_explicit = WorkStream(device_id=cp.cuda.Device().id)
        assert ctx_default.device_id == 0
        assert ctx_explicit.device_id == 1

    def test_handle_reference(self):
        ctx = WorkStream()
        ref = weakref.ref(ctx._handle)
        ctx = None
        assert ref() is None
