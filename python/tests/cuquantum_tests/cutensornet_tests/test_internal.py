# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import sys
import threading

import cupy as cp
from cupy.cuda.runtime import getDevice, setDevice
import pytest

from cuquantum.cutensornet import _internal
from cuquantum.cutensornet._internal import utils
from cuquantum._utils import WHITESPACE_UNICODE


class TestDeviceCtx:

    @pytest.mark.skipif(
        cp.cuda.runtime.getDeviceCount() < 2, reason='not enough GPUs')
    def test_device_ctx(self):
        assert getDevice() == 0
        with utils.device_ctx(0):
            assert getDevice() == 0
            with utils.device_ctx(1):
                assert getDevice() == 1
                with utils.device_ctx(0):
                    assert getDevice() == 0
                assert getDevice() == 1
            assert getDevice() == 0
        assert getDevice() == 0

        with utils.device_ctx(1):
            assert getDevice() == 1
            setDevice(0)
            with utils.device_ctx(1):
                assert getDevice() == 1
            assert getDevice() == 0
        assert getDevice() == 0

    @pytest.mark.skipif(
        cp.cuda.runtime.getDeviceCount() < 2, reason='not enough GPUs')
    def test_thread_safe(self):
        # adopted from https://github.com/cupy/cupy/blob/master/tests/cupy_tests/cuda_tests/test_device.py
        # recall that the CUDA context is maintained per-thread, so when each thread
        # starts it is on the default device (=device 0).
        t0_setup = threading.Event()
        t1_setup = threading.Event()
        t0_first_exit = threading.Event()

        t0_exit_device = []
        t1_exit_device = []

        def t0_seq():
            with utils.device_ctx(0):
                with utils.device_ctx(1):
                    t0_setup.set()
                    t1_setup.wait()
                    t0_exit_device.append(getDevice())
                t0_exit_device.append(getDevice())
                t0_first_exit.set()
            assert getDevice() == 0

        def t1_seq():
            t0_setup.wait()
            with utils.device_ctx(1):
                with utils.device_ctx(0):
                    t1_setup.set()
                    t0_first_exit.wait()
                    t1_exit_device.append(getDevice())
                t1_exit_device.append(getDevice())
            assert getDevice() == 0

        try:
            cp.cuda.runtime.setDevice(1)
            t0 = threading.Thread(target=t0_seq)
            t1 = threading.Thread(target=t1_seq)
            t1.start()
            t0.start()
            t0.join()
            t1.join()
            assert t0_exit_device == [1, 0]
            assert t1_exit_device == [0, 1]
        finally:
            cp.cuda.runtime.setDevice(0)

    def test_one_shot(self):
        dev = utils.device_ctx(0)
        with dev:
            pass
        # CPython raises AttributeError, but we should not care here
        with pytest.raises(Exception):
            with dev:
                pass


class TestGetSymbol:

    def test_no_whitespace(self):
        # Note: max(whitespace_s) = 12288
        out = []
        for i in range(0, 30000):
            s = _internal.circuit_converter_utils._get_symbol(i)
            assert not s.isspace()
            out.append(s)

        # check the mapping is unique
        assert len(set(out)) == 30000

    def test_whitespace_unicode_consistency(self):
        all_s = ''.join(chr(c) for c in range(sys.maxunicode+1))
        whitespace_s = ''.join(re.findall(r'\s', all_s))
        assert WHITESPACE_UNICODE == whitespace_s
