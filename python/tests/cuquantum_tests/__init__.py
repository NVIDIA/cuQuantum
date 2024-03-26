# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import atexit
import glob
import os
import sys
import tempfile

try:
    import cffi
except ImportError:
    cffi = None
import cupy
import numpy
import pytest

from cuquantum import ComputeType, cudaDataType


if cffi:
    # if the Python binding is not installed in the editable mode (pip install
    # -e .), the cffi tests would fail as the modules cannot be imported
    sys.path.append(os.getcwd())


def clean_up_cffi_files():
    files = glob.glob(os.path.join(os.getcwd(), "cuquantum_test_cffi*"))
    for f in files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass


dtype_to_data_type = {
    numpy.float16: cudaDataType.CUDA_R_16F,
    numpy.float32: cudaDataType.CUDA_R_32F,
    numpy.float64: cudaDataType.CUDA_R_64F,
    numpy.complex64: cudaDataType.CUDA_C_32F,
    numpy.complex128: cudaDataType.CUDA_C_64F,
}


dtype_to_compute_type = {
    numpy.float16: ComputeType.COMPUTE_16F,
    numpy.float32: ComputeType.COMPUTE_32F,
    numpy.float64: ComputeType.COMPUTE_64F,
    numpy.complex64: ComputeType.COMPUTE_32F,
    numpy.complex128: ComputeType.COMPUTE_64F,
}


# we don't wanna recompile for every test case...
_cffi_mod1 = None
_cffi_mod2 = None

def _can_use_cffi():
    if cffi is None or os.environ.get('CUDA_PATH') is None:
        return False
    else:
        return True


class MemoryResourceFactory:

    def __init__(self, source):
        self.source = source

    def get_dev_mem_handler(self):
        if self.source == "py-callable":
            return (*self._get_cuda_callable(), self.source)
        elif self.source == "cffi":
            # ctx is not needed, so set to NULL
            return (0, *self._get_functor_address(), self.source)
        elif self.source == "cffi_struct":
            return self._get_handler_address()
        # TODO: add more different memory sources
        else:
            raise NotImplementedError

    def _get_cuda_callable(self):
        def alloc(size, stream):
            return cupy.cuda.runtime.mallocAsync(size, stream)

        def free(ptr, size, stream):
            cupy.cuda.runtime.freeAsync(ptr, stream)

        return alloc, free

    def _get_functor_address(self):
        if not _can_use_cffi():
            raise RuntimeError

        global _cffi_mod1
        if _cffi_mod1 is None:
            import importlib
            mod_name = f"cuquantum_test_{self.source}"
            ffi = cffi.FFI()
            ffi.set_source(mod_name, """
                #include <cuda_runtime.h>

                // cffi limitation: we can't use the actual type cudaStream_t because
                // it's considered an "incomplete" type and we can't get the functor
                // address by doing so...

                int my_alloc(void* ctx, void** ptr, size_t size, void* stream) {
                    return (int)cudaMallocAsync(ptr, size, stream);
                }

                int my_free(void* ctx, void* ptr, size_t size, void* stream) {
                    return (int)cudaFreeAsync(ptr, stream);
                }
                """,
                include_dirs=[os.environ['CUDA_PATH']+'/include'],
                library_dirs=[os.environ['CUDA_PATH']+'/lib64'],
                libraries=['cudart'],
            )
            ffi.cdef("""
                int my_alloc(void* ctx, void** ptr, size_t size, void* stream);
                int my_free(void* ctx, void* ptr, size_t size, void* stream);
            """)
            ffi.compile(verbose=True)
            self.ffi = ffi
            _cffi_mod1 = importlib.import_module(mod_name)
        self.ffi_mod = _cffi_mod1
        atexit.register(clean_up_cffi_files)

        alloc_addr = self._get_address("my_alloc")
        free_addr = self._get_address("my_free")
        return alloc_addr, free_addr

    def _get_handler_address(self):
        if not _can_use_cffi():
            raise RuntimeError

        global _cffi_mod2
        if _cffi_mod2 is None:
            import importlib
            mod_name = f"cuquantum_test_{self.source}"
            ffi = cffi.FFI()
            ffi.set_source(mod_name, """
                #include <cuda_runtime.h>

                // cffi limitation: we can't use the actual type cudaStream_t because
                // it's considered an "incomplete" type and we can't get the functor
                // address by doing so...

                int my_alloc(void* ctx, void** ptr, size_t size, void* stream) {
                    return (int)cudaMallocAsync(ptr, size, stream);
                }

                int my_free(void* ctx, void* ptr, size_t size, void* stream) {
                    return (int)cudaFreeAsync(ptr, stream);
                }

                typedef struct {
                    void* ctx;
                    int (*device_alloc)(void* ctx, void** ptr, size_t size, void* stream);
                    int (*device_free)(void* ctx, void* ptr, size_t size, void* stream);
                    char name[64];
                } myHandler;

                myHandler* init_myHandler(myHandler* h, const char* name) {
                    h->ctx = NULL;
                    h->device_alloc = my_alloc;
                    h->device_free = my_free;
                    memcpy(h->name, name, 64);
                    return h;
                }
                """,
                include_dirs=[os.environ['CUDA_PATH']+'/include'],
                library_dirs=[os.environ['CUDA_PATH']+'/lib64'],
                libraries=['cudart'],
            )
            ffi.cdef("""
                typedef struct {
                    ...;
                } myHandler;

                myHandler* init_myHandler(myHandler* h, const char* name);
            """)
            ffi.compile(verbose=True)
            self.ffi = ffi
            _cffi_mod2 = importlib.import_module(mod_name)
        self.ffi_mod = _cffi_mod2
        atexit.register(clean_up_cffi_files)

        h = self.handler = self.ffi_mod.ffi.new("myHandler*")
        self.ffi_mod.lib.init_myHandler(h, self.source.encode())
        return self._get_address(h)

    def _get_address(self, func_name_or_ptr):
        if isinstance(func_name_or_ptr, str):
            func_name = func_name_or_ptr
            data = str(self.ffi_mod.ffi.addressof(self.ffi_mod.lib, func_name))
        else:
            ptr = func_name_or_ptr  # ptr to struct
            data = str(self.ffi_mod.ffi.addressof(ptr[0]))
        # data has this format: "<cdata 'int(*)(void *, void * *, size_t, void *)' 0x7f6c5da37300>"
        return int(data.split()[-1][:-1], base=16)


class MemHandlerTestBase:

    mod = None
    prefix = None  # TODO: remove me
    error = None

    def _test_set_get_device_mem_handler(self, source, handle):
        if (isinstance(source, str) and source.startswith('cffi')
                and not _can_use_cffi()):
            pytest.skip("cannot run cffi tests")

        if source is not None:
            mr = MemoryResourceFactory(source)
            handler = mr.get_dev_mem_handler()
            self.mod.set_device_mem_handler(handle, handler)
            # round-trip test
            queried_handler = self.mod.get_device_mem_handler(handle)
            if source == 'cffi_struct':
                # I'm lazy, otherwise I'd also fetch the functor addresses here...
                assert queried_handler[0] == 0  # ctx is NULL
                assert queried_handler[-1] == source
            else:
                assert queried_handler == handler
        else:
            with pytest.raises(self.error) as e:
                queried_handler = self.mod.get_device_mem_handler(handle)
            assert 'NO_DEVICE_ALLOCATOR' in str(e.value), f"{str(e.value)=}"


class LoggerTestBase:

    mod = None
    prefix = None

    def test_logger_set_level(self):
        self.mod.logger_set_level(6)  # on
        self.mod.logger_set_level(0)  # off

    def test_logger_set_mask(self):
        self.mod.logger_set_mask(16)  # should not raise

    def test_logger_set_callback_data(self):
        # we also test logger_open_file() here to avoid polluting stdout

        def callback(level, name, message, my_data, is_ok=False):
            log = f"{level}, {name}, {message} (is_ok={is_ok}) -> logged\n"
            my_data.append(log)

        handle = None
        my_data = []
        is_ok = True

        with tempfile.TemporaryDirectory() as temp:
            file_name = os.path.join(temp, f"{self.prefix}_test")
            self.mod.logger_open_file(file_name)
            self.mod.logger_set_callback_data(callback, my_data, is_ok=is_ok)
            self.mod.logger_set_level(6)

            try:
                handle = self.mod.create()
                self.mod.destroy(handle)
            except:
                if handle:
                    self.mod.destroy(handle)
                raise
            finally:
                self.mod.logger_force_disable()  # to not affect the rest of tests

            with open(file_name) as f:
                log_from_f = f.read()

        # check the log file
        assert f'[{self.prefix}Create]' in log_from_f
        assert f'[{self.prefix}Destroy]' in log_from_f

        # check the captured data (note we log 2 APIs)
        log = ''.join(my_data)
        assert log.count("-> logged") >= 2
        assert log.count("is_ok=True") >= 2
