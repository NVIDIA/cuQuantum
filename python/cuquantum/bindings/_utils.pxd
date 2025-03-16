# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

cimport cpython
from libc.stdint cimport int32_t, int64_t, intptr_t, uint32_t
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libcpp cimport nullptr_t, nullptr
from libcpp.memory cimport unique_ptr

from .cycutensornet cimport cutensornetTensorQualifiers_t

cdef extern from "driver_types.h" nogil:
    ctypedef void* Stream 'cudaStream_t'


cdef extern from * nogil:
    """
    template<typename T>
    class nullable_unique_ptr {
      public:
        nullable_unique_ptr() noexcept = default;

        nullable_unique_ptr(std::nullptr_t) noexcept = delete;

        explicit nullable_unique_ptr(T* data, bool own_data):
            own_data_(own_data)
        {
            if (own_data)
                manager_.reset(data);
            else
                raw_data_ = data;
        }

        nullable_unique_ptr(const nullable_unique_ptr&) = delete;

        nullable_unique_ptr& operator=(const nullable_unique_ptr&) = delete;

        nullable_unique_ptr(nullable_unique_ptr&& other) noexcept
        {
            own_data_ = other.own_data_;
            other.own_data_ = false;  // ownership is transferred
            if (own_data_)
            {
                manager_ = std::move(other.manager_);
                raw_data_ = nullptr;  // just in case
            }
            else
            {
                manager_.reset(nullptr);  // just in case
                raw_data_ = other.raw_data_;
            }
        }

        nullable_unique_ptr& operator=(nullable_unique_ptr&& other) noexcept
        {
            own_data_ = other.own_data_;
            other.own_data_ = false;  // ownership is transferred
            if (own_data_)
            {
                manager_ = std::move(other.manager_);
                raw_data_ = nullptr;  // just in case
            }
            else
            {
                manager_.reset(nullptr);  // just in case
                raw_data_ = other.raw_data_;
            }
            return *this;
        }

        ~nullable_unique_ptr() = default;

        void reset(T* data, bool own_data)
        {
            own_data_ = own_data;
            if (own_data_)
            {
                manager_.reset(data);
                raw_data_ = nullptr;
            }
            else
            {
                manager_.reset(nullptr);
                raw_data_ = data;
            }
        }

        void swap(nullable_unique_ptr& other) noexcept
        {
            std::swap(manager_, other.manager_);
            std::swap(raw_data_, other.raw_data_);
            std::swap(own_data_, other.own_data_);
        }

        /*
         * Get the pointer to the underlying object (this is different from data()!).
         */
        T* get() const noexcept
        {
            if (own_data_)
                return manager_.get();
            else
                return raw_data_;
        }

        /*
         * Get the pointer to the underlying buffer (this is different from get()!).
         */
        void* data() noexcept
        {
            if (own_data_)
                return manager_.get()->data();
            else
                return raw_data_;
        }

        T& operator*()
        {
            if (own_data_)
                return *manager_;
            else
                return *raw_data_;
        }

      private:
        std::unique_ptr<T> manager_{};
        T* raw_data_{nullptr};
        bool own_data_{false};
    };
    """
    # xref: cython/Cython/Includes/libcpp/memory.pxd
    cdef cppclass nullable_unique_ptr[T]:
        nullable_unique_ptr()
        nullable_unique_ptr(T*, cppbool)
        nullable_unique_ptr(nullable_unique_ptr[T]&)

        # Modifiers
        void reset(T*, cppbool)
        void swap(nullable_unique_ptr&)

        # Observers
        T* get()
        T& operator*()
        void* data()


ctypedef fused ResT:
    int
    double
    intptr_t
    int32_t
    int64_t
    uint32_t
    size_t
    cutensornetTensorQualifiers_t

ctypedef fused PtrT:
    void
    int32_t
    int64_t
    (void*)

cdef cppclass nested_resource[T]:
    nullable_unique_ptr[ vector[intptr_t] ] ptrs
    nullable_unique_ptr[ vector[vector[T]] ] nested_resource_ptr


# accepts the output pointer as input to use the return value for exception propagation
cdef int get_resource_ptr(nullable_unique_ptr[vector[ResT]] &in_out_ptr, object obj, ResT* __unused) except 1
cdef int get_resource_ptrs(nullable_unique_ptr[ vector[PtrT*] ] &in_out_ptr, object obj, PtrT* __unused) except 1
cdef int get_nested_resource_ptr(nested_resource[ResT] &in_out_ptr, object obj, ResT* __unused) except 1


# Cython limitation: need standalone typedef if we wanna use it for casting
ctypedef int (*DeviceAllocType)(void*, void**, size_t, Stream)
ctypedef int (*DeviceFreeType)(void*, void*, size_t, Stream)


cdef bint is_nested_sequence(data)
cdef int cuqnt_alloc_wrapper(void* ctx, void** ptr, size_t size, Stream stream) with gil
cdef int cuqnt_free_wrapper(void* ctx, void* ptr, size_t size, Stream stream) with gil
cdef void logger_callback_with_data(
        int32_t log_level, const char* func_name, const char* message,
        void* func_arg) with gil
cdef void* get_buffer_pointer(buf, Py_ssize_t size, readonly=*) except*
