/* Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "pybind11/pybind11.h"

#include "cudensitymat_jax.h"


namespace py = pybind11;


template<typename T>
py::capsule EncapsulateFfiHandler(T* func)
{
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
        "Encapsulated function must be an XLA FFI handler");
    return py::capsule(reinterpret_cast<void*>(func));
}

py::dict Registrations()
{
    py::dict dict;
    dict["operator_action"] = EncapsulateFfiHandler(OperatorActionHandler);
    dict["operator_action_backward_diff"] = EncapsulateFfiHandler(OperatorActionBackwardDiffHandler);
    return dict;
}


PYBIND11_MODULE(cudensitymat_jax, m)
{
    m.def("registrations", &Registrations);
}
