# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
JAX base primitive.
"""

from abc import ABCMeta, abstractmethod
from functools import partial

import jax
from jax.extend import core
from jax.interpreters import xla, mlir
from jax._src import dispatch

from cuquantum.lib import cudensitymat_jax


class BasePrimitive(metaclass=ABCMeta):
    """
    JAX primitive base class.
    """

    name = None

    @staticmethod
    @abstractmethod
    def abstract():
        """
        Abstract evaluation of the inner primitive.
        """
        return NotImplemented

    @classmethod
    def outer_abstract(cls, *args, **kwargs):
        """
        Abstract evaluation of the outer primitive.
        """
        return cls.abstract(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def lowering():
        """
        Lowering rule of the primitive.
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def impl():
        """
        Primal evaluation of the primitive.
        """
        return NotImplemented


def register_primitive(cls):
    """
    Register a JAX primitive.
    """

    def name_of_wrapper_p():
        return cls.name + "_wrapper"

    inner_p = core.Primitive(cls.name)
    dispatch.prim_requires_devices_during_lowering.add(inner_p)
    inner_p.multiple_results = cls.inner_multiple_results
    inner_p.def_impl(partial(xla.apply_primitive, inner_p))
    inner_p.def_abstract_eval(cls.abstract)
    mlir.register_lowering(inner_p, cls.lowering, platform="cuda")
    cls.inner_primitive = inner_p

    outer_p = core.Primitive(name_of_wrapper_p())
    dispatch.prim_requires_devices_during_lowering.add(outer_p)
    outer_p.multiple_results = cls.outer_multiple_results
    outer_p.def_impl(cls.impl)
    outer_p.def_abstract_eval(cls.outer_abstract)
    mlir.register_lowering(
        outer_p,
        mlir.lower_fun(cls.impl, multiple_results=cls.outer_multiple_results),
        platform="cuda"
    )
    cls.outer_primitive = outer_p


for _name, _value in cudensitymat_jax.registrations().items():
    jax.ffi.register_ffi_target(_name, _value, platform="CUDA")
