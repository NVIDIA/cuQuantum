# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Sequence, Literal, Optional
import math
from dataclasses import dataclass
import numpy as np

from nvmath.internal import tensor_wrapper
from nvmath.internal.tensor_wrapper import wrap_operand, TensorHolder, maybe_register_package
from nvmath.internal.tensor_ifc_numpy import NumpyTensor
import nvmath.memory as nvmath_memory
import nvmath.internal.utils as nvmath_utils
from nvmath.internal.mem_limit import check_memory_str

import cuquantum.bindings.cupauliprop as cupp
from cuquantum.bindings.cupauliprop import SortOrder
from ._internal import typemaps
from ._internal.utils import (
    is_all_zeros, is_contiguous, create_truncation_strategies, convert_truncation_strategies, 
    register_finalizer, SortOrderLiteral, sort_order_to_cupp, sort_order_from_cupp
)
from ._internal.work import Workspace
from .handles import LibraryHandle
from .operators import QuantumOperator
from .truncation import Truncation

__all__ = [
    "PauliExpansion", "PauliExpansionView", "PauliExpansionOptions",
    "RehearsalInfo", "GateApplicationRehearsalInfo",
    "TraceBackwardRehearsalInfo", "ProductTraceBackwardRehearsalInfo",
    "SortOrder",
]


@dataclass
class PauliExpansionOptions:
    """
    Options controlling Pauli expansion construction and execution.
    """
    allocator: nvmath_memory.BaseCUDAMemoryManagerAsync | None = None
    memory_limit: int | str = "80%"
    blocking: bool = False

    def __post_init__(self):
        check_memory_str(self.memory_limit, "memory limit")
        if self.allocator is not None and not isinstance(self.allocator, nvmath_memory.BaseCUDAMemoryManagerAsync):
            raise TypeError("allocator must fulfill the BaseCUDAMemoryManagerAsync protocol.")
        if not isinstance(self.blocking, bool):
            raise TypeError("blocking must be a boolean value.")

@dataclass
class RehearsalInfo:
    """
    Information about the required resources for a rehearsed operation.
    """
    device_scratch_workspace_required: int
    host_scratch_workspace_required: int

    def __or__(self, other: "RehearsalInfo") -> "RehearsalInfo":
        """Combine two RehearsalInfo by taking the max of each field.

        When the two operands are the same concrete subclass, the result
        preserves that subclass (merging all its fields).  When one
        operand is the exact base class, the subclass is preserved
        (the base has no extra fields to lose).  When both are
        *different* subclasses, the result decays to the base
        :class:`RehearsalInfo`, retaining only the workspace fields
        that are common to every subclass.
        """
        if not isinstance(other, RehearsalInfo):
            raise TypeError("other must be a RehearsalInfo or subclass thereof")
        # If self is the exact base and other is a subclass, let the
        # subclass handle it so it can preserve its extra fields.
        if type(self) is RehearsalInfo and type(other) is not RehearsalInfo:
            return other | self
        return RehearsalInfo(
            device_scratch_workspace_required=max(self.device_scratch_workspace_required, other.device_scratch_workspace_required),
            host_scratch_workspace_required=max(self.host_scratch_workspace_required, other.host_scratch_workspace_required),
        )

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes in human-readable form."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(num_bytes) < 1024:
                return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.1f} PB"

    def __str__(self) -> str:
        return (
            f"RehearsalInfo(\n"
            f"  device_scratch_workspace_required: {self._format_bytes(self.device_scratch_workspace_required)}\n"
            f"  host_scratch_workspace_required:   {self._format_bytes(self.host_scratch_workspace_required)}\n"
            f")"
        )

@dataclass
class TraceBackwardRehearsalInfo(RehearsalInfo):
    """
    Rehearsal information for backward differentiation of a trace operation.

    Includes the required number of terms for the cotangent expansion output(s).
    """
    cotangent_num_terms: int

    def __str__(self) -> str:
        return (
            f"TraceBackwardRehearsalInfo(\n"
            f"  cotangent_num_terms:               {self.cotangent_num_terms:,}\n"
            f"  device_scratch_workspace_required: {self._format_bytes(self.device_scratch_workspace_required)}\n"
            f"  host_scratch_workspace_required:   {self._format_bytes(self.host_scratch_workspace_required)}\n"
            f")"
        )

    def __or__(self, other: "RehearsalInfo") -> "TraceBackwardRehearsalInfo | RehearsalInfo":
        if not isinstance(other, RehearsalInfo):
            raise TypeError("other must be a RehearsalInfo or subclass thereof")
        if isinstance(other, TraceBackwardRehearsalInfo):
            return TraceBackwardRehearsalInfo(
                cotangent_num_terms=max(self.cotangent_num_terms, other.cotangent_num_terms),
                device_scratch_workspace_required=max(self.device_scratch_workspace_required, other.device_scratch_workspace_required),
                host_scratch_workspace_required=max(self.host_scratch_workspace_required, other.host_scratch_workspace_required),
            )
        if type(other) is RehearsalInfo:
            return TraceBackwardRehearsalInfo(
                cotangent_num_terms=self.cotangent_num_terms,
                device_scratch_workspace_required=max(self.device_scratch_workspace_required, other.device_scratch_workspace_required),
                host_scratch_workspace_required=max(self.host_scratch_workspace_required, other.host_scratch_workspace_required),
            )
        return super().__or__(other)

@dataclass
class ProductTraceBackwardRehearsalInfo(RehearsalInfo):
    """
    Rehearsal information for backward differentiation of a product trace operation.

    Includes the required number of terms for both cotangent expansion outputs.
    """
    cotangent_num_terms1: int
    cotangent_num_terms2: int

    def __str__(self) -> str:
        return (
            f"ProductTraceBackwardRehearsalInfo(\n"
            f"  cotangent_num_terms1:              {self.cotangent_num_terms1:,}\n"
            f"  cotangent_num_terms2:              {self.cotangent_num_terms2:,}\n"
            f"  device_scratch_workspace_required: {self._format_bytes(self.device_scratch_workspace_required)}\n"
            f"  host_scratch_workspace_required:   {self._format_bytes(self.host_scratch_workspace_required)}\n"
            f")"
        )

    def __or__(self, other: "RehearsalInfo") -> "ProductTraceBackwardRehearsalInfo | RehearsalInfo":
        if not isinstance(other, RehearsalInfo):
            raise TypeError("other must be a RehearsalInfo or subclass thereof")
        if isinstance(other, ProductTraceBackwardRehearsalInfo):
            return ProductTraceBackwardRehearsalInfo(
                cotangent_num_terms1=max(self.cotangent_num_terms1, other.cotangent_num_terms1),
                cotangent_num_terms2=max(self.cotangent_num_terms2, other.cotangent_num_terms2),
                device_scratch_workspace_required=max(self.device_scratch_workspace_required, other.device_scratch_workspace_required),
                host_scratch_workspace_required=max(self.host_scratch_workspace_required, other.host_scratch_workspace_required),
            )
        if type(other) is RehearsalInfo:
            return ProductTraceBackwardRehearsalInfo(
                cotangent_num_terms1=self.cotangent_num_terms1,
                cotangent_num_terms2=self.cotangent_num_terms2,
                device_scratch_workspace_required=max(self.device_scratch_workspace_required, other.device_scratch_workspace_required),
                host_scratch_workspace_required=max(self.host_scratch_workspace_required, other.host_scratch_workspace_required),
            )
        return super().__or__(other)

@dataclass
class GateApplicationRehearsalInfo(RehearsalInfo):
    """
    Information about the required resources for a gate application rehearsal.
    """
    num_terms_required: int
    
    def __str__(self) -> str:
        return (
            f"GateApplicationRehearsalInfo(\n"
            f"  num_terms_required:                {self.num_terms_required:,}\n"
            f"  device_scratch_workspace_required: {self._format_bytes(self.device_scratch_workspace_required)}\n"
            f"  host_scratch_workspace_required:   {self._format_bytes(self.host_scratch_workspace_required)}\n"
            f")"
        )
    
    def __or__(self, other: "RehearsalInfo") -> "GateApplicationRehearsalInfo | RehearsalInfo":
        """Combine two RehearsalInfo instances.

        Same-type merges preserve :class:`GateApplicationRehearsalInfo`.
        Merging with the exact base :class:`RehearsalInfo` also preserves
        the subclass (the base has no extra fields to lose).  Mixed
        subclass merges decay to the base :class:`RehearsalInfo`.
        """
        if not isinstance(other, RehearsalInfo):
            raise TypeError("other must be a RehearsalInfo or subclass thereof")
        if isinstance(other, GateApplicationRehearsalInfo):
            return GateApplicationRehearsalInfo(
                num_terms_required=max(self.num_terms_required, other.num_terms_required),
                device_scratch_workspace_required=max(self.device_scratch_workspace_required, other.device_scratch_workspace_required),
                host_scratch_workspace_required=max(self.host_scratch_workspace_required, other.host_scratch_workspace_required),
            )
        if type(other) is RehearsalInfo:
            return GateApplicationRehearsalInfo(
                num_terms_required=self.num_terms_required,
                device_scratch_workspace_required=max(self.device_scratch_workspace_required, other.device_scratch_workspace_required),
                host_scratch_workspace_required=max(self.host_scratch_workspace_required, other.host_scratch_workspace_required),
            )
        return super().__or__(other)


class PauliExpansion:
    """
    A Pauli operator expansion.
    
    This is a container for a set of Pauli strings and their coefficients, to which quantum operators can be applied.

    Args:
        library_handle: The library handle.
        num_qubits: The number of qubits.
        num_terms: The number of terms.
        xz_bits: The XZ bits buffer.
        coeffs: The coefficients buffer.
        sort_order: The sort order of the expansion (``None``, ``"internal"``, or ``"little_endian_bitwise"``).
        has_duplicates: Whether the expansion may contain duplicate Pauli strings.
        options: Either a :class:`PauliExpansionOptions` object or a ``dict`` with
            matching keywords (``allocator``, ``memory_limit``, ``blocking``).
    """
    def __init__(self, library_handle: LibraryHandle, num_qubits: int, num_terms: int, xz_bits, coeffs, sort_order: SortOrder | SortOrderLiteral = None, has_duplicates: bool = True, *, options: PauliExpansionOptions | dict | None = None) -> None:
        self._ptr : int | None = None
        self._xz_bits = xz_bits if isinstance(xz_bits, TensorHolder) else wrap_operand(xz_bits)
        self._coefs = coeffs if isinstance(coeffs, TensorHolder) else wrap_operand(coeffs)
        
        # Verify buffers are contiguous (any layout)
        if not is_contiguous(self._xz_bits.shape, self._xz_bits.strides):
            raise ValueError("xz_bits buffer must be contiguous")
        if not is_contiguous(self._coefs.shape, self._coefs.strides):
            raise ValueError("coefs buffer must be contiguous")
        
        if self._xz_bits.name != self._coefs.name:
            raise TypeError(f"xz_bits_buffer and coef_buffer must have the same package, got {self._xz_bits.name} and {self._coefs.name}")
        
        # Infer array package from tensor type
        # Map "nvmath" to "cuda" (nvmath wraps cuda.ndarray internally)
        detected_package = self._xz_bits.name
        self._package: str = "cuda" if detected_package == "nvmath" else detected_package
        self._library_handle = library_handle
        options = nvmath_utils.check_or_create_options(PauliExpansionOptions, options, "pauli expansion options")
        self._ptr : int | None= cupp.create_pauli_expansion(
            int(self._library_handle),
            num_qubits,
            self._xz_bits.data_ptr,
            self._xz_bits.size * self._xz_bits.itemsize,
            self._coefs.data_ptr,
            self._coefs.size * self._coefs.itemsize,
            typemaps.NAME_TO_DATA_TYPE[self._coefs.dtype],
            num_terms,
            sort_order_to_cupp(sort_order),
            int(has_duplicates))
        self._logger.debug(f"C API cupaulipropCreatePauliExpansion returned ptr={self._ptr}")
        self._update_stamp: int = 0
        self._is_rehearsal: bool = False
        # Create allocator based on array package (use "cuda" as fallback for CPU tensors)
        allocator_package = self._package if self._package != "numpy" else "cuda"
        if options.allocator is not None:
            allocator = options.allocator
        else:
            maybe_register_package(allocator_package)
            allocator = nvmath_memory._MEMORY_MANAGER[allocator_package](self._library_handle.device_id, self._library_handle._logger)
        self._workspace = Workspace(self._library_handle, allocator, options.memory_limit)
        self._blocking = options.blocking
        # Register cleanup finalizer for safe resource release
        self._finalizer = register_finalizer(self, cupp.destroy_pauli_expansion, self._ptr, self._logger, "PauliExpansion")
        self._logger.info(f"PauliExpansion created: {num_qubits} qubits, {num_terms} terms, dtype={self._coefs.dtype}, sort_order={sort_order}")

    @classmethod
    def empty(
        cls,
        library_handle: LibraryHandle,
        num_qubits: int,
        num_terms: int,
        dtype: str = "complex128",
        sort_order: SortOrder | SortOrderLiteral = None,
        has_duplicates: bool = True,
        *,
        options: "PauliExpansionOptions | dict | None" = None
    ) -> "PauliExpansion":
        """
        Create a rehearsal-only expansion for determining resource requirements.
        
        This expansion cannot be used for actual computation. All compute methods will
        raise an error if ``rehearse=False``. To execute operations, create a normal
        (non-rehearsal) expansion using :meth:`from_empty` on the rehearsal object.
        
        Args:
            library_handle: The library handle.
            num_qubits: The number of qubits.
            num_terms: The number of terms (used for rehearsal calculations).
            dtype: The data type for coefficients (defaults to "complex128").
            sort_order: The sort order of the expansion (defaults to ``None``).
            has_duplicates: Whether the expansion may contain duplicate Pauli strings (defaults to True).
            options: Either a :class:`PauliExpansionOptions` object or a ``dict`` with
                matching keywords.
        
        Returns:
            A rehearsal-only PauliExpansion instance.
        """
        # Create minimal 1-element dummy buffers (smallest valid allocation on CPU)
        xz_ints_per_term = cupp.get_num_packed_integers(num_qubits) * 2
        dummy_xz = np.zeros((1, xz_ints_per_term), dtype=np.uint64)
        dummy_coefs = np.zeros(1, dtype=dtype)
        
        # We need to bypass the normal constructor to pass fake buffer sizes to the C API
        # Create the expansion object without calling __init__
        expansion = object.__new__(cls)
        expansion._ptr = None
        expansion._xz_bits = wrap_operand(dummy_xz)
        expansion._coefs = wrap_operand(dummy_coefs)
        expansion._package = "numpy"
        expansion._library_handle = library_handle
        
        # Compute fake buffer sizes that would be required for num_terms
        fake_xz_size = num_terms * xz_ints_per_term * 8  # 8 bytes per uint64
        fake_coef_size = num_terms * np.dtype(dtype).itemsize
        
        # Store original options before defaulting (for from_empty propagation)
        expansion._original_options = options
        options = nvmath_utils.check_or_create_options(PauliExpansionOptions, options, "pauli expansion options")
        expansion._ptr = cupp.create_pauli_expansion(
            int(library_handle),
            num_qubits,
            expansion._xz_bits.data_ptr,
            fake_xz_size,
            expansion._coefs.data_ptr,
            fake_coef_size,
            typemaps.NAME_TO_DATA_TYPE[expansion._coefs.dtype],
            num_terms,
            sort_order_to_cupp(sort_order),
            int(has_duplicates)
        )
        library_handle.logger.debug(f"C API cupaulipropCreatePauliExpansion (rehearsal) returned ptr={expansion._ptr}")
        expansion._update_stamp = 0
        expansion._is_rehearsal = True
        
        if options.allocator is not None:
            allocator = options.allocator
        else:
            allocator = nvmath_memory._MEMORY_MANAGER["cuda"](library_handle.device_id, library_handle._logger)
        expansion._workspace = Workspace(library_handle, allocator, options.memory_limit)
        expansion._blocking = options.blocking
        expansion._finalizer = register_finalizer(expansion, cupp.destroy_pauli_expansion, expansion._ptr, expansion._logger, "PauliExpansion")
        expansion._logger.info(f"PauliExpansion (rehearsal) created: {num_qubits} qubits, {num_terms} terms")
        
        return expansion

    def from_empty(
        self,
        xz_bits,
        coefs,
        num_terms: int = 0,
        sort_order: SortOrder | SortOrderLiteral = None,
        has_duplicates: bool | None = None,
        *,
        options: "PauliExpansionOptions | dict | None" = None,
    ) -> "PauliExpansion":
        """
        Create a PauliExpansion backed by user-provided (preallocated) buffers.

        This is an out-of-place conversion from a rehearsal-only expansion to a normal
        (non-rehearsal) expansion backed by real buffers.

        Args:
            xz_bits: The XZ bits buffer.
            coefs: The coefficient buffer.
            num_terms: The initial number of valid terms (defaults to 0).
            sort_order: The sort order of the expansion (defaults to the rehearsal expansion's sort_order).
            has_duplicates: Whether the expansion may contain duplicate Pauli strings (defaults to the rehearsal expansion's value).
            options: Either a :class:`PauliExpansionOptions` object or a ``dict`` with
                matching keywords. If not provided, uses the original options passed
                to :meth:`empty` (which may be ``None`` or partial). The constructor
                then derives any unset defaults (e.g., allocator) from the provided buffers.

        Returns:
            A non-rehearsal PauliExpansion instance.
        """
        if not self._is_rehearsal:
            raise RuntimeError("from_empty() can only be called on rehearsal expansions")
        if sort_order is None:
            sort_order = self.sort_order
        if has_duplicates is None:
            has_duplicates = self.has_duplicates
        # If user didn't provide options to from_empty, use the original options from empty().
        # This preserves user-specified fields while letting the constructor derive defaults
        # (like allocator) from the actual buffers provided.
        if options is None:
            options = self._original_options
        self._logger.info(f"Creating expansion from rehearsal: {self.num_qubits} qubits, {num_terms} terms")
        return PauliExpansion(
            self._library_handle,
            self.num_qubits,
            num_terms,
            xz_bits,
            coefs,
            sort_order=sort_order,
            has_duplicates=has_duplicates,
            options=options,
        )

    @property
    def is_rehearsal(self) -> bool:
        """
        Whether this is a rehearsal-only expansion (created via :meth:`empty`).
        
        A rehearsal expansion can only be used with ``rehearse=True``. To execute
        operations, create a normal (non-rehearsal) expansion using :meth:`from_empty`.
        """
        return self._is_rehearsal

    @property
    def _logger(self):
        """Internal logger accessor, delegates to the library handle's logger."""
        return self._library_handle.logger

    def _get_num_terms(self):
        return cupp.pauli_expansion_get_num_terms(int(self._library_handle), int(self))
    
    def _get_num_qubits(self):
        return cupp.pauli_expansion_get_num_qubits(int(self._library_handle), int(self))
    
    def _get_data_type(self):
        return cupp.pauli_expansion_get_data_type(int(self._library_handle), int(self))
    
    def _get_sort_order(self):
        return cupp.pauli_expansion_get_sort_order(int(self._library_handle), int(self))
    
    def _get_is_deduplicated(self):
        return cupp.pauli_expansion_is_deduplicated(int(self._library_handle), int(self))

    @property
    def xz_bits(self):
        """
        The XZ bits buffer.
        """
        if self.package == "cuda":
            raise RuntimeError("Cannot access XZ bits buffer on GPU for \"cuda\" package. Transfer to CPU first using .to(device, package=...)")
        return self._xz_bits.tensor

    def valid_xz_bits(self):
        """
        A view on the XZ bits buffer that only contains the valid terms.
        """
        raise NotImplementedError

    @property
    def capacity(self) -> int:
        """
        The maximum number of terms that the Pauli expansion can accommodate.
        """
        # XZ bits may be stored as 1D or 2D; use the wrapper's unified size and ints-per-term.
        xz_elems = self._xz_bits.size
        #xz_elems = size_attr() if callable(size_attr) else size_attr
        ints_per_term = 2 * cupp.get_num_packed_integers(self.num_qubits)
        cap_xz = xz_elems // ints_per_term

        cap_coef = self._coefs.size  # number of coefficient elements
        return min(cap_xz, cap_coef)
    
    @property
    def coeffs(self):
        """
        The coefficient buffer.
        """
        if self.package == "cuda":
            raise RuntimeError("Cannot access coefficient buffer on GPU for \"cuda\" package. Transfer to CPU first using .to(device, package=...)")
        return self._coefs.tensor

    @property
    def valid_coeffs(self):
        """
        A view on the coefficients buffer that only contains the valid terms.
        """
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        """
        The number of qubits for this Pauli expansion.
        """
        #may consider storing this value since it's const and doesn't need to be queried
        return self._get_num_qubits()

    @property
    def num_terms(self) -> int:
        """
        The current number of valid terms in the Pauli expansion.
        """
        return self._get_num_terms()

    @property
    def dtype(self) -> str:
        """
        Returns the data type of the coefficients.
        """
        return typemaps.DATA_TYPE_TO_NAME[self._get_data_type()]
    
    @property
    def is_sorted(self) -> bool:
        """
        Returns whether the Pauli expansion is sorted (any sort order).
        """
        return self._get_sort_order() != cupp.SortOrder.NONE
    
    @property
    def sort_order(self) -> SortOrderLiteral:
        """
        Returns the sort order of the Pauli expansion.
        
        Returns:
            ``None``, ``"internal"``, or ``"little_endian_bitwise"``.
        """
        return sort_order_from_cupp(self._get_sort_order())
    
    @property
    def has_duplicates(self) -> bool:
        """
        Returns whether the Pauli expansion may contain duplicate Pauli strings.
        
        After calling :meth:`deduplicate`, this will return ``False``.
        """
        # _get_is_deduplicated returns True if there are NO duplicates, so we negate
        return not self._get_is_deduplicated()

    @property
    def package(self) -> str:
        """
        The backend package for this Pauli expansion, inferred from the expansion's tensor type, e.g. "numpy", "cupy", "torch", or "cuda".
        """
        return self._package

    def __int__(self) -> int:
        """
        Returns the pointer to the Pauli expansion.
        """
        if self._ptr is None:
            return 0
        else:
            return self._ptr

    def to(self, device: Literal["cpu"] | Literal["gpu"], package: Literal["cupy", "torch", "cuda"], stream=None, *, options: PauliExpansionOptions | dict | None = None) -> "PauliExpansion":
        """
        Copy the Pauli expansion to the specified device.

        Args:
            device: Target device. Use ``"cpu"`` for host memory, or the library handle's
                current GPU (``"gpu"``).
            package: Array package to use for the target tensors.
                Must be one of ``"cupy"``, ``"torch"``, ``"numpy"`` or ``"cuda"``.
                Note that "``cuda``" and ``"cupy"`` are only valid for transfers to GPU, while ``"numpy"`` is only valid for transfers to CPU.
            stream: Optional stream (package stream object or pointer) to use for the transfer
                when copying from host to device. Ignored for host-to-host copies.

        Returns:
            A PauliExpansion on the requested location. If already on that location,
            the current object is returned; otherwise a new PauliExpansion is created
            on host or on the handle's device. Transfers between different GPU devices
            are not supported.
        """
        
        # Normalize device selector
        self._logger.info(f"Starting transfer: device={device}, package={package}")
        options = nvmath_utils.check_or_create_options(PauliExpansionOptions, options, "pauli expansion options")
        if device not in {"cpu", "gpu"}:
            raise TypeError('device must be "cpu" or "gpu"')
        if package not in {"cupy", "torch", "cuda", "numpy"}:
            raise ValueError(f"package must be 'cupy', 'torch', 'cuda', or 'numpy', got '{package}'")
        if device == "gpu":
            device = self._library_handle.device_id
        transfer_from_device: bool = self.storage_location == "gpu"
        transfer_to_device: bool = device != "cpu"
        if transfer_to_device and package not in {"cupy", "torch", "cuda"}:
            raise ValueError(f"package must be 'cupy', 'torch', or 'cuda', got '{package}'")
        if not transfer_to_device and package not in {"numpy", "torch"}:
            raise ValueError(f"package must be 'numpy' or 'torch', got '{package}'")
        
        # Use the provided package for GPU operations
        array_package = package
        gpu_array_package = package if package in {"cupy", "torch", "cuda"} else "cuda"
        
        maybe_register_package(gpu_array_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, gpu_array_package)
        
        with nvmath_utils.device_ctx(self._library_handle.device_id):
            if transfer_to_device:
                if device != self._library_handle.device_id:
                    raise ValueError("Transfers between different GPU devices are not supported")
                if transfer_from_device:
                    out = self
                else:
                    tensor_wrapper_type = tensor_wrapper._TENSOR_TYPES[array_package]
                    if array_package != "torch":
                        xz_bits_wrapper = tensor_wrapper_type.create_from_host(
                            self._xz_bits, self._library_handle.device_id, stream_holder
                        )
                        coefs_wrapper = tensor_wrapper_type.create_from_host(
                            self._coefs, self._library_handle.device_id, stream_holder
                        )
                    else:
                        if isinstance(self._xz_bits, NumpyTensor):
                            xz_bits_wrapper = wrap_operand(tensor_wrapper_type.module.from_numpy(self._xz_bits.tensor))
                            coefs_wrapper = wrap_operand(tensor_wrapper_type.module.from_numpy(self._coefs.tensor))
                        else:
                            xz_bits_wrapper = self._xz_bits
                            coefs_wrapper = self._coefs
                        xz_bits_wrapper = xz_bits_wrapper.to(device, stream_holder)
                        coefs_wrapper = coefs_wrapper.to(device, stream_holder)
                    out = PauliExpansion(
                        self._library_handle,
                        self.num_qubits,
                        self.num_terms,
                        xz_bits_wrapper,
                        coefs_wrapper,
                        sort_order=self.sort_order,
                        has_duplicates=self.has_duplicates,
                        options=options,
                    )
            else:
                if not transfer_from_device:
                    out = self
                else:
                    if self._package != "torch":
                        xz_bits_wrapper = NumpyTensor.create_host_from(self._xz_bits.to("cpu", stream_holder), stream_holder)
                        coefs_wrapper = NumpyTensor.create_host_from(self._coefs.to("cpu", stream_holder), stream_holder)
                    else:
                        xz_bits_wrapper = self._xz_bits.to("cpu", stream_holder)
                        coefs_wrapper = self._coefs.to("cpu", stream_holder)
                    out = PauliExpansion(
                        self._library_handle,
                        self.num_qubits,
                        self.num_terms,
                        xz_bits_wrapper,
                        coefs_wrapper,
                        sort_order=self.sort_order,
                        has_duplicates=self.has_duplicates,
                        options=options,
                    )
        self._logger.info(f"Transfer completed: {self.num_terms} terms to {device}")
        return out

    def populate_from(self, other: "PauliExpansion | PauliExpansionView", stream=None) -> None:
        """
        Write coefficients and xzbits from another Pauli expansion to this Pauli expansion or Pauli expansion view.
        This will reset the number of valid terms to the number of terms in the other Pauli expansion or Pauli expansion view.
        
        Raises:
            RuntimeError: If this expansion is a rehearsal expansion.
        """
        if self._is_rehearsal:
            raise RuntimeError(
                "Cannot populate a rehearsal expansion. "
                "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...)."
            )
        if other._library_handle != self._library_handle:
            raise ValueError(f"Other Pauli expansion must be on the same device as the current Pauli expansion, got {other.library_handle} and {self._library_handle}")
        if isinstance(other, PauliExpansionView):
            if other.base == self:
                raise ValueError(f"Cannot populate Pauli expansion from a view on itself")
            other_num_terms = other.end_index - other.start_index
        else:
            if other == self:
                raise ValueError(f"Cannot populate Pauli expansion from itself")
            other_num_terms = other.num_terms
            other = other.view()
        
        self._logger.info(f"Starting populate_from: {other_num_terms} terms")
        
        # Use GPU package for stream (fallback to "cuda" for CPU tensors)
        stream_package = self._package if self._package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)
        
        timing = bool(self._logger.handlers)
        with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (_, elapsed):
            cupp.pauli_expansion_populate_from_view(int(self._library_handle), int(other), int(self), stream_holder.ptr)
        self._update_stamp += 1
        
        if elapsed.data is not None:
            self._logger.info(f"Populate completed in {elapsed.data:.3f} ms")
        else:
            self._logger.info("Populate completed")

    def view(self, start_index: int = 0, end_index: int = None, *, options: PauliExpansionOptions | dict | None = None) -> "PauliExpansionView":
        """
        Creates a non-owning view on a contiguous range of Pauli terms inside a Pauli operator expansion.

        Args:
            start_index: The start index of the view (inclusive). If not specified, the view will start at the beginning of the Pauli expansion.
            end_index: The end index of the view (exclusive, one past the last element). If not specified, the view will end at the end of the currently valid terms in the Pauli expansion.

        Returns:
            A view on a contiguous range of Pauli terms inside a Pauli operator expansion. The view covers the range [start_index, end_index).
        """
        if end_index is None:
            end_index = self.num_terms
        elif end_index > self.num_terms:
            raise ValueError(f"End index must be less than or equal to the number of terms, got {end_index} and {self.num_terms}")
        return PauliExpansionView(
            self,
            start_index,
            end_index,
            options=options if options is not None else PauliExpansionOptions(
                allocator=self._workspace._allocator,
                memory_limit=self._workspace.memory_limit,
                blocking=self._blocking,
            ),
        )

    @property
    def storage_location(self) -> Literal["cpu", "gpu"]:
        """
        The storage location of the Pauli expansion.
        """
        ret = cupp.pauli_expansion_get_storage_buffer(int(self._library_handle), int(self))
        storage_location = ret[-1]
        if storage_location == cupp.Memspace.HOST:
            return "cpu"
        elif storage_location == cupp.Memspace.DEVICE:
            return "gpu"
        else:
            raise ValueError(f"Invalid storage location: {storage_location}")

    # -------------------------------------------------------------------------
    # Convenience wrappers for PauliExpansionView compute methods
    # -------------------------------------------------------------------------

    def apply_gate(self,
                   gate: QuantumOperator,
                   truncation: Truncation | None = None,
                   expansion_out: Optional["PauliExpansion"] = None,
                   adjoint: bool = False,
                   sort_order: SortOrder | SortOrderLiteral = None,
                   keep_duplicates: bool = False,
                   rehearse: bool | None = None,
                   stream=None) -> "GateApplicationRehearsalInfo | PauliExpansion":
        """
        Applies a quantum operator to the Pauli expansion (out-of-place).

        This operation does not modify the input expansion; the result is written
        to a new or provided output expansion.

        This is a convenience method that creates a default view covering all terms
        and delegates to :meth:`PauliExpansionView.apply_gate`.

        Args:
            gate: The gate to apply.
            truncation: Truncation strategy to apply (optional).
            expansion_out: The Pauli expansion to write the result to. If not provided,
                a new expansion will be allocated with the required capacity.
            adjoint: Whether to apply the adjoint of the gate (defaults to False).
            sort_order: Sort order for the output expansion. One of ``None`` or ``"internal"``
                (defaults to ``None``).
            keep_duplicates: Whether to keep duplicate Pauli strings in the output (defaults to False).
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.

        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: The output PauliExpansion.
        """
        if expansion_out is self:
            raise ValueError("expansion_out must not alias the input expansion (got expansion_out is self)")
        return self.view().apply_gate(
            gate=gate,
            truncation=truncation,
            expansion_out=expansion_out,
            adjoint=adjoint,
            sort_order=sort_order,
            keep_duplicates=keep_duplicates,
            rehearse=rehearse,
            stream=stream,
        )
        
    @property
    def _zero_state(self) -> Sequence[int]:
        """
        The zero state ``|0...0>``.
        """
        return [0] * self.num_qubits
    
    def trace_with_zero_state(self, rehearse: bool | None = None, stream=None) -> "RehearsalInfo | tuple[float | complex, float]":
        """
        Computes the trace of the Pauli expansion with the zero state ``|0...0>``.

        This computes :math:`\\langle 0 | \\rho | 0 \\rangle` where :math:`\\rho` is the operator
        represented by this Pauli expansion and :math:`|0\\rangle = |0...0\\rangle` is the
        all-zeros computational basis state.

        This is a convenience method that creates a default view covering all terms
        and delegates to :meth:`PauliExpansionView.trace_with_zero_state`.

        Args:
            rehearse: If True, only rehearse the operation to determine resource requirements.
                If None (default), automatically set to True for rehearsal expansions,
                False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.

        Returns:
            If rehearse=True: A :class:`RehearsalInfo` with the required workspace sizes.
            If rehearse=False: A tuple ``(trace_significand, trace_exponent)`` where
            ``trace = trace_significand * pow(2, trace_exponent)``.

        Raises:
            RuntimeError: If the base expansion is a rehearsal expansion and rehearse=False.
        """
        return self.view().trace_with_zero_state(rehearse=rehearse, stream=stream)

    def _trace_with_basis_state(self, comp_basis_state: Sequence[int], rehearse: bool | None = None, stream=None) -> "RehearsalInfo | tuple[float | complex, float]":
        """
        Computes the trace of the Pauli expansion with a given quantum state
        expressed in the computational basis.

        This is a convenience method that creates a default view covering all terms
        and delegates to :meth:`PauliExpansionView._trace_with_basis_state`.

        Currently only the zero state ``|0...0>`` is supported.

        Args:
            comp_basis_state: Quantum state in computational basis (bit-string).
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.

        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: A tuple ``(trace_significand, trace_exponent)`` where
            ``trace = trace_significand * pow(2, trace_exponent)``.
        """
        return self.view()._trace_with_basis_state(
            comp_basis_state=comp_basis_state,
            rehearse=rehearse,
            stream=stream,
        )

    def product_trace(self, other: "PauliExpansion | PauliExpansionView", adjoint: bool = False, rehearse: bool | None = None, stream=None) -> "RehearsalInfo | tuple[float | complex, float]":
        """
        Computes the product trace Tr(self^† * other) or Tr(self * other) with another Pauli expansion.

        This is a convenience method that creates default views covering all terms
        and delegates to :meth:`PauliExpansionView.product_trace`.

        Args:
            other: The other Pauli expansion (or view) to compute the product trace with.
            adjoint: If True, computes Tr(self^† * other). If False, computes Tr(self * other). Defaults to False.
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.

        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: A tuple ``(trace_significand, trace_exponent)`` where
            ``trace = trace_significand * pow(2, trace_exponent)``.
        """
        other_view = other if isinstance(other, PauliExpansionView) else other.view()
        return self.view().product_trace(
            other=other_view,
            adjoint=adjoint,
            rehearse=rehearse,
            stream=stream,
        )

    def sort(self, expansion_out: Optional["PauliExpansion"] = None, sort_order: SortOrder | SortOrderLiteral = "internal", rehearse: bool | None = None, stream=None) -> "RehearsalInfo | PauliExpansion":
        """
        Sorts the Pauli expansion by the specified sorting order (out-of-place).

        This operation does not modify the input expansion; the result is written
        to a new or provided output expansion.

        This is a convenience method that creates a default view covering all terms
        and delegates to :meth:`PauliExpansionView.sort`.

        Args:
            expansion_out: The Pauli expansion to write the sorted result to. If not provided,
                a new expansion will be allocated with the required capacity.
            sort_order: Sort order to apply. One of ``"internal"`` or ``"little_endian_bitwise"``
                (defaults to ``"internal"``). Note: ``None`` is not valid for sort operations.
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.

        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: The output PauliExpansion.
        """
        if expansion_out is self:
            raise ValueError("expansion_out must not alias the input expansion (got expansion_out is self)")
        return self.view().sort(
            expansion_out=expansion_out,
            sort_order=sort_order,
            rehearse=rehearse,
            stream=stream,
        )

    def deduplicate(self, expansion_out: Optional["PauliExpansion"] = None, sort_order: SortOrder | SortOrderLiteral = None, rehearse: bool | None = None, stream=None) -> "RehearsalInfo | PauliExpansion":
        """
        Deduplicates the Pauli expansion (out-of-place).

        Removes duplicate Pauli strings and sums their coefficients. This operation
        does not modify the input expansion; the result is written to a new or
        provided output expansion.

        This is a convenience method that creates a default view covering all terms
        and delegates to :meth:`PauliExpansionView.deduplicate`.

        Note: The expansion must be sorted before calling this method.

        Args:
            expansion_out: The Pauli expansion to write the deduplicated result to. If not provided,
                a new expansion will be allocated with the required capacity.
            sort_order: Sort order for the output expansion. One of ``None`` or ``"internal"``
                (defaults to ``None``).
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.

        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: The output PauliExpansion.
        """
        if expansion_out is self:
            raise ValueError("expansion_out must not alias the input expansion (got expansion_out is self)")
        return self.view().deduplicate(
            expansion_out=expansion_out,
            sort_order=sort_order,
            rehearse=rehearse,
            stream=stream,
        )

    def truncate(self, expansion_out: Optional["PauliExpansion"] = None, truncation: Truncation | None = None, rehearse: bool | None = None, stream=None) -> "RehearsalInfo | PauliExpansion":
        """
        Truncates the Pauli expansion based on the given truncation strategy (out-of-place).

        This operation does not modify the input expansion; the result is written
        to a new or provided output expansion.

        This is a convenience method that creates a default view covering all terms
        and delegates to :meth:`PauliExpansionView.truncate`.

        Args:
            expansion_out: The Pauli expansion to write the truncated result to. If not provided,
                a new expansion will be allocated with the required capacity.
            truncation: Truncation strategy to apply (optional).
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.

        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: The output PauliExpansion.
        """
        if expansion_out is self:
            raise ValueError("expansion_out must not alias the input expansion (got expansion_out is self)")
        return self.view().truncate(
            expansion_out=expansion_out,
            truncation=truncation,
            rehearse=rehearse,
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Backward differentiation convenience methods
    # ------------------------------------------------------------------

    def trace_with_zero_state_backward_diff(
        self,
        cotangent_trace,
        cotangent_trace_exponent,
        /,
        cotangent_expansion: "PauliExpansion | None" = None,
        rehearse: bool | None = None,
        stream=None,
    ) -> "TraceBackwardRehearsalInfo | PauliExpansion":
        """Backward pass for :meth:`trace_with_zero_state`.

        Convenience method that creates a default view and delegates to
        :meth:`PauliExpansionView.trace_with_zero_state_backward_diff`.

        Args:
            cotangent_trace: Scalar cotangent :math:`\\tilde{t} = dL/dt`.
            cotangent_trace_exponent: Scalar cotangent for the trace-exponent
                output.
            cotangent_expansion: Pre-allocated expansion to receive coefficient
                cotangents, or ``None`` to auto-allocate.
            rehearse: If True, only rehearse.
            stream: Stream object, pointer, or None.

        Returns:
            If ``rehearse=True``: a :class:`TraceBackwardRehearsalInfo`.
            If ``rehearse=False``: the cotangent :class:`PauliExpansion`.
        """
        return self.view().trace_with_zero_state_backward_diff(
            cotangent_trace,
            cotangent_trace_exponent,
            cotangent_expansion=cotangent_expansion,
            rehearse=rehearse,
            stream=stream,
        )

    def product_trace_backward_diff(
        self,
        other: "PauliExpansion | PauliExpansionView",
        cotangent_trace,
        cotangent_trace_exponent,
        /,
        cotangent_expansion1: "PauliExpansion | None" = None,
        cotangent_expansion2: "PauliExpansion | None" = None,
        adjoint: bool = False,
        rehearse: bool | None = None,
        stream=None,
    ) -> "ProductTraceBackwardRehearsalInfo | tuple[PauliExpansion, PauliExpansion]":
        """Backward pass for :meth:`product_trace`.

        Convenience method that creates default views and delegates to
        :meth:`PauliExpansionView.product_trace_backward_diff`.

        Args:
            other: The other Pauli expansion (or view).
            cotangent_trace: Scalar cotangent.
            cotangent_trace_exponent: Scalar cotangent for the trace-exponent
                output.
            cotangent_expansion1: Receives cotangents for ``self``, or ``None``
                to auto-allocate.
            cotangent_expansion2: Receives cotangents for ``other``, or ``None``
                to auto-allocate.
            adjoint: Must match the forward call.
            rehearse: If True, only rehearse.
            stream: Stream object, pointer, or None.

        Returns:
            If ``rehearse=True``: a :class:`ProductTraceBackwardRehearsalInfo`.
            If ``rehearse=False``: a tuple ``(cotangent_expansion1, cotangent_expansion2)``.
        """
        other_view = other if isinstance(other, PauliExpansionView) else other.view()
        return self.view().product_trace_backward_diff(
            other_view,
            cotangent_trace,
            cotangent_trace_exponent,
            cotangent_expansion1=cotangent_expansion1,
            cotangent_expansion2=cotangent_expansion2,
            adjoint=adjoint,
            rehearse=rehearse,
            stream=stream,
        )

    def apply_gate_backward_diff(
        self,
        gate: QuantumOperator,
        cotangent_out: "PauliExpansionView",
        truncation: Truncation | None = None,
        cotangent_in: Optional["PauliExpansion"] = None,
        adjoint: bool = False,
        sort_order: "SortOrder | SortOrderLiteral" = None,
        keep_duplicates: bool = False,
        rehearse: bool | None = None,
        stream=None,
    ) -> "GateApplicationRehearsalInfo | PauliExpansion":
        """Backward pass for :meth:`apply_gate`.

        Convenience method that creates a default view and delegates to
        :meth:`PauliExpansionView.apply_gate_backward_diff`.

        Args:
            gate: The quantum operator (must match the forward call).
            cotangent_out: Cotangent of the forward output.
            truncation: Must match the forward call.
            cotangent_in: Pre-allocated expansion for the input cotangent, or None.
            adjoint: Must match the forward call.
            sort_order: Sort order for the cotangent expansion.
            keep_duplicates: Whether duplicates are allowed.
            rehearse: If True, only rehearse.
            stream: Stream object, pointer, or None.

        Returns:
            If ``rehearse=True``: a :class:`GateApplicationRehearsalInfo`.
            If ``rehearse=False``: the *cotangent_in* expansion.
        """
        return self.view().apply_gate_backward_diff(
            gate=gate,
            cotangent_out=cotangent_out,
            truncation=truncation,
            cotangent_in=cotangent_in,
            adjoint=adjoint,
            sort_order=sort_order,
            keep_duplicates=keep_duplicates,
            rehearse=rehearse,
            stream=stream,
        )


class PauliExpansionView:
    """
    A view on a contiguous range of Pauli terms inside a Pauli operator expansion.

    Args:
        pauli_expansion: The Pauli expansion.
        start_index: The start index of the view.
        end_index: The end index of the view.
        options: Either a :class:`PauliExpansionOptions` object or a ``dict`` with
            matching keywords (``allocator``, ``memory_limit``, ``blocking``). If not
            provided, defaults inherit from the parent expansion.
    """
    def __init__(self, pauli_expansion: PauliExpansion, start_index: int, end_index: int, *, options: PauliExpansionOptions | dict | None = None):
        self._pauli_expansion = pauli_expansion
        self._start_index = start_index
        self._end_index = end_index
        self._library_handle = pauli_expansion._library_handle
        self._ptr : int | None= cupp.pauli_expansion_get_contiguous_range(
            int(self._library_handle),
            int(self._pauli_expansion),
            self._start_index,
            self._end_index
        )
        self._library_handle.logger.debug(f"C API cupaulipropPauliExpansionGetContiguousRange returned ptr={self._ptr}")
        self._update_stamp = self.base._update_stamp
        options = nvmath_utils.check_or_create_options(PauliExpansionOptions, options, "pauli expansion options")
        # Create allocator based on base expansion's array package (use "cuda" as fallback for CPU tensors)
        allocator_package = self.base.package if self.base.package != "numpy" else "cuda"
        if options.allocator is not None:
            allocator = options.allocator
        else:
            maybe_register_package(allocator_package)
            allocator = nvmath_memory._MEMORY_MANAGER[allocator_package](self._library_handle.device_id, self._library_handle._logger)
        self._workspace = Workspace(self._library_handle, allocator, options.memory_limit)
        self._blocking = options.blocking if options.blocking is not None else pauli_expansion._blocking
        self._last_compute_event = None
        # Register cleanup finalizer for safe resource release
        self._finalizer = register_finalizer(self, cupp.destroy_pauli_expansion_view, self._ptr, self._logger, "PauliExpansionView")
        self._logger.debug(f"PauliExpansionView created: range [{start_index}, {end_index})")

    @property
    def _logger(self):
        """Internal logger accessor, delegates to the library handle's logger."""
        return self._library_handle.logger

    @property
    def is_valid(self) -> bool:
        """
        Whether the view is valid.

        A view becomes invalid if the library performs a mutating operation on the underlyingPauli expansion after the view was created.
        """
        if int(self._library_handle) == 0:
            return False
        elif int(self._pauli_expansion) == 0:
            return False
        elif int(self) == 0:
            return False
        elif self._update_stamp != self.base._update_stamp:
            return False
        else:
            return True
        
    @property
    def start_index(self) -> int:
        """
        The start index (inclusive) of the view.
        """
        return self._start_index
    
    @property
    def end_index(self) -> int:
        """
        The end index (exclusive) of the view.
        """
        return self._end_index

    @property
    def base(self) -> PauliExpansion:
        """
        The Pauli expansion that this view is based on.
        """
        return self._pauli_expansion

    @property
    def is_sorted(self) -> bool:
        """
        Returns whether the underlying Pauli expansion is sorted.
        
        Note: This reflects the sort status of the entire base expansion, not just the view's range.
        """
        return self.base.is_sorted

    @property
    def sort_order(self) -> SortOrderLiteral:
        """
        Returns the sort order of the underlying Pauli expansion.
        
        Note: This reflects the sort order of the entire base expansion, not just the view's range.
        
        Returns:
            ``None``, ``"internal"``, or ``"little_endian_bitwise"``.
        """
        return self.base.sort_order

    def _allocate_expansion(self, num_terms: int, sort_order: SortOrder | SortOrderLiteral = None, has_duplicates: bool = True, stream=None) -> PauliExpansion:
        """
        Allocates a new PauliExpansion with the specified capacity.
        
        The allocation uses the tensor wrapper's empty method, which delegates to the
        array package's native allocator (e.g., cupy.cuda.alloc, torch.empty).
        Users can customize allocation by configuring their package's allocator.
        
        Args:
            num_terms: The number of terms the expansion should be able to hold.
            sort_order: The sort order of the expansion (defaults to ``None``).
            has_duplicates: Whether the expansion may contain duplicates (defaults to True).
            stream: Optional stream for allocation.
            
        Returns:
            A new PauliExpansion with the specified capacity.
            
        Raises:
            RuntimeError: If the base expansion is on CPU (numpy arrays).
        """
        num_qubits = self.base.num_qubits
        dtype = self.base.dtype
        device_id = self._library_handle.device_id
        backend = self.base.package
        if backend == "numpy":
            raise RuntimeError("Cannot allocate expansion: base expansion is on CPU. Transfer to GPU first using .to(device, package=...)")
        # Get the tensor holder type for the backend
        tensor_type = tensor_wrapper._TENSOR_TYPES[backend]
        
        # Calculate buffer shapes
        ints_per_string = cupp.get_num_packed_integers(num_qubits)
        xz_bits_shape = (num_terms, 2 * ints_per_string)
        coefs_shape = (num_terms,)
        
        # Allocate tensors using the package's native allocator
        maybe_register_package(backend)
        stream_holder = nvmath_utils.get_or_create_stream(device_id, stream, backend)
        xz_bits_holder = tensor_type.empty(
            xz_bits_shape, device_id=device_id, dtype="uint64", stream_holder=stream_holder
        )
        coefs_holder = tensor_type.empty(
            coefs_shape, device_id=device_id, dtype=dtype, stream_holder=stream_holder
        )
        
        return PauliExpansion(
            self._library_handle,
            num_qubits,
            0,  # num_terms starts at 0, capacity is determined by buffer size
            xz_bits_holder,
            coefs_holder,
            sort_order=sort_order,
            has_duplicates=has_duplicates,
            options=PauliExpansionOptions(
                allocator=self._workspace._allocator,
                memory_limit=self._workspace.memory_limit,
                blocking=self._blocking,
            ),
        )

    def apply_gate(self,
                   gate: QuantumOperator,
                   truncation: Truncation | None = None,
                   expansion_out: Optional["PauliExpansion"] = None,
                   adjoint: bool = False,
                   sort_order: SortOrder | SortOrderLiteral = None,
                   keep_duplicates: bool = False,
                   rehearse: bool | None = None,
                   stream = None) -> GateApplicationRehearsalInfo | PauliExpansion:
        """
        Applies a quantum operator to the Pauli expansion view.

        The truncation is applied during gate application.
        
        If ``rehearse=True``, only the prepare phase is executed to determine resource requirements.
        
        Args:
            gate: The gate to apply.
            truncation: Truncation strategy to apply (optional).
            expansion_out: The Pauli expansion to write the result to. If not provided,
                a new expansion will be allocated with the required capacity.
            adjoint: Whether to apply the adjoint of the gate (defaults to False).
            sort_order: Sort order for the output expansion. One of ``None`` or ``"internal"``
                (defaults to ``None``).
            keep_duplicates: Whether to keep duplicate Pauli strings in the output (defaults to False).
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.
            
        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: The output PauliExpansion.
        
        Raises:
            RuntimeError: If the base expansion is a rehearsal expansion and rehearse=False.
        """
        if expansion_out is self.base:
            raise ValueError("expansion_out must not alias the input expansion (got expansion_out is view.base)")
        # Resolve rehearse default based on base expansion's rehearsal status
        if rehearse is None:
            rehearse = self.base._is_rehearsal
        if self.base._is_rehearsal and not rehearse:
            raise RuntimeError(
                "Cannot perform computation on a rehearsal expansion. "
                "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
            )
        
        self._logger.info(f"Starting apply_gate: gate={gate}, adjoint={adjoint}, sort_order={sort_order}, rehearse={rehearse}")
        
        stream_package = self.base._package if self.base._package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)
        
        truncation_strategies = create_truncation_strategies(truncation)
        cupp_sort_order = sort_order_to_cupp(sort_order)
        
        with gate._as_c_operator(self._library_handle) as gate_ptr:
            # Prepare phase
            self._logger.debug("Preparing operator application...")
            xz_size, coef_size = cupp.pauli_expansion_view_prepare_operator_application(
                int(self._library_handle),
                int(self),
                gate_ptr,
                int(cupp_sort_order),
                int(keep_duplicates),
                len(truncation_strategies),
                convert_truncation_strategies(truncation_strategies),
                self._workspace.memory_limit,
                int(self._workspace))
            
            device_ws, host_ws = self._workspace.get_required_sizes()
            # Avoid accessing .coeffs for package="cuda" (buffers are intentionally not user-exposed).
            num_terms = coef_size // self.base._coefs.itemsize
            self._logger.debug(f"Prepare complete: required {num_terms} terms, device_ws={device_ws}, host_ws={host_ws}")
            
            if rehearse:
                self._logger.info(f"Rehearsal complete: {num_terms} terms required")
                return GateApplicationRehearsalInfo(device_ws, host_ws, num_terms)
            
            # Compute phase - allocate expansion_out if not provided
            if expansion_out is None:
                self._logger.debug(f"Allocating output expansion with capacity {num_terms}")
                expansion_out = self._allocate_expansion(num_terms)
            elif expansion_out.capacity < num_terms:
                raise ValueError(f"Expansion out capacity is too small, required {num_terms} terms, got {expansion_out.capacity} terms")
            
            if host_ws > 0 and not self._blocking:
                raise RuntimeError("Host workspace requires blocking execution.")
            
            timing = bool(self._logger.handlers)
            with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
                with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                    cupp.pauli_expansion_view_compute_operator_application(
                        int(self._library_handle),
                        int(self),
                        int(expansion_out),
                        gate_ptr,
                        int(adjoint),
                        int(cupp_sort_order),
                        int(keep_duplicates),
                        len(truncation_strategies),
                        convert_truncation_strategies(truncation_strategies),
                        int(self._workspace),
                        stream_holder.ptr
                    )
                expansion_out._update_stamp += 1
                if elapsed.data is not None:
                    self._logger.info(f"Gate application completed in {elapsed.data:.3f} ms, output has {expansion_out.num_terms} terms")
                else:
                    self._logger.info(f"Gate application completed, output has {expansion_out.num_terms} terms")
                
                return expansion_out

    def __int__(self) -> int:
        """
        Returns the pointer to the Pauli expansion view.
        """
        if self._ptr is None:
            return 0
        else:
            return self._ptr
    
    @property
    def num_terms(self) -> int:
        """
        The number of terms in the Pauli expansion view.
        """
        return cupp.pauli_expansion_view_get_num_terms(int(self._library_handle), self._ptr)
    
    @property
    def storage_location(self) -> Literal["cpu", "gpu"]:
        """
        The storage location of the Pauli expansion view.

        Returns:
            Either ``"cpu"`` or ``"gpu"``. This is inherited from the base expansion.
        """
        return self.base.storage_location
    
    @property
    def _term(self, term_index: int) -> "PauliTerm":
        """
        The Pauli term at the specified index.
        """
        return cupp.pauli_expansion_view_get_term(int(self._library_handle), self._ptr, term_index)
    
    @property
    def _zero_state(self) -> Sequence[int]:
        """
        The zero state ``|0...0>``.
        """
        return self.base._zero_state
    
    def trace_with_zero_state(self, rehearse: bool | None = None, stream = None) -> RehearsalInfo | tuple[float | complex, float]:
        """
        Computes the trace of the Pauli expansion view with the zero state ``|0...0>``.

        This computes :math:`\\langle 0 | \\rho | 0 \\rangle` where :math:`\\rho` is the operator
        represented by the terms in this view and :math:`|0\\rangle = |0...0\\rangle` is the
        all-zeros computational basis state.

        Args:
            rehearse: If True, only rehearse the operation to determine resource requirements.
                If None (default), automatically set to True for rehearsal expansions,
                False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.

        Returns:
            If rehearse=True: A :class:`RehearsalInfo` with the required workspace sizes.
            If rehearse=False: A tuple ``(trace_significand, trace_exponent)`` where
            ``trace = trace_significand * pow(2, trace_exponent)``.

        Raises:
            RuntimeError: If the base expansion is a rehearsal expansion and rehearse=False.
        """
        return self._trace_with_basis_state(self._zero_state, rehearse=rehearse, stream=stream)

    def _trace_with_basis_state(self, comp_basis_state: Sequence[int], rehearse: bool | None = None, stream = None) -> RehearsalInfo | tuple[float | complex, float]:
        """
        Computes the trace of the Pauli expansion view with a given quantum state
        expressed in the computational basis.
        
        Currently only the zero state ``|0...0>`` is supported. The computational basis
        state can be None (to indicate the zero state) or an array of zeros.
        
        If ``rehearse=True``, only the prepare phase is executed to determine resource requirements.
        
        Args:
            comp_basis_state: Quantum state in computational basis (bit-string).
                Can be None to indicate the zero state ``|0...0>``, or an array-like
                of uint32 values (must all be zeros for now). If provided, should be
                on host memory.
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.
        
        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: A tuple ``(trace_significand, trace_exponent)`` where
            ``trace = trace_significand * pow(2, trace_exponent)``.
        
        Raises:
            ValueError: If a state other than the zero state is provided (not yet supported).
            RuntimeError: If the base expansion is a rehearsal expansion and rehearse=False.
        """
        # Resolve rehearse default based on base expansion's rehearsal status
        if comp_basis_state == None:
            comp_basis_state = self._zero_state
        if comp_basis_state != self._zero_state:
            raise ValueError("Only the zero state |0...0> is currently supported")
        if rehearse is None:
            rehearse = self.base._is_rehearsal
        if self.base._is_rehearsal and not rehearse:
            raise RuntimeError(
                "Cannot perform computation on a rehearsal expansion. "
                "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
            )
        
        self._logger.info(f"Starting trace_with_zero_state: {self.base.num_terms} terms, rehearse={rehearse}")
        
        stream_package = self.base.package if self.base.package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)

        # Convert comp_basis_state to host array if provided
        if comp_basis_state is not None:
            comp_basis_state = np.asarray(comp_basis_state, dtype=np.uint32)
        
        # Prepare workspace
        self._logger.debug("Preparing trace computation...")
        cupp.pauli_expansion_view_prepare_trace_with_zero_state(
            int(self._library_handle),
            int(self),
            self._workspace.memory_limit,
            int(self._workspace))
        
        device_ws, host_ws = self._workspace.get_required_sizes()
        self._logger.debug(f"Prepare complete: device_ws={device_ws}, host_ws={host_ws}")
        
        if rehearse:
            self._logger.info("Rehearsal complete for trace_with_zero_state")
            return RehearsalInfo(device_ws, host_ws)
        
        # Compute phase
        if host_ws > 0 and not self._blocking:
            raise RuntimeError("Host workspace requires blocking execution.")
        
        timing = bool(self._logger.handlers)
        with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
            # Allocate output buffers on host: trace significand and base-2 exponent.
            trace_significand_buffer = np.zeros(1, dtype=self.base.dtype)
            trace_exponent_buffer = np.zeros(1, dtype=np.float64)
            
            # Compute trace (C++ implementation copies result from device to host pointer)
            with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                cupp.pauli_expansion_view_compute_trace_with_zero_state(
                    int(self._library_handle),
                    int(self),
                    trace_significand_buffer.ctypes.data,
                    trace_exponent_buffer.ctypes.data,
                    int(self._workspace),
                    stream_holder.ptr)
        
        if elapsed.data is not None:
            self._logger.info(f"Trace computation completed in {elapsed.data:.3f} ms")
        else:
            self._logger.info("Trace computation completed")
        
        return (trace_significand_buffer[0].item(), float(trace_exponent_buffer[0]))
    

    def product_trace(self, other: "PauliExpansionView", adjoint: bool = False, rehearse: bool | None = None, stream = None) -> RehearsalInfo | tuple[float | complex, float]:
        """
        Computes the product trace Tr(self^† * other) or Tr(self * other) of two Pauli expansion views.
        
        This function computes the trace of the product of two Pauli operators by:
        1. Concatenating both views into workspace memory
        2. Sorting the combined terms by Pauli strings
        3. Identifying Pauli strings that appear exactly twice (once in each expansion)
        4. Multiplying the coefficients of matching pairs (with optional adjoint) and summing them
        
        If ``rehearse=True``, only the prepare phase is executed to determine resource requirements.
        
        Args:
            other: The other Pauli expansion view to compute the product trace with.
            adjoint: If True, computes Tr(self^† * other) by taking the complex conjugate of 
                coefficients from self. If False, computes Tr(self * other). Defaults to False.
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True if either expansion is a rehearsal expansion, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.
        
        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: A tuple ``(trace_significand, trace_exponent)`` where
            ``trace = trace_significand * pow(2, trace_exponent)``.
        
        Raises:
            ValueError: If the views have different numbers of qubits or data types.
            RuntimeError: If either expansion is a rehearsal expansion and rehearse=False.
        """
        # Resolve rehearse default: True if either base expansion is a rehearsal expansion
        if rehearse is None:
            rehearse = self.base._is_rehearsal or other.base._is_rehearsal
        if not rehearse:
            if self.base._is_rehearsal:
                raise RuntimeError(
                    "Cannot perform computation on a rehearsal expansion. "
                    "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
                )
            if other.base._is_rehearsal:
                raise RuntimeError(
                    "Cannot perform computation with a rehearsal expansion. "
                    "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
                )
        
        self._logger.info(f"Starting product_trace: {self.base.num_terms} x {other.base.num_terms} terms, adjoint={adjoint}, rehearse={rehearse}")
        
        stream_package = self.base.package if self.base.package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)
        
        # Validate that both views have the same number of qubits and data type
        if self.base.num_qubits != other.base.num_qubits:
            raise ValueError(f"Views must have the same number of qubits, got {self.base.num_qubits} and {other.base.num_qubits}")
        
        if self.base.dtype != other.base.dtype:
            raise ValueError(f"Views must have the same data type, got {self.base.dtype} and {other.base.dtype}")
        
        # Prepare workspace
        self._logger.debug("Preparing product trace computation...")
        cupp.pauli_expansion_view_prepare_trace_with_expansion_view(
            int(self._library_handle),
            int(self),
            int(other),
            self._workspace.memory_limit,
            int(self._workspace))
        
        device_ws, host_ws = self._workspace.get_required_sizes()
        self._logger.debug(f"Prepare complete: device_ws={device_ws}, host_ws={host_ws}")
        
        if rehearse:
            self._logger.info("Rehearsal complete for product_trace")
            return RehearsalInfo(device_ws, host_ws)
        
        # Compute phase
        if host_ws > 0 and not self._blocking:
            raise RuntimeError("Host workspace requires blocking execution.")
        
        timing = bool(self._logger.handlers)
        with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
            # Allocate output buffers on host: trace significand and base-2 exponent.
            trace_significand_buffer = np.zeros(1, dtype=self.base.dtype)
            trace_exponent_buffer = np.zeros(1, dtype=np.float64)
            
            # Compute product trace (C++ implementation copies result from device to host pointer)
            with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                cupp.pauli_expansion_view_compute_trace_with_expansion_view(
                    int(self._library_handle),
                    int(self),
                    int(other),
                    int(adjoint),  # takeAdjoint1: take adjoint of first view if requested
                    trace_significand_buffer.ctypes.data,
                    trace_exponent_buffer.ctypes.data,
                    int(self._workspace),
                    stream_holder.ptr)
            
            if elapsed.data is not None:
                self._logger.info(f"Product trace completed in {elapsed.data:.3f} ms")
            else:
                self._logger.info("Product trace completed")
            
            return (trace_significand_buffer[0].item(), float(trace_exponent_buffer[0]))
    
    def sort(self, expansion_out: Optional["PauliExpansion"] = None, sort_order: SortOrder | SortOrderLiteral = "internal", rehearse: bool | None = None, stream = None) -> RehearsalInfo | PauliExpansion:
        """
        Sorts the Pauli expansion view by the specified sorting order and writes the result to the output expansion.
        
        If ``rehearse=True``, only the prepare phase is executed to determine resource requirements.
        
        Args:
            expansion_out: The Pauli expansion to write the sorted result to. If not provided,
                a new expansion will be allocated with the required capacity.
            sort_order: Sort order to apply. One of ``"internal"`` or ``"little_endian_bitwise"``
                (defaults to ``"internal"``). Note: ``None`` is not valid for sort operations.
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.
            
        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: The output PauliExpansion.
        
        Raises:
            ValueError: If sort_order is ``None``.
            RuntimeError: If the base expansion is a rehearsal expansion and rehearse=False.
        """
        if expansion_out is self.base:
            raise ValueError("expansion_out must not alias the input expansion (got expansion_out is view.base)")
        if sort_order is None:
            raise ValueError("sort_order cannot be None for sort operations")
        # Resolve rehearse default based on base expansion's rehearsal status
        if rehearse is None:
            rehearse = self.base._is_rehearsal
        if self.base._is_rehearsal and not rehearse:
            raise RuntimeError(
                "Cannot perform computation on a rehearsal expansion. "
                "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
            )
        
        num_terms = self.end_index - self.start_index
        self._logger.info(f"Starting sort: {num_terms} terms, sort_order={sort_order}, rehearse={rehearse}")
        
        stream_package = self.base.package if self.base.package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)
        
        cupp_sort_order = sort_order_to_cupp(sort_order)
        
        # Prepare workspace
        self._logger.debug("Preparing sort...")
        cupp.pauli_expansion_view_prepare_sort(
            int(self._library_handle),
            int(self),
            cupp_sort_order,
            self._workspace.memory_limit,
            int(self._workspace))
        
        device_ws, host_ws = self._workspace.get_required_sizes()
        self._logger.debug(f"Prepare complete: device_ws={device_ws}, host_ws={host_ws}")
        
        if rehearse:
            self._logger.info("Rehearsal complete for sort")
            return GateApplicationRehearsalInfo(device_ws, host_ws, num_terms)
        
        # Compute phase - allocate expansion_out if not provided
        # Sort output has same number of terms as input
        if expansion_out is None:
            self._logger.debug(f"Allocating output expansion with capacity {num_terms}")
            expansion_out = self._allocate_expansion(num_terms)
        
        if host_ws > 0 and not self._blocking:
            raise RuntimeError("Host workspace requires blocking execution.")
        
        timing = bool(self._logger.handlers)
        with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
            # Execute sort
            with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                cupp.pauli_expansion_view_execute_sort(
                    int(self._library_handle),
                    int(self),
                    int(expansion_out),
                    cupp_sort_order,
                    int(self._workspace),
                    stream_holder.ptr)
            expansion_out._update_stamp += 1
            if elapsed.data is not None:
                self._logger.info(f"Sort completed in {elapsed.data:.3f} ms")
            else:
                self._logger.info("Sort completed")
        
        return expansion_out

    
    def deduplicate(self, expansion_out: Optional["PauliExpansion"] = None, sort_order: SortOrder | SortOrderLiteral = None, rehearse: bool | None = None, stream = None) -> RehearsalInfo | PauliExpansion:
        """
        Deduplicates the Pauli expansion view (removes duplicate Pauli strings and 
        sums their coefficients) and writes the result to the output expansion.
        
        Note: The input view must be sorted before calling this method.
        
        If ``rehearse=True``, only the prepare phase is executed to determine resource requirements.
        
        Args:
            expansion_out: The Pauli expansion to write the deduplicated result to. If not provided,
                a new expansion will be allocated with the required capacity.
            sort_order: Sort order for the output expansion. One of ``None`` or ``"internal"``
                (defaults to ``None``).
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.
        
        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: The output PauliExpansion.
        
        Raises:
            ValueError: If the input view is not sorted.
            RuntimeError: If the base expansion is a rehearsal expansion and rehearse=False.
        """
        if expansion_out is self.base:
            raise ValueError("expansion_out must not alias the input expansion (got expansion_out is view.base)")
        # Resolve rehearse default based on base expansion's rehearsal status
        if rehearse is None:
            rehearse = self.base._is_rehearsal
        if self.base._is_rehearsal and not rehearse:
            raise RuntimeError(
                "Cannot perform computation on a rehearsal expansion. "
                "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
            )
        
        num_terms_in = self.end_index - self.start_index
        self._logger.info(f"Starting deduplicate: {num_terms_in} terms, sort_order={sort_order}, rehearse={rehearse}")
        
        stream_package = self.base.package if self.base.package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)
        
        cupp_sort_order = sort_order_to_cupp(sort_order)
        
        # Prepare workspace
        self._logger.debug("Preparing deduplication...")
        cupp.pauli_expansion_view_prepare_deduplication(
            int(self._library_handle),
            int(self),
            int(cupp_sort_order),
            self._workspace.memory_limit,
            int(self._workspace))
        
        device_ws, host_ws = self._workspace.get_required_sizes()
        self._logger.debug(f"Prepare complete: device_ws={device_ws}, host_ws={host_ws}")
        
        # Deduplication output has at most as many terms as input
        if rehearse:
            self._logger.info("Rehearsal complete for deduplicate")
            return GateApplicationRehearsalInfo(device_ws, host_ws, num_terms_in)
        
        # Compute phase - allocate expansion_out if not provided
        if expansion_out is None:
            self._logger.debug(f"Allocating output expansion with capacity {num_terms_in}")
            expansion_out = self._allocate_expansion(num_terms_in)
        
        if host_ws > 0 and not self._blocking:
            raise RuntimeError("Host workspace requires blocking execution.")
        
        timing = bool(self._logger.handlers)
        with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
            # Execute deduplication
            with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                cupp.pauli_expansion_view_execute_deduplication(
                    int(self._library_handle),
                    int(self),
                    int(expansion_out),
                    int(cupp_sort_order),
                    int(self._workspace),
                    stream_holder.ptr)
            expansion_out._update_stamp += 1
            if elapsed.data is not None:
                self._logger.info(f"Deduplicate completed in {elapsed.data:.3f} ms, {num_terms_in} -> {expansion_out.num_terms} terms")
            else:
                self._logger.info(f"Deduplicate completed, {num_terms_in} -> {expansion_out.num_terms} terms")
        
        return expansion_out
    
    def truncate(self, expansion_out: Optional["PauliExpansion"] = None, truncation: Truncation | None = None, rehearse: bool | None = None, stream = None) -> RehearsalInfo | PauliExpansion:
        """
        Truncates the Pauli expansion view based on the given truncation strategy
        and writes the result to the output expansion.
        
        If ``rehearse=True``, only the prepare phase is executed to determine resource requirements.
        
        Args:
            expansion_out: The Pauli expansion to write the truncated result to. If not provided,
                a new expansion will be allocated with the required capacity.
            truncation: Truncation strategy to apply (optional).
            rehearse: If True, only rehearse the operation. If None (default), automatically
                set to True for rehearsal expansions, False otherwise.
            stream: A stream object from the array package, a stream pointer (int), or None
                to use the package's current stream.
                
        Returns:
            If rehearse=True: A RehearsalInfo with the required resources.
            If rehearse=False: The output PauliExpansion.
        
        Raises:
            RuntimeError: If the base expansion is a rehearsal expansion and rehearse=False.
        """
        if expansion_out is self.base:
            raise ValueError("expansion_out must not alias the input expansion (got expansion_out is view.base)")
        # Resolve rehearse default based on base expansion's rehearsal status
        if rehearse is None:
            rehearse = self.base._is_rehearsal
        if self.base._is_rehearsal and not rehearse:
            raise RuntimeError(
                "Cannot perform computation on a rehearsal expansion. "
                "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
            )
        
        num_terms_in = self.end_index - self.start_index
        self._logger.info(f"Starting truncate: {num_terms_in} terms, truncation={truncation}, rehearse={rehearse}")
        
        stream_package = self.base.package if self.base.package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)
        
        truncation_strategies = create_truncation_strategies(truncation)
        
        # Prepare workspace
        self._logger.debug("Preparing truncation...")
        cupp.pauli_expansion_view_prepare_truncation(
            int(self._library_handle),
            int(self),
            len(truncation_strategies),
            convert_truncation_strategies(truncation_strategies),
            self._workspace.memory_limit,
            int(self._workspace))
        
        device_ws, host_ws = self._workspace.get_required_sizes()
        self._logger.debug(f"Prepare complete: device_ws={device_ws}, host_ws={host_ws}")
        
        # Truncation output has at most as many terms as input
        if rehearse:
            self._logger.info("Rehearsal complete for truncate")
            return GateApplicationRehearsalInfo(device_ws, host_ws, num_terms_in)
        
        # Compute phase - allocate expansion_out if not provided
        if expansion_out is None:
            self._logger.debug(f"Allocating output expansion with capacity {num_terms_in}")
            expansion_out = self._allocate_expansion(num_terms_in)
        
        if host_ws > 0 and not self._blocking:
            raise RuntimeError("Host workspace requires blocking execution.")
        
        timing = bool(self._logger.handlers)
        with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
            # Execute truncation
            with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                cupp.pauli_expansion_view_execute_truncation(
                    int(self._library_handle),
                    int(self),
                    int(expansion_out),
                    len(truncation_strategies),
                    convert_truncation_strategies(truncation_strategies),
                    int(self._workspace),
                    stream_holder.ptr)
            expansion_out._update_stamp += 1
            if elapsed.data is not None:
                self._logger.info(f"Truncate completed in {elapsed.data:.3f} ms, {num_terms_in} -> {expansion_out.num_terms} terms")
            else:
                self._logger.info(f"Truncate completed, {num_terms_in} -> {expansion_out.num_terms} terms")
            
            return expansion_out

    # ------------------------------------------------------------------
    # Backward differentiation methods
    # ------------------------------------------------------------------

    def trace_with_zero_state_backward_diff(
        self,
        cotangent_trace,
        cotangent_trace_exponent,
        /,
        cotangent_expansion: "PauliExpansion | None" = None,
        rehearse: bool | None = None,
        stream=None,
    ) -> "TraceBackwardRehearsalInfo | PauliExpansion":
        """Backward pass for :meth:`trace_with_zero_state`.

        Propagates the scalar cotangent of the trace value back to coefficient
        cotangents of this view.

        Args:
            cotangent_trace: Scalar cotangent :math:`\\tilde{t} = dL/dt` (same dtype
                as the expansion coefficients).  Can be a Python scalar or a
                1-element numpy array.
            cotangent_trace_exponent: Scalar cotangent for the trace-exponent output.
            cotangent_expansion: Pre-allocated :class:`PauliExpansion` to receive the
                coefficient cotangents, or ``None`` to auto-allocate.
            rehearse: If True, only the prepare phase runs (returns workspace
                requirements).  If None (default), automatically set based on the
                base expansion's rehearsal status.
            stream: A stream object, stream pointer (int), or None.

        Returns:
            If ``rehearse=True``: a :class:`TraceBackwardRehearsalInfo`.
            If ``rehearse=False``: the cotangent :class:`PauliExpansion`.
        """
        if rehearse is None:
            rehearse = self.base._is_rehearsal
        if self.base._is_rehearsal and not rehearse:
            raise RuntimeError(
                "Cannot perform computation on a rehearsal expansion. "
                "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
            )

        self._logger.info(f"Starting trace_with_zero_state_backward_diff, rehearse={rehearse}")

        stream_package = self.base.package if self.base.package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)

        # Prepare
        self._logger.debug("Preparing backward trace with zero state...")
        _, required_coef_bytes = cupp.pauli_expansion_view_prepare_trace_with_zero_state_backward_diff(
            int(self._library_handle),
            int(self),
            self._workspace.memory_limit,
            int(self._workspace))
        required_terms = required_coef_bytes // self.base._coefs.itemsize if required_coef_bytes > 0 else 0

        device_ws, host_ws = self._workspace.get_required_sizes()
        self._logger.debug(
            f"Prepare complete: required_terms={required_terms}, device_ws={device_ws}, host_ws={host_ws}"
        )

        if rehearse:
            self._logger.info(f"Rehearsal complete: cotangent needs {required_terms} terms")
            return TraceBackwardRehearsalInfo(device_ws, host_ws, required_terms)

        # Auto-allocate cotangent expansion if not provided
        if cotangent_expansion is None:
            alloc_capacity = max(required_terms, 1)
            self._logger.debug(f"Allocating cotangent_expansion with capacity {alloc_capacity}")
            cotangent_expansion = self._allocate_expansion(alloc_capacity, stream=stream)
        elif cotangent_expansion.capacity < required_terms:
            raise ValueError(
                f"cotangent_expansion capacity is too small, required {required_terms} terms, got {cotangent_expansion.capacity} terms"
            )

        # Compute
        if host_ws > 0 and not self._blocking:
            raise RuntimeError("Host workspace requires blocking execution.")

        # Ensure cotangents are host buffers with correct dtype.
        cotangent_trace_significand_buf = np.array([cotangent_trace], dtype=self.base.dtype).ravel()
        cotangent_trace_exponent_buf = np.array([cotangent_trace_exponent], dtype=np.float64).ravel()

        timing = bool(self._logger.handlers)
        with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
            with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                cupp.pauli_expansion_view_compute_trace_with_zero_state_backward_diff(
                    int(self._library_handle),
                    int(self),
                    cotangent_trace_significand_buf.ctypes.data,
                    cotangent_trace_exponent_buf.ctypes.data,
                    int(cotangent_expansion),
                    int(self._workspace),
                    stream_holder.ptr)

        if elapsed.data is not None:
            self._logger.info(f"trace_with_zero_state_backward_diff completed in {elapsed.data:.3f} ms")
        else:
            self._logger.info("trace_with_zero_state_backward_diff completed")

        return cotangent_expansion

    def product_trace_backward_diff(
        self,
        other: "PauliExpansionView",
        cotangent_trace,
        cotangent_trace_exponent,
        /,
        cotangent_expansion1: "PauliExpansion | None" = None,
        cotangent_expansion2: "PauliExpansion | None" = None,
        adjoint: bool = False,
        rehearse: bool | None = None,
        stream=None,
    ) -> "ProductTraceBackwardRehearsalInfo | tuple[PauliExpansion, PauliExpansion]":
        """Backward pass for :meth:`product_trace`.

        Propagates the scalar cotangent of the product-trace value back to
        coefficient cotangents for both input views.

        Args:
            other: The other Pauli expansion view (same as in the forward
                :meth:`product_trace` call).
            cotangent_trace: Scalar cotangent :math:`\\tilde{t} = dL/dt`.
            cotangent_trace_exponent: Scalar cotangent for the trace-exponent output.
            cotangent_expansion1: Pre-allocated :class:`PauliExpansion` to receive
                coefficient cotangents for ``self``, or ``None`` to auto-allocate.
            cotangent_expansion2: Pre-allocated :class:`PauliExpansion` to receive
                coefficient cotangents for ``other``, or ``None`` to auto-allocate.
            adjoint: Must match the *adjoint* flag used in the forward
                :meth:`product_trace` call.
            rehearse: If True, only the prepare phase runs.
            stream: A stream object, stream pointer (int), or None.

        Returns:
            If ``rehearse=True``: a :class:`ProductTraceBackwardRehearsalInfo`.
            If ``rehearse=False``: a tuple ``(cotangent_expansion1, cotangent_expansion2)``.
        """
        if rehearse is None:
            rehearse = self.base._is_rehearsal or other.base._is_rehearsal
        if not rehearse:
            if self.base._is_rehearsal:
                raise RuntimeError(
                    "Cannot perform computation on a rehearsal expansion. "
                    "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
                )
            if other.base._is_rehearsal:
                raise RuntimeError(
                    "Cannot perform computation with a rehearsal expansion. "
                    "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
                )

        self._logger.info(f"Starting product_trace_backward_diff, adjoint={adjoint}, rehearse={rehearse}")

        stream_package = self.base.package if self.base.package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)

        # Prepare
        self._logger.debug("Preparing backward product trace...")
        (_, required_coef_bytes1,
         _, required_coef_bytes2) = cupp.pauli_expansion_view_prepare_trace_with_expansion_view_backward_diff(
            int(self._library_handle),
            int(self),
            int(other),
            self._workspace.memory_limit,
            int(self._workspace))
        required_terms1 = required_coef_bytes1 // self.base._coefs.itemsize if required_coef_bytes1 > 0 else 0
        required_terms2 = required_coef_bytes2 // self.base._coefs.itemsize if required_coef_bytes2 > 0 else 0

        device_ws, host_ws = self._workspace.get_required_sizes()
        self._logger.debug(
            f"Prepare complete: required_terms1={required_terms1}, required_terms2={required_terms2}, "
            f"device_ws={device_ws}, host_ws={host_ws}"
        )

        if rehearse:
            self._logger.info(
                f"Rehearsal complete: cotangent1 needs {required_terms1} terms, "
                f"cotangent2 needs {required_terms2} terms"
            )
            return ProductTraceBackwardRehearsalInfo(device_ws, host_ws, required_terms1, required_terms2)

        # Auto-allocate cotangent expansions if not provided
        if cotangent_expansion1 is None:
            alloc_capacity = max(required_terms1, 1)
            self._logger.debug(f"Allocating cotangent_expansion1 with capacity {alloc_capacity}")
            cotangent_expansion1 = self._allocate_expansion(alloc_capacity, stream=stream)
        elif cotangent_expansion1.capacity < required_terms1:
            raise ValueError(
                f"cotangent_expansion1 capacity is too small, required {required_terms1} terms, got {cotangent_expansion1.capacity} terms"
            )
        if cotangent_expansion2 is None:
            alloc_capacity = max(required_terms2, 1)
            self._logger.debug(f"Allocating cotangent_expansion2 with capacity {alloc_capacity}")
            cotangent_expansion2 = other._allocate_expansion(alloc_capacity, stream=stream)
        elif cotangent_expansion2.capacity < required_terms2:
            raise ValueError(
                f"cotangent_expansion2 capacity is too small, required {required_terms2} terms, got {cotangent_expansion2.capacity} terms"
            )

        # Compute
        if host_ws > 0 and not self._blocking:
            raise RuntimeError("Host workspace requires blocking execution.")

        cotangent_trace_significand_buf = np.array([cotangent_trace], dtype=self.base.dtype).ravel()
        cotangent_trace_exponent_buf = np.array([cotangent_trace_exponent], dtype=np.float64).ravel()

        timing = bool(self._logger.handlers)
        with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
            with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                cupp.pauli_expansion_view_compute_trace_with_expansion_view_backward_diff(
                    int(self._library_handle),
                    int(self),
                    int(other),
                    int(adjoint),
                    cotangent_trace_significand_buf.ctypes.data,
                    cotangent_trace_exponent_buf.ctypes.data,
                    int(cotangent_expansion1),
                    int(cotangent_expansion2),
                    int(self._workspace),
                    stream_holder.ptr)

        if elapsed.data is not None:
            self._logger.info(f"product_trace_backward_diff completed in {elapsed.data:.3f} ms")
        else:
            self._logger.info("product_trace_backward_diff completed")

        return cotangent_expansion1, cotangent_expansion2

    def apply_gate_backward_diff(
        self,
        gate: QuantumOperator,
        cotangent_out: "PauliExpansionView",
        truncation: "Truncation | None" = None,
        cotangent_in: "PauliExpansion | None" = None,
        param_grads_out: "np.ndarray | None" = None,
        adjoint: bool = False,
        sort_order: "SortOrder | SortOrderLiteral" = None,
        keep_duplicates: bool = False,
        rehearse: bool | None = None,
        stream=None,
    ) -> "GateApplicationRehearsalInfo | tuple[PauliExpansion, np.ndarray | None]":
        """Backward pass for :meth:`apply_gate`.

        Computes the input cotangent (written to *cotangent_in*) and accumulates
        parameter gradients into *param_grads_out* (or an auto-allocated buffer).

        Args:
            gate: The quantum operator (must match the forward :meth:`apply_gate` call).
            cotangent_out: Cotangent of the forward output, as a
                :class:`PauliExpansionView`.
            truncation: Truncation strategy (must match the forward call).
            cotangent_in: Pre-allocated :class:`PauliExpansion` for the input
                cotangent.  If ``None``, one is allocated automatically.
            param_grads_out: A numpy array to accumulate parameter gradients into.
                Must have shape ``(gate.num_differentiable_params,)`` and dtype
                matching the expansion coefficient type.  If ``None`` (default),
                a zeroed numpy array of the correct shape/dtype is allocated
                automatically.
            adjoint: Must match the *adjoint* flag used in the forward call.
            sort_order: Sort order for the output cotangent expansion.
            keep_duplicates: Whether the output may contain duplicates.
            rehearse: If True, only the prepare phase runs.
            stream: A stream object, stream pointer (int), or None.

        Returns:
            If ``rehearse=True``: a :class:`GateApplicationRehearsalInfo`.
            If ``rehearse=False``: a tuple ``(cotangent_in, param_grads)`` where
            *cotangent_in* is the input cotangent :class:`PauliExpansion` and
            *param_grads* is a numpy array of parameter gradients (or ``None``
            for non-differentiable operators).
        """
        if rehearse is None:
            rehearse = self.base._is_rehearsal
        if self.base._is_rehearsal and not rehearse:
            raise RuntimeError(
                "Cannot perform computation on a rehearsal expansion. "
                "Create a non-rehearsal expansion using rehearsal_expansion.from_empty(...) or use rehearse=True."
            )

        self._logger.info(
            f"Starting apply_gate_backward_diff: gate={gate}, adjoint={adjoint}, "
            f"sort_order={sort_order}, rehearse={rehearse}"
        )

        stream_package = self.base.package if self.base.package != "numpy" else "cuda"
        maybe_register_package(stream_package)
        stream_holder = nvmath_utils.get_or_create_stream(
            self._library_handle.device_id, stream, stream_package)

        truncation_strategies = create_truncation_strategies(truncation)
        cupp_sort_order = sort_order_to_cupp(sort_order)

        # Infer gradient dtype from the expansion's coefficient type
        grad_dtype = str(self.base._coefs.dtype)

        with gate._as_c_operator_with_grad(self._library_handle, param_grads_out, grad_dtype) as (gate_ptr, param_grads):
            # Prepare: get required buffer sizes for cotangent_in and workspace
            self._logger.debug("Preparing backward operator application...")
            _, coef_size = cupp.pauli_expansion_view_prepare_operator_application_backward_diff(
                int(self._library_handle),
                int(self),                          # viewIn
                int(cotangent_out),                 # cotangentOut
                gate_ptr,                           # quantumOperator
                int(cupp_sort_order),               # sortOrder
                int(keep_duplicates),               # keepDuplicates
                len(truncation_strategies) if truncation else 0,
                convert_truncation_strategies(truncation_strategies),
                self._workspace.memory_limit,
                int(self._workspace))

            device_ws, host_ws = self._workspace.get_required_sizes()
            num_terms = coef_size // self.base._coefs.itemsize if coef_size > 0 else 0
            self._logger.debug(
                f"Prepare complete: cotangent_in needs {num_terms} terms, "
                f"device_ws={device_ws}, host_ws={host_ws}"
            )

            if rehearse:
                self._logger.info(f"Rehearsal complete: cotangent_in needs {num_terms} terms")
                return GateApplicationRehearsalInfo(device_ws, host_ws, num_terms)

            # Allocate cotangent_in if not provided (capacity >= 1 to avoid zero-sized buffers)
            if cotangent_in is None:
                alloc_capacity = max(num_terms, 1)
                self._logger.debug(f"Allocating cotangent_in with capacity {alloc_capacity} (num_terms={num_terms})")
                cotangent_in = self._allocate_expansion(alloc_capacity, stream=stream)
            elif cotangent_in.capacity < num_terms:
                raise ValueError(
                    f"cotangent_in capacity too small: need {num_terms} terms, "
                    f"got {cotangent_in.capacity}"
                )

            if host_ws > 0 and not self._blocking:
                raise RuntimeError("Host workspace requires blocking execution.")

            timing = bool(self._logger.handlers)
            with self._workspace.scratch_context(device_ws, host_ws, stream_holder) as (_, _dev_buf, _host_buf):
                with nvmath_utils.cuda_call_ctx(stream_holder, self._blocking, timing) as (self._last_compute_event, elapsed):
                    cupp.pauli_expansion_view_compute_operator_application_backward_diff(
                        int(self._library_handle),
                        int(self),                      # viewIn
                        int(cotangent_out),              # cotangentOut
                        int(cotangent_in),               # cotangentIn
                        gate_ptr,                        # quantumOperator
                        int(adjoint),
                        int(cupp_sort_order),
                        int(keep_duplicates),
                        len(truncation_strategies) if truncation else 0,
                        convert_truncation_strategies(truncation_strategies),
                        int(self._workspace),
                        stream_holder.ptr)

            if elapsed.data is not None:
                self._logger.info(f"apply_gate_backward_diff completed in {elapsed.data:.3f} ms")
            else:
                self._logger.info("apply_gate_backward_diff completed")

            return cotangent_in, param_grads

