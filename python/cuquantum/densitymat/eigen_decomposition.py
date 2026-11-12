# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Operator eigen-decomposition."""

from dataclasses import dataclass
from typing import Optional, Sequence, Union
import collections
import warnings
import weakref

import numpy as np
import cupy as cp

from nvmath.internal import utils as nvmath_utils
from nvmath.internal.tensor_wrapper import wrap_operand

from cuquantum.bindings import cudensitymat as cudm
from .operators import Operator, _handle_callback_params
from .state import State
from .svd import SVDConfig, _build_svd_config_handle
from .work_stream import WorkStream
from ._internal import utils
from ._internal.utils import (
    InvalidObjectState,
    NDArrayType,
    resolve_enum as _resolve_enum,
    set_config_attribute as _set_config_attribute,
)
from ._internal.typemaps import CUDENSITYMAT_COMPUTE_TYPE_MAP


__all__ = [
    "DMRGConfig",
    "EigenDecomposition",
    "EigenDecompositionApproachKrylovConfig",
    "EigenDecompositionApproachLinearConfig",
    "EigenDecompositionResult",
]


_SPECTRUM_KIND_MAP = {
    "SA": cudm.EigenDecompositionSpectrumKind.EIGEN_SPECTRUM_SMALLEST_REAL,
    "LA": cudm.EigenDecompositionSpectrumKind.EIGEN_SPECTRUM_LARGEST_REAL,
}

_SCOPE_KIND_MAP = {
    "full": cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_FULL,
    "split": cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_SPLIT,
}

_APPROACH_KIND_MAP = {
    "krylov": cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_KRYLOV,
    "linear": cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_LINEAR,
}

_SPLIT_KIND_MAP = {
    "dmrg": cudm.EigenDecompositionScopeSplitKind.EIGEN_SCOPE_SPLIT_DMRG,
    "shift_invert_dmrg": cudm.EigenDecompositionScopeSplitKind.EIGEN_SCOPE_SPLIT_SHIFT_INVERT_DMRG,
}


@dataclass
class EigenDecompositionApproachKrylovConfig:
    """A data class for the block-Krylov approach configuration of :class:`EigenDecomposition`.

    Attributes:
        max_dim: Maximum ratio of the total number of blocks in the Krylov subspace to the number of requested eigenvalues.
            If not specified, a default value will be chosen. Must be greater than ``1``. Defaults to ``5``.
        max_restarts: Maximum number of restart cycles allowed during the iterative eigenvalue computation.
            If not specified, a default value will be chosen. Defaults to ``19``.
        min_block_size: Minimum number of Krylov subspace vectors to use in the block iterative method.
            A larger value may improve convergence but increases memory usage and computational cost.
            If not specified, a default value will be chosen. Defaults to ``1``.
    """
    max_dim: Optional[int] = None
    max_restarts: Optional[int] = None
    min_block_size: Optional[int] = None


@dataclass
class EigenDecompositionApproachLinearConfig:
    """A data class for the iterative linear-solver approach configuration of :class:`EigenDecomposition`.

    Used with ``approach="linear"`` (currently MINRES), as required by the shift-invert
    DMRG split kind. The requested per-site relative normal-equation tolerance is
    supplied by :meth:`EigenDecomposition.compute`; it is not a config field.

    Attributes:
        max_iterations: Maximum number of iterations for each local linear
            solve.
            If not specified, a default value will be chosen. Must be greater than ``0``.
            Defaults to ``200``.
    """
    max_iterations: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError(
                "EigenDecompositionApproachLinearConfig.max_iterations must be greater than zero."
            )


@dataclass
class DMRGConfig:
    """Configuration shared by the DMRG split decompositions.

    Args:
        num_sites: Number of sites swept per local update. ``1`` for 1-site
            DMRG, ``2`` for 2-site DMRG. Defaults to ``1`` when ``None``.
            Shift-invert supports both values for MPS containing at least two
            tensors.
        svd_config: SVD truncation policy for the 2-site sweep
            (:class:`SVDConfig`).
        max_sweeps: Maximum number of full L-R-L sweeps. For shift-invert,
            this is the fitting-sweep cap for each inverse application.
            Defaults to ``20`` when ``None``.
        energy_tolerance: Convergence threshold on the change in variational
            energy between consecutive sweeps for ground-state DMRG, or
            between consecutive outer Rayleigh quotients for shift-invert.
            Defaults to ``1e-10`` when ``None``.
        max_power_iterations: Maximum number of outer power iterations on
            ``(H - sigma)^{-1}`` for the shift-invert DMRG split kind; ignored by
            the ground-state DMRG split kind. Defaults to ``10`` when ``None``.
    """
    num_sites: Optional[int] = None
    svd_config: Optional[SVDConfig] = None
    max_sweeps: Optional[int] = None
    energy_tolerance: Optional[float] = None
    max_power_iterations: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_sites is not None and self.num_sites not in (1, 2):
            raise ValueError("DMRGConfig.num_sites must be 1 or 2.")
        if self.max_sweeps is not None and self.max_sweeps <= 0:
            raise ValueError("DMRGConfig.max_sweeps must be greater than zero.")
        if self.max_power_iterations is not None and self.max_power_iterations <= 0:
            raise ValueError("DMRGConfig.max_power_iterations must be greater than zero.")
        if self.energy_tolerance is not None:
            if not np.isfinite(self.energy_tolerance) or self.energy_tolerance < 0.0:
                raise ValueError("DMRGConfig.energy_tolerance must be finite and non-negative.")
        if self.svd_config is not None and not isinstance(self.svd_config, SVDConfig):
            raise TypeError(
                f"DMRGConfig.svd_config must be an SVDConfig instance, got "
                f"{type(self.svd_config).__name__}."
            )


@dataclass
class EigenDecompositionResult:
    """A data class for capturing the results of an :meth:`EigenDecomposition.compute` call.

    This class encapsulates all outputs from the eigenvalue/eigenstate computation, including
    convergence information and residual norms for analysis of solution quality.

    Attributes:
        evals: The computed eigenvalues as a 1D array. For batched computations,
            this will be a 2D array with shape ``[num_eigvals, batch_size]``. Elements are ordered 
            according to the ``which`` parameter used in the solver initialization.
        evecs: The computed eigenstates as a sequence of :class:`State` objects.
            Each state corresponds to one (or batch of) eigenvalue(s) and contains the eigenstate data. For batched
            computations, each State object contains all batch elements for that particular eigenstate.
        residual_norms: Achieved solver residuals with shape
            ``[num_eigvals, batch_size]``, always returned as a NumPy array.
            For full-state eigen-decomposition, each value is the global
            eigenpair residual ``||A*x - lambda*x||``. For ground-state DMRG,
            it is the achieved local Krylov residual from the final local
            update, not the global MPS eigenpair residual. For shift-invert
            DMRG, it is the maximum achieved absolute post-insertion (and, for
            a 2-site update, post-truncation) local normal-equation residual
            over the final complete fitting sweep of the final outer
            iteration; it does not certify outer convergence.
    """
    evals: NDArrayType
    evecs: Sequence[State]
    residual_norms: np.ndarray


class EigenDecomposition:
    """
    EigenDecomposition(operator, which="SA", is_hermitian=True, scope="full", approach="krylov", split="dmrg", scope_config=None, approach_config=None)

    Eigen-decomposition of a quantum operator.

    Args:
        operator: The :class:`Operator` whose eigenvalues and eigenstates are to be computed.
        which: Specifies which eigenvalues to compute. Accepted values are:

            - ``"SA"``: Smallest Algebraic - eigenvalues with smallest real parts,
              ordered ascending. For shift-invert DMRG this inherited default means
              the eigenvalue nearest ``target_energy``.
            - ``"LA"``: Largest Algebraic - eigenvalues with largest real parts, ordered descending

            A ``cudm.EigenDecompositionSpectrumKind`` enum value is also accepted.
        is_hermitian: Whether the operator is Hermitian. Currently only Hermitian operators are supported.
        scope: Decomposition scope. Accepts ``"full"``, ``"split"``, or
            a ``cudm.EigenDecompositionScopeKind`` enum value.
        approach: Decomposition approach. Accepts ``"krylov"``, ``"linear"`` or
            a ``cudm.EigenDecompositionApproachKind`` enum value. The ``"linear"``
            (iterative linear solver) approach is used by shift-invert DMRG.
        split: Split-scope kind (only relevant when ``scope="split"``). Accepts
            ``"dmrg"``, ``"shift_invert_dmrg"`` or a
            ``cudm.EigenDecompositionScopeSplitKind`` enum value. Shift-invert DMRG
            pairs ``scope="split"``, ``split="shift_invert_dmrg"``, ``approach="linear"``.
        scope_config: Optional scope-specific configuration
            (:class:`DMRGConfig`; shared by both split kinds).
        approach_config: Optional approach-specific configuration
            (:class:`EigenDecompositionApproachKrylovConfig` or
            :class:`EigenDecompositionApproachLinearConfig`). Its class must
            match the selected ``approach``.
    """

    def __init__(
        self,
        operator: Operator,
        which: Union[str, "cudm.EigenDecompositionSpectrumKind"] = "SA",
        is_hermitian: bool = True,
        scope: Union[str, "cudm.EigenDecompositionScopeKind"] = "full",
        approach: Union[str, "cudm.EigenDecompositionApproachKind"] = "krylov",
        split: Union[str, "cudm.EigenDecompositionScopeSplitKind"] = "dmrg",
        scope_config: Union[DMRGConfig, None] = None,
        approach_config: Union[
            EigenDecompositionApproachKrylovConfig, EigenDecompositionApproachLinearConfig, None
        ] = None,
    ) -> None:
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()

        self.operator = operator
        self._is_hermitian = is_hermitian
        if not is_hermitian:
            raise NotImplementedError(
                "EigenDecomposition does not currently support non-Hermitian operators."
            )

        self._scope_kind = _resolve_enum(scope, _SCOPE_KIND_MAP, "scope")
        self._approach_kind = _resolve_enum(approach, _APPROACH_KIND_MAP, "approach")
        self._split_kind = _resolve_enum(split, _SPLIT_KIND_MAP, "split")
        if (
            self._scope_kind != cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_SPLIT
            and self._split_kind
            != cudm.EigenDecompositionScopeSplitKind.EIGEN_SCOPE_SPLIT_DMRG
        ):
            raise ValueError(
                "split is only meaningful when scope='split'; "
                "non-default split kinds cannot be used with another scope."
            )
        if (
            self._scope_kind == cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_FULL
            and self._approach_kind
            != cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_KRYLOV
        ):
            raise ValueError("scope='full' requires approach='krylov'.")
        if self._scope_kind == cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_SPLIT:
            shift_invert = (
                self._split_kind
                == cudm.EigenDecompositionScopeSplitKind.EIGEN_SCOPE_SPLIT_SHIFT_INVERT_DMRG
            )
            dmrg = (
                self._split_kind
                == cudm.EigenDecompositionScopeSplitKind.EIGEN_SCOPE_SPLIT_DMRG
            )
            linear = (
                self._approach_kind
                == cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_LINEAR
            )
            krylov = (
                self._approach_kind
                == cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_KRYLOV
            )
            if shift_invert and not linear:
                raise ValueError(
                    "split='shift_invert_dmrg' requires approach='linear'."
                )
            if dmrg and not krylov:
                raise ValueError("split='dmrg' requires approach='krylov'.")

        self._which = which
        is_shift_invert = (
            self._scope_kind
            == cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_SPLIT
            and self._split_kind
            == cudm.EigenDecompositionScopeSplitKind.EIGEN_SCOPE_SPLIT_SHIFT_INVERT_DMRG
            and self._approach_kind
            == cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_LINEAR
        )
        if is_shift_invert and which == "SA":
            self._spectrum_kind = (
                cudm.EigenDecompositionSpectrumKind.EIGEN_SPECTRUM_SMALLEST
            )
        else:
            self._spectrum_kind = _resolve_enum(
                which, _SPECTRUM_KIND_MAP, "which"
            )
        if (
            is_shift_invert
            and self._spectrum_kind
            != cudm.EigenDecompositionSpectrumKind.EIGEN_SPECTRUM_SMALLEST
        ):
            raise ValueError(
                "Shift-invert DMRG requires the smallest-magnitude spectrum "
                "selection (the eigenvalue nearest target_energy)."
            )

        if self._batch_size != 1:
            raise NotImplementedError(
                "EigenDecomposition does not currently support batched operators."
            )

        self._scope_config = scope_config
        self._validate_approach_config(approach_config)
        self._approach_config = approach_config

        self._ctx: Optional[WorkStream] = None
        self._ptr = None
        self._last_compute_event: Optional[cp.cuda.Event] = None
        self._upstream_finalizers = collections.OrderedDict()
        self._requires_configuration = scope_config is not None or approach_config is not None
        self._current_compute_type: Optional[str] = None

    @property
    def _valid_state(self):
        return self._finalizer.alive

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise InvalidObjectState(
                "The EigenDecomposition instance cannot be used after resources are freed."
            )

    @property
    @nvmath_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr

    def _sync(self):
        if self._last_compute_event is not None:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    @property
    def _batch_size(self) -> int:
        return self.operator._batch_size

    @property
    def is_hermitian(self) -> bool:
        """Whether the operator is Hermitian."""
        return self._is_hermitian

    @property
    def which(self) -> Union[str, "cudm.EigenDecompositionSpectrumKind"]:
        """Which eigenvalues this solver computes."""
        return self._which

    def configure(
        self,
        scope_config: Union[DMRGConfig, None] = None,
        approach_config: Union[
            EigenDecompositionApproachKrylovConfig, EigenDecompositionApproachLinearConfig, None
        ] = None,
    ) -> None:
        """
        Update the eigen-decomposition configuration.

        :meth:`prepare` must be invoked before the next :meth:`compute` call.

        Args:
            scope_config: Scope-specific configuration (:class:`DMRGConfig`).
            approach_config: Approach-specific configuration
                (:class:`EigenDecompositionApproachKrylovConfig` or
                :class:`EigenDecompositionApproachLinearConfig`).
        """
        self._validate_approach_config(approach_config)
        apply_scope_config = scope_config is not None
        apply_approach_config = approach_config is not None
        if scope_config is not None:
            self._scope_config = scope_config
        if approach_config is not None:
            self._approach_config = approach_config
        if scope_config is None and approach_config is None:
            return
        if self._valid_state:
            self._apply_configs(
                apply_split_kind=False,
                apply_scope_config=apply_scope_config,
                apply_approach_config=apply_approach_config,
            )
        else:
            self._requires_configuration = True

    def _validate_approach_config(
        self,
        approach_config: Union[
            EigenDecompositionApproachKrylovConfig,
            EigenDecompositionApproachLinearConfig,
            None,
        ],
    ) -> None:
        """Validate that the approach configuration matches the selected approach."""
        if approach_config is None:
            return
        if self._approach_kind == cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_KRYLOV:
            expected_type = EigenDecompositionApproachKrylovConfig
            approach_name = "krylov"
        else:
            expected_type = EigenDecompositionApproachLinearConfig
            approach_name = "linear"
        if not isinstance(approach_config, expected_type):
            raise TypeError(
                f"approach={approach_name!r} requires approach_config to be an "
                f"{expected_type.__name__} instance, got "
                f"{type(approach_config).__name__}."
            )

    def _resolve_target_energy(self, target_energy: Optional[float]) -> float:
        """Validate sigma for shift-invert and reject it for every other mode."""
        is_shift_invert = (
            self._scope_kind == cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_SPLIT
            and self._split_kind
            == cudm.EigenDecompositionScopeSplitKind.EIGEN_SCOPE_SPLIT_SHIFT_INVERT_DMRG
            and self._approach_kind
            == cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_LINEAR
        )
        if not is_shift_invert:
            if target_energy is not None:
                raise ValueError(
                    "target_energy is only valid for shift-invert DMRG."
                )
            return 0.0
        if target_energy is None:
            raise ValueError(
                "target_energy must be explicitly supplied for shift-invert DMRG."
            )
        if np.iscomplexobj(target_energy):
            raise ValueError("target_energy must be a finite real scalar.")
        try:
            target_energy = float(target_energy)
        except (TypeError, ValueError) as error:
            raise ValueError("target_energy must be a finite real scalar.") from error
        if not np.isfinite(target_energy):
            raise ValueError("target_energy must be finite.")
        return target_energy

    def _configure_ed_attribute(self, attr_enum, config_ptr):
        """Pass a config handle to the eigen-decomposition object via eigen_decomposition_configure."""
        handle = self._ctx._handle._validated_ptr
        dtype = cudm.get_eigen_decomposition_attribute_dtype(attr_enum)
        val_arr = np.array([config_ptr], dtype=dtype)
        cudm.eigen_decomposition_configure(
            handle, self._ptr, attr_enum, val_arr.ctypes.data, val_arr.dtype.itemsize,
        )

    def _apply_configs(
        self,
        *,
        apply_split_kind: bool = True,
        apply_scope_config: bool = True,
        apply_approach_config: bool = True,
    ):
        """Materialize sub-config C handles, attach them, and destroy them."""
        handle = self._ctx._handle._validated_ptr

        # --- Split kind (DMRG vs shift-invert DMRG) ---
        # Only meaningful for split scope. Set before attaching the shared DMRG scope
        # config so a shift-invert selection is not clobbered by the DMRG_CONFIG attach.
        if (
            apply_split_kind
            and self._scope_kind
            == cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_SPLIT
        ):
            self._configure_ed_attribute(
                cudm.EigenDecompositionAttribute.EIGEN_SPLIT_SCOPE_KIND,
                int(self._split_kind),
            )

        # --- Scope (DMRG) config ---
        if apply_scope_config and self._scope_config is not None:
            if not isinstance(self._scope_config, DMRGConfig):
                raise TypeError(
                    f"scope_config must be a DMRGConfig instance, got "
                    f"{type(self._scope_config).__name__}."
                )
            dmrg_ptr = cudm.create_eigen_decomposition_scope_split_dmrg_config(handle)
            try:
                _DMRG_SCALAR_FIELDS = {
                    "num_sites": cudm.EigenDecompositionScopeSplitDMRGConfigAttribute.EIGEN_SPLIT_SCOPE_DMRG_NUM_SITES,
                    "max_sweeps": cudm.EigenDecompositionScopeSplitDMRGConfigAttribute.EIGEN_SPLIT_SCOPE_DMRG_MAX_SWEEPS,
                    "energy_tolerance": cudm.EigenDecompositionScopeSplitDMRGConfigAttribute.EIGEN_SPLIT_SCOPE_DMRG_ENERGY_TOLERANCE,
                    "max_power_iterations": cudm.EigenDecompositionScopeSplitDMRGConfigAttribute.EIGEN_SPLIT_SCOPE_DMRG_MAX_POWER_ITERATIONS,
                }
                for field_name, enum_val in _DMRG_SCALAR_FIELDS.items():
                    value = getattr(self._scope_config, field_name)
                    if value is not None:
                        _set_config_attribute(
                            cudm.eigen_decomposition_scope_split_dmrg_config_set_attribute,
                            cudm.get_eigen_decomposition_scope_split_dmrg_config_attribute_dtype,
                            handle, dmrg_ptr, enum_val, value,
                        )
                # SVDConfig nested attach 
                if self._scope_config.svd_config is not None:
                    if not isinstance(self._scope_config.svd_config, SVDConfig):
                        raise TypeError(
                            f"DMRGConfig.svd_config must be an SVDConfig instance, got "
                            f"{type(self._scope_config.svd_config).__name__}."
                        )
                    svd_ptr = _build_svd_config_handle(handle, self._scope_config.svd_config)
                    try:
                        _set_config_attribute(
                            cudm.eigen_decomposition_scope_split_dmrg_config_set_attribute,
                            cudm.get_eigen_decomposition_scope_split_dmrg_config_attribute_dtype,
                            handle, dmrg_ptr,
                            cudm.EigenDecompositionScopeSplitDMRGConfigAttribute.EIGEN_SPLIT_SCOPE_DMRG_SVD_CONFIG,
                            svd_ptr,
                        )
                    finally:
                        cudm.destroy_svd_config(svd_ptr)

                self._configure_ed_attribute(
                    cudm.EigenDecompositionAttribute.EIGEN_SPLIT_SCOPE_DMRG_CONFIG, dmrg_ptr,
                )
            finally:
                cudm.destroy_eigen_decomposition_scope_split_dmrg_config(dmrg_ptr)

        # --- Approach config (Krylov eigensolver or Linear solver) ---
        if apply_approach_config and self._approach_config is not None:
            if isinstance(self._approach_config, EigenDecompositionApproachKrylovConfig):
                krylov_ptr = cudm.create_eigen_decomposition_approach_krylov_config(handle)
                try:
                    _KRYLOV_FIELDS = {
                        "max_dim": cudm.EigenDecompositionApproachKrylovConfigAttribute.EIGEN_APPROACH_KRYLOV_MAX_DIM,
                        "max_restarts": cudm.EigenDecompositionApproachKrylovConfigAttribute.EIGEN_APPROACH_KRYLOV_MAX_RESTARTS,
                        "min_block_size": cudm.EigenDecompositionApproachKrylovConfigAttribute.EIGEN_APPROACH_KRYLOV_MIN_BLOCK_SIZE,
                    }
                    for field_name, enum_val in _KRYLOV_FIELDS.items():
                        value = getattr(self._approach_config, field_name)
                        if value is not None:
                            _set_config_attribute(
                                cudm.eigen_decomposition_approach_krylov_config_set_attribute,
                                cudm.get_eigen_decomposition_approach_krylov_config_attribute_dtype,
                                handle, krylov_ptr, enum_val, value,
                            )
                    self._configure_ed_attribute(
                        cudm.EigenDecompositionAttribute.EIGEN_APPROACH_KRYLOV_CONFIG, krylov_ptr,
                    )
                finally:
                    cudm.destroy_eigen_decomposition_approach_krylov_config(krylov_ptr)
            elif isinstance(self._approach_config, EigenDecompositionApproachLinearConfig):
                linear_ptr = cudm.create_eigen_decomposition_approach_linear_config(handle)
                try:
                    _LINEAR_FIELDS = {
                        "max_iterations": cudm.EigenDecompositionApproachLinearConfigAttribute.EIGEN_APPROACH_LINEAR_MAX_ITERATIONS,
                    }
                    for field_name, enum_val in _LINEAR_FIELDS.items():
                        value = getattr(self._approach_config, field_name)
                        if value is not None:
                            _set_config_attribute(
                                cudm.eigen_decomposition_approach_linear_config_set_attribute,
                                cudm.get_eigen_decomposition_approach_linear_config_attribute_dtype,
                                handle, linear_ptr, enum_val, value,
                            )
                    self._configure_ed_attribute(
                        cudm.EigenDecompositionAttribute.EIGEN_APPROACH_LINEAR_CONFIG, linear_ptr,
                    )
                finally:
                    cudm.destroy_eigen_decomposition_approach_linear_config(linear_ptr)
            else:
                raise TypeError(
                    f"approach_config must be an EigenDecompositionApproachKrylovConfig or "
                    f"EigenDecompositionApproachLinearConfig instance, got "
                    f"{type(self._approach_config).__name__}."
                )

        self._requires_configuration = False

    def _maybe_instantiate(self, ctx: WorkStream) -> None:
        if self._valid_state:
            if self._ctx != ctx:
                raise ValueError(
                    "EigenDecomposition objects can only be used with a single WorkStream. "
                    "Switching WorkStream is not supported."
                )
            return

        self._ctx = ctx
        self.operator._maybe_instantiate(ctx)

        self._ptr = cudm.create_eigen_decomposition(
            self._ctx._handle._validated_ptr,
            self.operator._validated_ptr,
            int(self._is_hermitian),
            self._spectrum_kind,
            self._scope_kind,
            self._approach_kind,
        )

        self._apply_configs()

        self._finalizer = weakref.finalize(
            self,
            utils.generic_finalizer,
            self._ctx.logger,
            self._upstream_finalizers,
            (cudm.destroy_eigen_decomposition, self._ptr),
            msg=f"Destroying EigenDecomposition instance {self}, ptr: {self._ptr}",
        )
        utils.register_with(self, self._ctx, self._ctx.logger)
        utils.register_with(self, self.operator, self._ctx.logger)

    def prepare(
        self,
        ctx: WorkStream,
        repr_state: State,
        max_num_eigvals: int = 1,
        compute_type: Optional[str] = None,
    ) -> None:
        """
        Prepare the eigen-decomposition for computation.

        Args:
            ctx: Library context containing workspace, stream and other configuration. See :class:`WorkStream`.
            repr_state: Representative quantum state used to size the plan.
            max_num_eigvals: Maximum number of eigenpairs that subsequent
                :meth:`compute` calls may request.
            compute_type: CUDA compute type string (e.g. ``"complex128"``).
                Defaults to ``ctx.compute_type`` when set, otherwise to the
                operator's dtype.
        """
        if max_num_eigvals < 1:
            raise ValueError(
                f"max_num_eigvals must be >= 1, got {max_num_eigvals}."
            )

        if not self._valid_state:
            self._maybe_instantiate(ctx)
        else:
            if self._ctx != ctx:
                raise ValueError(
                    "EigenDecomposition objects can only be used with a single WorkStream. "
                    "Switching WorkStream is not supported."
                )

        if self._requires_configuration:
            self._apply_configs()

        if repr_state.batch_size != self._batch_size:
            raise ValueError(
                f"Batch size of representative state ({repr_state.batch_size}) does not "
                f"match the operator batch size ({self._batch_size})."
            )
        if self.operator.hilbert_space_dims != repr_state.hilbert_space_dims:
            raise ValueError(
                f"Hilbert space dimensions of Operator, {self.operator.hilbert_space_dims}, "
                f"and representative State, {repr_state.hilbert_space_dims}, do not match."
            )

        default_compute_type = (
            self._ctx.compute_type if self._ctx.compute_type is not None else self.operator.dtype
        )
        self._current_compute_type = compute_type if compute_type else default_compute_type

        cudm.eigen_decomposition_prepare(
            self._ctx._handle._validated_ptr,
            self._ptr,
            max_num_eigvals,
            repr_state._validated_ptr,
            CUDENSITYMAT_COMPUTE_TYPE_MAP[self._current_compute_type],
            self._ctx._memory_limit,
            self._ctx._validated_ptr,
            0,
        )
        self._work_size, _ = self._ctx._update_required_size_upper_bound()
        self._max_num_eigvals = max_num_eigvals

    def compute(
        self,
        t: float,
        params: Union[NDArrayType, Sequence[float], None],
        states: Sequence[State],
        tol: Optional[Union[float, np.ndarray]] = None,
        target_energy: Optional[float] = None,
    ) -> EigenDecompositionResult:
        """
        Compute eigenpairs of the operator.

        Args:
            t: Current time value passed to all callback functions.
            params: Additional callback parameters. Element type must be ``float64``.
            states: Initial state or block of linearly-independent initial
                states for the selected iterative solver. The states are
                updated in place with the final eigenstates.
                The number of requested eigenstates is equal to the number of provided initial states.
            tol: Requested solver tolerance. For full-state eigen-decomposition,
                this is the global eigenpair residual tolerance; for ground-state
                DMRG, the local Krylov residual tolerance; and for shift-invert
                DMRG, the per-site relative normal-equation residual tolerance.
                Can be a scalar or an array of shape
                ``[num_eigvals, batch_size]``. If ``None``, the square root of
                machine precision is used.
            target_energy: Target energy sigma for the shift-invert DMRG split
                kind. It must be explicitly supplied, finite, and real. The returned
                eigenpair is the one whose eigenvalue is nearest sigma.
                Convergence requires a nonexact target uniquely nearest to one
                distinct eigenvalue.
                Supplying a target for any other decomposition mode is an error.
                Changing only this value after a successful Prepare requires
                no Configure or Prepare.

        Returns:
            :class:`EigenDecompositionResult`: Dataclass holding the requested
                eigenvalues, in-place-updated eigenstates, and solver-defined
                residual outputs.
        """
        if self._ctx is None:
            raise RuntimeError(
                "This EigenDecomposition has not been used with a WorkStream. "
                "Call its ``prepare`` method once before calls to ``compute``."
            )
        _ = self._validated_ptr

        target_energy = self._resolve_target_energy(target_energy)

        for state in states:
            if self._ctx != state._ctx:
                raise ValueError(
                    "This EigenDecomposition's WorkStream and an input state's WorkStream do not match."
                )
            if self._batch_size != state.batch_size:
                raise ValueError(
                    f"Inconsistent input state batch size. Expected {self._batch_size}, "
                    f"got {state.batch_size}."
                )

        num_eigvals = len(states)
        if self._max_num_eigvals is not None and self._max_num_eigvals < num_eigvals:
            warnings.warn(
                "Performing compute with more requested eigenstates than the prior "
                "prepare's max_num_eigvals is undocumented behavior and may raise "
                "in future releases.",
                UserWarning,
            )

        if (
            self._scope_kind
            != cudm.EigenDecompositionScopeKind.EIGEN_SCOPE_SPLIT
            or self._split_kind
            != cudm.EigenDecompositionScopeSplitKind.EIGEN_SCOPE_SPLIT_SHIFT_INVERT_DMRG
            or self._approach_kind
            != cudm.EigenDecompositionApproachKind.EIGEN_APPROACH_LINEAR
        ):
            self.prepare(
                self._ctx, states[0], num_eigvals, self._current_compute_type
            )

        self._ctx._maybe_allocate()

        with nvmath_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):
            params, num_params, _params_batch_size = _handle_callback_params(
                params, self._batch_size,
            )
            params_ptr = wrap_operand(params).data_ptr

            evals_shape = (num_eigvals, self._batch_size)
            evals_dtype_internal = self.operator.dtype
            evals_dtype_external = "float64" if self._is_hermitian else "complex128"
            # The eigenvalues buffer is in/out: the shift-invert DMRG split kind reads the
            # target energy sigma from it on input (other split kinds/approaches ignore the
            # input value), and every path overwrites it with the computed eigenvalue(s) on
            # output. A real sigma is stored in the complex slot with a zero imaginary part.
            evals_internal = cp.full(
                evals_shape, target_energy, dtype=evals_dtype_internal, order="F"
            )
            evals_ptr = evals_internal.data.ptr
            if evals_dtype_internal != evals_dtype_external:
                evals_external = cp.zeros(evals_shape, dtype=evals_dtype_external, order="F")
            else:
                evals_external = evals_internal

            if tol is None:
                tol = float(np.sqrt(np.finfo(np.dtype(self.operator.dtype)).eps))
            if isinstance(tol, (int, float)):
                tol_array = np.full(evals_shape, float(tol), dtype="float64", order="F")
            else:
                tol_array = np.asarray(tol, dtype="float64")
                if tol_array.shape != evals_shape:
                    raise ValueError(
                        f"tol must be a scalar or have shape {evals_shape}, "
                        f"got shape {tol_array.shape}."
                    )
                tol_array = np.ascontiguousarray(tol_array, dtype="float64").copy(order="F")
            output_tol_array = tol_array
            tol_ptr = output_tol_array.ctypes.data

            self._ctx._last_compute_event = self._last_compute_event
            for state in states:
                state._last_compute_event = self._last_compute_event
            self.operator._update_last_compute_event_downstream(self._last_compute_event)

            cudm.eigen_decomposition_compute(
                self._ctx._handle._validated_ptr,
                self._ptr,
                t,
                self._batch_size,
                num_params,
                params_ptr,
                num_eigvals,
                [state._validated_ptr for state in states],
                evals_ptr,
                tol_ptr,
                self._ctx._validated_ptr,
                self._ctx._stream_holder.ptr,
            )

        evals_result = evals_external
        if evals_dtype_internal != evals_dtype_external:
            if not self._is_hermitian:
                raise RuntimeError(
                    "Non-Hermitian operators must use a complex numerical dtype."
                )
            evals_external[:] = evals_internal.real
            if not cp.allclose(evals_internal.imag, 0.0):
                warnings.warn(
                    "Hermitian operator produced eigenvalues with non-negligible "
                    "imaginary parts.",
                    RuntimeWarning,
                )

        return EigenDecompositionResult(
            evals=evals_result,
            evecs=list(states),
            residual_norms=output_tol_array,
        )
