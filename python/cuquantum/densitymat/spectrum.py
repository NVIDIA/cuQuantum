# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


from typing import Iterable, Optional, Tuple, List, Set, Union, Sequence, Callable, Any
from numbers import Number
import collections
import dataclasses
from dataclasses import dataclass

import numpy as np
import cupy as cp
import weakref
import warnings

from nvmath.internal import utils as nvmath_utils
from nvmath.internal import typemaps

from cuquantum.bindings import cudensitymat as cudm
from .state import DensePureState
from .work_stream import WorkStream
from .operators import Operator
from ._internal import utils
from ._internal.typemaps import CUDENSITYMAT_COMPUTE_TYPE_MAP
from ._internal.utils import NDArrayType, InvalidObjectState, check_and_get_batchsize, wrap_operand
from .operators import _handle_callback_params

__all__ = ["OperatorSpectrumResult", "OperatorSpectrumConfig", "OperatorSpectrumSolver"]

WHICH_EIGENVALUE_MAP = {
    "SA": cudm.OperatorSpectrumKind.OPERATOR_SPECTRUM_SMALLEST_REAL, # Smallest Algebraic (real part)
    "LA": cudm.OperatorSpectrumKind.OPERATOR_SPECTRUM_LARGEST_REAL,  # Largest Algebraic (real part)
}

@dataclass
class OperatorSpectrumResult:
    """A data class for capturing the results of spectral computation from the :class:`OperatorSpectrumSolver` eigensolver.

    This class encapsulates all outputs from the eigenvalue/eigenstate computation, including
    convergence information and residual norms for analysis of solution quality.

    Attributes:
        evals: The computed eigenvalues as a 1D array. For batched computations,
            this will be a 2D array with shape ``[num_eigvals, batch_size]``. Elements are ordered 
            according to the ``which`` parameter used in the solver initialization.
        evecs: The computed eigenstates as a sequence of :class:`DensePureState` objects. 
            Each state corresponds to one (or batch of) eigenvalue(s) and contains the eigenstate data. For batched 
            computations, each State object contains all batch elements for that particular eigenstate.
        residual_norms: The residual norms ``||A*x - lambda*x||`` for each computed 
            eigenvalue-eigenstate pair. Shape is ``[num_eigvals, batch_size]`` where smaller values 
            indicate better convergence. Always returned as a numpy array regardless of the backend used. 
    """
    evals: NDArrayType
    evecs: Sequence[DensePureState]
    residual_norms: np.ndarray # shape: [num_eigvals, batch_size]  


@dataclass
class OperatorSpectrumConfig:
    """A data class for providing configuration options to the :class:`OperatorSpectrumSolver` eigensolver.

    Attributes:
        min_krylov_block_size: Minimum number of Krylov subspace vectors to use in the block iterative method.
            A larger value may improve convergence but increases memory usage and computational cost.
            If not specified, a default value will be chosen. Defaults to 1.
        max_buffer_ratio: Maximum ratio of the total number of blocks in the Krylov subspace to the number of requested eigenvalues.
            If not specified, a default value will be chosen. Must be greater than 1. Defaults to 5.
        max_restarts: Maximum number of restart cycles allowed during the iterative eigenvalue computation.
            If not specified, a default value will be chosen. Defaults to 20.
    """
    min_krylov_block_size: Optional[int] = None
    max_buffer_ratio: Optional[int] = None
    max_restarts: Optional[int] = None
    
    def _check_int(self, attribute, name, min_value=0):
        message = f"Invalid value ({attribute}) for '{name}'. Expect non-zero integer or None."
        if not isinstance(attribute, (type(None), int)):
            raise ValueError(message)
        if isinstance(attribute, int) and not attribute > min_value: 
            raise ValueError(message)

    def __post_init__(self):
        self._check_int(self.min_krylov_block_size, "min_krylov_block_size",0)
        self._check_int(self.max_buffer_ratio, "max_buffer_ratio",1)
        self._check_int(self.max_restarts, "max_restarts",0)
    
    @classmethod
    def _option_to_enum(cls, name):
        return {
            "min_krylov_block_size": cudm.OperatorSpectrumConfig.MIN_BLOCK_SIZE,
            "max_buffer_ratio": cudm.OperatorSpectrumConfig.MAX_EXPANSION,
            "max_restarts": cudm.OperatorSpectrumConfig.MAX_RESTARTS,
        }[name]


class OperatorSpectrumSolver:
    """
    Eigenvalue solver for computing the spectrum (eigenvalues and eigenstates) of quantum operators.
    
    This class provides an interface for finding eigenvalues and eigenstates of :class:`Operator` objects
    using iterative methods.

    Args:
        operator: The quantum operator whose eigenvalues and eigenstates are to be computed.
            Must be an instance of :class:`Operator`.
        which: Specifies which eigenvalues to compute. Accepted values are:
            - ``"SA"``: Smallest Algebraic - eigenvalues with smallest real parts, ordered ascending
            - ``"LA"``: Largest Algebraic - eigenvalues with largest real parts, ordered descending
        hermitian: Whether the operator is Hermitian. This affects the choice of eigenvalue algorithm
            and can improve performance for Hermitian operators. Currently, only Hermitian operators are supported.
        config: Specify options for the eigensolver as a :class:`OperatorSpectrumConfig` object. Alternatively, a ``dict``
            containing the parameters for the :class:`OperatorSpectrumConfig` constructor can also be provided. If not specified,
            the defaults documented in :class:`OperatorSpectrumConfig` are used.
    """
    def __init__(self,
        operator: Operator,
        which: str,
        hermitian: bool,
        config: Union[OperatorSpectrumConfig, dict] | None = None
        ) -> None:
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()
        self._ctx: "WorkStream" = None
        self._ptr = None
        self.operator = operator
        self._which = which
        self._is_hermitian = hermitian
        if not hermitian:
            raise NotImplementedError("OperatorSpectrumSolver does not currently support non-Hermitian operators.")
        self._last_compute_event = None
        if self._batch_size != 1:
            raise NotImplementedError("OperatorSpectrumSolver does not currently support batched operators.")
        self._upstream_finalizers = collections.OrderedDict()
        self._info = {}
        if config is None:
            self._requires_configuration = False
            config = OperatorSpectrumConfig()
        else:
            self._requires_configuration = True
        config = nvmath_utils.check_or_create_options(OperatorSpectrumConfig, config, "OperatorSpectrumConfig")
        self._config = config
        
    def configure(self, config: Union[OperatorSpectrumConfig, dict]) -> None:
        """
        Configure the eigensolver.

        Args:
            config: Configuration for the eigensolver. Can be instance of :class:`OperatorSpectrumConfig` or ``dict``.
        """
        config = nvmath_utils.check_or_create_options(OperatorSpectrumConfig, config, "OperatorSpectrumConfig")
        if utils.check_equal_config(self._config, config):
            pass
        else: 
            self._config = config
            self._requires_configuration = True
        
    def _set_spectrum_config_options(self):
        """
        Set OperatorSpectrumConfig options if the value is not None.

        Args:
            options: An OperatorSpectrumConfig object.
        """
        for field in dataclasses.fields(self._config):
            name, value = field.name, getattr(self._config, field.name)
            if value is None:
                continue

            enum = self._config._option_to_enum(name)
            self._set_spectrum_config_option(name, enum, value)
        self._requires_configuration = False

    def _set_spectrum_config_option(self, name, enum, value):
        """
        Set a single OperatorSpectrumConfig option if the value is not None.

        Args:
            name: The name of the attribute.
            enum: A OperatorSpectrumConfigAttribute to set.
            value: The value to which the attribute is set to.
        """
        dtype = cudm.get_operator_spectrum_config_dtype(enum) 
        value = np.array((value,), dtype=dtype)
        cudm.operator_spectrum_configure(self._ctx._handle._validated_ptr, self._validated_ptr, enum, value.ctypes.data, value.dtype.itemsize)
        self._ctx.logger.info(f"The operator spectrum config attribute '{name}' has been set to {value[0]}.")
    
    def _maybe_instantiate(self, ctx: WorkStream):
        if self._valid_state:
            if self._ctx != ctx:
                raise ValueError(
                    "OperatorSpectrumSolver objects can only be used with a single WorkStream, and this instance was originally used with another WorkStream. Switching WorkStream is not supported."
                )
        else: 
            self._ctx = ctx  
            self.operator._maybe_instantiate(ctx)
            self._ptr = cudm.create_operator_spectrum(
                    self._ctx._handle._validated_ptr,
                    self.operator._validated_ptr,
                    self._is_hermitian,
                    WHICH_EIGENVALUE_MAP[self._which]
                )
            self._finalizer = weakref.finalize(
                    self,
                    utils.generic_finalizer,
                    self._ctx.logger,
                    self._upstream_finalizers,
                    (cudm.destroy_operator_spectrum, self._ptr),
                    msg=f"Destroying OperatorSpectrumSolver instance {self}, ptr: {self._ptr}",
                )
            utils.register_with(self, self._ctx, self._ctx.logger)

    def _sync(self):
        if self._last_compute_event is not None:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    def _check_valid_state(self, *args, **kwargs):
        """ """
        if not self._valid_state:
            raise InvalidObjectState("The OperatorSpectrumSolver cannot be used after resources have been freed!")

    @property
    def _valid_state(self):
        return self._finalizer.alive 

    @property
    @nvmath_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        """
        The pointer to this instance's C-API counterpart.
        """
        return self._ptr 

    def prepare(
        self,
        ctx: WorkStream,
        repr_state: DensePureState,
        max_num_eigvals: int = 1,
        ) -> None:
        """
        Prepare the eigensolver for computation.

        Args:
            ctx: :class:`WorkStream` to use for computation. This includes library handle, workspace buffer allocation as well stream on which the computation will be performed.
            repr_state: Representative state to use for computation.
            max_num_eigvals: Maximum number of eigenvalues to compute.
        """
        if not self._valid_state:
            self._maybe_instantiate(ctx)
        else:
            if self._ctx != ctx:
                raise ValueError(
                    "OperatorSpectrumSolver objects can only be used with a single WorkStream, and this instance was originally used with another WorkStream. Switching WorkStream is not supported."
                ) 
        if self._batch_size != 1:
            raise NotImplementedError("OperatorSpectrumSolver does not currently support batched operators.")
        if repr_state.batch_size != self._batch_size:
            raise ValueError(f"Inconsistent batch size of representative state {repr_state.batch_size} and operator {self._batch_size}. For spectrum computation, the batch size of the representative state must match the batch size of the operator.")
        _compute_type = CUDENSITYMAT_COMPUTE_TYPE_MAP[self.operator.dtype]
        if self._requires_configuration:
            self._set_spectrum_config_options()
        cudm.operator_spectrum_prepare(
            self._ctx._handle._validated_ptr,
            self._validated_ptr,
            max_num_eigvals,
            repr_state._validated_ptr,
            _compute_type,  
            self._ctx._memory_limit,
            self._ctx._validated_ptr,
            0)

        self._work_size, _ = self._ctx._update_required_size_upper_bound()
        self._max_num_eigvals = max_num_eigvals
        self._ctx = ctx

    def compute(
        self,
        t: float,
        params: NDArrayType | Sequence[float] | None,
        states: Sequence[DensePureState],
        tol: Optional[Union[float, np.ndarray]] = None,
        ) -> OperatorSpectrumResult:
        """
        Compute the spectrum of the provided :class:`Operator` using block Krylov iteration. This function is blocking, i.e. it will only return after the computation is complete.

        Args:
            t: Time argument to be passed to all callback functions.
            params: Additional arguments to be passed to all callback functions. The element type is required to be float (i.e ``"float64"`` for arrays).
                If batched operators or coefficients are used, they need to be passed as 2-dimensional the last dimension of which is the batch size.
                To avoid copy of the argument array, it can be passed as a fortran-contiguous GPU array.
            states: Block of linearly-independent initial states for Krylov iteration, which will be in-place updated with the final eigenstates.
                The number of requested eigenstates is equal to the number of provided initial states.
            tol: Tolerance for residuals ``||A*x - lambda*x||``. Can be either scalar or array of shape [num_eigvals] or [num_eigvals, batch_size]. If None, square root of machine precision is used.
        Returns:
            :class:`OperatorSpectrumResult`: Dataclass holding the requested eigenvalues and eigenstates of the :class:`Operator`.
        """
        if self._ctx is None:
            raise RuntimeError(
                "This instance has not been used with a ``WorkStream``, please call its ``prepare`` method once before calls to this method."
            )
        for state in states:
            if self._ctx != state._ctx:
                raise ValueError("This OperatorSpectrumSolver's WorkStream and the WorkStream of an input state do not match.")
            if self._batch_size != state.batch_size:
                raise ValueError(f"Inconsistent input state batch size. Expected {self._batch_size}, got {state.batch_size}.")
        
        if self._batch_size != 1:
            raise NotImplementedError("OperatorSpectrumSolver does not currently support batched operators.")
        if self._max_num_eigvals != None and self._max_num_eigvals < len(states):
            warnings.warn("Performing compute calls without prior prepare call for a larger or equal number of requested eigenstates is undocumented behavior and may raise an error in future releases.", UserWarning)
        
        self.prepare(self._ctx, states[0], len(states))
        
        self._ctx._maybe_allocate()

        with nvmath_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):

            params, num_params, batch_size = _handle_callback_params(params, self._batch_size) 
            params_ptr = wrap_operand(params).data_ptr

            evals_shape = (self._max_num_eigvals, self._batch_size)
            evals_dtype_internal = self.operator.dtype
            evals_dtype_external = 'float64' if self._is_hermitian else 'complex128'
            evals_internal = cp.zeros(evals_shape, dtype=evals_dtype_internal)
            evals_ptr = evals_internal.data.ptr
            if evals_dtype_internal != evals_dtype_external:
                evals_external = cp.zeros(evals_shape, dtype=evals_dtype_external)
            else:
                evals_external = evals_internal
            if tol is None:
                tol = np.sqrt(np.finfo(np.dtype(self.operator.dtype)).eps)
            if isinstance(tol, float):
                tol_shape = (self._max_num_eigvals, self._batch_size)
                tol_dtype = 'float64' 
                tol_array = np.full(tol_shape, tol, dtype=tol_dtype)
            else:
                tol_array = np.asarray(tol)
            output_tol_array = tol_array.copy(order='F')
            tol_ptr = output_tol_array.ctypes.data

           
            # update last event in participating elementary/general operators to ensure proper stream synchronization and shutdown order
            self._ctx._last_compute_event = self._last_compute_event
            for state in states:
                state._last_compute_event = self._last_compute_event
            self.operator._update_last_compute_event_downstream(self._last_compute_event)

            cudm.operator_spectrum_compute(
                self._ctx._handle._validated_ptr,
                self._validated_ptr,
                t,
                self._batch_size,
                num_params,
                params_ptr,
                self._max_num_eigvals,
                [state._validated_ptr for state in states],
                evals_ptr,
                tol_ptr,
                self._ctx._validated_ptr,
                self._ctx._stream_holder.ptr,
            )

        evals_result = evals_external
        if evals_dtype_internal != evals_dtype_external:
            if not self._is_hermitian:
                raise RuntimeError("Non-hermitian operators need to be computed with complex numerical datatype.")
            evals_external[:] = evals_internal.real
            if not cp.allclose(evals_internal.imag, 0.0):
                warnings.warn("Complex hermitian operator did not generate purely real eigenvalues.", RuntimeWarning)
        evecs_result = list(states)
        residual_norms_result = output_tol_array

        return OperatorSpectrumResult(evals=evals_result, evecs=evecs_result, residual_norms=residual_norms_result)
    
    @property
    def _batch_size(self) -> int:
        return self.operator._batch_size
    
    @property
    def is_hermitian(self) -> bool:
        """
        Whether the operator is Hermitian.
        """
        return self._is_hermitian
    
    @property
    def config(self) -> OperatorSpectrumConfig:
        """
        Configuration options for this solver.
        Direct modification of the return argument by the user leads to undefined behaviour. Call :meth:`OperatorSpectrumSolver.configure` to modify the configuration.
        """
        return self._config
    
    @property
    def which(self) -> str:
        """
        Which eigenvalues this solver computes.
        """
        return self._which

 