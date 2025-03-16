# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause


from typing import Iterable, Optional, Tuple, List, Set, Union, Sequence, Callable, Any
from numbers import Number
import weakref
import collections

import numpy as np
import cupy as cp

from .._internal import typemaps as cutn_typemaps
from .._internal import utils as cutn_utils
from .._internal.tensor_wrapper import wrap_operand

from cuquantum.bindings import cudensitymat as cudm
from .elementary_operator import ElementaryOperator, DenseOperator, MultidiagonalOperator
from .matrix_operator import MatrixOperator, LocalDenseMatrixOperator
from .state import State
from .work_stream import WorkStream
from ._internal.callbacks import CallbackCoefficient
from .callbacks import Callback
from ._internal import utils
from ._internal.utils import NDArrayType, InvalidObjectState, check_and_get_batchsize

__all__ = [
    "full_matrix_product",
    "tensor_product",
    "OperatorTerm",
    "Operator",
    "OperatorAction",
]

ScalarCallbackType = Callable[[Number, Sequence], Union[Number, np.ndarray]]
CoefficientType = Union[Number, NDArrayType, Callback, Tuple[NDArrayType, Callback]]
_CoefficientTypeRuntimeCheckable = Union[Number, NDArrayType, Callback, Tuple]


class OperatorTerm:
    """
    Operator term consisting of tensor products of elementary operators.

    An :class:`OperatorTerm` containing a tensor product of elementary operators can be obtained from the free function :func:`tensor_product`. Sums of more than a single product are obtained by in-place (``+=``) or out-of-place addition (``+``) of :class:`OperatorTerm` objects.

    Args:
        dtype: Numeric data type of the underlying elementary operators' data. Defaults to ``None`` and will be inferred from the appended tensor products of elementary operators.

    .. note::
        - Scalar operators, for which no product is appended, require specification of ``dtype`` at construction.
    """

    # override to avoid calling into ndarray dundered methods instead of this classes
    __array_ufunc__ = None

    def __init__(self, dtype: Optional[str] = None):
        """
        Initialize an operator term consisting of tensor products of elementary operators.
        """
        self.terms = []
        self._term_types = []
        self.modes = []
        self.duals = []
        self._conjugations = []
        self._batch_size = 1
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()

        self._coefficients: CallbackCoefficient = []
        self._static_coefficients = []
        self._dtype: Optional[str] = dtype  # TODO: check for validity
        self._hilbert_space_dims = None
        self._ptr = None
        self._ctx: "WorkStream" = None
        self._last_compute_event = None
        self._using_ops: set[ElementaryOperator | MatrixOperator] = set()
        self._upstream_finalizers = collections.OrderedDict()

    def _check_valid_state(self, *args, **kwargs):
        if not self._valid_state:
            raise InvalidObjectState("The operator term cannot be used after resources are freed")

    @property
    def _valid_state(self) -> bool:
        return self._finalizer.alive

    @property
    def hilbert_space_dims(self) -> Tuple[int]:
        """
        Hilbert space dimensions of this `OperatorTerm`.
        """
        return self._hilbert_space_dims

    @property
    def dtype(self) -> str:
        """
        Data type of this :class:`OperatorTerm`.
        """
        return self._dtype

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        return self._ptr

    def append_elementary_product(
        self,
        elem_ops: Iterable[ElementaryOperator],
        modes: Iterable[Sequence[int]],
        duals: Iterable[Sequence[bool]],
        coeff: CoefficientType = complex(1),
        batch_size: int = 1,
    ) -> None:
        """
        Append a product of elementary operators to this operator term.

        Args:
            elem_ops: An iterable of elementary operators.
            modes: An iterable of the mode indices of the elementary operators.
            duals: Dualities of the elementary operators.
            coeff: Coefficient associated with this tensor product of elementary operators.
            batch_size: Batch size associated with this product of elementary operators.
        
        .. note::
            The relation among operators in ``elem_ops`` is matrix multiplication/tensor contraction,
            which is equivalent to tensor product when there are no overlapping modes. 
        """
        coeff = CallbackCoefficient.create(coeff, batch_size)
        conjugations = None
        self._append(elem_ops, modes, conjugations, duals, coeff)

    def append_matrix_product(
        self,
        matrix_ops: Iterable[MatrixOperator],
        conjugations: Iterable[bool],
        duals: Iterable[bool],
        coeff: CoefficientType = complex(1),
        batch_size: int = 1,
    ) -> None:
        """
        Append a product of matrix operators to this operator term.

        Args:
            matrix_ops: An iterable of matrix operators.
            conjugations: An iterable of whether each matrix operator is conjugated.
            duals: Dualities of the matrix operators.
            coeff: Coefficient associated with this tensor product of matrix operators.
            batch_size: Batch size associated with this product of matrix operators.
        
        .. note::
            The relation among operators in ``matrix_ops`` is matrix multiplication. 
        """
        coeff = CallbackCoefficient.create(coeff, batch_size)
        self._append(matrix_ops, None, conjugations, duals, coeff)

    def _append(
        self,
        ops: Iterable[ElementaryOperator] | Iterable[MatrixOperator],
        modes: Iterable[Sequence[int]] | None,
        conjugations: None | Iterable[bool],
        duals: Iterable[Sequence[bool]] | Iterable[bool],
        coeff: CallbackCoefficient,
    ):
        if self._valid_state:
            raise RuntimeError("Cannot append to OperatorTerm after its instantiate method has been called.")
        elementary_only = None
        matrix_only = None
        is_scalar_op = True
        # check pureness of operator product early
        for op in ops:
            is_scalar_op = False
            if isinstance(op, ElementaryOperator):
                if matrix_only is None:
                    elementary_only = True
                else:
                    raise ValueError("Mixed products between elementary and matrix operators are not currently supported.")
            elif isinstance(op, MatrixOperator):
                if elementary_only is None:
                    matrix_only = True
                else:
                    raise ValueError("Mixed products between elementary and matrix operators are not currently supported.")
        self._check_dtype(ops)
        if elementary_only or is_scalar_op:
            product_of = ElementaryOperator
            # check shapes
            for operand, operand_modes in zip(ops, modes):
                _shape = operand.shape
                assert len(_shape) % 2 == 0 and len(_shape) // 2 == len(operand_modes)
            elementary_only = True
        elif matrix_only:
            product_of = MatrixOperator
        # check batch sizes
        _batch_size = self._batch_size
        for operand in ops:
            _batch_size = check_and_get_batchsize(_batch_size, operand.batch_size)
        _batch_size = check_and_get_batchsize(_batch_size, coeff.batch_size)
        self._batch_size = check_and_get_batchsize(_batch_size, self._batch_size)
        # all checks passed, start inplace modification
        self.terms.append(ops)
        self.duals.append(duals)
        if elementary_only:
            self.modes.append(modes)
            self._conjugations.append(
                [
                    None,
                ]
                * len(ops)
            )
        if matrix_only:
            self.modes.append(
                [
                    None,
                ]
                * len(ops)
            )
            self._conjugations.append(conjugations)
        self._coefficients.append(coeff)
        self._term_types.append(product_of)

    def _check_dtype(self, operands):
        """
        Checks that the operands to be appended to self.term are of the same dtype, and that the latter is the same dtype as self._dtype .
        If self._dtype has not been set yet, this method will set it (unless empty operands are passed)
        """
        # handle case of empty operator
        if len(operands) == 0:
            if self._dtype is None:
                raise TypeError("OperatorTerms consisting of scalar terms need to specify a data type.")
            return
        # check consistency of operands dtypes
        dtypes = {op.dtype for op in operands}
        dtype = dtypes.pop()

        if len(dtypes) != 0:
            raise TypeError(
                "The provided operands have more than one data type, which is not supported. Please cast to same data type."
            )

        # check consistency of operands dtypes with this instances dtype
        if self._dtype is None:
            self._dtype = dtype
        elif dtype is not None:
            try:
                self._dtype != dtype
            except AssertionError as e:
                raise TypeError(
                    "The provided operands are required to have the same data type as this OperatorTerm instance."
                ) from e

    def _append_matrix_product(
        self,
        matrix_ops: Sequence[MatrixOperator],
        conjugations: Sequence[bool],
        duals: Sequence[bool],
        coeff: CallbackCoefficient,
        hilbert_space_dims: Optional[Sequence[int]] = None,  # allows checking against hilbert space dim of upstream user
    ):
        ptrs = []
        batch_size = coeff.batch_size

        with cutn_utils.device_ctx(self._ctx.device_id):
            for matrix_op in matrix_ops:
                matrix_op._maybe_instantiate(self._ctx)
                assert matrix_op.data.flags["F_CONTIGUOUS"]
                _batch_size = check_and_get_batchsize(matrix_op.batch_size, batch_size)
                _ = check_and_get_batchsize(self._batch_size, _batch_size)

                if hilbert_space_dims is not None:
                    if not all(
                        map(
                            lambda item: item[0] == item[1],
                            zip(hilbert_space_dims, matrix_op.hilbert_space_dims),
                        )
                    ):
                        raise RuntimeError(
                            "Hilbert space dimensions of matrix operators in product are inconsistent with Hilbert space dimensions of an Operator which contains this OperatorTerm, {self}."
                        )
                self._using_ops.add(matrix_op)
                utils.register_with(self, matrix_op, self._ctx.logger)
                ptrs.append(matrix_op._validated_ptr)
            if batch_size > 1:
                cudm.operator_term_append_matrix_product_batch(
                    self._ctx._handle._validated_ptr,
                    self._ptr,
                    len(matrix_ops),
                    ptrs,
                    conjugations,
                    duals,
                    coeff.batch_size,
                    coeff.static_coeff_ptr,
                    coeff.dynamic_coeff_ptr,
                    coeff.wrapper,
                )
            else:
                cudm.operator_term_append_matrix_product(
                    self._ctx._handle._validated_ptr,
                    self._ptr,
                    len(matrix_ops),
                    ptrs,
                    conjugations,
                    duals,
                    coeff.static_coeff,
                    coeff.wrapper,
                )

    def _append_elementary_product(
        self,
        elem_ops: Sequence[ElementaryOperator],
        modes: Sequence[int],
        duals: Sequence[bool],
        coeff: CallbackCoefficient,
    ):
        """
        Appends `term`, i.e. product of ElementaryOperator to C-API counterpart of this OperatorTerm.
        Before appending, the creation of the C-API counterpart for any ElementaryOperator in `term` is triggered if necessary.
        """
        ptrs = []
        flattened_modes = []
        flattened_duals = []
        batch_size = coeff.batch_size
        with cutn_utils.device_ctx(self._ctx.device_id):
            for elem_op, _modes, _duals in zip(elem_ops, modes, duals):
                elem_op._maybe_instantiate(self._ctx)
                _batch_size = check_and_get_batchsize(elem_op.batch_size, batch_size)
                _ = check_and_get_batchsize(self._batch_size, _batch_size)
                assert elem_op.data.flags["F_CONTIGUOUS"]
                self._using_ops.add(elem_op)
                utils.register_with(self, elem_op, self._ctx.logger)
                ptrs.append(elem_op._validated_ptr)
                flattened_modes.extend(_modes)
                flattened_duals.extend(map(lambda i: int(i), _duals))
            if batch_size > 1:
                cudm.operator_term_append_elementary_product_batch(
                    self._ctx._handle._validated_ptr,
                    self._ptr,
                    len(elem_ops),
                    ptrs,
                    flattened_modes,
                    flattened_duals,
                    coeff.batch_size,
                    coeff.static_coeff_ptr,
                    coeff.dynamic_coeff_ptr,
                    coeff.wrapper,
                )
            if batch_size == 1:
                cudm.operator_term_append_elementary_product(
                    self._ctx._handle._validated_ptr,
                    self._ptr,
                    len(elem_ops),
                    ptrs,
                    flattened_modes,
                    flattened_duals,
                    coeff.static_coeff,
                    coeff.wrapper,
                )

    def _maybe_instantiate(self, ctx: "WorkStream", hilbert_space_dims: Tuple[int]) -> None:
        """
        Create C-API equivalent of this instance (and potentially of its downstream dependencies) and store pointer as attribute.

        Args:
            ctx: WorkStream
                Library context, workspace, stream and other configuration information.
            hilbert_space_dims: Tuple[int]
                The local hilbert space dimensions as an iterable.
            stream: Optional[int]
                The stream to use for moving tensor storage from host to device,
                which is potentially triggered in downstream dependencies.
        """
        if not self._valid_state:
            self._ctx = ctx
            self._hilbert_space_dims = tuple(hilbert_space_dims)
            num_space_modes = len(hilbert_space_dims)
            if self._dtype is None:
                raise RuntimeError("Cannot use an OperatorTerm with unspecified datatype.")
            self._ptr = cudm.create_operator_term(self._ctx._handle._validated_ptr, num_space_modes, self.hilbert_space_dims)
            self._finalizer = weakref.finalize(
                self,
                utils.generic_finalizer,
                self._ctx.logger,
                self._upstream_finalizers,
                (cudm.destroy_operator_term, self._ptr),
                msg=f"Destroying OperatorTerm instance {self}, ptr: {self._ptr}",
            )
            utils.register_with(self, self._ctx, self._ctx.logger)
            for term, term_type, modes, conjugations, duals, coeff in zip(
                self.terms,
                self._term_types,
                self.modes,
                self._conjugations,
                self.duals,
                self._coefficients,
            ):
                if term_type == ElementaryOperator:
                    self._append_elementary_product(term, modes, duals, coeff)
                elif term_type == MatrixOperator:
                    self._append_matrix_product(term, conjugations, duals, coeff, hilbert_space_dims)

        else:
            try:
                assert self._ctx == ctx
            except AssertionError as e:
                raise ValueError(
                    "Using an object with a different WorkStream than it was originally used with is not supported."
                ) from e
            try:
                assert self._hilbert_space_dims == tuple(hilbert_space_dims)
            except AssertionError as e:
                raise ValueError(
                    "Using an object from an object with different Hilbert space dimensions is not supported."
                ) from e

    def __add__(self, other: "OperatorTerm") -> "OperatorTerm":
        """
        Return a new :class:`OperatorTerm` equal to the sum of this :class:`OperatorTerm` and another :class:`OperatorTerm`.
        """
        if not isinstance(other, OperatorTerm):
            raise TypeError(f"Cannot add {type(other)} to OperatorTerm. OperatorTerm only supports addition of OperatorTerm.")
        if self._dtype is None or self._dtype != other.dtype:
            raise TypeError(f"Cannot add OperatorTerm of datatype {self._dtype}  and datatype {other._dtype}.")
        new_terms = [*self.terms, *other.terms]
        new_modes = [*self.modes, *other.modes]
        new_duals = [*self.duals, *other.duals]
        new_conjugations = [*self._conjugations, *other._conjugations]
        new_coefficients = [*self._coefficients, *other._coefficients]
        new_opterm = OperatorTerm(dtype=self._dtype)
        # append method will raise error if dtypes are not compatible
        for term, modes, conjugations, duals, coeff in zip(new_terms, new_modes, new_conjugations, new_duals, new_coefficients):
            new_opterm._append(term, modes, conjugations, duals, coeff)
        return new_opterm

    def __iadd__(self, other: "OperatorTerm") -> "OperatorTerm":
        """
        Inplace add another :class:`OperatorTerm` into this :class:`OperatorTerm`.
        """
        if self._valid_state:
            raise RuntimeError(
                "Cannot in-place add to this OperatorTerm after either\n\
                               a) a prepare or compute method has been executed on an Operator depending on this instance, or\n\
                               b) an OperatorAction has been created that depends on an Operator that depends on this instance."
            )
        if not isinstance(other, OperatorTerm):
            raise TypeError(
                f"Cannot in-place add {type(other)} to OperatorTerm. OperatorTerm only supports in-place addition of OperatorTerm."
            )
        if self._dtype and other._dtype:
            assert self._dtype == other.dtype
        assert (
            self._dtype and self._dtype == other.dtype
        )  # TODO [FUTURE]: allow self to have indefinite dtype if other has definite dtype
        for term, modes, conjugations, duals, coeff in zip(
            other.terms, other.modes, other._conjugations, other.duals, other._coefficients
        ):
            self._append(term, modes, conjugations, duals, coeff)
        if self._dtype is None and other._dtype is not None:
            self._dtype = other._dtype
        return self

    def __mul__(self, other: CoefficientType | "OperatorTerm") -> "OperatorTerm":
        """
        Multiply this :class:`OperatorTerm` with another OperatorTerm or with a coefficient.
        Note that multiplication by a Callable that outputs a batched coefficient vector requires all static coefficients to have the the same `batch_size` as the output of the callable.
        """
        if isinstance(other, _CoefficientTypeRuntimeCheckable):
            other_coeff = CallbackCoefficient.create(other)
            new_opterm = OperatorTerm(dtype=self._dtype)
            for term, modes, conjugations, duals, coeff in zip(
                self.terms, self.modes, self._conjugations, self.duals, self._coefficients
            ):
                # what about batch size passed to callback_helper here?
                new_opterm._append(term, modes, conjugations, duals, other_coeff * coeff)
        elif isinstance(other, OperatorTerm):
            if other.dtype is not None:
                if self._dtype is None:
                    dtype = other.dtype
                elif self._dtype != other.dtype:
                    raise ValueError(
                        f"Data types of OperatorTerms to be multiplied, {self.dtype} and {other.dtype}, do not match."
                    )
                else:
                    dtype = self._dtype
            else:
                dtype = self._dtype
            new_opterm = OperatorTerm(dtype=dtype)
            for term_l, modes_l, conjugations_l, duals_l, coeff_l in zip(
                self.terms, self.modes, self._conjugations, self.duals, self._coefficients
            ):
                for term_r, modes_r, conjugations_r, duals_r, coeff_r in zip(
                    other.terms, other.modes, other._conjugations, other.duals, other._coefficients
                ):
                    new_terms = [*term_l, *term_r]
                    new_modes = [*modes_l, *modes_r]
                    new_conjugations = [*conjugations_l, *conjugations_r]
                    new_duals = [*duals_l, *duals_r]
                    new_opterm._append(new_terms, new_modes, new_conjugations, new_duals, coeff_l * coeff_r)
        else:
            raise TypeError(
                f"Cannot multiply OperatorTerm by {type(other)}. OperatorTerm only supports multiplication by CoefficientType or OperatorTerm."
            )
        return new_opterm

    def __rmul__(self, other: CoefficientType | "OperatorTerm") -> "OperatorTerm":
        """
        Multiply this :class:`OperatorTerm` with a number, callable or another :class:`OperatorTerm` on the right.
        """
        return self * other

    def dag(self) -> "OperatorTerm":
        """
        Return a new :class:`OperatorTerm` equal to the complex conjugate of this :class:`OperatorTerm`.

        .. warning::
            A error will be raised if the :class:`OperatorTerm` contains tensor products of elementary operators acting on both bra and ket modes at the same time.
        """
        if not all([all(len(set(_dual)) == 1 for _dual in dual) for dual in self.duals[::-1]]):
            raise NotImplementedError(
                "OperatorTerm's `dag` method is only supported if none of its products contains ElementaryOperators acting on both bra and ket modes at the same time."
            )
        new_opterm = OperatorTerm(self._dtype)
        for term, term_type, modes, conjugations, duals, coeff in zip(
            self.terms,
            self._term_types,
            self.modes,
            self._conjugations,
            self.duals,
            self._coefficients,
        ):
            if term_type is ElementaryOperator:
                new_opterm._append(
                    [op.dag() for op in term[::-1]],
                    modes[::-1],
                    conjugations[::-1],
                    duals[::-1],
                    coeff.conjugate(),
                )
            elif term_type is MatrixOperator:
                new_opterm._append(
                    [op for op in term[::-1]],
                    modes[::-1],
                    [not conjugation for conjugation in conjugations[::-1]],
                    duals[::-1],
                    coeff.conjugate(),
                )
        return new_opterm

    def dual(self) -> "OperatorTerm":
        """
        Return a new :class:`OperatorTerm` with duality reversed.
        """
        new_opterm = OperatorTerm(self._dtype)
        for term, modes, conjugations, duals, coeff in zip(
            self.terms, self.modes, self._conjugations, self.duals, self._coefficients
        ):
            recursive_logical_not = lambda x: (not x if isinstance(x, bool) else list(map(recursive_logical_not, x)))
            new_opterm._append(
                term[::-1],
                modes[::-1],
                conjugations[::-1],
                recursive_logical_not(duals[::-1]),
                coeff,
            )
        return new_opterm

    def _sync(self):
        if self._last_compute_event is not None:
            self._last_compute_event.synchronize()
            self._last_compute_event = None


class Operator:
    """
    Operator(hilbert_space_dims, *terms)

    Operator representing a collection of :class:`OperatorTerm` objects.

    The action of an :class:`Operator` maps a ``State`` to another ``State``.
    An :class:`Operator` acts on an instance of ``State`` through its ``compute`` method after its ``prepare`` method is called on the same instance of ``State``.

    Args:
        hilbert_space_dims: Hilbert space dimensions of the physical system.
        terms: A sequence of tuples specifying each term.
            Each tuple can consist of a single element (:class:`OperatorTerm`), two elements (:class:`OperatorTerm` and coefficient),
            three elements (:class:`OperatorTerm`, coefficient and duality) or four elements (:class:`OperatorTerm`, coefficient, duality and batch size).
            If less than four elements are given, the remaining ones will be set to their default values (``coefficient=1``, ``duality=False``, ``batch_size=1``).
    """

    # override to avoid calling into ndarray dundered methods instead of this classes
    __array_ufunc__ = None

    def __init__(
        self,
        hilbert_space_dims: Sequence[int],
        *terms: Tuple[OperatorTerm]
        | Tuple[OperatorTerm, CoefficientType]
        | Tuple[OperatorTerm, CoefficientType, bool]
        | Tuple[OperatorTerm, CoefficientType, bool, int],
    ) -> None:
        """
        Initialize an operator representing a collection of :class:`OperatorTerm` objects.
        """
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()
        self._using_terms: Set[OperatorTerm] = set()
        self._using_ops: Set[ElementaryOperator | MatrixOperator] = set()
        self._ctx = None

        self._hilbert_space_dims: Tuple[int] = tuple(hilbert_space_dims)
        self._batch_size: int = 1
        self._prepared_compute_batch_size: Optional[int] = None
        self._prepared_expectation_batch_size: Optional[int] = None

        self._dtype = None  # str
        self.terms: List[OperatorTerm] = []
        self._coefficients: List[CallbackCoefficient] = []
        self.dualities: List[bool] = []

        self._ptr = None
        self._expectation_ptr = None
        self._work_size = None
        self._expectation_work_size = None
        self._last_compute_event = None

        self._current_expectation_compute_type = None
        self._current_action_compute_type = None
        self._upstream_finalizers = collections.OrderedDict()

        for term in terms:
            self.append(*term)

    def _check_valid_state(self, *args, **kwargs):
        """ """
        if not self._valid_state:
            raise InvalidObjectState("The operator cannot be used after resources have been freed!")

    @property
    def _valid_state(self):
        return self._finalizer.alive

    @property
    def dtype(self):
        """
        Data type of this :class:`Operator`.
        """
        return self._dtype

    @property
    def hilbert_space_dims(self):
        """
        Hilbert space dimension of this :class:`Operator`.
        """
        return self._hilbert_space_dims

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _validated_ptr(self):
        """
        The pointer to this instance's C-API counterpart.
        """
        return self._ptr

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _validated_expectation_ptr(self):
        """
        The pointer to this instance's C-API counterpart.
        """
        return self._expectation_ptr

    def _sync(self):
        if self._last_compute_event is not None:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    def append(
        self,
        term: OperatorTerm,
        coeff: CoefficientType = 1.0,
        duality: bool = False,
        batch_size: Optional[int] = 1,
    ) -> None:
        """
        Append an :class:`OperatorTerm` to this :class:`Operator`.

        Args:
            term: The :class:`OperatorTerm` to be appended.
            coeff: The coefficient associated with this :class:`OperatorTerm`.
            duality: Whether the elementary operators in ``term`` are applied on ket modes (``False``) or bra modes (``True``).
            batch_size: Batch size associated with the operator term.
        """
        coeff = CallbackCoefficient.create(coeff, batch_size)
        self._append(term, coeff, duality)

    def _append(
        self,
        term: OperatorTerm,
        coeff: CallbackCoefficient,
        duality: bool = False,
    ) -> None:
        """
        Appends an OperatorTerm to this Operator.

        Args:
            term: OperatorTerm
                The OperatorTerm to be appended.
            coeff: Union[Number, Callable, CallbackCoefficient]
                The coefficient associated with this term.
                A static coefficient is provided as a number.
                A dynamic coefficient is provided as a callable with signature (t,args: Tuple) -> Number.
            duality: Optional[bool]
                Specifies whether the tensor operators in ``term`` are applied on ket or bra modes
                as specified for its constituents (False) or the opposite (True).
        """
        if self._valid_state:
            # TODO[FUTURE]/TODO[OPTIONAL]: Maybe relax this in the future, requires sync
            raise RuntimeError(
                "Cannot inplace add to this Operator after either\n\
                               a) its prepare or compute method has been called or\n\
                               b) an OperatorAction has been created that depends on this Operator."
            )
        else:
            if term.dtype is not None:
                if self._dtype is None:
                    self._dtype = term.dtype
                elif self._dtype != term.dtype:
                    raise ValueError(
                        "Data type of OperatorTerm to be appended to Operator does not match data type of Operator."
                    )
            elif self._dtype is None:
                raise ValueError("Cannot append OperatorTerm without definite datatype to Operator without definite datatype.")
            self._batch_size = check_and_get_batchsize(self._batch_size, coeff.batch_size)
            self._batch_size = check_and_get_batchsize(self._batch_size, term._batch_size)
            self.terms.append(term)
            self._coefficients.append(coeff)
            self.dualities.append(duality)

    def _append_internal(self, term: OperatorTerm, coeff: CallbackCoefficient, dual: bool):
        """
        Appends `term` to C-API counterpart.
        If OperatorTerm instances in self.terms are not instantiated,
        this method will instantiate them.
        """
        if not isinstance(term, OperatorTerm):
            raise TypeError("Can only append instances of OperatorTerm to Operator.")
        # side effect on entries of self.terms
        term._maybe_instantiate(self._ctx, self.hilbert_space_dims)
        utils.register_with(self, term, self._ctx.logger)
        self._using_terms.add(term)
        self._using_ops = self._using_ops.union(term._using_ops)
        if coeff.batch_size > 1:
            cudm.operator_append_term_batch(
                self._ctx._handle._validated_ptr,
                self._ptr,
                term._validated_ptr,
                int(dual),
                coeff.batch_size,
                coeff.static_coeff_ptr,
                coeff.dynamic_coeff_ptr,
                coeff.wrapper,
            )
        else:
            cudm.operator_append_term(
                self._ctx._handle._validated_ptr,
                self._ptr,
                term._validated_ptr,
                int(dual),
                coeff.static_coeff,
                coeff.wrapper,
            )

    def _maybe_instantiate(self, ctx: "WorkStream") -> None:
        """
        Creates the C-API counterpart of this instance, stores its pointer as attribute and appends the terms in
        self.terms to this instance's C-API counterpart, triggering the terms' instantiations if they haven't been
        instantiated yet.

        Args:
            ctx: WorkStream
                Library context, workspace, stream and other configuration information.
        """
        if self._valid_state:
            if self._ctx != ctx:
                raise ValueError(
                    "Operator objects can only be used with a single WorkStream, and this instance was originally used with another WorkStream (either directly or via an OperatorAction). Switching WorkStream is not supported."
                )
        else:
            self._ctx = ctx

            try:
                assert self.dtype is not None
            except AssertionError as e:
                raise RuntimeError(
                    "Operator must have a definite data type before indirect usage through OperatorAction or calls to its prepare or compute methods."
                ) from e

            self._ptr = cudm.create_operator(
                self._ctx._handle._validated_ptr,
                len(self.hilbert_space_dims),
                self.hilbert_space_dims,
            )

            self._expectation_ptr = cudm.create_expectation(self._ctx._handle._validated_ptr, self._ptr)

            self._finalizer = weakref.finalize(
                self,
                utils.generic_finalizer,
                self._ctx.logger,
                self._upstream_finalizers,
                (cudm.destroy_expectation, self._expectation_ptr),
                (cudm.destroy_operator, self._ptr),
                msg=f"Destroying Operator instance {self}, ptr: {self._ptr}",
            )
            utils.register_with(self, self._ctx, self._ctx.logger)

            for term, coeff, dual in zip(self.terms, self._coefficients, self.dualities):
                self._append_internal(term, coeff, dual)

    def dual(self) -> "Operator":
        """
        Return a shallow copy of this :class:`Operator` with flipped duality for each term.
        """
        dual_op = Operator(self._hilbert_space_dims)
        for args in zip(self.terms, self._coefficients, [not (duality) for duality in self.dualities]):
            dual_op._append(*args)
        return dual_op

    def prepare_action(
        self,
        ctx: "WorkStream",
        state: "State",
        state_out: Optional["State"] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        """
        Prepare the action of this :class:`Operator` on an input state and accumulate into the output state.

        Args:
            ctx: Library context, which contains workspace, stream and other configuration information.
            state: The input quantum state to which the :class:`Operator` is to be applied.
            state_out: The output quantum state to which the action is to be accumulated. Defaults to ``state``.
            compute_type: The CUDA compute type to be used by the computation.

        .. attention::
            The ``compute_type`` argument is currently not used and will default to the data type.
        """
        if not self._valid_state:
            self._maybe_instantiate(ctx)
        else:
            if self._ctx != ctx:
                raise ValueError(
                    "Operator objects can only be used with a single WorkStream, and this instance was originally used with another WorkStream. Switching WorkStream is not supported."
                )

        if self.hilbert_space_dims != state.hilbert_space_dims:
            raise ValueError(
                f"Hilbert space dimensions of Operator, {self.hilbert_space_dims}, and input State, {state.hilbert_space_dims}, instances are not matching."
            )
        if state_out is not None and self.hilbert_space_dims != state_out.hilbert_space_dims:
            raise ValueError(
                f"Hilbert space dimensions of Operator, {self.hilbert_space_dims}, and output State, {state.hilbert_space_dims}, instances are not matching."
            )
        self._prepared_compute_batch_size = check_and_get_batchsize(self._batch_size, state.batch_size)

        default_compute_type = self._ctx.compute_type if self._ctx.compute_type is not None else self.dtype
        self._current_action_compute_type = compute_type if compute_type else default_compute_type

        cudm.operator_prepare_action(
            self._ctx._handle._validated_ptr,
            self._ptr,
            state._validated_ptr,
            state_out._validated_ptr if state_out else state._validated_ptr,
            cutn_typemaps.NAME_TO_COMPUTE_TYPE[self._current_action_compute_type],
            self._ctx._memory_limit,
            self._ctx._validated_ptr,
            0,
        )
        self._expectation_work_size = None
        self._work_size, _ = self._ctx._update_required_size_upper_bound()

    def prepare_expectation(
        self,
        ctx: "WorkStream",
        state: "State",
        compute_type: Optional[str] = None,
    ) -> None:
        """
        Prepare the computation of an expectation value of this :class:`Operator` on a state.

        Args:
            ctx: Library context, which contains workspace, stream and other configuration information.
            state: The quantum state on which the expectation value is evaluated.
            compute_type: The CUDA compute type to be used by the computation.

        .. attention::
            The ``compute_type`` argument is currently not used and will default to the data type.
        """
        if not self._valid_state:
            self._maybe_instantiate(ctx)
        else:
            if self._ctx != ctx:
                raise ValueError(
                    "Operator objects can only be used with a single WorkStream, and this instance was originally used with another WorkStream. Switching WorkStream is not supported."
                )

        if self.hilbert_space_dims != state.hilbert_space_dims:
            raise ValueError(
                f"Hilbert space dimensions of Operator, {self.hilbert_space_dims}, and State, {state.hilbert_space_dims}, instances are not matching."
            )

        self._prepared_expectation_batch_size = check_and_get_batchsize(self._batch_size, state.batch_size)

        default_compute_type = self._ctx.compute_type if self._ctx.compute_type is not None else self.dtype
        self._current_expectation_compute_type = compute_type if compute_type else default_compute_type

        cudm.expectation_prepare(
            self._ctx._handle._validated_ptr,
            self._expectation_ptr,
            state._validated_ptr,
            cutn_typemaps.NAME_TO_COMPUTE_TYPE[self._current_expectation_compute_type],
            self._ctx._memory_limit,
            self._ctx._validated_ptr,
            0,
        )
        self._work_size = None
        self._expectation_work_size, _ = self._ctx._update_required_size_upper_bound()
        return

    # we don't want to precondition here to avoid hard to parse error message, instead a check for self._ctx is done inside the function body
    # @cutn_utils.precondition(_check_valid_state)
    def compute_action(
        self,
        t: float,
        params: NDArrayType | Sequence[float] | None,
        state_in: "State",
        state_out: "State",
    ) -> None:
        """
        Compute the action of this :class:`Operator` on an input state and accumulate into the output state.

        Args:
            t: Time argument to be passed to all callback functions.
            params: Additional arguments to be passed to all callback functions. The element type is required to be float (i.e "float64" for arrays).
                If batched operators or coefficients are used, they need to be passed as 2-dimensional the last dimension of which is the batch size.
                To avoid copy of the argument array, it can be passed as a fortran-contiguous cp.ndarray.
            state: The input quantum state to which the :class:`Operator` is to be applied.
            state_out: The output quantum state to which the action is to be accumulated. Defaults to ``state``.
        """
        # lack of preceding prepare_action call
        if self._ctx is None:
            raise RuntimeError(
                "This instance has not been used with a WorkStream, please call its ``prepare_expectation`` or ``_prepare_action`` method once before calls to this method."
            )
        _ = self._validated_ptr  # just check the instance hasn't been finalized yet
        if self._ctx != state_in._ctx:
            raise ValueError("This Operator's WorkStream and the WorkStream of State on which to compute action do not match.")
        if self._ctx != state_out._ctx:
            raise ValueError(
                "This Operator's WorkStream and the WorkStream of State in which to accumulate action do not match."
            )
        self.prepare_action(self._ctx, state_in, state_out, self._current_action_compute_type)
        # unnecessary due to prepare call in each compute call, but keep in case we decide to only prepare here if required
        if not self._prepared_compute_batch_size == check_and_get_batchsize(self._batch_size, state_in.batch_size):
            raise ValueError("This Operator's prepared batchsize does not match the input state's batchsize.")
        if not self._prepared_compute_batch_size == check_and_get_batchsize(self._batch_size, state_out.batch_size):
            raise ValueError("This Operator's prepared batchsize does not match the output state's batchsize.")

        self._ctx._maybe_allocate()

        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):

            params, num_params, batch_size = _handle_callback_params(params, self._batch_size)
            params_ptr = wrap_operand(params).data_ptr

            # update last event in participating elementary/general operators to ensure proper stream synchronization and shutdown order
            self._ctx._last_compute_event = self._last_compute_event
            state_in._last_compute_event = self._last_compute_event
            state_out._last_compute_event = self._last_compute_event
            for _op in self._using_ops:
                _op._last_compute_event = self._last_compute_event
            # update last event for contained OperatorTerms as well
            for term in set(self.terms):
                term._last_compute_event = self._last_compute_event
            _ = check_and_get_batchsize(self._batch_size, batch_size)
            cudm.operator_compute_action(
                self._ctx._handle._validated_ptr,
                self._validated_ptr,
                t,
                self._batch_size,
                num_params,
                params_ptr,
                state_in._validated_ptr,
                state_out._validated_ptr,
                self._ctx._validated_ptr,
                self._ctx._stream_holder.ptr,
            )

    @cutn_utils.precondition(_check_valid_state)
    def compute_expectation(
        self,
        t: float,
        params: NDArrayType | Sequence[float] | None,
        state: "State",
        out: Optional[cp.ndarray] = None,
    ) -> cp.ndarray:
        """
        Compute the expectation value of this :class:`Operator` on a state.

        Args:
            t: Time argument to be passed to all callback functions.
            params: Additional arguments to be passed to all callback functions. The element type is required to be float (i.e "float64" for arrays). 
                If batched operators or coefficients are used, they need to be passed as 2-dimensional the last dimension of which is the batch size.
                To avoid copy of the argument array, it can be passed as a fortran-contiguous cp.ndarray.
            state: The quantum state on which the expectation value is evaluated.

        Returns:
            The computed expectation value wrapped in a :class:`cupy.ndarray`.

        .. note::
            Currently, this method executes in blocking manner, returning the expectation value only after the computation is finished.
        """
        if self._ctx is None:
            raise RuntimeError(
                "This instance has not been used with a WorkStream, please call its ``prepare_expectation`` or ``_prepare_action`` method once before calls to this method."
            )
        elif self._ctx != state._ctx:
            raise RuntimeError(
                "This Operator's WorkStream and the WorkStream of ``state`` for which to compute expectation value do not match."
            )
        _ = self._validated_expectation_ptr  # just check the instance hasn't been finalized yet
        self.prepare_expectation(self._ctx, state, self._current_expectation_compute_type)
        if not self._prepared_expectation_batch_size == check_and_get_batchsize(self._batch_size, state.batch_size):
            raise ValueError("This Operator's prepared expectation batchsize does not match state's batchsize.")
        if out is not None and self._prepared_expectation_batch_size != out.size:
            raise ValueError(
                f"The array to which to write the (batched) expectation value is of incorrect shape.\
                Expected shape is ({self._prepared_expectation_batch_size},), received output array of shape {out.shape}."
            )

        self._ctx._maybe_allocate()

        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx, blocking=True) as (
            self._last_compute_event,
            elapsed,
        ):

            params, num_params, batch_size = _handle_callback_params(params, self._batch_size)
            params_ptr = wrap_operand(params).data_ptr

            # update last event in participating elementary/general operators to ensure proper stream synchronization and shutdown order
            self._ctx._last_compute_event = self._last_compute_event
            state._last_compute_event = self._last_compute_event
            for _op in self._using_ops:
                _op._last_compute_event = self._last_compute_event
            # update last event for contained OperatorTerms as well
            for term in set(self.terms):
                term._last_compute_event = self._last_compute_event

            out = cp.ndarray((self._prepared_expectation_batch_size,), dtype=state.dtype)

            cudm.expectation_compute(
                self._ctx._handle._validated_ptr,
                self._validated_expectation_ptr,
                t,
                self._batch_size,
                len(params),
                params_ptr,
                state._validated_ptr,
                out.data.ptr,
                self._ctx._validated_ptr,
                self._ctx._stream_holder.ptr,
            )
        return out

    def __add__(self, other: "Operator") -> "Operator":
        """
        Return a new :class:`Operator` equal to the sum of this :class:`Operator` with another :class:`Operator`.
        """
        if not isinstance(other, Operator):
            raise TypeError("Only Operator instances can be out-of-place added to Operator")
        if self.hilbert_space_dims != other.hilbert_space_dims:
            raise ValueError("Addition of two Operators with mismatching Hilbert space dimensions is not supported.")
        return Operator(self.hilbert_space_dims, *_unpack_operator(self), *_unpack_operator(other))

    def __iadd__(self, other: Union["Operator", "OperatorTerm"]) -> None:
        """
        Inplace add another :class:`Operator` or :class:`OperatorTerm` into this :class:`Operator`.
        """
        if isinstance(other, OperatorTerm):
            self._append(other)
        elif isinstance(other, Operator):
            for term, coeff, duality in _unpack_operator(other):
                self.append(term, coeff=coeff, duality=duality)
        else:
            raise TypeError("Only Operator and OperatorTerm instances can be in-place added to Operator")
        return self

    def __neg__(self) -> "Operator":
        """
        Return a new :class:`Operator` equal to this :class:`Operator` with all terms negated.
        """
        return self * -1

    def __sub__(self, other: "Operator") -> "Operator":
        """
        Return the difference of this :class:`Operator` with another :class:`Operator`.
        """
        return Operator(self.hilbert_space_dims, *_unpack_operator(self), *_unpack_operator(-other))

    def __mul__(self, factor: CoefficientType) -> "Operator":
        """
        Return a new :class:`Operator` equal to this :class:`Operator` multiplied by a scalar or a batch of scalars on the left.
        """
        factor = CallbackCoefficient.create(factor)

        return Operator(
            self.hilbert_space_dims,
            *(
                tuple(
                    zip(
                        self.terms,
                        ((factor * coeff).unpack() for coeff in self._coefficients),  # convert back to CoefficientType
                        self.dualities,
                    )
                )
            ),
        )

    def __rmul__(self, scalar) -> "Operator":
        """
        Return a new :class:`Operator` equal to this :class:`Operator` multiplied by a scalar on the right.
        """
        return self * scalar


class OperatorAction:
    """
    OperatorAction(ctx, operators)

    Operator action representing the action of a set of :class:`Operator` objects on a set of input states, accumulated into a single output state.

    Args:
        ctx: Library context, which contains workspace, stream and other configuration information.
        operators: A sequence of :class:`Operator` objects, the length of which is identical to the length of sequence of input states accepted when computing this instance's action.
    """

    def __init__(
        self,
        ctx: WorkStream,
        operators: Tuple[Operator],
    ):
        """
        Initialize an operator action representing the action of a set of :class:`Operator` objects on a set of input states, accumulated into a single output state.
        """
        self._finalizer = weakref.finalize(self, lambda: None)
        self._finalizer.detach()
        self.operators = []

        self._dtype = None
        self._set_or_check_dtype(operators)
        self.operators = operators
        if self.dtype is None:
            raise ValueError("Datatype of OperatorAction cannot be inferred from its constituent Operators.")
        _hilbert_space_dims = set(op.hilbert_space_dims for op in operators)
        if len(_hilbert_space_dims) != 1:
            raise RuntimeError("Operator's constituting this OperatorAction have mismatching Hilbert space dimensions.")
        self._hilbert_space_dims = tuple(_hilbert_space_dims.pop())

        self._ctx = ctx
        self._default_compute_type = self._ctx.compute_type if self._ctx.compute_type is not None else self._dtype
        self._current_compute_type = None
        self._last_compute_event = None
        self._work_size = None
        self._upstream_finalizers = collections.OrderedDict()  # future proofing
        self._ptr = None
        operators = []
        self._batch_size = 1
        self._prepared_batch_size: Optional[int] = None
        for op in self.operators:
            op._maybe_instantiate(self._ctx)
            self._batch_size = check_and_get_batchsize(self._batch_size, op._batch_size)
            operators.append(op._validated_ptr)
        self._ptr = cudm.create_operator_action(self._ctx._handle._validated_ptr, len(self.operators), operators)
        self._finalizer = weakref.finalize(
            self,
            utils.generic_finalizer,
            self._ctx.logger,
            self._upstream_finalizers,
            (cudm.destroy_operator_action, self._ptr),
            msg=f"Destroying OperatorAction instance {self}, ptr: {self._ptr}",
        )
        utils.register_with(self, self._ctx, self._ctx.logger)

        for op in self.operators:
            utils.register_with(self, op, self._ctx.logger)
            # op._upstream_finalizers[self._finalizer] = weakref.ref(self)

        self._using_tensor_ops = set()
        self._using_terms = set()
        for op in self.operators:
            self._using_terms = self._using_terms.union(set(op.terms))
            self._using_tensor_ops = self._using_tensor_ops.union(op._using_ops)

    def _check_valid_state(self, *args, **kwargs) -> None:
        """ """
        if not self._valid_state:
            raise InvalidObjectState("The operator action cannot be used after resources are free'd")

    @property
    def _valid_state(self):
        return self._finalizer.alive

    @property
    @cutn_utils.precondition(_check_valid_state)
    def _validated_ptr(self) -> int:
        """
        The pointer to this instances C-API counterpart.
        """
        return self._ptr

    @property
    def hilbert_space_dims(self):
        """
        Hilbert space dimension of this :class:`OperatorAction`.
        """
        return self._hilbert_space_dims

    @property
    def dtype(self):
        """
        Data type of this :class:`OperatorAction`.
        """
        return self._dtype

    def _sync(self):
        if self._last_compute_event:
            self._last_compute_event.synchronize()
            self._last_compute_event = None

    def _set_or_check_dtype(self, operands) -> None:
        """
        Checks that the operands to be appended to self.term are of the same dtype, and that the latter is the same dtype as self.dtype .
        If self.dtype has not been set yet, this method will set it (unless empty operands are passed)
        """
        # check consistency of operands dtypes
        dtypes = {op.dtype for op in operands}
        try:
            dtype = dtypes.pop()
        except KeyError:
            dtype = None
        if len(dtypes) != 0:
            raise ValueError(
                "The provided operands have more than one dtype, which is not supported. Please cast to same dtype."
            )
        # check consistency of operands dtypes with this instances dtype
        if self.dtype is None:
            self._dtype = dtype
        elif dtype is not None:
            try:
                assert self.dtype != dtype
            except AssertionError as e:
                raise TypeError(
                    "The provided operands are required to have the same dtype as this OperatorTerm instance."
                ) from e

    @cutn_utils.precondition(_check_valid_state)
    def prepare(
        self,
        ctx: "WorkStream",
        states_in: Sequence["State"],
        state_out: Optional["State"] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        """
        Prepare the action of this instance on input states.

        Args:
            ctx: Library context, which contains workspace, stream and other configuration information.
            states_in: The input quantum states to which the action is to be applied.
            state_out: The output quantum state to which the action is to be accumulated. Defaults to the first element of ``state_in``.
            compute_type: The CUDA compute type to be used by the computation.

        .. attention::
            The ``compute_type`` argument is currently not used and will default to the data type.
        """
        if self._ctx != ctx:
            raise ValueError(
                "OperatorAction objects can only be used with a single WorkStream, and this instance was originally used with another WorkStream. Switching WorkStream is not supported."
            )
        self._current_compute_type = compute_type if compute_type else self._default_compute_type

        _state_hilbert_spaces = set(state.hilbert_space_dims for state in states_in)
        if len(_state_hilbert_spaces) != 1:
            raise ValueError("Input states have mismatching Hilbert space dimensions.")
        elif state_out is not None:
            _state_hilbert_spaces.add(state_out.hilbert_space_dims)
            if len(_state_hilbert_spaces) != 1:
                raise ValueError("Output state's Hilbert space dimensions do not match input states'.")
        if set((self.hilbert_space_dims,)) != _state_hilbert_spaces:
            raise ValueError(
                f"Hilbert space dimensions of OperatorAction, {self.hilbert_space_dims}, and of input states, {_state_hilbert_spaces.pop()},  are not matching."
            )
        for state_in in states_in:
            _batch_size = check_and_get_batchsize(self._batch_size, state_in.batch_size)
            if _batch_size != state_in.batch_size:
                raise ValueError("Inconsistent input state batch size.")
        if state_out and _batch_size != state_out.batch_size:
            raise ValueError("Inconsistent output state batch size.")
        self._prepared_batch_size = _batch_size
        cudm.operator_action_prepare(
            self._ctx._handle._validated_ptr,
            self._ptr,
            [state._validated_ptr for state in states_in],
            state_out._validated_ptr if state_out else states_in[0]._validated_ptr,
            cutn_typemaps.NAME_TO_COMPUTE_TYPE[self._current_compute_type],
            self._ctx._memory_limit,
            self._ctx._validated_ptr,
            0,
        )
        self._work_size, _ = self._ctx._update_required_size_upper_bound()

        return

    @cutn_utils.precondition(_check_valid_state)
    def compute(
        self,
        t: float,
        params: NDArrayType | Sequence[float] | None,
        states_in: Sequence["State"],
        state_out: "State",
    ) -> None:
        """
        Compute the action of this instance on a sequence of input states and accumulate the results into an output state.

        Args:
            t: Time argument to be passed to all callback functions.
            params: Additional arguments to be passed to all callback functions. The element type is required to be float (i.e "float64" for arrays).
                If batched operators or coefficients are used, they need to be passed as 2-dimensional the last dimension of which is the batch size.
                To avoid copy of the argument array, it can be passed as a fortran-contiguous cp.ndarray.
            states_in: The quantum states to which the :class:`OperatorAction` is applied.
            state_out: The quantum state into which the result is accumulated.
        """
        for state_in in states_in:
            if self._ctx != state_in._ctx:
                raise ValueError("This OperatorAction's WorkStream and the WorkStream of an input state do not match.")
        if self._ctx != state_out._ctx:
            raise ValueError("This OperatorAction's WorkStream and the WorkStream of output state do not match.")
        _ = self._ctx._validated_ptr
        self.prepare(self._ctx, states_in, state_out, self._current_compute_type)
        for state_in in states_in:
            if self._prepared_batch_size != state_in.batch_size:
                raise ValueError("Inconsistent input state batch size.")
        if self._prepared_batch_size != state_out.batch_size:
            raise ValueError("Inconsistent output state batch size.")
        self._ctx._maybe_allocate()

        with cutn_utils.device_ctx(self._ctx.device_id), utils.cuda_call_ctx(self._ctx) as (
            self._last_compute_event,
            elapsed,
        ):

            params, num_params, batch_size = _handle_callback_params(params, self._batch_size)
            params_ptr = wrap_operand(params).data_ptr

            # update last event in participating elementary/general operators to ensure proper stream synchronization and shutdown order
            self._ctx._last_compute_event = self._last_compute_event
            for state_in in states_in:
                state_in._last_compute_event = self._last_compute_event
            state_out._last_compute_event = self._last_compute_event
            for _op in self._using_tensor_ops:
                _op._last_compute_event = self._last_compute_event
            # update last event for contained OperatorTerms as well
            for _term in self._using_terms:
                _term._last_compute_event = self._last_compute_event
            for op in self.operators:
                op._last_compute_event = self._last_compute_event

            cudm.operator_action_compute(
                self._ctx._handle._validated_ptr,
                self._ptr,
                t,
                self._batch_size,
                len(params),
                params_ptr,
                [state._validated_ptr for state in states_in],
                state_out._validated_ptr,
                self._ctx._validated_ptr,
                self._ctx._stream_holder.ptr,
            )


def _unpack_operator(op):
    return zip(op.terms, (coeff.unpack() for coeff in op._coefficients), op.dualities)


def full_matrix_product(
    *operands: Tuple[MatrixOperator, Optional[bool], Optional[bool]],
    coeff: CoefficientType = 1.0,
    batch_size: int = 1,
) -> OperatorTerm:
    """
    Return an :class:`OperatorTerm` from a product of matrix operators defined on the full Hilbert space.

    Args:
        operands: Operands in the product. Each operand is a tuple of length 1 to 3 of the form ``(matrix, conjugation, dual)``, where ``matrix`` is an instance of :class:`MatrixOperator` and ``conjugation`` and ``dual`` are optional booleans and default to `False`.

            - `conjugation=True` implies that the complex conjugate transpose of the class:`MatrixOperator` is applied.
            - `dual=True` implies that the MatrixOperator acts from the right on the bra modes of a mixed quantum state, while `dual=False` (default) the :class:`MatrixOperator` acts from the left on the ket modes of the pure or mixed quantum state.
                
        coeff: Coefficient(s) associated with this :class:`OperatorTerm`.
        batch_size: Batch size of coefficient `coeff`, needs to be specified only if `coeff` is a :class:`Callback`.

    Returns:
        An :class:`OperatorTerm` constructed from the product of :class:`MatrixOperator`s.
    """
    matrices = []
    conjugations = []
    duals = []
    if len(operands) == 0:
        raise ValueError(
            "Empty matrix product is not supported. If you intend to express a term proportional to the identity, this is possibly by calling `tensor_product` with keyword arguments only."
        )
    _hilbert_space_dims = None
    dtype = None
    _batch_size = 1
    if not isinstance(coeff, _CoefficientTypeRuntimeCheckable):
        raise TypeError(f"Unsupported input type, {type(coeff)}, for `coeff`.")

    for op in operands:
        matrices.append(op[0])
        _batch_size = check_and_get_batchsize(_batch_size, matrices[-1].batch_size)
        if _hilbert_space_dims is not None:
            if not all(
                map(
                    lambda item: item[0] == item[1],
                    zip(_hilbert_space_dims, matrices[-1].hilbert_space_dims),
                )
            ):
                raise ValueError("Matrices in matrix product act on inconsistent Hilbert spaces.")
        if dtype is None:
            dtype = matrices[-1].dtype
        elif dtype != matrices[-1].dtype:
            raise ValueError("Matrices in matrix product have inconsistent numerical data types.")
        if len(op) > 1:
            conjugations.append(op[1])
            if len(op) > 2:
                duals.append(op[2])
            else:
                duals.append(False)
        else:
            conjugations.append(False)
            duals.append(False)

    term = OperatorTerm(dtype=dtype)
    term.append_matrix_product(matrices, conjugations, duals, coeff=coeff, batch_size=batch_size)
    return term


def tensor_product(
    *operands: Tuple[
        Union[ElementaryOperator, NDArrayType, Tuple[NDArrayType, Callback]], Sequence[int], Optional[Sequence[bool]]
    ],
    coeff: CoefficientType = 1,
    batch_size: int = 1,
    dtype: Optional[str] = None,
) -> OperatorTerm:
    """
    Return an :class:`OperatorTerm` from a tensor product of elementary operators.

    Args:
        operands: Operands in the tensor product. Each operand is a tuple of length 2 or 3 of the form ``(tensor, modes, dual)``, where ``dual`` is optional. ``tensor`` contains the numerical data of the elementary operator and an optional callback function providing the tensor data. Accepted inputs for ``tensor`` are

            - Subclass of ``ElementaryOperator``, i.e. :class:`DenseOperator` and :class:`MultidiagonalOperator`
            - ``NDArrayType``, which will be converted to a :class:`DenseOperator`
            - ``Tuple[NDArrayType, Callable]``, which will be passed to the initializer of :class:`DenseOperator`

        coeff: Coefficient(s) associated with this :class:`OperatorTerm`.
        dtype: Data type of this :class:`OperatorTerm`. Default value is inferred from input operands unless this function returns a scalar :class:`OperatorTerm`, in which case ``dtype`` is required.
        batch_size: Batch size of coefficient `coeff`, needs to be specified only if `coeff` is a :class:`Callback`.

    Returns:
        An :class:`OperatorTerm` constructed from the tensor product of elementary operators.
    """
    tensors = []
    modes = []
    duals = []
    _batch_size = 1
    if not isinstance(coeff, _CoefficientTypeRuntimeCheckable):
        raise ValueError("Unsupported input type for `coeff`.")
    for op in operands:
        if not isinstance(op, tuple):
            raise TypeError("`tensor_product` expect 2-tuple or 3-tuple as inputs for operands.")

        if len(op) == 2:
            tensor, _modes = op
            _duals = (False,) * len(_modes)
        elif len(op) == 3:
            tensor, _modes, _duals = op
            assert len(modes) == len(duals)
        else:
            raise ValueError(
                f"`tensor_product` expect 2-tuple or 3-tuple as inputs for operands. Received a tuple of length {len(op)}"
            )

        if not isinstance(tensor, ElementaryOperator):
            # MultidiagonalOperators need to be wrapped before passing
            # safe to specialize to DenseOperator here
            if isinstance(tensor, tuple):
                tensor = DenseOperator(*tensor)
            else:
                tensor = DenseOperator(tensor)
            _batch_size = check_and_get_batchsize(_batch_size, tensor.batch_size)
        check_and_get_batchsize(_batch_size, batch_size)
        tensors.append(tensor)
        modes.append(_modes)
        duals.append(_duals)

    if len(operands) == 0 and dtype is None:
        raise ValueError("A data type needs to be specified when creating an OperatorTerm proportional to the identity.")
    term = OperatorTerm(dtype=dtype)
    term.append_elementary_product(tensors, modes, duals, coeff=coeff, batch_size=batch_size)
    return term


def _handle_callback_params(params, user_batch_size, prepared_batch_size=None):
    if params is None:
        params_ptr = 0
        num_params = 0
        batch_size = user_batch_size
        params = cp.asarray([])
    elif not (isinstance(params, NDArrayType)):
        num_params = len(params)
        batch_size = 1
        params = cp.asarray(params)
    else:
        batch_size = params.shape[1]
        num_params = params.shape[0]
        params = cp.asarray(params, order="F")
    return params, num_params, batch_size
