# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ElementaryOperator."""

from itertools import product

import pytest
import numpy as np
import cupy as cp

from cuquantum.densitymat import DenseOperator, MultidiagonalOperator, CPUCallback, GPUCallback


np.random.seed(42)
cp.random.seed(42)


@pytest.fixture(scope="class")
def callback_args():
    t = 1.0
    args = np.array([1.0, 2.0, 3.0]).reshape(3,1)
    return t, args


@pytest.fixture
def dense_operator(request):
    hilbert_space_dims, order, package, has_callback, callback_is_inplace, batch_size = request.param
    shape = (*hilbert_space_dims, *hilbert_space_dims, batch_size)
    if has_callback:
        data = package.zeros(shape, order=order)
        def callback(t, args):
            _data = package.zeros(shape, order=order)
            matdim = np.prod(shape[:len(shape)//2])
            _data_mat = _data.reshape(matdim, matdim, batch_size)
            for i in range(matdim):
                for j in range(matdim):
                    for batch_index in range(batch_size):
                        _data_mat[i, j, batch_index] = package.sin((i + 2 * j) * t * package.sum(args[...,batch_index]))
            return _data

        def inplace_callback(t, args, arr):
            shape = arr.shape
            batch_size = shape[-1]
            matdim = np.prod(shape[:len(shape)//2])
            arr_mat = arr.reshape(matdim, matdim, batch_size)
            for i in range(matdim):
                for j in range(matdim):
                    for batch_index in range(batch_size):
                        arr_mat[i, j, batch_index] = package.sin((i + 2 * j) * t * package.sum(args[...,batch_index]))
            
        

        if package == np:
            CallbackType = CPUCallback
        else:
            CallbackType = GPUCallback
        callback = CallbackType(inplace_callback if callback_is_inplace else callback, is_inplace=callback_is_inplace)

    else:
        data = package.asarray(np.random.rand(*shape), order=order)
        callback = None
    return DenseOperator(data, callback)


@pytest.fixture
def dense_operator_(request):
    hilbert_space_dims, order, package, has_callback, callback_is_inplace, batch_size = request.param
    shape = (*hilbert_space_dims, *hilbert_space_dims, batch_size)
    if has_callback:
        data = package.empty(shape, order=order)
        def callback(t, args):
            _data = package.empty(shape, order=order)
            matdim = np.prod(shape[:len(shape)//2])
            _data_mat = _data.reshape(matdim, matdim, batch_size)
            for i in range(matdim):
                for j in range(matdim):
                    for batch_index in range(batch_size):
                        _data_mat[i, j, batch_index] = package.cos((i + 2 * j) * t * package.sum(args[...,batch_index]))
            return _data
    

        def inplace_callback(t, args, arr):
            shape = arr.shape
            batch_size = shape[-1]
            matdim = np.prod(shape[:len(shape)//2])
            arr_mat = arr.reshape(matdim, matdim, batch_size)
            for i in range(matdim):
                for j in range(matdim):
                    for batch_index in range(batch_size):
                        arr_mat[i, j, batch_index] = package.cos((i + 2 * j) * t * package.sum(args[...,batch_index]))
        if package == np:
            CallbackType = CPUCallback
        else:
            CallbackType = GPUCallback
        callback = CallbackType(inplace_callback if callback_is_inplace else callback, is_inplace=callback_is_inplace)

    else:
        data = package.asarray(np.random.rand(*shape), order=order)
        callback = None
    return DenseOperator(data, callback)


@pytest.fixture
def multidiagonal_operator(request):
    dim, num_diags, order, package, has_callback, callback_is_inplace, batch_size = request.param
    print(dim, num_diags, order, package, has_callback, batch_size)
    if has_callback:
        data = package.empty((dim, num_diags, batch_size), order=order)
        shape = data.shape

        def callback(t, args):
            _data = package.empty(shape, order=order)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for batch_index in range(shape[2]):
                        _data[i, j, batch_index] = package.sin((i + 2 * j) * t * package.sum(args[...,batch_index]))
            return _data

        def inplace_callback(t, args, arr):
            shape = arr.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        arr[i, j, k] = package.sin((i + 2 * j) * t * package.sum(args[...,k]))

        if package == np:
            CallbackType = CPUCallback
        else:
            CallbackType = GPUCallback
        callback = CallbackType(inplace_callback if callback_is_inplace else callback, is_inplace=callback_is_inplace)

    else:
        data = package.asarray(np.random.random((dim, num_diags)), order=order)
        callback = None
    offsets = list(np.random.choice(range(-dim + 1, dim + 1), size=num_diags, replace=False))
    return MultidiagonalOperator(data, offsets, callback=callback)


@pytest.fixture
def multidiagonal_operator_(request):
    dim, num_diags, order, package, has_callback, callback_is_inplace, batch_size = request.param
    if has_callback:
        data = package.empty((dim, num_diags, batch_size), order=order)
        shape = data.shape
        def callback(t, args):
            _data = package.empty(shape, order=order)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(batch_size):
                        print(_data.shape, i, j, k, args.shape)
                        _data[i, j, k] = package.cos((i + 2 * j) * t * package.sum(args[...,k]))
            return _data

        def inplace_callback(t, args, arr):
            shape = arr.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(batch_size):
                        arr[i, j, k] = package.cos((i + 2 * j) * t * package.sum(args[...,k]))
        
        if package == np:
            CallbackType = CPUCallback
        else:
            CallbackType = GPUCallback
        callback = CallbackType(inplace_callback if callback_is_inplace else callback, is_inplace=callback_is_inplace)
    else:
        data = package.random.random((dim, num_diags, batch_size))
        callback = None
    offsets = list(np.random.choice(range(-dim + 1, dim + 1), size=num_diags, replace=False))
    return MultidiagonalOperator(data, offsets, callback=callback)


@pytest.fixture
def dia_matrix(request):
    dim, num_diags, dia_matrix_func, package = request.param
    data = package.random.random((num_diags, dim))
    offsets = list(np.random.choice(range(-dim + 1, dim + 1), size=num_diags, replace=False))
    return dia_matrix_func((data, offsets), shape=(dim, dim))


@pytest.mark.usefixtures("callback_args")
@pytest.mark.parametrize(
    "dense_operator",
    list(product(
            [(3,), (3, 4)],
            ["C", "F"],
            [np, cp],
            [False, True],
            [False,],
            [1,2],
        )
        )
    +
    list(
            pytest.param(item, marks=pytest.mark.xfail) for item in product(
            [(3,), (3, 4)],
            ["F"],
            [np, cp],
            [True,],
            [True,],
            [1,2],
        )
    )
    ,
    indirect=True,
)
class TestDenseOperatorUnaryOperations:

    @pytest.mark.parametrize("scalar", [2.3, np.array([2.3,1.7]), cp.asarray([2.3,1.7])])
    def test_left_scalar_multiplication(self, dense_operator, callback_args, scalar):
        dense_op = dense_operator
        t, args = callback_args
        batch_size = dense_op.batch_size
        if isinstance(scalar, (np.ndarray, cp.ndarray)):
            batch_size = max(batch_size, scalar.size)
        
        args = args * np.arange(1, 1+batch_size)
        scaled_dense_op = scalar * dense_op

        dense_op_arr = dense_op.to_array(t, args, device="cpu")
        if isinstance(scalar, cp.ndarray):
            ref = scalar.get() * dense_op_arr 
        else:
            ref = scalar * dense_op_arr
        np.testing.assert_allclose(scaled_dense_op.to_array(t, args, device="cpu"), ref)
        if dense_op.batch_size > 1:
            with pytest.raises(ValueError):
                scaled_dense_op = np.array([2.3,1.7,3.5]) * dense_op


    @pytest.mark.parametrize("scalar", [2.3, np.array([2.3,1.7]), cp.asarray([2.3,1.7])])
    def test_right_scalar_multiplication(self, dense_operator, callback_args, scalar):
        dense_op = dense_operator
        t, args = callback_args
        batch_size = dense_op.batch_size
        if isinstance(scalar, (np.ndarray, cp.ndarray)):
            batch_size = max(batch_size, scalar.size)
        
        args = args * np.arange(1, 1+batch_size)
        

        scaled_dense_op = dense_op * scalar

        dense_op_arr = dense_op.to_array(t, args, device="cpu")
        if isinstance(scalar, cp.ndarray):
            ref = dense_op_arr * scalar.get()
        else:
            ref = dense_op_arr * scalar
        np.testing.assert_allclose(scaled_dense_op.to_array(t, args,  device="cpu"), ref)

        if dense_op.batch_size > 1:
            with pytest.raises(ValueError):
                scaled_dense_op = dense_op * np.array([2.3,1.7,3.5]) 
    
    def test_conjugate_transpose(self, dense_operator, callback_args):
        dense_op = dense_operator
        t, args = callback_args
        batch_size = dense_op.batch_size
        args = args * np.arange(1, 1+batch_size)
        
        dense_op_arr = dense_operator.to_array(t, args,  device="cpu")

        dense_op_dag = dense_op.dag()

        n = len(dense_op.shape)
        indices = list(range(n // 2, n)) + list(range(n // 2))
        ref = dense_op_arr.transpose(*indices, n).conj()
        np.testing.assert_allclose(dense_op_dag.to_array(t, args, device="cpu"), ref)


@pytest.mark.parametrize(
    "dense_operator,dense_operator_",
    # CPU Callback, full functionality
    list(
        product(
            list(product([(3,)], ["C", "F"], [np], [False, True], [False,], [1,2])),
            list(product([(3,)], ["C", "F"], [np], [False, True], [False,], [1,2])),
        )
    )
    + # no callback, full functionality
    list(
        product(
            list(product([(3,)], ["C", "F"], [cp,np], [False,], [False,], [1,2])),
            list(product([(3,)], ["C", "F"], [cp], [False,], [False,], [1,2])),
        )
    )
    + # only CPU callback, static data on GPU full functionality 
    list(
        product(
            list(product([(3,)], ["C", "F"], [np], [False,True], [False,], [1,2])),
            list(product([(3,)], ["C", "F"], [cp], [False,], [False,], [1,2])),
        )
    )
    + # GPU x {CPU,GPU} Callback not supported
    list(
        pytest.param(*item,marks=pytest.mark.xfail(raises=ValueError)) for item in product(
            list(product([(3,)], ["F"], [np, cp], [True], [False,], [1,2])),
            list(product([(3,)], ["F"], [cp], [True], [False,], [1,2])),
        )
    )
    + # inplace callback not supported
    list(
        pytest.param(*item,marks=pytest.mark.xfail(raises=ValueError)) for item in product(
            list(product([(3,)], ["F"], [np], [True,], [True,], [1,2])),
            list(product([(3,)], ["F"], [np], [True], [True,], [1,2])),
        )
    )
    + # incompatible batchsizes
    list(
        pytest.param(*item,marks=pytest.mark.xfail(raises=ValueError)) for item in product(
            list(product([(3,)], ["F"], [np], [True,], [False,], [2,])),
            list(product([(3,)], ["F"], [np], [True,], [False,], [3])),
        )
    )
    + # incompatible mode dims
    list(
        pytest.param(*item,marks=pytest.mark.xfail(raises=ValueError)) for item in product(
            list(product([(3,)], ["F"], [np], [True, False], [False,], [1])),
            list(product([(4,)], ["F"], [np], [True, False], [False,], [1])),
        )
    )
    ,
    indirect=True,
)
class TestDenseOperatorBinaryOperations:

    def test_addition(self, dense_operator, dense_operator_, callback_args):
        dense_op1 = dense_operator
        dense_op2 = dense_operator_
        batch_size = max(dense_op1.batch_size, dense_op2.batch_size)

        t, args = callback_args
        args = args * np.arange(1,1+batch_size)
        dense_op_sum = dense_op1 + dense_op2

        dense_op1_arr = dense_op1.to_array(t, args, device="cpu")
        dense_op2_arr = dense_op2.to_array(t, args, device="cpu")
        ref = dense_op1_arr + dense_op2_arr
        np.testing.assert_allclose(dense_op_sum.to_array(t, args, device="cpu"), ref)

    def test_subtraction(self, dense_operator, dense_operator_, callback_args):
        dense_op1 = dense_operator
        dense_op2 = dense_operator_
        batch_size = max(dense_op1.batch_size, dense_op2.batch_size)

        t, args = callback_args
        args = args * np.arange(1,1+batch_size)

        dense_op_diff = dense_op1 - dense_op2

        dense_op1_arr = dense_op1.to_array(t, args, device="cpu")
        dense_op2_arr = dense_op2.to_array(t, args, device="cpu")
        ref = dense_op1_arr - dense_op2_arr
        np.testing.assert_allclose(dense_op_diff.to_array(t, args, device="cpu"), ref)

    def test_matrix_multiplication(self, dense_operator, dense_operator_, callback_args):
        dense_op1 = dense_operator
        dense_op2 = dense_operator_
        batch_size = max(dense_op1.batch_size, dense_op2.batch_size)

        t, args = callback_args
        args = args * np.arange(1,1+batch_size)

        dense_op_prod = dense_op1 @ dense_op2
        #FIXME: not ideal, more generic solution via matricization (see above) would be better
        # alternatively, unit test the utility function that's already implemented and used in the implementation
        if len(dense_op1.shape) == 2:
            subscripts = "abi,bci->aci"
        elif len(dense_op1.shape) == 4:
            subscripts = "abcdi,cdefi->abefi"

        dense_op1_arr = dense_op1.to_array(t, args, device="cpu")
        dense_op2_arr = dense_op2.to_array(t, args, device="cpu")
        ref = np.einsum(subscripts, dense_op1_arr, dense_op2_arr)
        np.testing.assert_allclose(dense_op_prod.to_array(t, args, device="cpu"), ref)


@pytest.mark.parametrize(
    "multidiagonal_operator",
    list(product([4], [3], ["C", "F"], [np, cp], [False, True], [False,], [1,2])) +
    list(
            pytest.param(item, marks=pytest.mark.xfail) for item in product(
            [4],
            [3],
            ["F"],
            [np, cp],
            [True,],
            [True,],
            [1,2],
        )
    ),
    indirect=True,
)
class TestMultidiagonalOperatorUnaryOperations:

    @pytest.mark.parametrize("scalar", [2.3, np.array([2.3,1.7]), cp.asarray([2.3,1.7])])
    def test_left_scalar_multiplication(self, multidiagonal_operator, callback_args, scalar):
        dia_op = multidiagonal_operator
        t, args = callback_args
        batch_size = dia_op.batch_size
        if isinstance(scalar, (np.ndarray, cp.ndarray)):
            batch_size = max(batch_size, scalar.size)
        
        args = args * np.arange(1, 1+batch_size)

        dia_op_arr = dia_op.to_array(t, args, device="cpu")
        scaled_dia_op = scalar * dia_op
        if isinstance(scalar, cp.ndarray):
            ref = scalar.get() * dia_op_arr
        else:
            ref = scalar * dia_op_arr
        np.testing.assert_allclose(scaled_dia_op.to_array(t, args, device="cpu"), ref)

    @pytest.mark.parametrize("scalar", [2.3, np.array([2.3,1.7]), cp.asarray([2.3,1.7])])
    def test_right_scalar_multiplication(self, multidiagonal_operator, callback_args, scalar):
        dia_op = multidiagonal_operator
        t, args = callback_args
        batch_size = dia_op.batch_size
        if isinstance(scalar, (np.ndarray, cp.ndarray)):
            batch_size = max(batch_size, scalar.size)
        
        args = args * np.arange(1, 1+batch_size)

        dia_op_arr = multidiagonal_operator.to_array(t, args, device="cpu")
        scaled_dia_op = dia_op * scalar
        if isinstance(scalar, cp.ndarray):
            ref = dia_op_arr * scalar.get()
        else:
            ref = dia_op_arr * scalar
        np.testing.assert_allclose(scaled_dia_op.to_array(t, args, device="cpu"), ref)

    def test_conjugate_transpose(self, multidiagonal_operator, callback_args):
        dia_op = multidiagonal_operator
        t, args = callback_args
        batch_size = dia_op.batch_size       
        args = args * np.arange(1, 1+batch_size)
        dia_op_arr = dia_op.to_array(t, args, device="cpu")

        dia_op_dag = dia_op.dag()
        n = len(dia_op.shape)
        indices = list(range(n // 2, n)) + list(range(n // 2))
        ref = dia_op_arr.transpose(*indices, n).conj()
        np.testing.assert_allclose(dia_op_dag.to_array(t, args, device="cpu"), ref)


@pytest.mark.parametrize(
    "multidiagonal_operator,multidiagonal_operator_",
    # CPU Callback, full functionality
    list(
        product(
            list(product([4], [3], ["C", "F"], [np], [False, True], [False,], [1,2])),
            list(product([4], [2], ["C", "F"], [np], [False, True], [False,], [1,2])),
        )
    )
    + # no callback, full functionality
    list(
        product(
            list(product([4], [3], ["C", "F"], [cp,np], [False,], [False,], [1,2])),
            list(product([4], [2], ["C", "F"], [cp], [False,], [False,], [1,2])),
        )
    )
    + # only CPU callback, static data on GPU full functionality 
    list(
        product(
            list(product([4], [3], ["C", "F"], [np], [False,True], [False,], [1,2])),
            list(product([4], [2], ["C", "F"], [cp], [False,], [False,], [1,2])),
        )
    )
    + # GPU x {CPU,GPU} Callback not supported
    list(
        pytest.param(*item,marks=pytest.mark.xfail(raises=ValueError)) for item in product(
            list(product([4], [3], ["F"], [np, cp], [True], [False,], [1,2])),
            list(product([4], [2], ["F"], [cp], [True], [False,], [1,2])),
        )
    )
    + # inplace callback not supported
    list(
        pytest.param(*item,marks=pytest.mark.xfail(raises=ValueError)) for item in product(
            list(product([4], [3], ["F"], [np], [True,], [True,], [1,2])),
            list(product([4], [2], ["F"], [np], [True], [True,], [1,2])),
        )
    )
    + # incompatible batchsizes
    list(
        pytest.param(*item,marks=pytest.mark.xfail(raises=ValueError)) for item in product(
            list(product([4], [3], ["F"], [np], [True,], [False,], [2,])),
            list(product([4], [2], ["F"], [np], [True,], [False,], [3])),
        )
    )
    + # incompatible mode dims
    list(
        pytest.param(*item,marks=pytest.mark.xfail(raises=ValueError)) for item in product(
            list(product([5], [3], ["F"], [np], [True, False], [False,], [1])),
            list(product([4], [2], ["F"], [np], [True, False], [False,], [1])),
        )
    )
    ,
    indirect=True,
)
class TestMultidiagonalOperatorBinaryOperations:

    def test_addition(self, multidiagonal_operator, multidiagonal_operator_, callback_args):
        dia_op1 = multidiagonal_operator
        dia_op2 = multidiagonal_operator_
        batch_size = max(dia_op1.batch_size, dia_op2.batch_size)

        t, args = callback_args
        args = args * np.arange(1,1+batch_size)
        dia_op1_arr = dia_op1.to_array(t, args, device="cpu")
        dia_op2_arr = dia_op2.to_array(t, args, device="cpu")

        dia_op_sum = dia_op1 + dia_op2

        ref = dia_op1_arr + dia_op2_arr
        np.testing.assert_allclose(dia_op_sum.to_array(t, args, device="cpu"), ref)

    def test_subtraction(self, multidiagonal_operator, multidiagonal_operator_, callback_args):
        dia_op1 = multidiagonal_operator
        dia_op2 = multidiagonal_operator_
        batch_size = max(dia_op1.batch_size, dia_op2.batch_size)

        t, args = callback_args
        args = args * np.arange(1,1+batch_size)
        
        dia_op1_arr = dia_op1.to_array(t, args, device="cpu")
        dia_op2_arr = dia_op2.to_array(t, args, device="cpu")

        dia_op_diff = dia_op1 - dia_op2

        ref = dia_op1_arr - dia_op2_arr
        np.testing.assert_allclose(dia_op_diff.to_array(t, args, device="cpu"), ref)

    def test_matrix_multiplication(
        self, multidiagonal_operator, multidiagonal_operator_, callback_args
    ):
        dia_op1 = multidiagonal_operator
        dia_op2 = multidiagonal_operator_
        batch_size = max(dia_op1.batch_size, dia_op2.batch_size)

        t, args = callback_args
        args = args * np.arange(1,1+batch_size)
        
        dia_op1_arr = dia_op1.to_array(t, args, device="cpu")
        dia_op2_arr = dia_op2.to_array(t, args, device="cpu")

        subscripts = "abi,bci->aci"
        ref = np.einsum(subscripts, dia_op1_arr , dia_op2_arr )
        dia_op_prod = dia_op1 @ dia_op2
        np.testing.assert_allclose(dia_op_prod.to_array(t, args, device="cpu"), ref)


@pytest.mark.parametrize(
    "dense_operator,multidiagonal_operator",
    list(
        product(
            list(product([(4,)], ["C", "F"], [np], [False, True], [False,], [1,])),
            list(product([4], [2], ["C", "F"], [np], [False, True], [False,], [1,])),
        )
    )
    #since this dispatches through dense x dense operator binary ops, no need to cover other cases
    ,
    indirect=True,
)
class TestMixedOperations:

    def test_dense_multidiagonal_addition(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        batch_size = max(dia_op.batch_size, dense_op.batch_size)
        
        t, args = callback_args
        args = args * np.arange(1,1+batch_size)

        dense_op_arr = dense_op.to_array(t, args, device="cpu")
        dia_op_arr = dia_op.to_array(t, args, device="cpu")

        dense_dia_op_sum = dense_op + dia_op

        ref = dense_op_arr + dia_op_arr
        np.testing.assert_allclose(dense_dia_op_sum.to_array(t, args, device="cpu"), ref)

    def test_multidiagonal_dense_addition(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        batch_size = max(dia_op.batch_size, dense_op.batch_size)
        
        t, args = callback_args
        args = args * np.arange(1,1+batch_size)

        dense_op_arr = dense_op.to_array(t, args, device="cpu")
        dia_op_arr = dia_op.to_array(t, args, device="cpu")

        dense_dia_op_sum = dia_op + dense_op

        ref = dense_op_arr + dia_op_arr
        np.testing.assert_allclose(dense_dia_op_sum.to_array(t, args, device="cpu"), ref)

    def test_dense_multidiagonal_subtraction(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        batch_size = max(dia_op.batch_size, dense_op.batch_size)
        
        t, args = callback_args
        args = args * np.arange(1,1+batch_size)

        dense_op_arr = dense_op.to_array(t, args, device="cpu")
        dia_op_arr = dia_op.to_array(t, args, device="cpu")

        dense_dia_op_diff = dense_op - dia_op

        ref = dense_op_arr - dia_op_arr
        np.testing.assert_allclose(dense_dia_op_diff.to_array(t, args, device="cpu"), ref)

    def test_multidiagonal_dense_subtraction(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        batch_size = max(dia_op.batch_size, dense_op.batch_size)
        
        t, args = callback_args
        args = args * np.arange(1,1+batch_size)
        
        dense_op_arr = dense_op.to_array(t, args, device="cpu")
        dia_op_arr = dia_op.to_array(t, args, device="cpu")

        dense_dia_op_diff = dia_op - dense_op

        ref = dia_op_arr - dense_op_arr
        np.testing.assert_allclose(dense_dia_op_diff.to_array(t, args, device="cpu"), ref)

    def test_dense_multidiagonal_matrix_multiplication(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        batch_size = max(dia_op.batch_size, dense_op.batch_size)
        
        t, args = callback_args
        args = args * np.arange(1,1+batch_size)
        
        dense_op_arr = dense_op.to_array(t, args, device="cpu")
        dia_op_arr = dia_op.to_array(t, args, device="cpu")
        dense_dia_op_prod = dense_op @ dia_op
        #fair to use here, since DIA op restricts dense also to single mode
        subscripts = "abi,bci->aci"
        ref = np.einsum(subscripts, dense_op_arr , dia_op_arr )
        np.testing.assert_allclose(dense_dia_op_prod.to_array(t, args, device="cpu"), ref)

    def test_multidiagonal_dense_matrix_multiplication(
        self, dense_operator, multidiagonal_operator, callback_args
    ):
        dense_op = dense_operator
        dia_op = multidiagonal_operator
        batch_size = max(dia_op.batch_size, dense_op.batch_size)
        
        t, args = callback_args
        args = args * np.arange(1,1+batch_size)

        dense_op_arr = dense_op.to_array(t, args, device="cpu")
        dia_op_arr = dia_op.to_array(t, args, device="cpu")

        dense_dia_op_prod = dia_op @ dense_op

        #fair to use here, since DIA op restricts dense also to single mode
        subscripts = "abi,bci->aci"
        ref = np.einsum(subscripts, dia_op_arr , dense_op_arr )
        np.testing.assert_allclose(dense_dia_op_prod.to_array(t, args, device="cpu"), ref)
