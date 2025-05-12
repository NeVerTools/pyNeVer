"""Internal representation of a tensor

This module contains out internal representation of a tensor and some relevant tensor operations.

"""
from __future__ import annotations

import enum
from collections.abc import Iterable

import numpy
import torch
from numpy.typing import ArrayLike


class BackEnd(enum.Enum):
    """Support for possible multiple backends"""
    NUMPY = 0
    PYTORCH = 1


class NpTensor(numpy.ndarray):
    """Representation of a numpy tensor"""

    def __new__(cls, t: numpy.ndarray):
        obj = numpy.asarray(t).view(cls)
        # If attributes are added they should be added here
        # The following is an example with an info attribute
        # obj.info = "value"
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # If attributes are added they should be also initialized here
        # The following is an example with an info attribute
        # self.info = getattr(obj, 'info', None)


class PtTensor(torch.Tensor):
    """Representation of a pytorch tensor"""

    def __getitem__(self, item):
        try:
            return super().__getitem__(item).item()
        except RuntimeError:
            return super().__getitem__(item)


class Tensor:
    """Our internal representation of a Tensor.

    It's a different type of tensor depending on the chosen backend.
    """

    def __new__(cls, t: numpy.ndarray | torch.Tensor):
        # Set common parameters as parameters of cls
        cls.shape = t.shape

        match BACKEND:
            case BACKEND.NUMPY:
                return NpTensor(t)
            case BACKEND.PYTORCH:
                return PtTensor(t)
            case _:
                raise NotImplementedError

    def __add__(self, other: Tensor) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor) -> Tensor:
        return self - other

    def __getitem__(self, item):
        return self[item]

    def __setitem__(self, key, value):
        self[key] = value


# TODO move to a configuration file
BACKEND = BackEnd.PYTORCH


def array(array_like: ArrayLike) -> Tensor:
    """Returns a Tensor from the ArrayLike object given as input.

    Parameters
    ----------
    array_like : ArrayLike
        The input ArrayLike object

    Returns
    -------
    Tensor
        A Tensor containing the elements of the ArrayLike object

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.array(array_like))
        case BackEnd.PYTORCH:
            return Tensor(torch.Tensor(array_like))
        case _:
            raise NotImplementedError


def loadtxt(fname: str, dtype=float, delimiter: str = ' ') -> Tensor:
    """Returns a Tensor filled with data from a text file

    Parameters
    ----------
    fname : str
        The pathname of the text file
    dtype: type
        The type of the elements of the returned Tensor
    delimiter: str
        The data separator character

    Returns
    -------
    Tensor
        A Tensor containing the data loaded from the text file

    """
    np_a = numpy.loadtxt(fname=fname, dtype=dtype, delimiter=delimiter)
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(np_a)
        case BackEnd.PYTORCH:
            return Tensor(torch.from_numpy(np_a))
        case _:
            raise NotImplementedError


def ones(shape: int | tuple[int], dtype=float) -> Tensor:
    """Returns a Tensor filled with ones.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to return
    dtype : type
        The type of its elements

    Returns
    -------
    Tensor
        The Tensor filled with ones

    """
    if isinstance(shape, int):
        shape = (shape,)

    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.ones(shape=shape, dtype=dtype))
        case BackEnd.PYTORCH:
            return Tensor(torch.ones(size=shape, dtype=dtype))
        case _:
            raise NotImplementedError


def zeros(shape: int | tuple[int], dtype=float) -> Tensor:
    """Returns a Tensor filled with zeroes.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to return
    dtype : type
        The type of its elements

    Returns
    -------
    Tensor
        The Tensor filled with zeroes

    """
    if isinstance(shape, int):
        shape = (shape,)

    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.zeros(shape=shape, dtype=dtype))
        case BackEnd.PYTORCH:
            return Tensor(torch.zeros(size=shape, dtype=dtype))
        case _:
            raise NotImplementedError


def identity(n: int, dtype=float) -> Tensor:
    """Returns a square identity matrix.

    Parameters
    ----------
    n : int
        The dimension of the square matrix
    dtype : type
        The type of its elements

    Returns
    -------
    Tensor
        The identity matrix

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.identity(n=n, dtype=dtype))
        case BackEnd.PYTORCH:
            return Tensor(torch.eye(n=n, dtype=dtype))
        case _:
            raise NotImplementedError


def diag(in_tensor: Tensor) -> Tensor:
    """Returns a square matrix with the diagonal specified in the input

    Parameters
    ----------
    in_tensor : Tensor
        The diagonal Tensor

    Returns
    ----------
    Tensor
        The square Tensor with the diagonal specified in the input

    """

    if len(in_tensor.shape) > 1:
        raise Exception('Only supports 1D tensors')

    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.diag(in_tensor))
        case BackEnd.PYTORCH:
            return Tensor(torch.diag(in_tensor))
        case _:
            raise NotImplementedError


def random_uniform(low: float | int, high: float | int, size: int | Iterable | tuple[int]) -> Tensor:
    """Returns a Tensor containing samples drawn from a random uniform distribution.

    Parameters
    ----------
    low : float | int
        The lower bound of the distribution
    high : float | int
        The upper bound of the distribution
    size : int | Iterable | tuple[int]
        The shape of the returned Tensor, specified as a tuple or an integer

    Returns
    -------
    Tensor
        The computed output

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.random.default_rng().uniform(low=low, high=high, size=size))
        case BackEnd.PYTORCH:
            return Tensor((high - low) * torch.rand(size) + low)
        case _:
            raise NotImplementedError


def random_normal(mean: float | Iterable[float], std: float | Iterable[float],
                  size: int | tuple[int]) -> Tensor:
    """Returns a Tensor containing samples drawn from a random normal distribution.

    Parameters
    ----------
    mean : float | Iterable[float]
        The mean of the normal distribution
    std : float | Iterable[float]
        The standard deviation of the distribution
    size : int | Iterable | tuple[int]
        The shape of the returned Tensor, specified as a tuple or an integer

    Returns
    -------
    Tensor
        The computed output

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.random.default_rng().normal(loc=mean, scale=std, size=size))
        case BackEnd.PYTORCH:
            return Tensor(torch.normal(mean=mean, std=std, size=size))
        case _:
            raise NotImplementedError


def random_standard_normal(size: int | Iterable | tuple[int], dtype=float) -> Tensor:
    """Returns a Tensor containing samples drawn from a standard random normal distribution (mean = 0, std = 1).

    Parameters
    ----------
    size : int | Iterable | tuple[int]
        The shape of the returned Tensor, specified as a tuple or an integer
    dtype
        The type of the elements of the distribution

    Returns
    -------
    Tensor
        The computed output

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.random.default_rng().standard_normal(size=size, dtype=dtype))
        case BackEnd.PYTORCH:
            return Tensor(torch.normal(mean=0, std=1, size=size, dtype=dtype))
        case _:
            raise NotImplementedError


# TODO torch does not support endpoint and axis
def linspace(start: Tensor | Iterable | int | float, stop: Tensor | Iterable | int | float,
             num: int = 50, endpoint: bool = True, axis: int = 0) -> Tensor:
    """Generates `num` evenly spaced values between `start` and `stop`.

    Parameters
    ----------
    start : Tensor | Iterable | int | float
        The first value
    stop : Tensor | Iterable | int | float
        The last value if `endpoint` is `True`, otherwise it's excluded
    num : int | float
        The number of values
    endpoint : bool
        Include `stop` if True
    axis : int
        The axis where the samples are stored, by default a new axis at the beginning

    Returns
    -------
    Tensor
        The generated values

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.linspace(start=start, stop=stop, num=num, endpoint=endpoint, axis=axis))
        case BackEnd.PYTORCH:
            return Tensor(torch.linspace(start=start, end=stop, steps=num))
        case _:
            raise NotImplementedError


def prod(in_tensor: Tensor, axis: int | tuple[int] | None, dtype=float) -> Tensor:
    """Returns the product of the Tensor elements along a specified axis (or axes).

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor on which we want to calculate the product
    axis : int | tuple[int] | None
        The axis (or axes) along which we calculate the product. If None is given the product of all the elements is
        returned
    dtype: type
        The type of the elements of the returned Tensor

    Returns
    -------
    Tensor
        A Tensor containing the product of the elements of the input Tensor along the axis

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.prod(a=in_tensor, axis=axis, dtype=dtype))
        case BackEnd.PYTORCH:
            return Tensor(torch.prod(input=in_tensor, dim=axis, dtype=dtype))
        case _:
            raise NotImplementedError


def matmul(x1: Tensor, x2: Tensor) -> Tensor:
    """Returns the matrix product of two tensors.

    Parameters
    ----------
    x1 : Tensor
        The first Tensor
    x2 : Tensor
        The second Tensor

    Returns
    -------
    Tensor
        The computed output

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.matmul(x1, x2))
        case BackEnd.PYTORCH:
            return Tensor(torch.matmul(x1, x2))
        case _:
            raise NotImplementedError


def dot(x1: Tensor, x2: Tensor) -> Tensor:
    """Returns the dot product of two (batches of) tensors.
    Uses torch.matmul for PyTorch and numpy.dot for numpy

    Parameters
    ----------
    x1 : Tensor
        The first Tensor
    x2 : Tensor
        The second Tensor

    Returns
    ----------
    Tensor
        The computed output
    """

    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.dot(x1, x2))
        case BackEnd.PYTORCH:
            return Tensor(torch.matmul(x1, x2))
        case _:
            raise NotImplementedError


def sqrt(in_tensor: Tensor) -> Tensor:
    """Returns the element-wise square root of a Tensor.

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor on which we want to calculate the square root

    Returns
    -------
    Tensor
        A Tensor containing the square root of the elements of the input Tensor

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.sqrt(in_tensor))
        case BackEnd.PYTORCH:
            return Tensor(torch.sqrt(in_tensor))
        case _:
            raise NotImplementedError


def reshape(in_tensor: Tensor, new_shape: int | Iterable | tuple[int]) -> Tensor:
    """Returns a reshaped Tensor with a specified new shape

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor to be reshaped
    new_shape : int | Iterable | tuple[int]
        The new shape of the Tensor

    Returns
    -------
    Tensor
        The reshaped Tensor

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.reshape(a=in_tensor, newshape=new_shape))
        case BackEnd.PYTORCH:
            return Tensor(torch.reshape(input=in_tensor, shape=new_shape))
        case _:
            raise NotImplementedError


def reduce_min(in_tensor: Tensor, axis: int | Iterable | tuple[int] = 0, keepdims: bool = False) -> Tensor:
    """Returns the minimum of a Tensor along a specified axis.

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor to evaluate
    axis : int | Iterable | tuple[int]
        The axis along which we want to calculate the minimum
    keepdims : bool
        Parameter to decide whether the output will have reduced dimensions or not

    Returns
    -------
    Tensor
        The Tensor containing the minimums

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.min(a=in_tensor, axis=axis, keepdims=keepdims))
        case BackEnd.PYTORCH:
            return Tensor(torch.min(input=in_tensor, dim=axis, keepdim=keepdims))
        case _:
            raise NotImplementedError


def reduce_max(in_tensor: Tensor, axis: int | Iterable | tuple[int] = 0, keepdims: bool = False) -> Tensor:
    """Returns the maximum of a Tensor along a specified axis.

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor to evaluate
    axis : int | Iterable | tuple[int]
        The axis along which we want to calculate the maximum
    keepdims : bool
        Parameter to decide whether the output will have reduced dimensions or not

    Returns
    -------
    Tensor
        The Tensor containing the maximums

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.max(a=in_tensor, axis=axis, keepdims=keepdims))
        case BackEnd.PYTORCH:
            return Tensor(torch.max(input=in_tensor, dim=axis, keepdim=keepdims))
        case _:
            raise NotImplementedError


def reduce_all(in_tensor: Tensor, axis: int | Iterable | tuple[int] | None = None) -> Tensor:
    """Checks whether all the elements of a Tensor over a given axis evaluate to True.

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor to evaluate
    axis : int | Iterable | tuple[int] | None
        The axis along which we want to evaluate the elements

    Returns
    -------
    Tensor
        A Tensor with reduced dimensions along the axis containing the results of the evaluation

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.all(a=in_tensor, axis=axis))
        case BackEnd.PYTORCH:
            return Tensor(torch.all(input=in_tensor, dim=axis))
        case _:
            raise NotImplementedError


def is_close(x1: Tensor, x2: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
    """Checks whether two Tensors are element-wise almost equal (i.e. equal within a tolerance value).

    Parameters
    ----------
    x1 : Tensor
        The first Tensor
    x2 : Tensor
        The second Tensor
    rtol : float
        The relative tolerance parameter
    atol : float
        The absolute tolerance parameter
    equal_nan : bool
        Should be set to True if NaNs are considered to be equal, False otherwise

    Returns
    -------
    Tensor
        True if they are close, False otherwise

    """
    return bool(numpy.isclose(x1, x2, rtol, atol, equal_nan).any())


def argmax(in_tensor: Tensor, axis: int | Iterable | tuple[int] = 0) -> Tensor:
    """Returns the index of the maximum of a Tensor along a specified axis.

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor to evaluate
    axis : int | Iterable | tuple[int]
        The axis along which we want to calculate the maximum

    Returns
    -------
    Tensor
        The Tensor containing the indices

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.argmax(a=in_tensor, axis=axis))
        case BackEnd.PYTORCH:
            return Tensor(torch.argmax(input=in_tensor, dim=axis))
        case _:
            raise NotImplementedError


def argmin(in_tensor: Tensor, axis: int | Iterable | tuple[int] = 0) -> Tensor:
    """Returns the index of the minimum of a Tensor along a specified axis.

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor to evaluate
    axis : int | Iterable | tuple[int]
        The axis along which we want to calculate the minimum

    Returns
    -------
    Tensor
        The Tensor containing the indices

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.argmin(a=in_tensor, axis=axis))
        case BackEnd.PYTORCH:
            return Tensor(torch.argmin(input=in_tensor, dim=axis))
        case _:
            raise NotImplementedError


def where(condition: Tensor | Iterable | float | int | bool,
          x: Tensor | Iterable | int | float,
          y: Tensor | Iterable | int | float) -> Tensor:
    """Returns elements of `x` if condition is True, otherwise returns elements of `y`.

    Parameters
    ----------
    condition : Tensor | Iterable | float | int | bool
        The condition to check for each element pair
    x : Tensor | Iterable | int | float
        The first element or collection of elements
    y : Tensor | Iterable | int | float
        The second element or collection of elements
    Returns
    -------
    Tensor
        The Tensor containing the resulting chosen elements based on condition

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.where(condition, x, y))
        case BackEnd.PYTORCH:
            return Tensor(torch.where(condition, x, y))
        case _:
            raise NotImplementedError


def vstack(tup: tuple[Tensor]) -> Tensor:
    """Stacks tensors row wise.

    Parameters
    ----------
    tup : tuple[Tensor]
        The sequence of tensors to stack

    Returns
    -------
    Tensor
        The stacked tensor

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.vstack(tup))
        case BackEnd.PYTORCH:
            return Tensor(torch.vstack(tup))
        case _:
            raise NotImplementedError


def hstack(tup: tuple[Tensor]) -> Tensor:
    """Stacks tensors column wise.

    Parameters
    ----------
    tup : tuple[Tensor]
        The sequence of tensors to stack

    Returns
    -------
    Tensor
        The stacked tensor

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.hstack(tup))
        case BackEnd.PYTORCH:
            return Tensor(torch.hstack(tup))
        case _:
            raise NotImplementedError


def flip(in_tensor: Tensor, axis: int | Iterable | tuple[int] | None = None) -> Tensor:
    """Reverses the order of elements of a Tensor.

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor to reverse
    axis : int | Iterable | tuple[int] | None
        The axis along which we want to reverse

    Returns
    -------
    Tensor
        The reversed Tensor

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.flip(m=in_tensor, axis=axis))
        case BackEnd.PYTORCH:
            return Tensor(torch.flip(input=in_tensor, dims=axis))
        case _:
            raise NotImplementedError


def argsort(in_tensor: Tensor, axis: int = -1) -> Tensor:
    """Returns the indices of elements of a Tensor in the order that would sort it.

    Parameters
    ----------
    in_tensor : Tensor
        The Tensor to sort
    axis : int | Iterable | tuple[int] | None
        The axis along which we want to sort

    Returns
    -------
    Tensor
        The indices that sort `in_tensor` along the specified axis

    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.argsort(a=in_tensor, axis=axis))
        case BackEnd.PYTORCH:
            return Tensor(torch.argsort(input=in_tensor, dim=axis))
        case _:
            raise NotImplementedError


def expand_dims(in_tensor: Tensor | Iterable | int | float, axis: int | Iterable | tuple[int]) -> Tensor:
    """Expands the shape of a Tensor by adding an axis.

    Parameters
    ----------
    in_tensor : Tensor | Iterable | int | float
        The Tensor to be expanded
    axis : int | Iterable | tuple[int]
        The axis to be added

    Returns
    -------
        The expanded Tensor
    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.expand_dims(a=in_tensor, axis=axis))
        case BackEnd.PYTORCH:
            return Tensor(torch.unsqueeze(in_tensor, axis))
        case _:
            raise NotImplementedError


def get_negative(in_tensor: Tensor) -> Tensor:
    """Procedure to return the negative coefficients of a Tensor

    Parameters
    ----------
    in_tensor : Tensor
        The input Tensor

    Returns
    ----------
    Tensor
        The computed output
    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.minimum(in_tensor, numpy.zeros(in_tensor.shape)))
        case BackEnd.PYTORCH:
            return Tensor(torch.clamp(in_tensor, max=0))
        case _:
            raise NotImplementedError


def get_positive(in_tensor: Tensor) -> Tensor:
    """Procedure to return the positive coefficients of a Tensor

    Parameters
    ----------
    in_tensor : Tensor
        The input Tensor

    Returns
    ----------
    Tensor
        The computed output
    """
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.maximum(in_tensor, numpy.zeros(in_tensor.shape)))
        case BackEnd.PYTORCH:
            return Tensor(torch.clamp(in_tensor, min=0))
        case _:
            raise NotImplementedError
