import enum
from collections.abc import Iterable

import numpy
import tensorflow


class BackEnd(enum.Enum):
    NUMPY = 0
    TENSORFLOW = 1


class Tensor(numpy.ndarray, tensorflow.Tensor):
    """
    Our internal representation of a Tensor. Right now it just a placeholder.
    """


# TODO move to configuration file
BACKEND = BackEnd.TENSORFLOW


def random_uniform(low: float | int, high: float | int, size: int | Iterable | tuple[int]) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.random.default_rng().uniform(low=low, high=high, size=size))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.random.uniform(shape=size, minval=low, maxval=high, dtype=tensorflow.float32))
        case _:
            raise NotImplementedError


def ones(shape: tuple, dtype=float) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.ones(shape=shape, dtype=dtype))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.ones(shape=shape, dtype=dtype))
        case _:
            raise NotImplementedError


def zeros(shape: tuple, dtype=float) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.zeros(shape=shape, dtype=dtype))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.zeros(shape=shape, dtype=dtype))
        case _:
            raise NotImplementedError


def prod(in_tensor: Tensor, axis: int | tuple[int] | None, dtype=float) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.prod(a=in_tensor, axis=axis, dtype=dtype))
        case BackEnd.TENSORFLOW:
            temp_tensor = in_tensor.numpy()
            return Tensor(tensorflow.convert_to_tensor(numpy.prod(a=temp_tensor, axis=axis, dtype=dtype)))
        case _:
            raise NotImplementedError


def sqrt(in_tensor: Tensor) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.sqrt(in_tensor))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.math.sqrt(in_tensor))
        case _:
            raise NotImplementedError


def reshape(in_tensor: Tensor, new_shape: int | Iterable | tuple[int]) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.reshape(a=in_tensor, newshape=new_shape))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.reshape(tensor=in_tensor, shape=new_shape))
        case _:
            raise NotImplementedError


def array(array_like) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.array(array_like))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.convert_to_tensor(array_like))
        case _:
            raise NotImplementedError


def loadtxt(fname: str, dtype=float, delimiter: str = ' ') -> Tensor:
    np_a = numpy.loadtxt(fname=fname, dtype=dtype, delimiter=delimiter)
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(np_a)
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.convert_to_tensor(np_a))
        case _:
            raise NotImplementedError


def random_normal(mean: float | Iterable[float], std: float | Iterable[float],
                  size: int | tuple[int]) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.random.default_rng().normal(loc=mean, scale=std, size=size))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.random.uniform(shape=size, mean=mean, stddev=std))
        case _:
            raise NotImplementedError


def identity(n: int, dtype=float) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.identity(n=n, dtype=dtype))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.eye(num_rows=n, dtype=dtype))
        case _:
            raise NotImplementedError


def matmul(x1: Tensor, x2: Tensor) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.matmul(x1, x2))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.linalg.matmul(x1, x2))
        case _:
            raise NotImplementedError


def reduce_all(in_tensor: Tensor, axis: int | Iterable | tuple[int] | None = None) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.all(a=in_tensor, axis=axis))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.math.reduce_all(input_tensor=in_tensor, axis=axis))
        case _:
            raise NotImplementedError


def random_standard_normal(size: int | Iterable | tuple[int], dtype=float) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.random.default_rng().standard_normal(size=size, dtype=dtype))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.random.normal(shape=size, dtype=dtype))
        case _:
            raise NotImplementedError


def is_close(x1: Tensor, x2: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
    return bool(numpy.isclose(x1, x2, rtol, atol, equal_nan).any())


def reduce_min(in_tensor: Tensor, axis: int | Iterable | tuple[int] = 0, keepdims: bool = False) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.min(a=in_tensor, axis=axis, keepdims=keepdims))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.reduce_min(input_tensor=in_tensor, axis=axis, keepdims=keepdims))
        case _:
            raise NotImplementedError


def reduce_max(in_tensor: Tensor, axis: int | Iterable | tuple[int] = 0, keepdims: bool = False) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.max(a=in_tensor, axis=axis, keepdims=keepdims))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.reduce_max(input_tensor=in_tensor, axis=axis, keepdims=keepdims))
        case _:
            raise NotImplementedError


def argmax(in_tensor: Tensor, axis: int | Iterable | tuple[int] = 0) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.argmax(a=in_tensor, axis=axis))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.math.argmax(input=in_tensor, axis=axis))
        case _:
            raise NotImplementedError


def argmin(in_tensor: Tensor, axis: int | Iterable | tuple[int] = 0) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.argmin(a=in_tensor, axis=axis))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.math.argmin(input=in_tensor, axis=axis))
        case _:
            raise NotImplementedError


def where(condition: Tensor | Iterable | float | int | bool,
          x: Tensor | Iterable | int | float,
          y: Tensor | Iterable | int | float) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.where(condition, x, y))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.where(condition=condition, x=x, y=y))
        case _:
            raise NotImplementedError


def vstack(tup: tuple) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.vstack(tup))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.convert_to_tensor(numpy.vstack(tup)))
        case _:
            raise NotImplementedError


def hstack(tup: tuple) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.hstack(tup))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.convert_to_tensor(numpy.hstack(tup)))
        case _:
            raise NotImplementedError


def stack(arrays: Iterable[Tensor], axis: int = 0) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.stack(arrays, axis=axis))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.stack(values=arrays, axis=axis))
        case _:
            raise NotImplementedError


def flip(in_tensor: Tensor, axis: int | Iterable | tuple[int] | None = None) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.flip(m=in_tensor, axis=axis))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.reverse(tensor=in_tensor, axis=axis))
        case _:
            raise NotImplementedError


def argsort(in_tensor: Tensor, axis: int = -1) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.argsort(a=in_tensor, axis=axis))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.argsort(values=in_tensor, axis=axis))
        case _:
            raise NotImplementedError


def linspace(start: Tensor | Iterable | int | float, stop: Tensor | Iterable | int | float,
             num: int = 50, endpoint: bool = True, axis: int = 0) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.linspace(start=start, stop=stop, num=num, endpoint=endpoint, axis=axis))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.convert_to_tensor(numpy.linspace(start=start, stop=stop, num=num,
                                                                      endpoint=endpoint, axis=axis)))
        case _:
            raise NotImplementedError


def expand_dims(in_tensor: Tensor | Iterable | int | float, axis: int | Iterable | tuple[int]) -> Tensor:
    match BACKEND:
        case BackEnd.NUMPY:
            return Tensor(numpy.expand_dims(a=in_tensor, axis=axis))
        case BackEnd.TENSORFLOW:
            return Tensor(tensorflow.expand_dims(input=in_tensor, axis=axis))
        case _:
            raise NotImplementedError
