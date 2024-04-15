import enum
from typing import Union, Iterable, Optional

import numpy
import tensorflow


class BackEnd(enum.Enum):
    NUMPY = 0
    TENSORFLOW = 1


class Tensor(numpy.ndarray, tensorflow.Tensor):
    """
    Our internal representation of a Tensor. Right now it just a placeholder.
    """


BACKEND = BackEnd.TENSORFLOW


def random_uniform(low: Union[float, int], high: Union[float, int], size: Union[int, Iterable, tuple[int]]):
    if BACKEND == BackEnd.NUMPY:
        return numpy.random.default_rng().uniform(low=low, high=high, size=size)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.random.uniform(shape=size, minval=low, maxval=high, dtype=tensorflow.float32)
    else:
        raise NotImplementedError


def ones(shape: Union[int, tuple[int]], dtype=float):
    if BACKEND == BackEnd.NUMPY:
        return numpy.ones(shape=shape, dtype=dtype)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.ones(shape=shape, dtype=dtype)
    else:
        raise NotImplementedError


def zeros(shape: Union[int, tuple[int]], dtype=float):
    if BACKEND == BackEnd.NUMPY:
        return numpy.zeros(shape=shape, dtype=dtype)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.zeros(shape=shape, dtype=dtype)
    else:
        raise NotImplementedError


def prod(in_tensor: Tensor, axis: Union[int, tuple[int], None], dtype=float):
    if BACKEND == BackEnd.NUMPY:
        return numpy.prod(a=in_tensor, axis=axis, dtype=dtype)
    elif BACKEND == BackEnd.TENSORFLOW:
        temp_tensor = in_tensor.numpy()
        return tensorflow.convert_to_tensor(numpy.prod(a=temp_tensor, axis=axis, dtype=dtype))
    else:
        raise NotImplementedError


def sqrt(in_tensor: Tensor):
    if BACKEND == BackEnd.NUMPY:
        return numpy.sqrt(in_tensor)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.math.sqrt(in_tensor)
    else:
        raise NotImplementedError


def reshape(in_tensor: Tensor, new_shape: Union[int, Iterable, tuple[int]]):
    if BACKEND == BackEnd.NUMPY:
        return numpy.reshape(a=in_tensor, newshape=new_shape)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.reshape(tensor=in_tensor, shape=new_shape)
    else:
        raise NotImplementedError


def array(array_like):
    if BACKEND == BackEnd.NUMPY:
        return numpy.array(array_like)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.convert_to_tensor(array_like)
    else:
        raise NotImplementedError


def loadtxt(fname: str, dtype=float, delimiter: str = ' '):
    np_a = numpy.loadtxt(fname=fname, dtype=dtype, delimiter=delimiter)
    if BACKEND == BackEnd.NUMPY:
        return np_a
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.convert_to_tensor(np_a)
    else:
        raise NotImplementedError


def random_normal(mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]],
                  size: Union[int, tuple[int]]):
    if BACKEND == BackEnd.NUMPY:
        return numpy.random.default_rng().normal(loc=mean, scale=std, size=size)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.random.uniform(shape=size, mean=mean, stddev=std)
    else:
        raise NotImplementedError


def identity(n: int, dtype=float):
    if BACKEND == BackEnd.NUMPY:
        return numpy.identity(n=n, dtype=dtype)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.eye(num_rows=n, dtype=dtype)
    else:
        raise NotImplementedError


def matmul(x1: Tensor, x2: Tensor):
    if BACKEND == BackEnd.NUMPY:
        return numpy.matmul(x1, x2)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.linalg.matmul(x1, x2)
    else:
        raise NotImplementedError


def reduce_all(in_tensor: Tensor, axis: Union[int, Iterable, tuple[int], None] = None):
    if BACKEND == BackEnd.NUMPY:
        return numpy.all(a=in_tensor, axis=axis)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.math.reduce_all(input_tensor=in_tensor, axis=axis)
    else:
        raise NotImplementedError


def random_standard_normal(size: Union[int, Iterable, tuple[int], None] = None, dtype=float):
    if BACKEND == BackEnd.NUMPY:
        return numpy.random.default_rng().standard_normal(size=size, dtype=dtype)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.random.normal(shape=size, dtype=dtype)
    else:
        raise NotImplementedError


def is_close(x1: Tensor, x2: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
    return numpy.isclose(x1, x2, rtol, atol, equal_nan)


def reduce_min(in_tensor: Tensor, axis: Union[int, Iterable, tuple[int], None] = None, keepdims: bool = False):
    if BACKEND == BackEnd.NUMPY:
        return numpy.min(a=in_tensor, axis=axis, keepdims=keepdims)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.reduce_min(input_tensor=in_tensor, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError


def reduce_max(in_tensor: Tensor, axis: Union[int, Iterable, tuple[int], None] = None, keepdims: bool = False):
    if BACKEND == BackEnd.NUMPY:
        return numpy.max(a=in_tensor, axis=axis, keepdims=keepdims)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.reduce_max(input_tensor=in_tensor, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError


def argmax(in_tensor: Tensor, axis: Union[int, Iterable, tuple[int], None] = None):
    if BACKEND == BackEnd.NUMPY:
        return numpy.argmax(a=in_tensor, axis=axis)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.math.argmax(input_tensor=in_tensor, axis=axis)
    else:
        raise NotImplementedError


def argmin(in_tensor: Tensor, axis: Union[int, Iterable, tuple[int], None] = None):
    if BACKEND == BackEnd.NUMPY:
        return numpy.argmin(a=in_tensor, axis=axis)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.math.argmin(input_tensor=in_tensor, axis=axis)
    else:
        raise NotImplementedError


def where(condition: Union[Tensor, Iterable, float, int, bool],
          x: Union[Tensor, Iterable, int, float],
          y: Union[Tensor, Iterable, int, float]):
    if BACKEND == BackEnd.NUMPY:
        return numpy.where(condition=condition, x=x, y=y)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.where(condition=condition, x=x, y=y)
    else:
        raise NotImplementedError


def vstack(tup: tuple):
    if BACKEND == BackEnd.NUMPY:
        return numpy.vstack(tup)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.convert_to_tensor(numpy.vstack(tup))
    else:
        raise NotImplementedError


def hstack(tup: tuple):
    if BACKEND == BackEnd.NUMPY:
        return numpy.hstack(tup)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.convert_to_tensor(numpy.hstack(tup))
    else:
        raise NotImplementedError


def stack(arrays: Union[Iterable[Tensor], Iterable, int, float], axis: Optional[int] = 0):
    if BACKEND == BackEnd.NUMPY:
        return numpy.stack(arrays=arrays, axis=axis)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.stack(values=arrays, axis=axis)
    else:
        raise NotImplementedError


def flip(in_tensor: Tensor, axis: Union[None, int, Iterable, tuple[int]] = None):
    if BACKEND == BackEnd.NUMPY:
        return numpy.flip(m=in_tensor, axis=axis)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.reverse(tensor=in_tensor, axis=axis)
    else:
        raise NotImplementedError


def argsort(in_tensor: Tensor, axis: Optional[int] = -1):
    if BACKEND == BackEnd.NUMPY:
        return numpy.argsort(a=in_tensor, axis=axis)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.argsort(values=in_tensor, axis=axis)
    else:
        raise NotImplementedError


def linspace(start: Union[Tensor, Iterable, int, float], stop: Union[Tensor, Iterable, int, float],
             num: Optional[int] = 50, endpoint: Optional[bool] = True, axis: Optional[int] = 0):
    if BACKEND == BackEnd.NUMPY:
        return numpy.linspace(start=start, stop=stop, num=num, endpoint=endpoint, axis=axis)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.convert_to_tensor(numpy.linspace(start=start, stop=stop, num=num,
                                                           endpoint=endpoint, axis=axis))
    else:
        raise NotImplementedError


def expand_dims(in_tensor: Union[Tensor, Iterable, int, float], axis: Union[int, Iterable, tuple[int]]):
    if BACKEND == BackEnd.NUMPY:
        return numpy.expand_dims(a=in_tensor, axis=axis)
    elif BACKEND == BackEnd.TENSORFLOW:
        return tensorflow.expand_dims(input=in_tensor, axis=axis)
    else:
        raise NotImplementedError
