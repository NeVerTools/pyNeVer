from typing import Tuple

import keras
import tensorflow as tf
from keras import layers
from tensorflow.python.ops.init_ops_v2 import glorot_uniform


class Sequential(keras.Sequential):
    """
    Custom representation of tensorflow Sequential Network.
    It adds the identifier for the network and for the input

    Attributes
    ----------
    identifier : str
        Identifier for the network
    input_id : str
        Identifier for the input of the network
    nn_layers : list
        Optional list of layers to add to the model

    """

    def __init__(self, identifier: str, input_id: str, nn_layers):
        super().__init__(layers=nn_layers)
        self.identifier = identifier
        self.input_id = input_id


class ReLU(layers.Activation):
    """
    Custom representation of a tensorflow ReLU Layer. It adds the identifier for the node, the input size and the output
    size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple):
        super(ReLU, self).__init__('relu')
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Sigmoid(layers.Activation):
    """
    Custom representation of a tensorflow Sigmoid Layer. It adds the identifier for the node, the input size and the
    output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple):
        super(Sigmoid, self).__init__('sigmoid')
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Linear(layers.Dense):
    """
    Custom representation of a tensorflow Linear Layer. It adds the identifier for the node, the input size and the
    output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, units: int, activation: str,
                 use_bias: bool = True):
        super(Linear, self).__init__(units, activation, use_bias)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class BatchNorm(layers.BatchNormalization):
    """
    Custom representation of a tensorflow BatchNorm Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, axis: int, momentum: float = 0.1,
                 epsilon: float = 1e-05, center: bool = True, scale: bool = True):
        super(BatchNorm, self).__init__(axis, momentum, epsilon, center, scale)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.trainable = True


class Conv1d(layers.Conv1D):
    """
    Custom representation of a tensorflow Conv1d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, filters: int,
                 kernel_size: Tuple, strides: Tuple, padding: Tuple, data_format: str, dilation: Tuple, groups: int,
                 use_bias: bool):
        super().__init__(filters, kernel_size, strides, 'valid', data_format, dilation, groups, use_bias=use_bias)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding


class Conv2d(layers.Conv2D):
    """
    Custom representation of tensorflow Conv2d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, filters: int,
                 kernel_size: Tuple, strides: Tuple, padding: str, data_format: str, dilation_rate: Tuple,
                 groups: int, use_bias: bool):
        super().__init__(filters, kernel_size, strides, 'valid', data_format, dilation_rate=dilation_rate,
                         groups=groups,
                         use_bias=use_bias)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding


class Conv3d(layers.Conv3D):
    """
    Custom representation of tensorflow Conv3d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, filters: int,
                 kernel_size: Tuple, strides: Tuple, padding: Tuple, data_format: str, dilation_rate: Tuple,
                 groups: int,
                 use_bias: bool):
        super().__init__(filters, kernel_size, strides, 'valid', data_format, dilation_rate=dilation_rate,
                         groups=groups,
                         use_bias=use_bias, kernel_initializer=glorot_uniform(seed=0))
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding


class AvgPool1d(layers.AvgPool1D):
    """
    Custom representation of tensorflow AvgPool1d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, pool_size: Tuple, strides: Tuple, padding: str,
                 data_format: str, ceil_mode: bool, count_include_pad: bool):
        super(AvgPool1d, self).__init__(pool_size, strides, 'valid', data_format)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad


class AvgPool2d(layers.AvgPool2D):
    """
    Custom representation of tensorflow AvgPool2d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, pool_size: Tuple, strides: Tuple, padding: str,
                 data_format: str, ceil_mode: bool, count_include_pad: bool):
        super(AvgPool2d, self).__init__(pool_size, strides, 'valid', data_format)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad


class AvgPool3d(layers.AvgPool3D):
    """
    Custom representation of tensorflow AvgPool3d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, pool_size: Tuple, strides: Tuple, padding: str,
                 data_format: str, ceil_mode: bool, count_include_pad: bool):
        super(AvgPool3d, self).__init__(pool_size, strides, 'valid', data_format)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad


class MaxPool1d(layers.MaxPooling1D):
    """
    Custom representation of tensorflow MaxPool1d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, pool_size: Tuple, strides: Tuple, padding: str,
                 data_format: str, dilation: Tuple, return_indices: bool, ceil_mode: bool):
        super(MaxPool1d, self).__init__(pool_size, strides, 'valid', data_format)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode


class MaxPool2d(layers.MaxPooling2D):
    """
    Custom representation of tensorflow MaxPool2d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, pool_size: Tuple, strides: Tuple, padding: str,
                 data_format: str, dilation: Tuple, return_indices: bool, ceil_mode: bool):
        super(MaxPool2d, self).__init__(pool_size, strides, 'valid', data_format)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode


class MaxPool3d(layers.MaxPooling3D):
    """
    Custom representation of tensorflow MaxPool3d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, pool_size: Tuple, strides: Tuple, padding: str,
                 data_format: str, dilation: Tuple, return_indices: bool, ceil_mode: bool):
        super(MaxPool3d, self).__init__(pool_size, strides, 'valid', data_format)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pad = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode


class LocalResponseNorm(layers.Layer):
    """
    Custom representation of tensorflow LocalResponseNorm Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, depth_radius: int, alpha: float, beta: float,
                 bias: float):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth_radius = depth_radius
        self.alpha = alpha
        self.beta = beta
        self.bias = bias

    def __call__(self, x: tf.Tensor):
        x = tf.nn.local_response_normalization(x, self.depth_radius, self.bias, self.alpha, self.beta)
        return x


class Softmax(layers.Softmax):
    """
    Custom representation of tensorflow Softmax Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, axis: int):
        super(Softmax, self).__init__(axis)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Unsqueeze(layers.Layer):
    """
    Custom class for tensorflow Unsqueeze layer. It conforms to our representation and ONNX.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output
    axis : Tuple
        List of indices at which to insert the singleton dimension.
    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, axis: Tuple):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.axis = axis

    def __call__(self, x: tf.Tensor):
        for ax in self.axis:
            x = tf.expand_dims(x, ax)
        return x


class Reshape(layers.Layer):
    """
    Custom class for tensorflow Reshape layer. It conforms to our representation and ONNX.
    Torch reshape function does not support zeros in the shape, therefore it cannot support the allow_zero attribute
    of our representation.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output
    shape : Tuple
        Tuple which specifies the output shape
    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, shape: Tuple):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.shape = shape

    def __call__(self, x: tf.Tensor):
        x = tf.reshape(x, self.shape)
        return x


class Dropout(layers.Dropout):
    """
    Custom representation of tensorflow Dropout Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, rate: float):
        super().__init__(rate)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Flatten(layers.Flatten):
    """
    Custom class for tensorflow Flatten layer.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : Tuple
        Tuple expressing the dimension of the input
    out_dim : Tuple
        Tuple expressing the dimension of the output

    In Flatten layer in tensorflow, it is always start_dim=1 and end_dim=-1.
    If you want to flatten other dimensions, you may use Reshape layer.
    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple, data_format: str):
        super().__init__(data_format)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.axis = 1
