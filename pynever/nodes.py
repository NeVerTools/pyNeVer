import abc
import copy
import math
import numpy as np
from typing import Tuple

from pynever.tensor import Tensor


class LayerNode(abc.ABC):
    """
    An abstract class used for our internal representation of a generic Layer of a Neural Network.
    Its concrete children correspond to real network layers.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.
    in_dim : Tuple
        Dimension of the input Tensor as a tuple (ndarray.shape like).
    out_dim : Tuple
        Dimension of the output Tensor as a tuple (ndarray.shape like).

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_dim: Tuple):
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __repr__(self):
        return f"{self.__class__.__name__}\n" \
               f"in_dim = {self.in_dim}\n" \
               f"out_dim = {self.out_dim}"

    def __str__(self):
        return self.__repr__()

    @abc.abstractmethod
    def update_input(self, in_dim: Tuple):
        pass


class ReLUNode(LayerNode):
    """
    A class used for our internal representation of a ReLU Layer of a Neural Network.

    Attributes
    ----------

    """

    def __init__(self, identifier: str, in_dim: Tuple):
        if not len(in_dim) >= 1:
            raise Exception("ReLUNode: in_dim cannot be empty")

        out_dim = copy.deepcopy(in_dim)
        super().__init__(identifier, in_dim, out_dim)

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim)


class SigmoidNode(LayerNode):
    """
    A class used for our internal representation of a Sigmoid Layer of a Neural Network.

    Attributes
    ----------

    """

    def __init__(self, identifier: str, in_dim: Tuple):
        if not len(in_dim) >= 1:
            raise Exception("SigmoidNode: in_dim cannot be void")

        out_dim = copy.deepcopy(in_dim)
        super().__init__(identifier, in_dim, out_dim)

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim)


class FullyConnectedNode(LayerNode):
    """
    A class used for our internal representation of a Fully Connected layer of a Neural Network

    Attributes
    ----------
    in_features : int
        Number of input features of the fully connected layer.
    out_features : int
        Number of output features of the fully connected layer.
    weight : Tensor, optional
        Tensor containing the weight parameters of the fully connected layer.
    bias : Tensor, optional
        Tensor containing the bias parameters of the fully connected layer.
    has_bias : bool, optional
        Flag True if the fully connected layer has bias, False otherwise (default: True)

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_features: int,
                 weight: Tensor = None, bias: Tensor = None, has_bias: bool = True):

        if not len(in_dim) >= 1:
            raise Exception("FullyConnectedNode: in_dim cannot be empty")

        """in_dim_error = f"Wrong value for in_features ({in_features}): " \
                       f"should be equal to the last element of in_dim ({in_dim[-1]})."
        if not in_dim[-1] == in_features:
            raise Exception(in_dim_error)"""

        temp = list(in_dim)
        temp[-1] = out_features
        out_dim = tuple(temp)
        super().__init__(identifier, in_dim, out_dim)

        in_features = in_dim[-1]
        self.in_features = in_features
        self.out_features = out_features

        # We assume the Linear operation is x * W^T
        if weight is None:
            weight = np.random.uniform(-math.sqrt(1 / in_features), math.sqrt(1 / in_features),
                                       size=[out_features, in_features])

        weight_error = f"Weight dimensions should be equal to out_features ({out_features}) " \
                       f"and in_features ({in_features}) respectively."

        if not (weight.shape[0] == out_features and weight.shape[1] == in_features):
            raise Exception(weight_error)

        if has_bias:
            if bias is None:
                bias = np.random.uniform(-math.sqrt(1 / in_features), math.sqrt(1 / in_features),
                                         size=[out_features])
            else:
                if not (bias.shape == (out_features,)):
                    raise Exception(f"Bias shape is wrong: it should be equal to ({out_features},)")
        else:
            bias = None

        self.weight = weight
        self.has_bias = has_bias
        self.bias = bias

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.out_features, self.weight, self.bias, self.has_bias)


class BatchNormNode(LayerNode):
    """
    A class used for our internal representation of a one dimensional Batch Normalization Layer.
    N.B. There are some problem for compatibility between pytorch and onnx: pytorch provide 3 different kind
    of batchnorm layers which supports [(N, C) or (N, C, L)], (N, C, H, W) and (N, C, D, H, W) dimensional inputs
    respectively (BatchNorm1D, BatchNorm2D and BatchNorm3D). The batchnorm operation is always applied to the
    C dimension (N is the batch dimension which we do not keep track of). ONNX accepts input in the form of
    (N, C, D1, ... , Dn) where N is the batch dimension and C is the dimension to which the batchnorm is applied.
    It should also be noted that at present the pytorch constructors do not support the setting of weight and
    bias explicitly.

    Attributes
    ----------

    num_features : int
        Number of input and output feature of the Batch Normalization Layer.
    weight : Tensor, optional
        Tensor containing the weight parameters of the Batch Normalization Layer. (default: None)
    bias : Tensor, optional
        Tensor containing the bias parameter of the Batch Normalization Layer. (default: None)
    running_mean : Tensor, optional
        Tensor containing the running mean parameter of the Batch Normalization Layer. (default: None)
    running_var : Tensor, optional
        Tensor containing the running var parameter of the Batch Normalization Layer. (default: None)
    eps : float, optional
        Value added to the denominator for numerical stability (default: 1e-5).
    momentum : float, optional
        Value used for the running_mean and running_var computation. Can be set to None
        for cumulative moving average (default: 0.1)
    affine : bool, optional
        When set to True, the module has learnable affine parameter (default: True).
    track_running_stats : bool, optional
        When set to True, the module tracks the running mean and variance, when set to false the module
        does not track such statistics and always uses batch statistics in both training and eval modes (default: True).

    """

    def __init__(self, identifier: str, in_dim: Tuple, weight: Tensor = None, bias: Tensor = None,
                 running_mean: Tensor = None, running_var: Tensor = None, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):

        # Since we don't consider the batch dimension in our representation we assume that the first dimension of
        # in_dim is always the dimension on which the normalization is applied.
        if not len(in_dim) >= 1:
            raise Exception("in_dim cannot be empty")
        out_dim = copy.deepcopy(in_dim)
        super().__init__(identifier, in_dim, out_dim)

        num_features = in_dim[0]
        self.num_features = num_features

        if track_running_stats:
            if running_mean is None:
                running_mean = np.ones(num_features)
            if running_var is None:
                running_var = np.zeros(num_features)

            if not running_var.shape[0] == num_features:
                raise Exception("The dimension of the running_var should be equal to num_features")
            if not running_mean.shape[0] == num_features:
                raise Exception("The dimension of the running_mean should be equal to num_features")

        else:
            running_mean = None
            running_var = None

        if weight is None:
            weight = np.ones(num_features)

        if bias is None:
            bias = np.zeros(num_features)

        if not weight.shape[0] == num_features:
            raise Exception("The dimension of the weight should be equal to num_features")

        if not bias.shape[0] == num_features:
            raise Exception("The dimension of the bias should be equal to num_features")

        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        self.running_var = running_var

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim,
                      self.weight, self.bias, self.running_mean, self.running_var,
                      self.eps, self.momentum, self.affine, self.track_running_stats)


class ConvNode(LayerNode):
    """
    A class used for our internal representation of a Convolutional layer of a Neural Network.
    Also in this case the pytorch and onnx representation present incompatibilities. As in Batchnorm pytorch
    provide 3 different class for convolution based on the dimensionality of the input considered.
    Moreover the padding is forced to be symmetric.
    The dimensionality supported for the input are (N, C, L), (N, C, H, W) and (N, C, D, H, W).
    In ONNX the padding can be asymmetric and the dimensionality supported is (N, C, D1, ... , Dn) where D1, ... Dn are
    the dimension on which the convolution is applied

    Attributes
    ----------
    in_channels : int
        Number of input channels in Conv Layer.
    out_channels : int
        Number of output channels in Conv Layer.
    kernel_size : Tuple
        The size of the kernel. Should have size equal to the number of dimension n
        (we don't count the channel dimension).
    stride : Tuple
        Stride along each spatial axis. Should have size equal to the number of dimension n
        (we don't count the channel dimension).
    padding : Tuple
        Padding for the beginning and ending along each spatial axis.
        Padding format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of
        pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`.
        Should have size equal to two times the number of dimension n (we don't count the channel dimension).
    dilation : Tuple
        Dilation value along each spatial axis of the filter
    groups : int
        Number of groups input channels and output channels are divided into
    has_bias : bool, optional
        Flag True if the convolutional layer has bias, False otherwise (default: False)
    bias : Tensor, optional
        Tensor containing the bias parameter of the Conv Layer (default: None)
    weight : Tensor, optional
        Tensor containing the weight parameters of the Conv layer (default: None)

    """

    def __init__(self, identifier: str, in_dim: Tuple, out_channels: int,
                 kernel_size: Tuple, stride: Tuple, padding: Tuple, dilation: Tuple, groups: int,
                 has_bias: bool = False, bias: Tensor = None, weight: Tensor = None):

        if not (len(in_dim) >= 2):
            raise Exception("The input dimension must be at least 2 (one for the channel and one for the rest)")

        if not (len(kernel_size) == len(in_dim) - 1):
            raise Exception("Size of kernel should be equal to the size of in_dim - 1")

        if not (len(stride) == len(in_dim) - 1):
            raise Exception("Size of stride should be equal to the size of in_dim - 1")

        if not (len(padding) == 2 * len(kernel_size)):
            raise Exception("Size of padding should be equal to 2 * size of kernel_size")

        if not (len(dilation) == len(in_dim) - 1):
            raise Exception("Size of dilation should be equal to the size of in_dim - 1")

        in_channels = in_dim[0]

        if not (in_channels % groups == 0 and out_channels % groups == 0):
            raise Exception("in_channels and out_channels must be divisible by groups")

        if not (in_channels == in_dim[0]):
            raise Exception("in_channel should be equals to the number of channels of the input "
                            f"(in_dim[0] = {in_dim[0]})")

        temp_out_dim = [out_channels]
        for i in range(1, len(in_dim)):
            aux = ((in_dim[i] + padding[i - 1] + padding[i + len(in_dim) - 2] -
                    dilation[i - 1] * (kernel_size[i - 1] - 1) - 1) / stride[i - 1]) + 1

            aux = math.floor(aux)

            temp_out_dim.append(aux)

        out_dim = tuple(temp_out_dim)

        super().__init__(identifier, in_dim, out_dim)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = has_bias

        k = groups / (in_channels * np.prod(kernel_size))

        weight_size = [out_channels, int(in_channels / groups)]
        for s in kernel_size:
            weight_size.append(s)
        weight_size = tuple(weight_size)
        if weight is None:
            weight = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=weight_size)

        if not (weight.shape == weight_size):
            raise Exception(f"Weight shape is wrong: it should be {weight_size}")

        if has_bias:
            if bias is None:
                bias = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=out_channels)
            else:
                if not (bias.shape == (out_channels,)):
                    raise Exception(f"Bias shape is wrong: it should be equal to ({out_channels},)")
        else:
            bias = None

        self.bias = bias
        self.weight = weight

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.out_channels,
                      self.kernel_size, self.stride, self.padding, self.dilation, self.groups,
                      self.has_bias, self.bias, self.weight)


class AveragePoolNode(LayerNode):
    """
    A class used for our internal representation of a AveragePool layer of a Neural Network.
    Also in this case the pytorch and onnx representation present incompatibilities. As in Batchnorm pytorch
    provide 3 different class for pooling based on the dimensionality of the input considered.
    Moreover the padding is forced to be symmetric and the parameter divisor_override is present (it is not clear
    what is its effect). The dimensionality supported for the input are (N, C, L), (N, C, H, W) and (N, C, D, H, W).
    In ONNX the padding can be asymmetric and the dimensionality supported is (N, C, D1, ... , Dn) where D1, ... Dn are
    the dimension on which the pooling is applied

    Attributes
    ----------
    kernel_size : Tuple
        The size of the kernel. Should have size equal to the number of dimension n
        (we don't count the channel dimension).
    stride : Tuple
        Stride along each spatial axis. Should have size equal to the number of dimension n
        (we don't count the channel dimension).
    padding : Tuple
        Padding for the beginning and ending along each spatial axis.
        Padding format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of
        pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`.
        Should have size equal to two times the number of dimension n (we don't count the channel dimension).
    ceil_mode : bool, optional
        In order to use ceil mode. (default: False)
    count_include_pad: bool, optional
        Whether include pad pixels when calculating values for the edges (default: False)

    """

    def __init__(self, identifier: str, in_dim: Tuple, kernel_size: Tuple, stride: Tuple,
                 padding: Tuple, ceil_mode: bool = False, count_include_pad: bool = False):

        if not (len(in_dim) >= 2):
            raise Exception("The input dimension must be at least 2 (one for the channel and one for the rest)")

        if not (len(kernel_size) == len(in_dim) - 1):
            raise Exception("Size of kernel should be equal to the size of in_dim - 1")

        if not (len(stride) == len(in_dim) - 1):
            raise Exception("Size of stride should be equal to the size of in_dim - 1")

        if not (len(padding) == 2 * len(kernel_size)):
            raise Exception("Size of padding should be equal to 2 * size of kernel_size")

        temp_out_dim = [in_dim[0]]
        for i in range(1, len(in_dim)):

            aux = ((in_dim[i] + padding[i - 1] + padding[i + len(in_dim) - 2] - kernel_size[i - 1]) / stride[i - 1]) + 1
            if ceil_mode:
                aux = math.ceil(aux)
            else:
                aux = math.floor(aux)

            temp_out_dim.append(aux)

        out_dim = tuple(temp_out_dim)

        super().__init__(identifier, in_dim, out_dim)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.kernel_size, self.stride,
                      self.padding, self.ceil_mode, self.count_include_pad)


class MaxPoolNode(LayerNode):
    """
    A class used for our internal representation of a MaxPool layer of a Neural Network.
    Also in this case the pytorch and onnx representation present incompatibilities. As in Batchnorm pytorch
    provide 3 different class for pooling based on the dimensionality of the input considered.
    Moreover the padding is forced to be symmetric. The dimensionality supported for the input
    are (N, C, L), (N, C, H, W) and (N, C, D, H, W).
    In ONNX the padding can be asymmetric and the dimensionality supported is (N, C, D1, ... , Dn) where D1, ... Dn are
    the dimension on which the pooling is applied

    Attributes
    ----------
    kernel_size : Tuple
        The size of the kernel. Should have size equal to the number of dimension n
        (we don't count the channel dimension).
    stride : Tuple
        Stride along each spatial axis. Should have size equal to the number of dimension n
        (we don't count the channel dimension).
    padding : Tuple
        Padding for the beginning and ending along each spatial axis.
        Padding format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of
        pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`.
        Should have size equal to two times the number of dimension n (we don't count the channel dimension).
    dilation : Tuple
        Dilation value along each spatial axis of the filter
    ceil_mode : bool, optional
        In order to use ceil mode. (default: False)
    return_indices: bool
        If True it will return the max indices along with the outputs (default: False)

    """

    def __init__(self, identifier: str, in_dim: Tuple, kernel_size: Tuple, stride: Tuple,
                 padding: Tuple, dilation: Tuple, ceil_mode: bool = False, return_indices: bool = False):

        if not (len(in_dim) >= 2):
            raise Exception("The input dimension must be at least 2 (one for the channel and one for the rest)")

        if not (len(kernel_size) == len(in_dim) - 1):
            raise Exception("Size of kernel should be equal to the size of in_dim - 1")

        if not (len(stride) == len(in_dim) - 1):
            raise Exception("Size of stride should be equal to the size of in_dim - 1")

        if not (len(padding) == 2 * len(kernel_size)):
            raise Exception("Size of padding should be equal to 2 * size of kernel_size")

        if not (len(dilation) == len(in_dim) - 1):
            raise Exception("Size of dilation should be equal to the size of in_dim - 1")

        temp_out_dim = [in_dim[0]]
        for i in range(1, len(in_dim)):

            aux = ((in_dim[i] + padding[i - 1] + padding[i + len(in_dim) - 2] -
                    dilation[i - 1] * (kernel_size[i - 1] - 1) - 1) / stride[i - 1]) + 1

            if ceil_mode:
                aux = math.ceil(aux)
            else:
                aux = math.floor(aux)

            temp_out_dim.append(aux)

        out_dim = tuple(temp_out_dim)

        super().__init__(identifier, in_dim, out_dim)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.dilation = dilation

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.kernel_size, self.stride,
                      self.padding, self.dilation, self.ceil_mode, self.return_indices)


class LRNNode(LayerNode):
    """
    A class used for our internal representation of a LocalResponseNormalization Layer of a Neural Network.

    Attributes
    ----------
    size : int
        Amount of neighbouring channels used for normalization
    alpha : float, optional
        Multiplicative factor (default: 0.0001)
    beta : float, optional
        Exponent. (default: 0.75)
    k : float, optional
        Additive factor (default: 1.0)

    """

    def __init__(self, identifier: str, in_dim: Tuple, size: int, alpha: float = 0.0001, beta: float = 0.75,
                 k: float = 1.0):
        if not (len(in_dim) >= 2):
            raise Exception("The input dimension must be at least 2 (one for the channel and one for the rest)")

        out_dim = copy.deepcopy(in_dim)
        super().__init__(identifier, in_dim, out_dim)
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.size, self.alpha, self.beta, self.k)


class SoftMaxNode(LayerNode):
    """
    A class used for our internal representation of a SoftMax Layer of a Neural Network.

    Attributes
    ----------
    axis : int, optional
        A dimension along which Softmax will be computed (so every slice along dim will sum to 1)

    """

    def __init__(self, identifier: str, in_dim: Tuple, axis: int = -1):

        if not (len(in_dim) >= 1):
            raise Exception("in_dim cannot be void")

        if not (-len(in_dim) <= axis <= len(in_dim) - 1):
            raise Exception(f"axis must be in [-{len(in_dim)}, {len(in_dim) - 1}]")

        out_dim = copy.deepcopy(in_dim)
        super().__init__(identifier, in_dim, out_dim)
        self.axis = axis

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.axis)


class UnsqueezeNode(LayerNode):
    """
    A class used for our internal representation of a Unsqueeze Layer.
    We follow the ONNX operator convention for attributes and definitions.
    Attributes
    ----------
    axes : Tuple
        List of indices at which to insert the singleton dimension.

    """

    def __init__(self, identifier: str, in_dim: Tuple, axes: Tuple):

        if not (len(in_dim) >= 1):
            raise Exception("in_dim cannot be void")

        if not (len(axes) == len(set(axes))):
            raise Exception("All elements in axes must be unique")

        if not (len(axes) > 0):
            raise Exception("axes cannot be void")

        check_axes_values = True
        for e in axes:
            if e < - (len(in_dim) + len(axes)) or e > (len(in_dim) + len(axes) - 1):
                check_axes_values = False

        if not check_axes_values:
            raise Exception(f"Every axes element must be in [{- (len(in_dim) + len(axes))}, "
                            f"{(len(in_dim) + len(axes) - 1)}]")

        # We add the singleton dimensions to the out_dim
        out_dim = copy.deepcopy(in_dim)
        out_dim = list(out_dim)
        temp_axes = list(axes)
        temp_axes.sort()
        for e in axes:
            out_dim.insert(e, 1)
        out_dim = tuple(out_dim)

        super().__init__(identifier, in_dim, out_dim)

        self.axes = axes

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.axes)


class ReshapeNode(LayerNode):
    """
    A class used for our internal representation of a Reshape layer of a Neural Network.
    We follow the ONNX operator convention for attributes and definitions.
    Attributes
    ----------
    shape : Tuple
        Tuple which specifies the output shape
    allow_zero : bool, optional
        By default, when any value in the 'shape' input is equal to zero the corresponding dimension value
        is copied from the input tensor dynamically. allowzero=1 indicates that if any value in the 'shape' input is
        set to zero, the zero value is honored, similar to NumPy. (default: False)


    """

    def __init__(self, identifier: str, in_dim: Tuple, shape: Tuple, allow_zero: bool = False):

        if not (list(shape).count(-1) <= 1):
            raise Exception("At most one dimension of the new shape can be -1")

        temp_shape = []
        for i in range(len(shape)):
            e = shape[i]
            if e == 0 and allow_zero:
                temp_shape.append(e)
            elif e == 0 and not allow_zero:
                if i >= len(in_dim):
                    raise Exception(f"0 value for new shape in position {i} but original shape has only "
                                    f"{len(in_dim)} elements")
                temp_shape.append(in_dim[i])
            else:
                temp_shape.append(e)

        # We leverage numpy reshape to compute our output dimension. If the reshape encounter a new shape which is
        # not valid numpy raise an exception which will be eventually caught in the gui.
        temp_input = np.ones(in_dim)
        temp_output = np.reshape(temp_input, temp_shape)
        out_dim = temp_output.shape

        super().__init__(identifier, in_dim, out_dim)
        self.shape = shape
        self.allow_zero = allow_zero

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.shape, self.allow_zero)


class FlattenNode(LayerNode):
    """
    A class used for our internal representation of a Flatten layer of a Neural Network. We follow the ONNX operator
    convention for attributes and definitions.
    Attributes
    ----------
    axis : int, optional
        Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output.
        The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value
        means counting dimensions from the back. When axis = 0, the shape of the output tensor is
        (1, (d_0 X d_1 ... d_n)), where the shape of the input tensor is (d_0, d_1, ... d_n).
        N.B: it works assuming the initial batch dimension. (default: 0)

    """

    def __init__(self, identifier: str, in_dim: Tuple, axis: int = 0):
        if not (-len(in_dim) <= axis <= len(in_dim)):
            raise Exception(f"Axis must be in [{-len(in_dim)}, {len(in_dim)}]")

        temp_input = np.ones(in_dim)
        new_shape = (-1,) if axis == 0 else (np.prod(in_dim[0:axis]).astype(int), -1)
        temp_output = np.reshape(temp_input, new_shape)

        # We leverage numpy reshape to compute our output dimension. If the reshape encounter a new shape which is
        # not valid numpy raise an exception which will be eventually caught in the gui.

        out_dim = temp_output.shape

        super().__init__(identifier, in_dim, out_dim)
        self.axis = axis

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.axis)


class DropoutNode(LayerNode):
    """
    A class used for our internal representation of a Dropout Layer of a Neural Network.
    The inplace parameter of pytorch and the seed attribute and training_mode of onnx are not supported.
    Attributes
    ----------
    p : float, optional
        Probability of an element to be zeroed (default: 0.5)

    """

    def __init__(self, identifier: str, in_dim: Tuple, p: float = 0.5):
        out_dim = copy.deepcopy(in_dim)
        super().__init__(identifier, in_dim, out_dim)
        if not (0 <= p <= 1):
            raise Exception("The p parameter must be between [0, 1]")
        self.p = p

    def update_input(self, in_dim: Tuple):
        self.__init__(self.identifier, in_dim, self.p)
