import torch
import torch.nn as nn


class Sequential(nn.Sequential):
    """
    Custom representation of pytorch Sequential Network. It adds the identifier for the network and for the input

    Attributes
    ----------
    identifier : str
        Identifier for the network
    input_id : str
        Identifier for the input of the network
    """

    def __init__(self, identifier: str, input_id: str, modules):
        super().__init__(*modules)
        self.identifier = identifier
        self.input_id = input_id


class ReLU(nn.ReLU):
    """
    Custom representation of pytorch ReLU Layer. It adds the identifier for the node, the input size and the output
    size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class ELU(nn.ELU):
    """
    Custom representation of pytorch ELU Layer. It adds the identifier for the node, the input size and the output
    size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, alpha: float):
        super().__init__(alpha)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class LeakyReLU(nn.LeakyReLU):
    """
    Custom representation of pytorch LeakyReLU Layer. It adds the identifier for the node, the input size and the output
    size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, negative_slope: float):
        super().__init__(negative_slope)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class CELU(nn.CELU):
    """
    Custom representation of pytorch CELU Layer. It adds the identifier for the node, the input size and the output
    size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, alpha: float):
        super().__init__(alpha)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Sigmoid(nn.Sigmoid):
    """
    Custom representation of pytorch Sigmoid Layer. It adds the identifier for the node, the input size and the output
    size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Tanh(nn.Tanh):
    """
    Custom representation of pytorch Tanh Layer. It adds the identifier for the node, the input size and the output
    size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Hardtanh(nn.Hardtanh):
    """
        Custom representation of pytorch Hardtanh Layer. It adds the identifier for the node,
        the input size and the output size.

        Attributes
        ----------
        identifier : str
            Identifier for the node
        in_dim : tuple
            tuple expressing the dimension of the input
        out_dim : tuple
            tuple expressing the dimension of the output

        """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__(min_val, max_val)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Linear(nn.Linear):
    """
    Custom representation of pytorch Linear Layer. It adds the identifier for the node, the input size and the output
    size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, in_features: int, out_features: int,
                 bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class BatchNorm1d(nn.BatchNorm1d):
    """
    Custom representation of pytorch BatchNorm1d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, num_features: int, eps: float = 1e-05,
                 momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class BatchNorm2d(nn.BatchNorm2d):
    """
    Custom representation of pytorch BatchNorm2d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, num_features: int, eps: float = 1e-05,
                 momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class BatchNorm3d(nn.BatchNorm3d):
    """
    Custom representation of pytorch BatchNorm3d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, num_features: int, eps: float = 1e-05,
                 momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Conv1d(nn.Conv1d):
    """
    Custom representation of pytorch Conv1d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, in_channels: int, out_channels: int,
                 kernel_size: tuple, stride: tuple, padding: tuple, dilation: tuple, groups: int, bias: bool):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Conv2d(nn.Conv2d):
    """
    Custom representation of pytorch Conv2d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, in_channels: int, out_channels: int,
                 kernel_size: tuple, stride: tuple, padding: tuple, dilation: tuple, groups: int, bias: bool):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Conv3d(nn.Conv3d):
    """
    Custom representation of pytorch Conv3d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, in_channels: int, out_channels: int,
                 kernel_size: tuple, stride: tuple, padding: tuple, dilation: tuple, groups: int, bias: bool):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class AvgPool1d(nn.AvgPool1d):
    """
    Custom representation of pytorch AvgPool1d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, kernel_size: tuple, stride: tuple,
                 padding: tuple,
                 ceil_mode: bool, count_include_pad: bool):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class AvgPool2d(nn.AvgPool2d):
    """
    Custom representation of pytorch AvgPool2d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, kernel_size: tuple, stride: tuple,
                 padding: tuple,
                 ceil_mode: bool, count_include_pad: bool):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class AvgPool3d(nn.AvgPool3d):
    """
    Custom representation of pytorch AvgPool3d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, kernel_size: tuple, stride: tuple,
                 padding: tuple,
                 ceil_mode: bool, count_include_pad: bool):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class MaxPool1d(nn.MaxPool1d):
    """
    Custom representation of pytorch MaxPool1d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, kernel_size: tuple, stride: tuple,
                 padding: tuple,
                 dilation: tuple, return_indices: bool, ceil_mode: bool):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class MaxPool2d(nn.MaxPool2d):
    """
    Custom representation of pytorch MaxPool2d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, kernel_size: tuple, stride: tuple,
                 padding: tuple,
                 dilation: tuple, return_indices: bool, ceil_mode: bool):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class MaxPool3d(nn.MaxPool3d):
    """
    Custom representation of pytorch MaxPool3d Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, kernel_size: tuple, stride: tuple,
                 padding: tuple,
                 dilation: tuple, return_indices: bool, ceil_mode: bool):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class LocalResponseNorm(nn.LocalResponseNorm):
    """
    Custom representation of pytorch LocalResponseNorm Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, size: int, alpha: float, beta: float, k: float):
        super().__init__(size, alpha, beta, k)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Softmax(nn.Softmax):
    """
    Custom representation of pytorch Softmax Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, dim: int):
        super().__init__(dim)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Dropout(nn.Dropout):
    """
    Custom representation of pytorch Dropout Layer. It adds the identifier for the node,
    the input size and the output size.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output

    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, p: float):
        super().__init__(p)
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim


class Unsqueeze(nn.Module):
    """
    Custom class for pytorch Unsqueeze layer. It conforms to our representation and ONNX.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output
    axes : tuple
        List of indices at which to insert the singleton dimension.
    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, axes: tuple):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.axes = axes

    def forward(self, x: torch.Tensor):
        for ax in self.axes:
            x = torch.unsqueeze(x, ax)
        return x


class Reshape(nn.Module):
    """
    Custom class for pytorch Reshape layer. It conforms to our representation and ONNX.
    Torch reshape function does not support zeros in the shape, therefore it cannot support the allow_zero attribute
    of our representation.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output
    shape : tuple
        tuple which specifies the output shape
    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, shape: tuple):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.shape = shape

    def forward(self, x: torch.Tensor):
        x = torch.reshape(x, self.shape)
        return x


class Flatten(nn.Module):
    """
    Custom class for pytorch Flatten layer. It conforms to our representation and ONNX.

    Attributes
    ----------
    identifier : str
        Identifier for the node
    in_dim : tuple
        tuple expressing the dimension of the input
    out_dim : tuple
        tuple expressing the dimension of the output
    axis : int
        Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output.
        The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value
        means counting dimensions from the back. Pytorch works assuming the presence of a batch dimension
        for its tensor therefore we set the default to 1 so that the batch dimension is preserved.(default: 1)
    """

    def __init__(self, identifier: str, in_dim: tuple, out_dim: tuple, axis: int = 1):
        super().__init__()
        self.identifier = identifier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.axis = axis

    def forward(self, x: torch.Tensor):
        if len(x.shape) > 1:
            # Given our representation we exclude the batch dimension from the operation.
            x = torch.flatten(x, self.axis, -1)

        x = torch.flatten(x, 0, self.axis - 1)

        return x
