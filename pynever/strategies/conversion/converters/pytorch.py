import pynever.strategies.conversion.converters.pytorch_layers as pyt_l
from pynever.strategies.conversion.representation import ConversionStrategy, PyTorchNetwork
import torch

import pynever.networks as networks
import pynever.nodes as nodes


class PyTorchConverter(ConversionStrategy):
    """
    A class used to represent the conversion strategy for PyTorch models.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to a PyTorchNetwork model.
    to_neural_network(PyTorchNetwork)
        Convert the PyTorchNetwork of interest to our internal representation of a Neural Network.

    """

    def from_neural_network(self, network: networks.NeuralNetwork) -> PyTorchNetwork:
        """
        Convert the neural network of interest to a PyTorch representation.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        PyTorchNetwork
            The PyTorch representation resulting from the conversion of the original network.

        """
        pytorch_network = None
        if isinstance(network, networks.SequentialNetwork):
            pytorch_layers = []
            for layer in network.nodes.values():
                if not isinstance(layer.get_input_dim(), tuple):
                    raise Exception('Only layers with a single input are currently supported')

                if isinstance(layer, nodes.ReLUNode):
                    new_layer = pyt_l.ReLU(layer.identifier, layer.get_input_dim(), layer.out_dim)

                elif isinstance(layer, nodes.ELUNode):
                    new_layer = pyt_l.ELU(layer.identifier, layer.get_input_dim(), layer.out_dim, layer.alpha)

                elif isinstance(layer, nodes.LeakyReLUNode):
                    new_layer = pyt_l.LeakyReLU(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                layer.negative_slope)

                elif isinstance(layer, nodes.CELUNode):
                    new_layer = pyt_l.CELU(layer.identifier, layer.get_input_dim(), layer.out_dim, layer.alpha)

                elif isinstance(layer, nodes.SigmoidNode):
                    new_layer = pyt_l.Sigmoid(layer.identifier, layer.get_input_dim(), layer.out_dim)

                elif isinstance(layer, nodes.TanhNode):
                    new_layer = pyt_l.Tanh(layer.identifier, layer.get_input_dim(), layer.out_dim)

                elif isinstance(layer, nodes.FullyConnectedNode):

                    if layer.bias is not None:
                        has_bias = True
                    else:
                        has_bias = False

                    new_layer = pyt_l.Linear(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                             in_features=layer.in_features, out_features=layer.out_features,
                                             bias=has_bias)

                    weight = torch.from_numpy(layer.weight)
                    new_layer.weight.data = weight

                    if has_bias:
                        bias = torch.from_numpy(layer.bias)
                        new_layer.bias.data = bias

                elif isinstance(layer, nodes.BatchNormNode):

                    if len(layer.get_input_dim()) == 1 or len(layer.get_input_dim()) == 2:

                        new_layer = pyt_l.BatchNorm1d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                      num_features=layer.num_features,
                                                      eps=layer.eps, momentum=layer.momentum,
                                                      affine=layer.affine,
                                                      track_running_stats=layer.track_running_stats)
                    elif len(layer.get_input_dim()) == 3:

                        new_layer = pyt_l.BatchNorm2d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                      num_features=layer.num_features,
                                                      eps=layer.eps, momentum=layer.momentum,
                                                      affine=layer.affine,
                                                      track_running_stats=layer.track_running_stats)

                    elif len(layer.get_input_dim()) == 4:

                        new_layer = pyt_l.BatchNorm3d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                      num_features=layer.num_features,
                                                      eps=layer.eps, momentum=layer.momentum,
                                                      affine=layer.affine,
                                                      track_running_stats=layer.track_running_stats)

                    else:
                        raise Exception("Pytorch does not support batchnorm layer for input with more than"
                                        "4 or less than 1 dimension excluding the batch dimension")

                    new_layer.weight.data = torch.from_numpy(layer.weight)
                    new_layer.bias.data = torch.from_numpy(layer.bias)
                    new_layer.running_mean.data = torch.from_numpy(layer.running_mean)
                    new_layer.running_var.data = torch.from_numpy(layer.running_var)

                elif isinstance(layer, nodes.ConvNode):

                    # Pytorch support only symmetric padding, therefore we assume that the padding given is
                    # symmetric. Padding mode is not supported in our representation therefore we let it be
                    # set to the default value.
                    padding = layer.padding[:int(len(layer.padding) / 2)]

                    if len(layer.get_input_dim()) == 2:

                        new_layer = pyt_l.Conv1d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                 layer.in_channels, layer.out_channels, layer.kernel_size,
                                                 layer.stride, padding, layer.dilation, layer.groups,
                                                 layer.has_bias)

                    elif len(layer.get_input_dim()) == 3:

                        new_layer = pyt_l.Conv2d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                 layer.in_channels, layer.out_channels, layer.kernel_size,
                                                 layer.stride, padding, layer.dilation, layer.groups,
                                                 layer.has_bias)

                    elif len(layer.get_input_dim()) == 4:

                        new_layer = pyt_l.Conv3d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                 layer.in_channels, layer.out_channels, layer.kernel_size,
                                                 layer.stride, padding, layer.dilation, layer.groups,
                                                 layer.has_bias)

                    else:
                        raise Exception("Pytorch does not support Conv layer for input with more than"
                                        "4 or less than 2 dimension excluding the batch dimension")

                    new_layer.weight.data = torch.from_numpy(layer.weight)
                    if layer.has_bias:
                        new_layer.bias.data = torch.from_numpy(layer.bias)

                elif isinstance(layer, nodes.AveragePoolNode):

                    # Pytorch support only symmetric padding, therefore we assume that the padding given is
                    # symmetric.
                    padding = layer.padding[:int(len(layer.padding) / 2)]

                    if len(layer.get_input_dim()) == 2:

                        new_layer = pyt_l.AvgPool1d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                    layer.kernel_size, layer.stride, padding,
                                                    layer.ceil_mode, layer.count_include_pad)

                    elif len(layer.get_input_dim()) == 3:

                        new_layer = pyt_l.AvgPool2d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                    layer.kernel_size, layer.stride, padding,
                                                    layer.ceil_mode, layer.count_include_pad)

                    elif len(layer.get_input_dim()) == 4:

                        new_layer = pyt_l.AvgPool3d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                    layer.kernel_size, layer.stride, padding,
                                                    layer.ceil_mode, layer.count_include_pad)

                    else:
                        raise Exception("Pytorch does not support AvgPool layer for input with more than"
                                        "4 or less than 2 dimension excluding the batch dimension")

                elif isinstance(layer, nodes.MaxPoolNode):

                    # Pytorch support only symmetric padding, therefore we assume that the padding given is
                    # symmetric.
                    padding = layer.padding[:int(len(layer.padding) / 2)]

                    if len(layer.get_input_dim()) == 2:

                        new_layer = pyt_l.MaxPool1d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                    layer.kernel_size, layer.stride, padding,
                                                    layer.dilation, layer.return_indices, layer.ceil_mode)

                    elif len(layer.get_input_dim()) == 3:

                        new_layer = pyt_l.MaxPool2d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                    layer.kernel_size, layer.stride, padding,
                                                    layer.dilation, layer.return_indices, layer.ceil_mode)

                    elif len(layer.get_input_dim()) == 4:

                        new_layer = pyt_l.MaxPool3d(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                    layer.kernel_size, layer.stride, padding,
                                                    layer.dilation, layer.return_indices, layer.ceil_mode)

                    else:
                        raise Exception("Pytorch does not support Conv layer for input with more than"
                                        "4 or less than 2 dimension excluding the batch dimension")

                elif isinstance(layer, nodes.LRNNode):

                    new_layer = pyt_l.LocalResponseNorm(layer.identifier, layer.get_input_dim(), layer.out_dim,
                                                        layer.size, layer.alpha, layer.beta, layer.k)

                elif isinstance(layer, nodes.SoftMaxNode):

                    # We need to scale the axis by one since our representation does not support the batch dimension
                    new_layer = pyt_l.Softmax(layer.identifier, layer.get_input_dim(), layer.out_dim, layer.axis + 1)

                elif isinstance(layer, nodes.UnsqueezeNode):

                    # Our representation does not consider batch dimension, therefore we need to scale
                    # the axes values.
                    axes = tuple([e + 1 for e in layer.axes])
                    new_layer = pyt_l.Unsqueeze(layer.identifier, layer.get_input_dim(), layer.out_dim, axes)

                elif isinstance(layer, nodes.ReshapeNode):

                    # Pytorch does not support the allow_zero attribute and the corresponding reshape with 0
                    # dimensions.
                    if layer.allow_zero:
                        raise Exception("allow_zero not supported by pytorch")

                    # Our representation does not consider batch dimension, therefore we need to add it to
                    # the shape.
                    shape = [1]
                    for e in layer.shape:
                        shape.append(e)
                    shape = tuple(shape)

                    new_layer = pyt_l.Reshape(layer.identifier, layer.get_input_dim(), layer.out_dim, shape)

                elif isinstance(layer, nodes.FlattenNode):

                    # We need to scale the axis by one since our representation does not support the batch dimension
                    new_layer = pyt_l.Flatten(layer.identifier, layer.get_input_dim(), layer.out_dim, layer.axis + 1)

                elif isinstance(layer, nodes.DropoutNode):

                    new_layer = pyt_l.Dropout(layer.identifier, layer.get_input_dim(), layer.out_dim, layer.p)

                else:
                    raise NotImplementedError

                if new_layer is not None:
                    pytorch_layers.append(new_layer)

            pytorch_network = pyt_l.Sequential(network.identifier, network.get_input_id(), pytorch_layers)

        if pytorch_network is None:
            raise Exception("WARNING: network to convert is not valid, the alternative representation is None")

        alt_net = PyTorchNetwork(network.identifier, pytorch_network)

        return alt_net

    def to_neural_network(self, alt_rep: PyTorchNetwork) -> networks.NeuralNetwork:
        """
        Convert the PyTorch representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : PyTorchNetwork
            The PyTorch Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of PyTorch Representation.

        """

        identifier = alt_rep.identifier
        if hasattr(alt_rep.pytorch_network, 'input_id'):
            input_id = alt_rep.pytorch_network.input_id
        else:
            input_id = 'X'

        network = networks.SequentialNetwork(identifier, input_id)

        node_index = 0
        alt_rep.pytorch_network.cpu()

        layer_in_dim = None

        for m in alt_rep.pytorch_network.modules():

            # Control input
            if hasattr(m, 'in_dim'):
                layer_in_dim = m.in_dim

            if layer_in_dim is None and not isinstance(m, pyt_l.Sequential):
                print('Please provide input dimension for the network:')
                layer_in_dim = input()
                layer_in_dim = (layer_in_dim,)

            if hasattr(m, 'identifier'):
                layer_id = m.identifier
            else:
                layer_id = f"Layer{node_index}"

            # Read node
            new_node = None

            if isinstance(m, pyt_l.ReLU):
                new_node = nodes.ReLUNode(layer_id, layer_in_dim)

            elif isinstance(m, pyt_l.ELU):
                new_node = nodes.ELUNode(layer_id, layer_in_dim, m.alpha)

            elif isinstance(m, pyt_l.LeakyReLU):
                new_node = nodes.LeakyReLUNode(layer_id, layer_in_dim, m.negative_slope)

            elif isinstance(m, pyt_l.CELU):
                new_node = nodes.CELUNode(layer_id, layer_in_dim, m.alpha)

            elif isinstance(m, pyt_l.Sigmoid):
                new_node = nodes.SigmoidNode(layer_id, layer_in_dim)

            elif isinstance(m, pyt_l.Tanh):
                new_node = nodes.TanhNode(m.identifier, m.in_dim)

            elif isinstance(m, pyt_l.Linear):
                out_features = m.out_features
                weight = m.weight.detach().numpy()
                bias = None
                has_bias = False
                if m.bias is not None:
                    bias = m.bias.detach().numpy()
                    has_bias = True
                new_node = nodes.FullyConnectedNode(layer_id, layer_in_dim, out_features, weight, bias, has_bias)

            elif isinstance(m, pyt_l.BatchNorm1d) or isinstance(m, pyt_l.BatchNorm2d) or \
                    isinstance(m, pyt_l.BatchNorm3d):

                eps = m.eps
                momentum = m.momentum
                track_running_stats = m.track_running_stats
                affine = m.affine

                weight = m.weight.detach().numpy()
                bias = m.bias.detach().numpy()
                running_mean = m.running_mean.numpy()
                running_var = m.running_var.numpy()

                new_node = nodes.BatchNormNode(layer_id, layer_in_dim, weight,
                                               bias, running_mean, running_var, eps, momentum, affine,
                                               track_running_stats)

            elif isinstance(m, pyt_l.Conv1d) or isinstance(m, pyt_l.Conv2d) or isinstance(m, pyt_l.Conv3d):

                out_channels = m.out_channels
                kernel_size = m.kernel_size
                stride = m.stride
                temp_padding = list(m.padding)
                for e in m.padding:
                    temp_padding.append(e)
                padding = tuple(temp_padding)
                dilation = m.dilation
                groups = m.groups
                weight = m.weight.detach().numpy()
                if m.bias is None:
                    has_bias = False
                    bias = None
                else:
                    has_bias = True
                    bias = m.bias.detach().numpy()

                new_node = nodes.ConvNode(layer_id, layer_in_dim, out_channels, kernel_size,
                                          stride, padding, dilation, groups, has_bias, bias, weight)

            elif isinstance(m, pyt_l.AvgPool1d) or isinstance(m, pyt_l.AvgPool2d) or \
                    isinstance(m, pyt_l.AvgPool3d):

                stride = m.stride
                temp_padding = list(m.padding)
                for e in m.padding:
                    temp_padding.append(e)
                padding = tuple(temp_padding)
                kernel_size = m.kernel_size
                ceil_mode = m.ceil_mode
                count_include_pad = m.count_include_pad

                new_node = nodes.AveragePoolNode(layer_id, layer_in_dim, kernel_size, stride, padding,
                                                 ceil_mode, count_include_pad)

            elif isinstance(m, pyt_l.MaxPool1d) or isinstance(m, pyt_l.MaxPool2d) or \
                    isinstance(m, pyt_l.MaxPool3d):

                stride = m.stride
                temp_padding = list(m.padding)
                for e in m.padding:
                    temp_padding.append(e)
                padding = tuple(temp_padding)
                kernel_size = m.kernel_size
                ceil_mode = m.ceil_mode
                dilation = m.dilation
                return_indices = m.return_indices

                new_node = nodes.MaxPoolNode(layer_id, layer_in_dim, kernel_size, stride, padding, dilation,
                                             ceil_mode, return_indices)

            elif isinstance(m, pyt_l.LocalResponseNorm):

                new_node = nodes.LRNNode(layer_id, layer_in_dim, m.size, m.alpha, m.beta, m.k)

            elif isinstance(m, pyt_l.Softmax):

                new_node = nodes.SoftMaxNode(layer_id, layer_in_dim, m.dim - 1)

            elif isinstance(m, pyt_l.Unsqueeze):

                axes = tuple([e - 1 for e in m.axes])
                new_node = nodes.UnsqueezeNode(layer_id, layer_in_dim, axes)

            elif isinstance(m, pyt_l.Reshape):

                shape = m.shape[1:]
                new_node = nodes.ReshapeNode(layer_id, layer_in_dim, shape)

            elif isinstance(m, pyt_l.Flatten):

                new_node = nodes.FlattenNode(layer_id, layer_in_dim, m.axis - 1)

            elif isinstance(m, pyt_l.Dropout):

                new_node = nodes.DropoutNode(layer_id, layer_in_dim, m.p)

            elif isinstance(m, pyt_l.Sequential):
                pass

            else:
                raise NotImplementedError

            if new_node is not None:
                node_index += 1
                network.append_node(new_node)
                layer_in_dim = network.get_last_node().out_dim

        return network
