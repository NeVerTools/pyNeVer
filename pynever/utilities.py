"""Utility functions collection

This module contains utility functions which may be used throughout the code.

"""
import copy

import numpy
import torch

from pynever import networks
from pynever import tensors
from pynever.strategies.conversion.converters import pytorch_layers as ptl
from pynever.strategies.conversion.converters.pytorch import PyTorchConverter
from pynever.strategies.conversion.representation import PyTorchNetwork


def execute_network(network: networks.NeuralNetwork, net_input: tensors.Tensor) -> tensors.Tensor:
    """Applies the neural network function to an input Tensor using pyTorch backend.

    Parameters
    ----------
    network : NeuralNetwork
        The network to execute
    net_input : Tensor
        The input value to feed

    Returns
    -------
    Tensor
        The computed output

    """

    if net_input.dtype == numpy.object_:
        print("net_input", net_input)

    input_t = torch.Tensor(net_input)

    py_net = PyTorchConverter().from_neural_network(network)
    py_net.pytorch_network.eval()
    py_net.pytorch_network.float()

    output = py_net.pytorch_network(input_t.float().permute(*torch.arange(input_t.ndim - 1, -1, -1)))

    return tensors.array(output.detach().numpy().T)


def combine_batchnorm1d(linear: ptl.Linear, batchnorm: ptl.BatchNorm1d) -> ptl.Linear:
    """
    Utility function to combine a BatchNorm1D node with a Linear node in a corresponding Linear node.
    Parameters
    ----------
    linear : Linear
        Linear to combine.
    batchnorm : BatchNorm1D
        BatchNorm1D to combine.
    Return
    ----------
    Linear
        The Linear resulting from the fusion of the two input nodes.

    """

    l_weight = linear.weight
    l_bias = linear.bias
    bn_running_mean = batchnorm.running_mean
    bn_running_var = batchnorm.running_var
    bn_weight = batchnorm.weight
    bn_bias = batchnorm.bias
    bn_eps = batchnorm.eps

    fused_bias = torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps))
    fused_bias = torch.mul(fused_bias, torch.sub(l_bias, bn_running_mean))
    fused_bias = torch.add(fused_bias, bn_bias)

    fused_weight = torch.diag(torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps)))
    fused_weight = torch.matmul(fused_weight, l_weight)

    has_bias = linear.bias is not None
    fused_linear = ptl.Linear(linear.identifier, linear.in_dim, linear.out_dim, linear.in_features, linear.out_features,
                              has_bias)

    p_fused_weight = torch.nn.Parameter(fused_weight, requires_grad=False)
    p_fused_bias = torch.nn.Parameter(fused_bias, requires_grad=False)

    fused_linear.weight = p_fused_weight
    fused_linear.bias = p_fused_bias

    return fused_linear


def combine_batchnorm1d_net(network: networks.SequentialNetwork) -> networks.NeuralNetwork:
    """
    Utilities function to combine all the FullyConnectedNodes followed by BatchNorm1DNodes in corresponding
    FullyConnectedNodes.
    Parameters
    ----------
    network : SequentialNetwork
        Sequential Network of interest of which we want to combine the nodes.
    Return
    ----------
    SequentialNetwork
        Corresponding Sequential Network with the combined nodes.

    """

    py_net = PyTorchConverter().from_neural_network(network)

    modules = [m for m in py_net.pytorch_network.modules()]
    modules = modules[1:]
    num_modules = len(modules)
    current_index = 0

    new_modules = []

    while current_index + 1 < num_modules:

        current_node = modules[current_index]
        next_node = modules[current_index + 1]

        if isinstance(current_node, ptl.Linear) and isinstance(next_node, ptl.BatchNorm1d):
            combined_node = combine_batchnorm1d(current_node, next_node)
            new_modules.append(combined_node)
            current_index = current_index + 1

        elif isinstance(current_node, ptl.Linear):
            new_modules.append(copy.deepcopy(current_node))

        elif isinstance(current_node, ptl.ReLU):
            new_modules.append(copy.deepcopy(current_node))

        else:
            raise Exception("Combine Batchnorm supports only ReLU, Linear and BatchNorm1D layers.")

        current_index = current_index + 1

    if not isinstance(modules[current_index], ptl.BatchNorm1d):
        new_modules.append(copy.deepcopy(modules[current_index]))

    temp_pynet = ptl.Sequential(py_net.pytorch_network.identifier, py_net.pytorch_network.input_id, new_modules)
    combined_pynet = PyTorchNetwork(py_net.identifier, temp_pynet)
    combined_network = PyTorchConverter().to_neural_network(combined_pynet)

    return combined_network
