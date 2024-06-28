"""Utility functions collection

This module contains utility functions which may be used throughout the code.

"""
import numpy
import torch

import pynever.networks as networks
from pynever.strategies.conversion.converters.pytorch import PyTorchConverter
from pynever.tensors import Tensor


def execute_network(network: networks.NeuralNetwork, net_input: Tensor) -> Tensor:
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

    return output.detach().numpy().T
