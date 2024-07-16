import abc
import copy
import os
from abc import abstractmethod
from copy import deepcopy

import onnx
import torch

import pynever.networks as networks


class AlternativeRepresentation(abc.ABC):
    """
    An abstract class used to represent an alternative representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation

    """

    def __init__(self, path: str, identifier: str | None = None):
        self.path = path
        self.identifier = identifier

        if identifier is None:
            self.identifier = '.'.join(os.path.basename(path).split('.')[:-1])

    @abstractmethod
    def save(self, new_path: str):
        raise NotImplementedError


class ONNXNetwork(AlternativeRepresentation):
    """
    A class used to represent a ONNX representation for a neural network.

    Attributes
    ----------
    onnx_network : onnx.ModelProto
        Real ONNX network.

    """

    def __init__(self, *args):
        """
        Two possible init methods:

        1. Path-based: the network is loaded from the path (len(args) == 1)
        2. Representation-based: the network is initialized with an identifier and a model (len(args) == 2)

        """

        self.onnx_network = None

        if len(args) == 1:
            """ The network is loaded from a path """
            if isinstance(args[0], str):
                super().__init__(args[0])
                self.onnx_network = onnx.load(self.path)

            else:
                raise ValueError('Incorrect path constructor')

        elif len(args) == 2:
            """ The network is loaded from a model """
            if isinstance(args[0], str) and isinstance(args[1], onnx.ModelProto):
                super().__init__(f'{args[0]}.onnx', args[0])
                self.onnx_network = copy.deepcopy(args[1])

            else:
                raise ValueError('Incorrect model constructor')

        else:
            raise Exception('Incorrect parameters passed to constructor')

    def save(self, new_path: str):
        onnx.save(self.onnx_network, new_path)


class PyTorchNetwork(AlternativeRepresentation):
    """
    A class used to represent a PyTorch representation for a neural network.

    Attributes
    ----------
        identifier for the alternative representation
    pytorch_network : torch.nn.Module
        Real PyTorch network.

    """

    def __init__(self, *args):
        """
        Two possible init methods:

        1. Path-based: the network is loaded from the path (len(args) == 1)
        2. Representation-based: the network is initialized with an identifier and a model (len(args) == 2)

        """

        self.pytorch_network = None

        if len(args) == 1:
            """ The network is loaded from a path """
            if isinstance(args[0], str):
                super().__init__(args[0])
                self.pytorch_network = torch.load(self.path, weights_only=True)

            else:
                raise ValueError('Incorrect path constructor')

        elif len(args) == 2:
            """ The network is loaded from a model """
            if isinstance(args[0], str) and isinstance(args[1], torch.nn.Module):
                super().__init__(f'{args[0]}.pt', args[0])
                self.pytorch_network = copy.deepcopy(args[1])

            else:
                raise ValueError('Incorrect model constructor')

        else:
            raise Exception('Incorrect parameters passed to constructor')

    def save(self, new_path: str):
        torch.save(self.pytorch_network, new_path)


class ConversionStrategy(abc.ABC):
    """
    An abstract class used to represent a Conversion Strategy.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to an alternative representation determined in the concrete children.
    to_neural_network(AlternativeRepresentation)
        Convert the alternative representation of interest to our internal representation of a Neural Network.

    """

    @abc.abstractmethod
    def from_neural_network(self, network: networks.NeuralNetwork) -> AlternativeRepresentation:
        """
        Convert the neural network of interest to an alternative representation determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        AlternativeRepresentation
            The alternative representation resulting from the conversion of the original network.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_neural_network(self, alt_rep: AlternativeRepresentation) -> networks.NeuralNetwork:
        """
        Convert the alternative representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : AlternativeRepresentation
            The Alternative Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of Alternative Representation.
        """
        raise NotImplementedError


def load_network_path(path: str) -> AlternativeRepresentation:
    """
    Method to load a network from a path in an Alternative Representation.

    Parameters
    ----------
    path : str
        Path to the network.

    Returns
    -------
    Optional[AlternativeRepresentation]
        The AlternativeRepresentation object if the network is supported, None otherwise.

    """

    extension = path.split('.')[-1]

    if extension in ['pt', 'pth']:
        return PyTorchNetwork(path)

    if extension == 'onnx':
        return ONNXNetwork(path)

    raise Exception('Only PyTorch and ONNX are currently supported')
