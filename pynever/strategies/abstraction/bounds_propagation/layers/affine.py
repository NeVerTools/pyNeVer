import torch

from pynever import tensors
from pynever.nodes import FullyConnectedNode
from pynever.strategies.abstraction.bounds_propagation import util
from pynever.strategies.abstraction.bounds_propagation.bounds import SymbolicLinearBounds
from pynever.strategies.abstraction.linearfunctions import LinearFunctions


def get_layer_equation(layer: FullyConnectedNode) -> LinearFunctions:
    return LinearFunctions(layer.weight, layer.bias)


def compute_dense_output_bounds(layer: FullyConnectedNode, inputs: SymbolicLinearBounds) -> SymbolicLinearBounds:
    weights = torch.from_numpy(layer.weight).float()
    weights_plus = tensors.get_positive(weights)
    weights_minus = tensors.get_negative(weights)

    lm = inputs.get_lower().get_matrix()
    um = inputs.get_upper().get_matrix()
    lo = inputs.get_lower().get_offset()
    uo = inputs.get_upper().get_offset()

    lower_matrix, lower_offset, upper_matrix, upper_offset = \
        util.compute_lower(weights_minus, weights_plus, lm, um), \
            util.compute_lower(weights_minus, weights_plus, lo, uo) + layer.bias, \
            util.compute_upper(weights_minus, weights_plus, lm, um), \
            util.compute_upper(weights_minus, weights_plus, lo, uo) + layer.bias

    return SymbolicLinearBounds(LinearFunctions(lower_matrix, lower_offset),
                                LinearFunctions(upper_matrix, upper_offset))
