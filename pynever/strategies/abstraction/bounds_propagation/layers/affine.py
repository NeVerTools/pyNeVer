import torch

from pynever import tensors
from pynever.strategies.abstraction.bounds_propagation.bounds import SymbolicLinearBounds
from pynever.strategies.abstraction.bounds_propagation import util
from pynever.strategies.abstraction.linearfunctions import LinearFunctions


def compute_dense_output_bounds(layer, inputs):
    weights = torch.from_numpy(layer.weight).float()
    weights_plus = tensors.get_positive(weights)
    weights_minus = tensors.get_negative(weights)

    if layer.bias is None:
        layer.bias = torch.zeros(layer.weight.shape[0], device=layer.weight.device)

    lm = inputs.get_lower_matrix()
    um = inputs.get_upper_matrix()
    lo = inputs.get_lower_offset()
    uo = inputs.get_upper_offset()

    lower_matrix, lower_offset, upper_matrix, upper_offset = \
        util.compute_lower(weights_minus, weights_plus, lm, um), \
            util.compute_lower(weights_minus, weights_plus, lo, uo) + layer.bias, \
            util.compute_upper(weights_minus, weights_plus, lm, um), \
            util.compute_upper(weights_minus, weights_plus, lo, uo) + layer.bias

    return SymbolicLinearBounds(LinearFunctions(lower_matrix, lower_offset),
                                LinearFunctions(upper_matrix, upper_offset))
