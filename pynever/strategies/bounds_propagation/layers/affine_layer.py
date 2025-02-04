from pynever.strategies.bounds_propagation.utility.functions import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions
from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds
import torch

def compute_dense_output_bounds(layer, inputs):
    weights = torch.from_numpy(layer.weight).float()
    weights_plus = get_positive_part(weights)
    weights_minus = get_negative_part(weights)

    if layer.bias is None:
        layer.bias = torch.zeros(layer.weight.shape[0], device=layer.weight.device)

    lower_matrix, lower_offset, upper_matrix, upper_offset = \
        compute_lin_lower_and_upper(weights_minus, weights_plus, layer.bias,
                                    inputs.get_lower().get_matrix(),
                                    inputs.get_upper().get_matrix(),
                                    inputs.get_lower().get_offset(),
                                    inputs.get_upper().get_offset())

    return SymbolicLinearBounds(LinearFunctions(lower_matrix, lower_offset),
                                LinearFunctions(upper_matrix, upper_offset))
