import torch

from pynever import tensors
from pynever.networks import NeuralNetwork
from pynever.nodes import FullyConnectedNode
from pynever.strategies.abstraction.bounds_propagation import util
from pynever.strategies.abstraction.bounds_propagation.bounds import SymbolicLinearBounds, HyperRectangleBounds
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.tensors import Tensor


def get_layer_equation(layer: FullyConnectedNode) -> LinearFunctions:
    """Return the linear equation of the layer

    Parameters
    ----------
    layer : FullyConnectedNode
        The linear layer

    Returns
    ----------
    LinearFunctions
        The equation of the layer
    """
    return LinearFunctions(layer.weight, layer.bias)


def substitute_one_step_back(equation: LinearFunctions, prev: SymbolicLinearBounds,
                             lower: bool = False) -> LinearFunctions:
    """Perform one substitution step
    Given an equation mapping R^n -> R^m in the form of a matrix and an offset, and
    previous equations mapping R^k to R^n,
    computes a new equation (in the form of a matrix and an offset) that
    maps R^k to R^m.

    """
    prev_lower = prev.get_lower()
    prev_upper = prev.get_upper()

    matrix_pos = tensors.get_positive(equation.get_matrix())
    matrix_neg = tensors.get_negative(equation.get_matrix())

    if lower:
        cur_matrix = tensors.dot(matrix_pos, prev_lower.get_matrix()) + \
                     tensors.dot(matrix_neg, prev_upper.get_matrix())
        cur_offset = tensors.dot(matrix_pos, prev_lower.get_offset()) + \
                     tensors.dot(matrix_neg, prev_upper.get_offset()) + equation.get_offset()
    else:
        cur_matrix = tensors.dot(matrix_pos, prev_upper.get_matrix()) + \
                     tensors.dot(matrix_neg, prev_lower.get_matrix())
        cur_offset = tensors.dot(matrix_pos, prev_upper.get_offset()) + \
                     tensors.dot(matrix_neg, prev_lower.get_offset()) + equation.get_offset()

    return LinearFunctions(cur_matrix, cur_offset)


def get_backwards_layer_equation(layer: FullyConnectedNode,
                                 network: NeuralNetwork,
                                 equations_in: dict[str, SymbolicLinearBounds],
                                 input_bounds: HyperRectangleBounds,
                                 lower: bool = False) -> tuple[LinearFunctions, Tensor]:
    """Compute the lower or upper bound equation of the layer by backwards substitution
    of the previous layers from the variables of the input layer

    Parameters
    ----------
    layer : FullyConnectedNode
        The linear layer
    network : NeuralNetwork
        The neural network
    equations_in: dict[str, SymbolicLinearBounds]
        The layer-to-layer symbolic bounds
    input_bounds: HyperRectangleBounds
        The input bounds
    lower: bool
        Flag to compute the lower bound or the upper bound

    Returns
    ----------
    tuple[LinearFunctions, Tensor]
        The bound equation from the input and the bound values
    """
    if lower:
        cur_matrix = equations_in[layer.identifier].get_lower().get_matrix()
        cur_offset = equations_in[layer.identifier].get_lower().get_offset()
    else:
        cur_matrix = equations_in[layer.identifier].get_upper().get_matrix()
        cur_offset = equations_in[layer.identifier].get_upper().get_offset()

    cur_equation = LinearFunctions(cur_matrix, cur_offset)

    # Linear layers have a single parent
    # TODO refactor method get_previous_id(...) in NeuralNetwork class
    prev_layer_id = network.get_parents(layer)[0].identifier

    while prev_layer_id is not None:
        cur_equation = substitute_one_step_back(cur_equation, equations_in[prev_layer_id], lower)
        prev_layer_id = network.get_parents(network.nodes[prev_layer_id])[0].identifier

    bound = cur_equation.compute_min_values(input_bounds) if lower else cur_equation.compute_max_values(input_bounds)
    return cur_equation, bound


def compute_dense_output_bounds(layer: FullyConnectedNode, inputs: SymbolicLinearBounds) -> SymbolicLinearBounds:
    """Compute the forwards symbolic output bounds for the layer

    Parameters
    ----------
    layer : FullyConnectedNode
        The linear layer
    inputs : SymbolicLinearBounds
        The input symbolic bounds

    Returns
    ----------
    SymbolicLinearBounds
        The symbolic output bounds for the layer
    """
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
