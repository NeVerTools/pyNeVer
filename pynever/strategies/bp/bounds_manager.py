from collections import OrderedDict

from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies.bp.bounds import SymbolicLinearBounds
from pynever.strategies.bp.linearfunctions import LinearFunctions
from pynever.strategies.bp.utils.property_converter import *
from pynever.strategies.bp.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper
from collections import OrderedDict

from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies.bp.bounds import SymbolicLinearBounds
from pynever.strategies.bp.linearfunctions import LinearFunctions
from pynever.strategies.bp.utils.property_converter import *
from pynever.strategies.bp.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper


class BoundsManager:
    def __init__(self, net, prop):
        self.numeric_bounds = None
        self.net = net
        self.prop = prop

    def __repr__(self):
        return str(self.numeric_bounds)

    def compute_bounds(self) -> dict:
        """
        precomputes bounds for all nodes using symbolic linear propagation
        """

        # Create HyperRectBounds from property
        property_converter = PropertyFormatConverter(self.prop)

        # HyperRectBounds input bounds
        input_hyper_rect = property_converter.get_vectors()

        # Get layers
        layers = net2list(self.net)

        input_size = input_hyper_rect.get_size()
        lower = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        upper = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        input_bounds = SymbolicLinearBounds(lower, upper)

        numeric_preactivation_bounds = dict()
        numeric_postactivation_bounds = OrderedDict()
        symbolic_bounds = dict()
        # TODO change the structure of symbolic?bounds

        current_input_bounds = input_bounds
        for i in range(0, len(layers)):

            if isinstance(layers[i], nodes.ReLUNode):
                symbolic_activation_output_bounds = self.compute_relu_output_bounds(symbolic_dense_output_bounds,
                                                                                    input_hyper_rect)
                postactivation_bounds = HyperRectangleBounds(np.maximum(preactivation_bounds.get_lower(), 0),
                                                             np.maximum(preactivation_bounds.get_upper(), 0))

            elif isinstance(layers[i], nodes.FullyConnectedNode):
                symbolic_dense_output_bounds = self.compute_dense_output_bounds(layers[i], current_input_bounds)
                preactivation_bounds = symbolic_dense_output_bounds.to_hyper_rectangle_bounds(input_hyper_rect)

                symbolic_activation_output_bounds = symbolic_dense_output_bounds
                postactivation_bounds = HyperRectangleBounds(preactivation_bounds.get_lower(),
                                                             preactivation_bounds.get_upper())

            else:
                raise Exception("Currently supporting bounds computation only for Relu and Linear activation functions")

            symbolic_bounds[layers[i].identifier] = symbolic_activation_output_bounds
            numeric_preactivation_bounds[layers[i].identifier] = preactivation_bounds
            numeric_postactivation_bounds[layers[i].identifier] = postactivation_bounds

            current_input_bounds = symbolic_activation_output_bounds
            self.numeric_bounds = numeric_postactivation_bounds

        return {
            'symbolic': symbolic_bounds,
            'numeric_pre': numeric_preactivation_bounds,
            'numeric_post': numeric_postactivation_bounds
        }

    def compute_dense_output_bounds(self, layer, inputs):
        weights_plus = get_positive_part(layer.weight)
        weights_minus = get_negative_part(layer.weight)

        if layer.bias is None:
            layer.bias = np.zeros(layer.weight.shape[0])

        lower_matrix, lower_offset, upper_matrix, upper_offset = \
            compute_lin_lower_and_upper(weights_minus, weights_plus, layer.bias,
                                        inputs.get_lower().get_matrix(),
                                        inputs.get_upper().get_matrix(),
                                        inputs.get_lower().get_offset(),
                                        inputs.get_upper().get_offset())

        return SymbolicLinearBounds(LinearFunctions(lower_matrix, lower_offset),
                                    LinearFunctions(upper_matrix, upper_offset))

    def compute_relu_output_bounds(self, inputs, input_hyper_rect):
        lower_l, lower_u, upper_l, upper_u = inputs.get_all_bounds(input_hyper_rect)
        lower, upper = self.compute_symb_lin_bounds_equations(inputs, lower_l, lower_u, upper_l, upper_u)

        return SymbolicLinearBounds(lower, upper)

    def compute_symb_lin_bounds_equations(self, inputs, lower_l, lower_u, upper_l, upper_u):
        k_lower, b_lower = get_array_lin_lower_bound_coefficients(lower_l, lower_u)
        k_upper, b_upper = get_array_lin_upper_bound_coefficients(upper_l, upper_u)

        lower_matrix = get_transformed_matrix(inputs.get_lower().get_matrix(), k_lower)
        upper_matrix = get_transformed_matrix(inputs.get_upper().get_matrix(), k_upper)
        #
        lower_offset = get_transformed_offset(inputs.get_lower().get_offset(), k_lower, b_lower)
        upper_offset = get_transformed_offset(inputs.get_upper().get_offset(), k_upper, b_upper)

        lower = LinearFunctions(lower_matrix, lower_offset)
        upper = LinearFunctions(upper_matrix, upper_offset)

        return lower, upper


def get_transformed_matrix(matrix, k):
    return matrix * k[:, None]


def get_transformed_offset(offset, k, b):
    return offset * k + b


def get_array_lin_lower_bound_coefficients(lower, upper):
    ks = np.zeros(len(lower))
    bs = np.zeros(len(lower))

    for i in range(len(lower)):
        k, b = get_lin_lower_bound_coefficients(lower[i], upper[i])
        ks[i] = k
        bs[i] = b

    return ks, bs


def get_array_lin_upper_bound_coefficients(lower, upper):
    ks = np.zeros(len(lower))
    bs = np.zeros(len(lower))

    for i in range(len(lower)):
        k, b = get_lin_upper_bound_coefficients(lower[i], upper[i])
        ks[i] = k
        bs[i] = b

    return ks, bs


def get_lin_lower_bound_coefficients(lower, upper):
    if lower >= 0:
        return 1, 0

    if upper <= 0:
        return 0, 0

    mult = upper / (upper - lower)

    return mult, 0


def get_lin_upper_bound_coefficients(lower, upper):
    if lower >= 0:
        return 1, 0

    if upper <= 0:
        return 0, 0

    mult = upper / (upper - lower)
    add = -mult * lower

    return mult, add


def net2list(network: SequentialNetwork) -> list:
    """
    Create the layers representation as a list

    Parameters
    ----------
    network : SequentialNetwork
        The network in the internal representation

    Returns
    ----------
    list
        The list of the layers

    """

    layers = list()
    node = network.get_first_node()
    layers.append(node)

    while node is not network.get_last_node():
        node = network.get_next_node(node)
        layers.append(node)

    return layers
