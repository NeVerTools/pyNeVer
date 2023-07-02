import pynever.strategies.abstraction
from pynever import nodes
import numpy as np
from pynever.strategies.bound_propagation_alternative.utils.property_converter import *
from pynever.strategies.bound_propagation_alternative.linear_function import LinearFunctions
from pynever.strategies.bound_propagation_alternative.bounds import SymbolicLinearBounds
from pynever.strategies.bound_propagation_alternative.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper
import logging

DEBUG = True
logger_bound = logging.getLogger("bound_propagation.bounds_menager")


class BoundsManager_1:

    def __init__(self, abst_net, prop):

        super().__init__()

        # Abstract neural net
        self.abs_net = abst_net

        # Create HyperRectBounds from property
        property_converter = PropertyFormatConverter(prop)

        # HyperRectBounds input bounds
        self.input_bounds = property_converter.get_vectors()

        self.numeric_bounds = None

        self.symbolic_bounds = None

    def __repr__(self):
        return str(self.numeric_bounds)

    def compute_bounds(self):

        # dict containing the numeric bounds of all layers of the abst_net whose identifier is the key
        self.numeric_bounds = dict()

        # dict containing the simbolic bound (SimbolicLinearBounds) of each layer
        self.symbolic_bounds = dict()

        # the layers on which the computation is performed
        layers = get_abstract_network(self.abs_net)

        # initialization for the input_bounds
        input_symbolic_bounds = self.initializeSymbolicInputBounds()

        # add input symbolic bounds and numeric input bounds in the corrispective dicts con key "input"
        # self.numeric_bounds["input"] = self.input_bounds
        # self.symbolic_bounds["input"] = input_symbolic_bounds

        # logger
        logger_bound.debug(input_symbolic_bounds)

        symbolic_bounds = input_symbolic_bounds
        last_numeric_bounds = None
        for i in range(len(layers)):

            if isinstance(layers[i], pynever.strategies.abstraction.AbsFullyConnectedNode):
                symbolic_bounds = self.compute_dense_output_bounds(layers[i], symbolic_bounds)
                numeric_bounds = symbolic_bounds.to_hyper_rectangle_bounds(self.input_bounds)
                last_numeric_bounds = numeric_bounds
                self.numeric_bounds[layers[i].identifier] = numeric_bounds
                self.symbolic_bounds[layers[i].identifier] = symbolic_bounds

                # logger
                logger_bound.debug([layers[i].identifier])
                logger_bound.debug("symbolic bounds")
                logger_bound.debug("upper matrix")
                logger_bound.debug(symbolic_bounds.upper.matrix)
                logger_bound.debug("upper offset")
                logger_bound.debug(symbolic_bounds.upper.offset)
                logger_bound.debug("lower matrix")
                logger_bound.debug(symbolic_bounds.lower.matrix)
                logger_bound.debug("lower offset")
                logger_bound.debug(symbolic_bounds.lower.offset)
                logger_bound.debug("numeric bounds")
                logger_bound.debug(numeric_bounds)


            elif isinstance(layers[i], pynever.strategies.abstraction.AbsReLUNode):
                symbolic_bounds = self.compute_relu_output_bounds(symbolic_bounds, last_numeric_bounds)
                numeric_bounds = symbolic_bounds.to_hyper_rectangle_bounds(self.input_bounds)
                last_numeric_fc_bounds = numeric_bounds
                self.numeric_bounds[layers[i].identifier] = numeric_bounds
                self.symbolic_bounds[layers[i].identifier] = symbolic_bounds

                # logger
                logger_bound.debug([layers[i].identifier])
                logger_bound.debug("symbolic bounds")
                logger_bound.debug("upper matrix")
                logger_bound.debug(symbolic_bounds.upper.matrix)
                logger_bound.debug("upper offset")
                logger_bound.debug(symbolic_bounds.upper.offset)
                logger_bound.debug("lower matrix")
                logger_bound.debug(symbolic_bounds.lower.matrix)
                logger_bound.debug("lower offset")
                logger_bound.debug(symbolic_bounds.lower.offset)
                logger_bound.debug("numeric bounds")
                logger_bound.debug(numeric_bounds)

        return self.symbolic_bounds, self.numeric_bounds

    def initializeSymbolicInputBounds(self):
        dim = self.input_bounds.get_size()
        lower = LinearFunctions(np.identity(dim), np.zeros(dim))
        upper = LinearFunctions(np.identity(dim), np.zeros(dim))
        to_ret = SymbolicLinearBounds(lower, upper)
        return to_ret

    def compute_dense_output_bounds(self, layer, inputs):
        weights = layer.ref_node.weight
        weights_plus = get_positive_part(weights)
        weights_minus = get_negative_part(weights)
        bias = layer.ref_node.bias

        lower_matrix, lower_offset, upper_matrix, upper_offset = \
            compute_lin_lower_and_upper(weights_minus, weights_plus, bias,
                                        inputs.get_lower().get_matrix(),
                                        inputs.get_upper().get_matrix(),
                                        inputs.get_lower().get_offset(),
                                        inputs.get_upper().get_offset())

        return SymbolicLinearBounds(LinearFunctions(lower_matrix, lower_offset),
                                    LinearFunctions(upper_matrix, upper_offset))

    def compute_relu_output_bounds(self, input_symb_bounds, input_numeric_bounds):
        row_number = input_symb_bounds.upper.matrix.shape[0]
        col_number = input_symb_bounds.upper.matrix.shape[1]
        new_upper_matrix = np.zeros(input_symb_bounds.upper.matrix.shape)
        new_lower_matrix = np.zeros(input_symb_bounds.upper.matrix.shape)
        new_upper_offset = np.zeros(input_symb_bounds.upper.matrix.shape[0])
        new_lower_offset = np.zeros(input_symb_bounds.upper.matrix.shape[0])

        for i in range(row_number):
            if input_numeric_bounds.lower[i] >= 0:
                new_upper_matrix[i, :] = input_symb_bounds.upper.matrix[i, :]
                new_lower_matrix[i, :] = input_symb_bounds.lower.matrix[i, :]

                new_upper_offset[i] = input_symb_bounds.upper.offset[i]
                new_lower_offset[i] = input_symb_bounds.lower.offset[i]

            elif input_numeric_bounds.upper[i] <= 0:
                new_upper_matrix[i, :] = 0
                new_lower_matrix[i, :] = 0
                new_upper_offset[i] = 0
                new_lower_offset[i] = 0

            else:
                coeff_lamda = input_numeric_bounds.upper[i] / \
                              (input_numeric_bounds.upper[i] - input_numeric_bounds.lower[i])
                coeff_offset = (-input_numeric_bounds.lower[i] * input_numeric_bounds.upper[i]) / \
                               (input_numeric_bounds.upper[i] - input_numeric_bounds.lower[i])

                new_upper_matrix[i, :] = input_symb_bounds.upper.matrix[i, :] * coeff_lamda
                new_lower_matrix[i, :] = 0

                new_upper_offset[i] = input_symb_bounds.upper.offset[i] * coeff_lamda + coeff_offset
                new_lower_offset[i] = 0

        return SymbolicLinearBounds(LinearFunctions(new_lower_matrix, new_lower_offset),
                                    LinearFunctions(new_upper_matrix, new_upper_offset))


def get_abstract_network(abst_network):
    # Create the layers representation and the input hyper rectangle
    layers = []
    node = abst_network.get_first_node()
    layers.append(node)

    while node is not abst_network.get_last_node():
        node = abst_network.get_next_node(node)
        layers.append(node)

    return layers


def get_number_of_stable_neurons(numeric_bounds_list: list):
    counter_list = list()
    total_numer_neurons = list()
    for index, layer in enumerate(numeric_bounds_list):
        counter = 0
        for j in range(len(layer.lower)):
            if layer.lower[j] >= 0 or layer.upper[j] <= 0:
                counter += 1
            else:
                # print(j)
                # print("lower:", layer.lower[j])
                # print("upper:", layer.upper[j])
                pass
        total_numer_neurons.append(len(layer.lower))
        counter_list.append(counter)
    return counter_list, total_numer_neurons
