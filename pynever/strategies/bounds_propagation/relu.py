from enum import Enum

import numpy as np

from pynever import nodes
from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds, HyperRectangleBounds
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions


class StabilityInfo(Enum):
    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2


class LinearizeReLU:

    USE_FIXED_NEURONS = True

    def __init__(self, fixed_neurons: dict, stable_count: int, input_hyper_rect: HyperRectangleBounds):
        self.fixed_neurons = fixed_neurons
        self.stable_count = stable_count
        self.input_hyper_rect = input_hyper_rect

    def compute_output_equation(self, input_eq: SymbolicLinearBounds) -> SymbolicLinearBounds:
        """
        Set the equations to zero for the neurons that have been fixed to 0
        This does not work well, at least for acas.
        It seems to mess up the equations in a strange way.
        For instance, when there are no stable neurons, the equations are different from
        what we get with abstract propagation.
        Not sure if there is a problem is with abstract propagation or here.
        Could be abstract propagation as the bug I was getting was because
        the counter-example after using abstract propagation was not valid.
        However, the bug does not appear when we don't incorporate info from the fixed neurons.

        """

        return LinearizeReLU.compute_relu_output_bounds(input_eq, self.input_hyper_rect)

    def compute_output_numeric(self, relu: nodes.ReLUNode, cur_numeric_bounds: HyperRectangleBounds,
                               cur_symbolic_bounds: SymbolicLinearBounds) -> tuple[HyperRectangleBounds, int]:

        layer_id = relu.identifier

        current_layer_inactive = LinearizeReLU.extract_layer_inactive_from_fixed_neurons(self.fixed_neurons, layer_id)

        cur_layer_output_num_bounds = HyperRectangleBounds(
            np.maximum(cur_numeric_bounds.get_lower(), 0),
            np.maximum(cur_numeric_bounds.get_upper(), 0))

        if LinearizeReLU.USE_FIXED_NEURONS:
            self.force_inactive_neurons2(cur_symbolic_bounds, cur_layer_output_num_bounds,
                                         current_layer_inactive)

        self.stable_count += self.get_layer_stability_stats(layer_id, cur_layer_input_num_bounds,
                                                       stability_info, overapprox_area)

        return cur_layer_output_num_bounds, self.stable_count

    @staticmethod
    def extract_layer_inactive_from_fixed_neurons(fixed_neurons, layer_id):
        # TODO make this a util method somewhere else
        return [neuron_n for ((lay_n, neuron_n), value) in fixed_neurons.items()
                if lay_n == layer_id and value == 0]

    @staticmethod
    def extract_layer_active_from_fixed_neurons(fixed_neurons, layer_id):
        return [neuron_n for ((lay_n, neuron_n), value) in fixed_neurons.items()
                if lay_n == layer_id and value == 1]

    @staticmethod
    def compute_relu_output_bounds(inputs, input_hyper_rect):
        lower_l, lower_u, upper_l, upper_u = inputs.get_all_bounds(input_hyper_rect)
        lower, upper = LinearizeReLU.compute_symb_lin_bounds_equations(inputs, lower_l, lower_u, upper_l, upper_u)

        return SymbolicLinearBounds(lower, upper)

    @staticmethod
    def compute_symb_lin_bounds_equations(inputs, lower_l, lower_u, upper_l, upper_u):
        k_lower, b_lower = LinearizeReLU.get_array_lin_lower_bound_coefficients(lower_l, lower_u)
        k_upper, b_upper = LinearizeReLU.get_array_lin_upper_bound_coefficients(upper_l, upper_u)

        lower_matrix = LinearizeReLU.get_transformed_matrix(inputs.get_lower().get_matrix(), k_lower)
        upper_matrix = LinearizeReLU.get_transformed_matrix(inputs.get_upper().get_matrix(), k_upper)

        lower_offset = LinearizeReLU.get_transformed_offset(inputs.get_lower().get_offset(), k_lower, b_lower)
        upper_offset = LinearizeReLU.get_transformed_offset(inputs.get_upper().get_offset(), k_upper, b_upper)

        lower = LinearFunctions(lower_matrix, lower_offset)
        upper = LinearFunctions(upper_matrix, upper_offset)

        return lower, upper

    @staticmethod
    def get_transformed_matrix(matrix, k):
        return matrix * k[:, None]

    @staticmethod
    def get_transformed_offset(offset, k, b):
        return offset * k + b

    @staticmethod
    def get_array_lin_lower_bound_coefficients(lower, upper):
        ks = np.zeros(len(lower))
        bs = np.zeros(len(lower))

        for i in range(len(lower)):
            k, b = LinearizeReLU.get_lin_lower_bound_coefficients(lower[i], upper[i])
            ks[i] = k
            bs[i] = b

        return ks, bs

    @staticmethod
    def get_array_lin_upper_bound_coefficients(lower, upper):
        ks = np.zeros(len(lower))
        bs = np.zeros(len(lower))

        for i in range(len(lower)):
            k, b = LinearizeReLU.get_lin_upper_bound_coefficients(lower[i], upper[i])
            ks[i] = k
            bs[i] = b

        return ks, bs

    @staticmethod
    def get_lin_lower_bound_coefficients(lower, upper):
        if lower >= 0:
            return 1, 0

        if upper >= - lower:
            mult = upper / (upper - lower)
            return mult, 0

        # upper <= 0:
        # or
        # -lower > upper, i.e., 0 is a tighter lower bound that the slope mult above
        return 0, 0

    @staticmethod
    def get_lin_upper_bound_coefficients(lower, upper):
        if lower >= 0:
            return 1, 0

        if upper <= 0:
            return 0, 0

        mult = upper / (upper - lower)
        add = -mult * lower

        return mult, add

