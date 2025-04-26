"""
This file contains specialized methods that provide
the linearization of ReLU activation functions

"""
import torch
from torch import Tensor

from pynever import nodes
from pynever.exceptions import FixedConflictWithBounds
from pynever.strategies.abstraction import ABSTRACTION_PRECISION_GUARD
from pynever.strategies.abstraction.bounds_propagation.bounds import SymbolicLinearBounds, HyperRectangleBounds
from pynever.strategies.abstraction.linearfunctions import LinearFunctions


class LinearizeReLU:
    """
    This class provides the linearization for the ReLU function enhanced by information
    about currently active and inactive neurons

    """

    USE_FIXED_NEURONS = False

    def __init__(self, fixed_neurons: dict, input_hyper_rect: HyperRectangleBounds):
        self.fixed_neurons = fixed_neurons
        self.input_hyper_rect = input_hyper_rect

    def compute_output_linear_bounds(self, input_eq: SymbolicLinearBounds) -> SymbolicLinearBounds:
        """
        Set the equations to zero for the neurons that have been fixed to 0
        This does not work well, at least for ACAS_XU.
        It seems to mess up the equations in a strange way.
        For instance, when there are no stable neurons, the equations are different from
        what we get with abstract propagation.
        Not sure if there is a problem is with abstract propagation or here.
        Could be abstract propagation as the bug I was getting was because
        the counter-example after using abstract propagation was not valid.
        However, the bug does not appear when we don't incorporate info from the fixed neurons.

        """

        lower_l, lower_u, upper_l, upper_u = input_eq.get_all_bounds(self.input_hyper_rect)
        lower, upper = LinearizeReLU.compute_symb_lin_bounds_equations(input_eq, lower_l, lower_u, upper_l, upper_u)

        return SymbolicLinearBounds(lower, upper)

    def compute_output_numeric_bounds(self, relu: nodes.ReLUNode, cur_numeric_bounds: HyperRectangleBounds,
                                      cur_symbolic_bounds: SymbolicLinearBounds) -> HyperRectangleBounds:
        """
        Compute the numeric post-activation bounds of the linearized ReLU function
        using the information about currently inactive neurons

        """

        layer_id = relu.identifier

        cur_layer_inactive = LinearizeReLU.extract_layer_inactive_from_fixed_neurons(self.fixed_neurons, layer_id)

        cur_layer_output_num_bounds = HyperRectangleBounds(
            torch.max(cur_numeric_bounds.get_lower(), torch.zeros(cur_numeric_bounds.get_size())),
            torch.max(cur_numeric_bounds.get_upper(), torch.zeros(cur_numeric_bounds.get_size())))

        if LinearizeReLU.USE_FIXED_NEURONS:
            LinearizeReLU.force_inactive_neurons(cur_symbolic_bounds, cur_layer_output_num_bounds, cur_layer_inactive)

        return cur_layer_output_num_bounds

    @staticmethod
    def compute_relu_equation(preact_num_lower, preact_num_upper):
        lower_relu_eq, postact_lower = LinearizeReLU.get_relu_relax_lower_bound_equation(preact_num_lower,
                                                                                         preact_num_upper)
        upper_relu_eq, postact_upper = LinearizeReLU.get_relu_relax_upper_bound_equation(preact_num_lower,
                                                                                         preact_num_upper)

        return SymbolicLinearBounds(lower_relu_eq, upper_relu_eq), HyperRectangleBounds(postact_lower, postact_upper)

    @staticmethod
    def get_relu_relax_lower_bound_equation(preact_lower_bounds, preact_upper_bounds):
        """
        The lower bound of unstable nodes is either 0, or
        the linear relaxation of the preactivation (hence, the slope).

        The latter is the case when the upper bound is greater than or equal to the absolute value of the lower bound,
        thus resulting in a triangle of smaller area than the one formed by 0.

        The former is the case when the absolute value of the lower bound is greater than the upper bound,
        thus resulting is a triangle of smaller area than the one formed by the slope.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = torch.eye(size)
        offset = torch.zeros(size)

        postact_lower_bounds = Tensor(preact_lower_bounds)

        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the lower bound is exactly the preactivation
                # it remains 1
                pass

            elif preact_upper_bounds[i] >= -preact_lower_bounds[i]:
                # Unstable node, lower bound is linear relaxation of the equation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i][i] = k
                postact_lower_bounds[i] *= k

            else:  # upper[i] <= 0 (inactive node)
                # or
                # -lower[i] > upper[i]
                # lower bound is 0
                matrix[i][i] = 0
                postact_lower_bounds[i] = 0

        return LinearFunctions(matrix, offset), postact_lower_bounds

    @staticmethod
    def get_relu_relax_upper_bound_equation(preact_lower_bounds, preact_upper_bounds):
        """
        Compute the resulting upper bound equation after relaxing ReLU,
        qiven a preactivation upper bound equation.

        input_bounds are required for computing the concrete bounds.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = torch.eye(size)
        offset = torch.zeros(size)

        postact_upper_bounds = Tensor(preact_upper_bounds)
        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the upper bound is exactly the preactivation
                # it remains 1
                pass

            elif preact_upper_bounds[i] >= 0:
                # Unstable node - linear relaxation of preactivation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i][i] = k
                offset[i] = - preact_lower_bounds[i] * k

            else:  # preact_upper_bounds[i] <= 0 (inactive node)
                # The upper bound is 0
                matrix[i][i] = 0
                postact_upper_bounds[i] = 0

        return LinearFunctions(matrix, offset), postact_upper_bounds

    @staticmethod
    def check_and_enforce_fixed_constraints(relu_input_eq, preact_bounds, fixed_neurons, layer_id):
        """
        We need to check if the bounds do not conflict with the currently fixed neurons.
        That could happen if we haven't detected that the current branch is infeasible.
        That could happen because we are dealing with approximated bounds.

        """

        current_layer_inactive = LinearizeReLU.extract_layer_inactive_from_fixed_neurons(fixed_neurons, layer_id)
        current_layer_active = LinearizeReLU.extract_layer_active_from_fixed_neurons(fixed_neurons, layer_id)

        new_bounds = preact_bounds.clone()

        new_eq = SymbolicLinearBounds(relu_input_eq.lower.clone(), relu_input_eq.upper.clone())

        for neuron_n in current_layer_active:
            if preact_bounds.upper[neuron_n] < 0:
                raise FixedConflictWithBounds("A neuron has been fixed to be positive, "
                                              "but the bounds are negative. The current branch is not viable.")

        for neuron_n in current_layer_inactive:
            if preact_bounds.lower[neuron_n] > 0:
                raise FixedConflictWithBounds("A neuron has been fixed to be negative, "
                                              "but the bounds are positive. The current branch is not viable.")

            if preact_bounds.upper[neuron_n] > 0:
                new_eq.lower.matrix[neuron_n] = 0 * new_eq.lower.matrix[neuron_n]
                new_eq.lower.offset[neuron_n] = 0

                new_eq.upper.matrix[neuron_n] = 0 * new_eq.upper.matrix[neuron_n]
                new_eq.upper.offset[neuron_n] = 0

                new_bounds.lower[neuron_n] = -ABSTRACTION_PRECISION_GUARD
                new_bounds.upper[neuron_n] = -ABSTRACTION_PRECISION_GUARD  # TODO is the sign correct?

        return new_eq, new_bounds

    @staticmethod
    def extract_layer_active_from_fixed_neurons(fixed_neurons: dict, layer_id: str) -> list[int]:
        return [neuron_n for ((lay_n, neuron_n), value) in fixed_neurons.items()
                if lay_n == layer_id and value == 1]

    @staticmethod
    def extract_layer_inactive_from_fixed_neurons(fixed_neurons: dict, layer_id: str) -> list[int]:
        return [neuron_n for ((lay_n, neuron_n), value) in fixed_neurons.items()
                if lay_n == layer_id and value == 0]

    @staticmethod
    def force_inactive_neurons(relu_eq, postact_bounds, current_layer_inactive):
        for neuron_n in current_layer_inactive:
            if postact_bounds.lower[neuron_n] > 0:
                raise Exception("A neuron is supposed to be fixed to be negative, "
                                "but the bounds are positive. A conflict must have been detected before.")

            if postact_bounds.upper[neuron_n] > 0:
                relu_eq.lower.matrix[neuron_n] = 0 * relu_eq.lower.matrix[neuron_n]
                relu_eq.lower.offset[neuron_n] = 0
                relu_eq.upper.matrix[neuron_n] = 0 * relu_eq.upper.matrix[neuron_n]
                relu_eq.upper.offset[neuron_n] = 0
                postact_bounds.lower[neuron_n] = 0
                postact_bounds.upper[neuron_n] = 0

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
        ks = torch.zeros(len(lower))
        bs = torch.zeros(len(lower))

        for i in range(len(lower)):
            k, b = LinearizeReLU.get_lin_lower_bound_coefficients(lower[i], upper[i])
            ks[i] = k
            bs[i] = b

        return ks, bs

    @staticmethod
    def get_array_lin_upper_bound_coefficients(lower, upper):
        ks = torch.zeros(len(lower))
        bs = torch.zeros(len(lower))

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
