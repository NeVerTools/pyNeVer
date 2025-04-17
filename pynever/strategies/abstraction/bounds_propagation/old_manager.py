"""
This file contains the bounds propagation methods for
the different layers of a neural network

"""

import numpy as np

from pynever import nodes
from pynever.exceptions import FixedConflictWithBounds
from pynever.networks import SequentialNetwork, NeuralNetwork
from pynever.strategies.abstraction.bounds_propagation import BOUNDS_LOGGER, BOUNDS_PRECISION_GUARD
from pynever.strategies.abstraction.bounds_propagation import util
from pynever.strategies.abstraction.bounds_propagation.bounds import SymbolicLinearBounds, HyperRectangleBounds, \
    VerboseBounds, BoundsStats
from pynever.strategies.abstraction.bounds_propagation.layers.affine import compute_dense_output_bounds
from pynever.strategies.abstraction.bounds_propagation.layers.convolution import LinearizeConv
from pynever.strategies.abstraction.bounds_propagation.layers.relu import LinearizeReLU
from pynever.strategies.abstraction.bounds_propagation.util import ReLUStatus
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.strategies.verification.ssbp.constants import BoundsDirection


class OldBoundsManager:
    """
    This class handles the propagation of symbolic bounds with
    both forwards and backwards propagation

    """

    def __init__(self, direction: BoundsDirection = BoundsDirection.BACKWARDS):
        self.direction = direction

        # Equations for each layer to do backward substitution
        self.layer2layer_equations = {}

        self.stability_info = {}
        self.overapprox_area = {}

        self.logger = BOUNDS_LOGGER

        self.reset_info()

    @staticmethod
    def check_stable(lb, ub) -> ReLUStatus:
        """
        Static method, given the bounds of a neuron, whether it is stable
        """

        # Positive stable
        if lb >= BOUNDS_PRECISION_GUARD:
            return ReLUStatus.ACTIVE

        # Negative stable
        elif ub <= -BOUNDS_PRECISION_GUARD:
            return ReLUStatus.INACTIVE

        # Unstable
        else:
            return ReLUStatus.UNSTABLE

    @staticmethod
    def get_symbolic_preact_bounds_at(bounds: VerboseBounds, layer_id: str,
                                      nn: SequentialNetwork) -> SymbolicLinearBounds:
        """
        This method retrieves the preactivation symbolic bounds in the bounds
        container at the specified target

        """

        return bounds.symbolic_bounds[nn.get_previous_id(layer_id)]

    def compute_bounds_from_property(self, network: NeuralNetwork, prop: 'NeverProperty') -> VerboseBounds | None:
        """
        Precomputes bounds for all nodes using symbolic linear propagation

        """

        # HyperRectBounds input bounds
        input_hyper_rect = prop.to_numeric_bounds()

        # Get layers
        if not isinstance(network, SequentialNetwork):
            raise NotImplementedError

        return self.compute_bounds(input_hyper_rect, network)

    def compute_bounds(self, input_hyper_rect: HyperRectangleBounds, network: SequentialNetwork,
                       fixed_neurons: dict = None) -> VerboseBounds | None:
        """
        This method computes the bounds for the neural network given the property,
        either using forwards propagation or backwards propagation

        """

        if fixed_neurons is None:
            fixed_neurons = dict()

        # Reset to default the stability info and overapprox area dictionaries
        self.reset_info()

        # We are collecting the bounds, symbolic and numeric, in this data structure
        all_bounds = VerboseBounds()

        # Initialising the current equations
        input_size = input_hyper_rect.get_size()
        lower_equation = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        upper_equation = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        cur_layer_input_eq = SymbolicLinearBounds(lower_equation, upper_equation)
        cur_layer_input_num_bounds = input_hyper_rect

        stable_count = 0

        # Reset equations
        self.layer2layer_equations = dict()

        # Iterate through the layers
        for layer in network.layers_iterator():
            bounds_pack = (
                self.compute_layer_bounds(network, layer, cur_layer_input_eq, cur_layer_input_num_bounds,
                                          input_hyper_rect, stable_count, fixed_neurons))
            if bounds_pack is None:
                return None
            else:
                cur_layer_output_eq, cur_layer_output_num_bounds, stable_count = bounds_pack

            # Store the current equations and numeric bounds
            all_bounds.symbolic_bounds[layer.identifier] = cur_layer_output_eq
            all_bounds.numeric_pre_bounds[layer.identifier] = cur_layer_input_num_bounds
            all_bounds.numeric_post_bounds[layer.identifier] = cur_layer_output_num_bounds

            # Update the current input equation and numeric bounds
            cur_layer_input_eq = cur_layer_output_eq
            cur_layer_input_num_bounds = cur_layer_output_num_bounds

        # sort the over-approximation areas ascending
        self.overapprox_area['sorted'] = sorted(self.overapprox_area['sorted'], key=lambda x: x[1])
        self.overapprox_area['volume'] = util.compute_overapproximation_volume(self.overapprox_area['map'])

        # Return all the computed bounds along to the statistics
        all_bounds.statistics = BoundsStats(self.stability_info, self.overapprox_area)

        return all_bounds

    def compute_layer_bounds(self, network: SequentialNetwork, layer: nodes.LayerNode,
                             layer_in_eq: SymbolicLinearBounds,
                             layer_in_num: HyperRectangleBounds, input_hyper_rect: HyperRectangleBounds,
                             stable_count: int, fixed_neurons: dict) \
            -> tuple[SymbolicLinearBounds, HyperRectangleBounds, int] | None:

        if isinstance(layer, nodes.FullyConnectedNode):
            """ Fully Connected layer """

            if self.direction == BoundsDirection.FORWARDS:
                cur_layer_output_eq = compute_dense_output_bounds(layer, layer_in_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)

            else:
                layer_eq = OldBoundsManager.get_linear_layer_equation(layer)
                self.layer2layer_equations[layer.identifier] = SymbolicLinearBounds(layer_eq, layer_eq)

                lower_eq_from_input, lower_bounds = (
                    OldBoundsManager._get_equation_from_input(network, layer.identifier, "lower",
                                                              self.layer2layer_equations, input_hyper_rect))
                upper_eq_from_input, upper_bounds = (
                    OldBoundsManager._get_equation_from_input(network, layer.identifier, "upper",
                                                              self.layer2layer_equations, input_hyper_rect))

                cur_layer_output_num_bounds = HyperRectangleBounds(lower_bounds, upper_bounds)
                cur_layer_output_eq = SymbolicLinearBounds(lower_eq_from_input, upper_eq_from_input)

        elif isinstance(layer, nodes.ConvNode):
            """ Convolutional layer """

            if self.direction == BoundsDirection.FORWARDS:
                cur_layer_output_eq = LinearizeConv().compute_output_equations(layer, layer_in_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)

            else:
                raise NotImplementedError('Backwards bounds propagation not yet implemented for convolutional layers')

        elif isinstance(layer, nodes.ReLUNode):
            """ ReLU layer """

            relu_lin = LinearizeReLU(fixed_neurons, input_hyper_rect)

            if self.direction == BoundsDirection.FORWARDS:
                cur_layer_output_eq = relu_lin.compute_output_linear_bounds(layer_in_eq)
                cur_layer_output_num_bounds = relu_lin.compute_output_numeric_bounds(layer, layer_in_num, layer_in_eq)

            else:
                if relu_lin.USE_FIXED_NEURONS:
                    try:
                        layer_in_eq, layer_in_num = LinearizeReLU.check_and_enforce_fixed_constraints(
                            layer_in_eq, layer_in_num, fixed_neurons, layer.identifier
                        )

                    except FixedConflictWithBounds:
                        # The current branch is not feasible. Return None
                        return None

                # Just to set a value to cur_layer_output_eq
                cur_layer_output_eq = layer_in_eq

                relu_eq, cur_layer_output_num_bounds = LinearizeReLU.compute_relu_equation(layer_in_num.get_lower(),
                                                                                           layer_in_num.get_upper())
                self.layer2layer_equations[layer.identifier] = relu_eq

            # Update stable count
            stable_count += self.get_layer_stability_stats(layer.identifier, layer_in_num)

        elif isinstance(layer, nodes.FlattenNode):
            """ Flatten layer """

            self.layer2layer_equations[layer.identifier] = layer_in_eq
            cur_layer_output_eq = layer_in_eq
            cur_layer_output_num_bounds = layer_in_num

        elif isinstance(layer, nodes.ReshapeNode):
            """ Reshape layer """

            self.layer2layer_equations[layer.identifier] = layer_in_eq
            cur_layer_output_eq = layer_in_eq
            cur_layer_output_num_bounds = layer_in_num

        else:
            raise Exception(
                "Currently supporting bounds computation only for FullyConnected, Convolutional, ReLU "
                "and Flatten layers.\n Instead got {}".format(layer.__class__))

        return cur_layer_output_eq, cur_layer_output_num_bounds, stable_count

    def get_layer_stability_stats(self, layer_id: str, numeric_preactivation_bounds: HyperRectangleBounds) -> int:
        stable_count = 0

        inactive = list()
        active = list()
        unstable = list()

        for neuron_n in range(numeric_preactivation_bounds.size):
            l, u = numeric_preactivation_bounds.get_dimension_bounds(neuron_n)

            stable_status = OldBoundsManager.check_stable(l, u)
            if stable_status == ReLUStatus.INACTIVE:
                inactive.append(neuron_n)
                stable_count += 1

            elif stable_status == ReLUStatus.ACTIVE:
                active.append(neuron_n)
                stable_count += 1

            else:  # stable_status == ReLUStatus.UNSTABLE
                unstable.append((layer_id, neuron_n))

                # Compute approximation area
                area = 0.5 * (u - l) * u
                self.overapprox_area['sorted'].append(((layer_id, neuron_n), area))
                self.overapprox_area['map'][(layer_id, neuron_n)] = area

        self.stability_info[util.ReLUStatus.INACTIVE][layer_id] = inactive
        self.stability_info[util.ReLUStatus.ACTIVE][layer_id] = active
        self.stability_info[util.ReLUStatus.UNSTABLE].extend(unstable)

        return stable_count

    def reset_info(self) -> None:
        # Here we save information about the stable and unstable neurons
        self.stability_info = {
            util.ReLUStatus.INACTIVE: dict(),
            util.ReLUStatus.ACTIVE: dict(),
            util.ReLUStatus.UNSTABLE: list()
        }

        self.overapprox_area = {
            'sorted': list(),
            'map': dict()
        }

    @staticmethod
    def _get_equation_from_input(network, layer_id, end, symbolic_bounds, input_bounds):
        """
        Given an equation for the current layer (which depends on the variables of the previous layer),
        computes the lower or the upper bound equation from the variables of the input layer
        by backwards substitution of the equations of the previous layers.

        Then, computes the concrete bounds of the obtained equation.

        end
            indicates if we want to compute the lower or the upper bound.

        """

        if end == "lower":
            current_matrix = symbolic_bounds[layer_id].get_lower().get_matrix()
            current_offset = symbolic_bounds[layer_id].get_lower().get_offset()
        else:
            current_matrix = symbolic_bounds[layer_id].get_upper().get_matrix()
            current_offset = symbolic_bounds[layer_id].get_upper().get_offset()

        prev_layer_id = network.get_previous_id(layer_id)

        while prev_layer_id is not None:
            current_matrix, current_offset = OldBoundsManager._substitute_one_step_back(
                current_matrix, current_offset, symbolic_bounds[prev_layer_id], end
            )
            prev_layer_id = network.get_previous_id(prev_layer_id)

        equation_from_input = LinearFunctions(current_matrix, current_offset)

        if end == "lower":
            bound = equation_from_input.compute_min_values(input_bounds)
        else:
            bound = equation_from_input.compute_max_values(input_bounds)

        return equation_from_input, bound

    @staticmethod
    def _substitute_one_step_back(current_matrix, current_offset, prev_equations, end):
        """
        Performs one substitution step.

        Given an equation mapping R^n -> R^m in the form of a matrix and an offset, and
        previous equations mapping R^k to R^n,
        computes a new equation (in the form of a matrix and an offset) that
        maps R^k to R^m.
        """
        prev_lower_eq = prev_equations.get_lower()
        prev_upper_eq = prev_equations.get_upper()

        matrix_pos = np.maximum(current_matrix, np.zeros(current_matrix.shape))
        matrix_neg = np.minimum(current_matrix, np.zeros(current_matrix.shape))

        if end == "lower":
            current_matrix = matrix_pos.dot(prev_lower_eq.get_matrix()) + matrix_neg.dot(prev_upper_eq.get_matrix())
            current_offset = matrix_pos.dot(prev_lower_eq.get_offset()) + matrix_neg.dot(prev_upper_eq.get_offset()) + \
                             current_offset

            # self._round_down(current_matrix)
            # self._round_down(current_bias)
        else:
            current_matrix = matrix_pos.dot(prev_upper_eq.get_matrix()) + matrix_neg.dot(prev_lower_eq.get_matrix())
            current_offset = matrix_pos.dot(prev_upper_eq.get_offset()) + matrix_neg.dot(prev_lower_eq.get_offset()) + \
                             current_offset

            # self._round_up(current_matrix)
            # self._round_up(current_bias)

        return current_matrix, current_offset

    @staticmethod
    def get_linear_layer_equation(layer):
        if layer.bias is None:
            layer.bias = np.zeros(layer.weight.shape[0])
        return LinearFunctions(layer.weight, layer.bias)
