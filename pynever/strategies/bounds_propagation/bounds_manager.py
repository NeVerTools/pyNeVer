from pynever import nodes
from pynever.exceptions import FixedConflictWithBounds
from pynever.networks import SequentialNetwork, NeuralNetwork
from pynever.strategies.bounds_propagation import LOGGER
from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds, PRECISION_GUARD
from pynever.strategies.bounds_propagation.convolution import LinearizeConv
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions
from pynever.strategies.bounds_propagation.relu import LinearizeReLU, StabilityInfo
from pynever.strategies.bounds_propagation.utils.property_converter import *
from pynever.strategies.bounds_propagation.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper
from pynever.strategies.verification.ssbp.constants import NeuronState


class BoundsManager:
    """
    This class handles the propagation of symbolic bounds with
    both forwards and backwards propagation

    """

    def __init__(self):
        # TODO add new data structure for bounds
        self.numeric_bounds = None
        self.logger = LOGGER
        self.stability_info = {}
        self.overapprox_area = {}
        self.reset_info()

    def __repr__(self):
        return str(self.numeric_bounds)

    @staticmethod
    def check_stable(lb, ub) -> NeuronState:
        """
        Static method, given the bounds of a neuron, whether it is stable
        """

        # Positive stable
        if lb >= PRECISION_GUARD:
            return NeuronState.POSITIVE_STABLE

        # Negative stable
        elif ub <= -PRECISION_GUARD:
            return NeuronState.NEGATIVE_STABLE

        # Unstable
        else:
            return NeuronState.UNSTABLE

    @staticmethod
    def get_input_bounds(prop: 'NeverProperty') -> HyperRectangleBounds:
        """
        This method computes the numeric bounds of the input layer

        Parameters
        ----------
        prop : NeverProperty
            The property to verify

        Returns
        ----------
        HyperRectangleBounds
            The numeric bounds of the input layer

        """

        # HyperRectBounds input bounds
        return PropertyFormatConverter(prop).get_vectors()

    @staticmethod
    def get_symbolic_preact_bounds_at(bounds: dict, layer_id: str, nn: SequentialNetwork) -> SymbolicLinearBounds:
        """
        This method retrieves the preactivation symbolic bounds in the bounds
        dictionary at the specified target

        """

        return bounds['symbolic'][nn.get_previous_id(layer_id)]

    def compute_bounds_from_property(self, network: NeuralNetwork, prop: 'NeverProperty') -> dict:
        """
        Precomputes bounds for all nodes using symbolic linear propagation

        """

        # HyperRectBounds input bounds
        input_hyper_rect = BoundsManager.get_input_bounds(prop)

        # Get layers
        if not isinstance(network, SequentialNetwork):
            raise NotImplementedError

        return self.compute_bounds(input_hyper_rect, network)

    def compute_bounds(self, input_hyper_rect: HyperRectangleBounds, network: SequentialNetwork,
                       fixed_neurons: dict = None, direction: str = 'backwards') -> dict:
        """
        This method computes the bounds for the neural network given the property,
        either using forwards propagation or backwards propagation

        """

        if fixed_neurons is None:
            fixed_neurons = dict()

        # We are collecting the bounds, symbolic and numeric, in these dictionaries
        symbolic_bounds = dict()
        num_preact_bounds = dict()
        num_postact_bounds = dict()

        # Reset to default the stability info and overapprox area dictionaries
        self.reset_info()

        # Initialising the current equations
        input_size = input_hyper_rect.get_size()
        lower_equation = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        upper_equation = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        cur_layer_input_eq = SymbolicLinearBounds(lower_equation, upper_equation)
        cur_layer_input_num_bounds = input_hyper_rect

        stable_count = 0

        # Iterate through the layers
        for layer in network.layers_iterator():
            cur_layer_output_eq, cur_layer_output_num_bounds, stable_count = (
                self.compute_layer_bounds(network, layer, cur_layer_input_eq, cur_layer_input_num_bounds,
                                          input_hyper_rect, direction, stable_count, fixed_neurons))

            # Store the current equations and numeric bounds
            symbolic_bounds[layer.identifier] = cur_layer_output_eq
            num_preact_bounds[layer.identifier] = cur_layer_input_num_bounds
            num_postact_bounds[layer.identifier] = cur_layer_output_num_bounds

            # Update the current input equation and numeric bounds
            cur_layer_input_eq = cur_layer_output_eq
            cur_layer_input_num_bounds = cur_layer_output_num_bounds

        # sort the over-approximation areas ascending
        self.overapprox_area['sorted'] = sorted(self.overapprox_area['sorted'], key=lambda x: x[1])
        self.overapprox_area['volume'] = compute_overapproximation_volume(self.overapprox_area['map'])

        # Put all the collected bounds in a dictionary and return it
        # TODO create data structure
        return {
            'symbolic': symbolic_bounds,
            'numeric_pre': num_preact_bounds,
            'numeric_post': num_postact_bounds,
            'stability_info': self.stability_info,
            'stable_count': stable_count,
            'overapproximation_area': self.overapprox_area
        }

    def compute_layer_bounds(self, network: SequentialNetwork, layer: nodes.LayerNode,
                             layer_in_eq: SymbolicLinearBounds,
                             layer_in_num: HyperRectangleBounds, input_hyper_rect: HyperRectangleBounds,
                             direction: str, stable_count: int, fixed_neurons: dict) \
            -> tuple[SymbolicLinearBounds, HyperRectangleBounds, int] | None:

        # Equations for each layer to do backward substitution
        layer2layer_equations = dict()

        if isinstance(layer, nodes.FullyConnectedNode):
            """ Fully Connected layer """

            if direction == 'forwards':
                cur_layer_output_eq = BoundsManager.compute_dense_output_bounds(layer, layer_in_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)

            else:
                layer_eq = BoundsManager.get_linear_layer_equation(layer)
                layer2layer_equations[layer.identifier] = SymbolicLinearBounds(layer_eq, layer_eq)

                lower_eq_from_input, lower_bounds = (
                    BoundsManager._get_equation_from_input(network, layer.identifier, "lower",
                                                           layer2layer_equations, input_hyper_rect))
                upper_eq_from_input, upper_bounds = (
                    BoundsManager._get_equation_from_input(network, layer.identifier, "upper",
                                                           layer2layer_equations, input_hyper_rect))

                cur_layer_output_num_bounds = HyperRectangleBounds(lower_bounds, upper_bounds)
                cur_layer_output_eq = SymbolicLinearBounds(lower_eq_from_input, upper_eq_from_input)

        elif isinstance(layer, nodes.ConvNode):
            """ Convolutional layer """

            if direction == 'forwards':
                cur_layer_output_eq = LinearizeConv().compute_output_equation(layer, layer_in_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)

            else:
                raise NotImplementedError('Backwards bounds propagation not yet implemented for convolutional layers')

        elif isinstance(layer, nodes.ReLUNode):
            """ ReLU layer """

            relu_lin = LinearizeReLU(fixed_neurons, input_hyper_rect)

            if direction == 'forwards':
                cur_layer_output_eq = relu_lin.compute_output_equation(layer_in_eq)
                cur_layer_output_num_bounds = relu_lin.compute_output_numeric(layer, layer_in_num, layer_in_eq)

            else:
                if relu_lin.USE_FIXED_NEURONS:
                    try:
                        layer_in_eq, layer_in_num = relu_lin.check_and_enforce_fixed_constraints(
                            layer_in_eq, layer_in_num, fixed_neurons, layer.identifier
                        )

                    except FixedConflictWithBounds:
                        # The current branch is not feasible. Return None
                        return None

                # Just to set a value to cur_layer_output_eq
                cur_layer_output_eq = layer_in_eq

                relu_eq, cur_layer_output_num_bounds = relu_lin.compute_relu_equation(layer_in_num.get_lower(),
                                                                                      layer_in_num.get_upper())
                layer2layer_equations[layer.identifier] = relu_eq

            stable_count += self.get_layer_stability_stats(layer.identifier, layer_in_num)


        elif isinstance(layer, nodes.FlattenNode):
            """ Flatten layer """

            layer2layer_equations[layer.identifier] = layer_in_eq
            cur_layer_output_eq = layer_in_eq
            cur_layer_output_num_bounds = layer_in_num

        elif isinstance(layer, nodes.ReshapeNode):
            """ Reshape layer """

            layer2layer_equations[layer.identifier] = layer_in_eq
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

            stable_status = BoundsManager.check_stable(l, u)
            if stable_status == NeuronState.NEGATIVE_STABLE:
                inactive.append(neuron_n)
                stable_count += 1

            elif stable_status == NeuronState.POSITIVE_STABLE:
                active.append(neuron_n)
                stable_count += 1

            else:  # stable_status == NeuronState.UNSTABLE
                unstable.append((layer_id, neuron_n))

                # Compute approximation area
                area = 0.5 * (u - l) * u
                self.overapprox_area['sorted'].append(((layer_id, neuron_n), area))
                self.overapprox_area['map'][(layer_id, neuron_n)] = area

        self.stability_info[StabilityInfo.INACTIVE][layer_id] = inactive
        self.stability_info[StabilityInfo.ACTIVE][layer_id] = active
        self.stability_info[StabilityInfo.UNSTABLE].extend(unstable)

        return stable_count

    def reset_info(self) -> None:
        # Here we save information about the stable and unstable neurons
        self.stability_info = {
            StabilityInfo.INACTIVE: dict(),
            StabilityInfo.ACTIVE: dict(),
            StabilityInfo.UNSTABLE: list()
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
            current_matrix, current_offset = BoundsManager._substitute_one_step_back(
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

    @staticmethod
    def compute_dense_output_bounds(layer, inputs):
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


def compute_layer_inactive_from_bounds_and_fixed_neurons(bounds, fixed_neurons, layer_id):
    return (bounds['stability_info'][StabilityInfo.INACTIVE][layer_id] +
            [i for (lay_id, i), value in fixed_neurons.items() if lay_id == layer_id and value == 0])


def extract_layer_unstable_from_bounds(bounds, layer_id):
    return {neuron_n for lay_id, neuron_n in bounds['stability_info'][StabilityInfo.UNSTABLE]
            if lay_id == layer_id}


def compute_layer_unstable_from_bounds_and_fixed_neurons(bounds, fixed_neurons, layer_id):
    return [neuron_n for neuron_n in extract_layer_unstable_from_bounds(bounds, layer_id)
            if (layer_id, neuron_n) not in fixed_neurons]


def compute_unstable_from_bounds_and_fixed_neurons(bounds: dict, fixed_neurons: dict) -> list:
    """
    Utility method

    """

    unstable = bounds['stability_info'][StabilityInfo.UNSTABLE]
    return [neuron for neuron in unstable if neuron not in fixed_neurons]


def compute_overapproximation_volume(areas_map: dict) -> float:
    return np.prod(list(areas_map.values()))
