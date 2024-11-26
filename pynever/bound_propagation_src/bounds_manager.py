import enum
from enum import Enum
import time
from  pynever.bound_propagation_src.neural_networks import nodes
from  pynever.bound_propagation_src.neural_networks.networks import SequentialNetwork, NeuralNetwork
from  pynever.bound_propagation_src.bounds import SymbolicLinearBounds
from  pynever.bound_propagation_src.convolution import ConvLinearization
from  pynever.bound_propagation_src.linearfunctions import LinearFunctions
from  pynever.bound_propagation_src.utils.property_converter import *
from  pynever.bound_propagation_src.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper

from  pynever.bound_propagation_src.bounds import HyperRectangleBounds
from  pynever.bound_propagation_src.utils.property_converter import PropertyFormatConverter

class StabilityInfo(Enum):
    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2

class NeuronState(enum.Enum):
    """
    This class is used as an enumerator to represent the state of a neuron.
    The possible states are:

    - positive stable when the lower bound is greater than zero
    - negative stable when the upper bound is lower than zero
    - unstable when the upper bound is greater than zero and the lower bound lower than zero

    """

    POSITIVE_STABLE = 1
    NEGATIVE_STABLE = -1
    UNSTABLE = 0


class BoundsManager:
    """
    This class handles the propagation of symbolic bounds and the refinement
    of the input bounds after a branch split

    """

    INPUT_DIMENSIONS_TO_REFINE = 50
    USE_FIXED_NEURONS = True
    PRECISION_GUARD = 10e-15

    def __init__(self):
        # TODO add new data structure for bounds
        self.numeric_bounds = None

    def __repr__(self):
        return str(self.numeric_bounds)

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
                       fixed_neurons: dict = dict()) -> dict:
        return self.compute_bounds_forwards(input_hyper_rect, network, fixed_neurons)

    def compute_bounds_forwards(self, input_hyper_rect: HyperRectangleBounds, network: SequentialNetwork,
                                fixed_neurons: dict = dict()) -> dict:
        """
        Given input hyper rectangle bounds, propagates them through the NN
        using forward symbolic bound propagation and
        returns a dictionary with the symbolic and numeric bounds as well
        as information on stability and refinement parameters

        """

        # We are collecting the bounds, symbolic and numeric, in these dictionaries
        symbolic_bounds = dict()
        num_preact_bounds = dict()
        num_postact_bounds = dict()

        # Here we save information about the stable and unstable neurons
        stability_info = {
            StabilityInfo.INACTIVE: dict(),
            StabilityInfo.ACTIVE: dict(),
            StabilityInfo.UNSTABLE: list()
        }
        overapprox_area = {
            'sorted': list(),
            'map': dict()
        }

        # Initialising the current equations
        input_size = input_hyper_rect.get_size()
        lower_equation = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        upper_equation = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        cur_layer_input_eq = SymbolicLinearBounds(lower_equation, upper_equation)
        cur_layer_input_num_bounds = input_hyper_rect

        stable_count = 0

        # Iterate through the layers
        for layer in network.layers_iterator():

            if isinstance(layer, nodes.FullyConnectedNode):
                """ Fully Connected layer """
                time_t = time.time()
                cur_layer_output_eq = BoundsManager.compute_dense_output_bounds(layer, cur_layer_input_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)
                print(f"dim: {cur_layer_output_num_bounds.lower.shape[0]}")
                print(f"Tempo FC: {time.time() - time_t}")

            elif isinstance(layer, nodes.ConvNode):
                """ Convolutional layer """
                time_t = time.time()
                cur_layer_output_eq = ConvLinearization().compute_output_equation(layer, cur_layer_input_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)
                print(f"dim: {cur_layer_output_num_bounds.lower.shape[0]}")
                print(f"Tempo ConvNode: {time.time() - time_t}")


            elif isinstance(layer, nodes.MaxPoolNode):
                pass


            elif isinstance(layer, nodes.ReLUNode):
                """ ReLU layer """

                time_t = time.time()
                layer_id = layer.identifier

                ## Set the equations to zero for the neurons that have been fixed to 0
                ## This does not work well, at least for acas.
                ## It seems to mess up the equations in a strange way.
                ## For instance, when there are no stable neurons, the equations are different from
                ## what we get with abstract propagation.
                ## Not sure if there problem is with abstract propagation or here.
                ## Could be abstract propagation as the bug I was getting was because
                ## the counter-example after using abstract propagation was not valid.
                ## However, the bug does not appear when we don't incorportate info from the fixed neurons.
                current_layer_inactive = BoundsManager.extract_layer_inactive_from_fixed_neurons(fixed_neurons, layer_id)

                cur_layer_output_eq = self.compute_relu_output_bounds(cur_layer_input_eq, input_hyper_rect)
                cur_layer_output_num_bounds = HyperRectangleBounds(
                    np.maximum(cur_layer_input_num_bounds.get_lower(), 0),
                    np.maximum(cur_layer_input_num_bounds.get_upper(), 0))

                if BoundsManager.USE_FIXED_NEURONS:
                    self.force_inactive_neurons(cur_layer_output_eq, cur_layer_output_num_bounds,
                                                 current_layer_inactive)

                stable_count += self.get_layer_stability_stats(layer_id, cur_layer_input_num_bounds,
                                                               stability_info, overapprox_area)

                print(f"dim: {cur_layer_output_num_bounds.lower.shape[0]}")
                print(f"Tempo ReLU: {time.time() - time_t}")
                # TODO: these bounds are somewhat useless. Perhaps copying input numeric bounds?

            elif isinstance(layer, nodes.FlattenNode):
                """ Flatten layer """
                time_t = time.time()
                cur_layer_output_eq = cur_layer_input_eq
                cur_layer_output_num_bounds = cur_layer_input_num_bounds
                print(f"dim: {cur_layer_output_num_bounds.lower.shape[0]}")
                print(f"Tempo FlattenNode: {time.time() - time_t}")

            elif isinstance(layer, nodes.ReshapeNode):
                """ Reshape layer """
                time_t = time.time()
                cur_layer_output_eq = cur_layer_input_eq
                cur_layer_output_num_bounds = cur_layer_input_num_bounds
                print(f"dim: {cur_layer_output_num_bounds.lower.shape[0]}")
                print(f"Tempo ReshapeNode: {time.time() - time_t}")

            else:
                raise Exception(
                    "Currently supporting bounds computation only for FullyConnected, Convolutional, ReLU "
                    "and Flatten layers.\n Instead got {}".format(layer.__class__))

            # Store the current equations and numeric bounds
            symbolic_bounds[layer.identifier] = cur_layer_output_eq
            num_preact_bounds[layer.identifier] = cur_layer_input_num_bounds
            num_postact_bounds[layer.identifier] = cur_layer_output_num_bounds

            # Update the current input equation and numeric bounds
            cur_layer_input_eq = cur_layer_output_eq
            cur_layer_input_num_bounds = cur_layer_output_num_bounds


        # sort the overapproximation areas ascending
        #overapprox_area['sorted'] = sorted(overapprox_area['sorted'], key=lambda x: x[1])
        #overapprox_area['volume'] = BoundsManager.compute_overapproximation_volume(overapprox_area['map'])

        # Put all the collected bounds in a dictionary and return it
        # TODO create data structure
        return {
            'symbolic': symbolic_bounds,
            'numeric_pre': num_preact_bounds,
            'numeric_post': num_postact_bounds,
            'stability_info': stability_info,
            'stable_count': stable_count
            #'overapproximation_area': overapprox_area
        }

    @staticmethod
    def extract_layer_inactive_from_fixed_neurons(fixed_neurons, layer_id):
        # TODO make this a util method somewhere else
        return [neuron_n for ((lay_n, neuron_n), value) in fixed_neurons.items()
                if lay_n == layer_id and value == 0]

    @staticmethod
    def compute_overapproximation_volume(areas_map: dict) -> float:
        import numpy
        return numpy.prod(list(areas_map.values()))

    @staticmethod
    def compute_relu_equation(preact_num_lower, preact_num_upper):
        lower_relu_eq, postact_lower = BoundsManager.get_relu_relax_lower_bound_equation(preact_num_lower,
                                                                                         preact_num_upper)
        upper_relu_eq, postact_upper = BoundsManager.get_relu_relax_upper_bound_equation(preact_num_lower,
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
        matrix = np.identity(size)
        offset = np.zeros(size)

        postact_lower_bounds = np.array(preact_lower_bounds)

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
        matrix = np.identity(size)
        offset = np.zeros(size)

        postact_upper_bounds = np.array(preact_upper_bounds)
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

    def _get_equation_from_input(self, network, layer_id, end, symbolic_bounds, input_bounds):
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
    def get_layer_equation(layer):
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

    @staticmethod
    def get_layer_stability_stats(layer_id, numeric_preactivation_bounds,
                                  stability_info, overapprox_area):
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
                overapprox_area['sorted'].append(((layer_id, neuron_n), area))
                overapprox_area['map'][(layer_id, neuron_n)] = area

        stability_info[StabilityInfo.INACTIVE][layer_id] = inactive
        stability_info[StabilityInfo.ACTIVE][layer_id] = active
        stability_info[StabilityInfo.UNSTABLE].extend(unstable)

        return stable_count

    @staticmethod
    def check_stable(lb, ub) -> NeuronState:
        """
        Static method, given the bounds of a neuron, whether it is stable
        """

        # Positive stable
        if lb >= BoundsManager.PRECISION_GUARD:
            return NeuronState.POSITIVE_STABLE

        # Negative stable
        elif ub <= -BoundsManager.PRECISION_GUARD:
            return NeuronState.NEGATIVE_STABLE

        # Unstable
        else:
            return NeuronState.UNSTABLE

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

    if upper >= - lower:
        mult = upper / (upper - lower)
        return mult, 0

    # upper <= 0:
    # or
    # -lower > upper, i.e., 0 is a tighter lower bound that the slope mult above
    return 0, 0


def get_lin_upper_bound_coefficients(lower, upper):
    if lower >= 0:
        return 1, 0

    if upper <= 0:
        return 0, 0

    mult = upper / (upper - lower)
    add = -mult * lower

    return mult, add


