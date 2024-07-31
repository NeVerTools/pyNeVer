from pynever import nodes
from pynever.exceptions import FixedConflictWithBounds
from pynever.networks import SequentialNetwork, NeuralNetwork
from pynever.strategies.bounds_propagation import LOGGER
from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds
from pynever.strategies.bounds_propagation.convolution import LinearizeConv
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions
from pynever.strategies.bounds_propagation.relu import LinearizeReLU, StabilityInfo
from pynever.strategies.bounds_propagation.utils.property_converter import *
from pynever.strategies.bounds_propagation.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper
from pynever.strategies.verification.ssbp.constants import NeuronState, RefinementTarget
from pynever.tensors import Tensor


class BoundsManager:
    """
    This class handles the propagation of symbolic bounds and the refinement
    of the input bounds after a branch split

    """

    INPUT_DIMENSIONS_TO_REFINE = 50
    PRECISION_GUARD = 10e-15

    def __init__(self):
        # TODO add new data structure for bounds
        self.numeric_bounds = None
        self.logger = LOGGER

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

    def __repr__(self):
        return str(self.numeric_bounds)

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

        # Equations for each layer to do backward substitution
        layer2layer_equations = dict()

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
                BoundsManager.compute_layer_bounds(layer, cur_layer_input_eq, cur_layer_input_num_bounds,
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

    @staticmethod
    def compute_layer_bounds(layer: nodes.LayerNode, layer_in_eq: SymbolicLinearBounds,
                             layer_in_num: HyperRectangleBounds, input_hyper_rect: HyperRectangleBounds,
                             direction: str, stable_count: int, fixed_neurons: dict) \
            -> tuple[SymbolicLinearBounds, HyperRectangleBounds, int]:

        # Buffer
        cur_layer_output_eq = layer_in_eq
        cur_layer_output_num_bounds = layer_in_num

        if isinstance(layer, nodes.FullyConnectedNode):
            """ Fully Connected layer """

            if direction == 'forwards':
                cur_layer_output_eq = BoundsManager.compute_dense_output_bounds(layer, layer_in_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)

        elif isinstance(layer, nodes.ConvNode):
            """ Convolutional layer """

            if direction == 'forwards':
                conv_lin = LinearizeConv()
                cur_layer_output_eq = conv_lin.compute_output_equation(layer, layer_in_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)

        elif isinstance(layer, nodes.ReLUNode):
            """ ReLU layer """

            if direction == 'forwards':
                relu_lin = LinearizeReLU(fixed_neurons, stable_count, input_hyper_rect)
                cur_layer_output_eq = relu_lin.compute_output_equation(layer_in_eq)
                cur_layer_output_num_bounds = relu_lin.compute_output_numeric(layer, layer_in_num, layer_in_eq)
                stable_count = BoundsManager.get_layer_stability_stats(layer.identifier, layer_in_num)

        elif isinstance(layer, nodes.FlattenNode):
            """ Flatten layer """
            pass

        elif isinstance(layer, nodes.ReshapeNode):
            """ Reshape layer """
            pass

        else:
            raise Exception(
                "Currently supporting bounds computation only for FullyConnected, Convolutional, ReLU "
                "and Flatten layers.\n Instead got {}".format(layer.__class__))

        return cur_layer_output_eq, cur_layer_output_num_bounds, stable_count

    def compute_bounds_backwards(self, input_hyper_rect: HyperRectangleBounds, network: SequentialNetwork,
                                 fixed_neurons: dict = None) -> dict:
        """
        Given input hyper rectangle bounds, propagates them through the NN
        using forward symbolic bound propagation and
        returns a dictionary with the symbolic and numeric bounds as well
        as information on stability and refinement parameters

        """

        if fixed_neurons is None:
            fixed_neurons = dict()

        # We are collecting the bounds, symbolic and numeric, in these dictionaries
        symbolic_bounds = dict()
        num_preact_bounds = dict()
        num_postact_bounds = dict()

        # Equations for each layer to do backward substitution
        layer2layer_equations = dict()

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

                layer_equation = BoundsManager.get_layer_equation(layer)
                layer2layer_equations[layer.identifier] = SymbolicLinearBounds(layer_equation, layer_equation)

                lower_eq_from_input, lower_bounds = (
                    self._get_equation_from_input(network, layer.identifier, "lower", layer2layer_equations,
                                                  input_hyper_rect))
                upper_eq_from_input, upper_bounds = (
                    self._get_equation_from_input(network, layer.identifier, "upper", layer2layer_equations,
                                                  input_hyper_rect))

                cur_layer_output_num_bounds = HyperRectangleBounds(lower_bounds, upper_bounds)
                cur_layer_output_eq = SymbolicLinearBounds(lower_eq_from_input, upper_eq_from_input)

            elif isinstance(layer, nodes.ReLUNode):
                """ ReLU layer """

                if BoundsManager.USE_FIXED_NEURONS:
                    try:
                        cur_layer_input_eq, cur_layer_input_num_bounds = self.check_and_enforce_fixed_constraints(
                            cur_layer_input_eq, cur_layer_input_num_bounds, fixed_neurons, layer.identifier
                        )
                    except FixedConflictWithBounds:
                        # The current branch is not feasible. Return None
                        return None

                relu_eq, cur_layer_output_num_bounds = \
                    self.compute_relu_equation(cur_layer_input_num_bounds.get_lower(),
                                               cur_layer_input_num_bounds.get_upper())

                layer2layer_equations[layer.identifier] = relu_eq

                stable_count += self.get_layer_stability_stats(layer.identifier, cur_layer_input_num_bounds,
                                                               stability_info, overapprox_area)

                # Just to set a value to cur_layer_output_eq
                cur_layer_output_eq = cur_layer_input_eq

            elif isinstance(layer, nodes.FlattenNode):
                """ Flatten layer """

                layer2layer_equations[layer.identifier] = cur_layer_input_eq
                cur_layer_output_eq = cur_layer_input_eq
                cur_layer_output_num_bounds = cur_layer_input_num_bounds

            elif isinstance(layer, nodes.ReshapeNode):
                """ Reshape layer """

                cur_layer_output_eq = cur_layer_input_eq
                cur_layer_output_num_bounds = cur_layer_input_num_bounds

            elif isinstance(layer, nodes.ConvNode):
                """ Convolutional layer """

                raise NotImplementedError('Not yet')

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
        overapprox_area['sorted'] = sorted(overapprox_area['sorted'], key=lambda x: x[1])
        overapprox_area['volume'] = compute_overapproximation_volume(overapprox_area['map'])

        # Put all the collected bounds in a dictionary and return it
        # TODO create data structure
        return {
            'symbolic': symbolic_bounds,
            'numeric_pre': num_preact_bounds,
            'numeric_post': num_postact_bounds,
            'stability_info': stability_info,
            'stable_count': stable_count,
            'overapproximation_area': overapprox_area
        }

    @staticmethod
    def check_and_enforce_fixed_constraints(relu_input_eq, preact_bounds, fixed_neurons, layer_id):
        """
        We need to check if the bounds do not conflict with the currently fixed neurons.
        That could happen if we haven't detected that the current branch is infeasible.
        That could happen because we are dealing with approximated bounds.
        """

        current_layer_inactive = extract_layer_inactive_from_fixed_neurons(fixed_neurons, layer_id)
        current_layer_active = extract_layer_active_from_fixed_neurons(fixed_neurons, layer_id)

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
                new_bounds.lower[neuron_n] = -BoundsManager.PRECISION_GUARD
                new_bounds.upper[neuron_n] = -BoundsManager.PRECISION_GUARD

        return new_eq, new_bounds

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
    def get_layer_stability_stats(layer_id, numeric_preactivation_bounds):
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

    def branch_update_bounds(self, pre_branch_bounds: dict, nn: SequentialNetwork, target: RefinementTarget,
                             fixed_neurons: dict) -> tuple[dict, dict]:
        """
        Update the bounds for after splitting the target neuron.
        Attempts to refine the input bounds for each of the two splits.
        If the input bounds have been updated, recomputes the bounds.
        """

        self.logger.debug("\tTarget {} "
                          "Overapprox. area {:10.4}".format(target, pre_branch_bounds['overapproximation_area']['map'][
            target.to_pair()]))

        input_bounds = pre_branch_bounds['numeric_pre'][nn.get_id_from_index(0)]

        # LOGGER.debug(f"--- Input bounds\n"
        #              f"{input_bounds} --- stable count {pre_branch_bounds['stable_count']}"
        #              f" Volume {pre_branch_bounds['overapproximation_area']['volume']}")

        negative_branch_input = BoundsManager.refine_input_bounds_after_split(
            pre_branch_bounds, nn, target, NeuronSplit.Negative, fixed_neurons
        )
        # negative_bounds = None if negative_branch_input is None else (
        #     pre_branch_bounds if (negative_branch_input == input_bounds and not BoundsManager.USE_FIXED_NEURONS) else
        #     self.compute_bounds(negative_branch_input, nn, fixed_neurons=fixed_neurons | {target.to_pair(): 0})
        # )
        negative_bounds = None if negative_branch_input is None else (
            pre_branch_bounds if negative_branch_input == input_bounds else
            self.compute_bounds(negative_branch_input, nn, fixed_neurons=fixed_neurons | {target.to_pair(): 0})
        )
        LOGGER.debug("\tNega Stable count  {}  Volume {} --- {}".format(
            None if negative_bounds is None else "{:4}".format(negative_bounds['stable_count']),
            None if negative_bounds is None else "{:10.4}".format(negative_bounds['overapproximation_area']['volume']),
            negative_branch_input))

        positive_branch_input = BoundsManager.refine_input_bounds_after_split(
            pre_branch_bounds, nn, target, NeuronSplit.Positive, fixed_neurons
        )
        positive_bounds = None if positive_branch_input is None else (
            pre_branch_bounds if positive_branch_input == input_bounds else
            self.compute_bounds(positive_branch_input, nn, fixed_neurons=fixed_neurons | {target.to_pair(): 1})
        )
        LOGGER.debug("\tPosi Stable count  {}  Volume {} --- {}".format(
            None if positive_bounds is None else "{:4}".format(positive_bounds['stable_count']),
            None if positive_bounds is None else "{:10.4}".format(positive_bounds['overapproximation_area']['volume']),
            positive_branch_input))

        return negative_bounds, positive_bounds

    def branch_bisect_input(self, bounds, nn: SequentialNetwork, fixed_neurons):
        input_bounds = bounds['numeric_pre'][nn.get_first_node().identifier]

        lower_half, upper_half = BoundsManager.bisect_an_input_dimension(input_bounds)

        negative_bounds = self.compute_bounds(lower_half, nn, fixed_neurons=fixed_neurons)
        positive_bounds = self.compute_bounds(upper_half, nn, fixed_neurons=fixed_neurons)

        LOGGER.debug("\tBisect1 Stable count  {}  Volume {} --- {}".format(
            None if negative_bounds is None else "{:4}".format(negative_bounds['stable_count']),
            None if negative_bounds is None else "{:10.4}".format(negative_bounds['overapproximation_area']['volume']),
            lower_half))
        LOGGER.debug("\tBisect2 Stable count  {}  Volume {} --- {}".format(
            None if positive_bounds is None else "{:4}".format(positive_bounds['stable_count']),
            None if positive_bounds is None else "{:10.4}".format(positive_bounds['overapproximation_area']['volume']),
            upper_half))
        return negative_bounds, positive_bounds

    @staticmethod
    def bisect_an_input_dimension(input_bounds):
        diff = input_bounds.get_upper() - input_bounds.get_lower()
        widest_dim = np.argmax(diff)
        mid = diff[widest_dim] / 2

        lower_half = input_bounds.clone()
        upper_half = input_bounds.clone()

        lower_half.upper[widest_dim] = lower_half.lower[widest_dim] + mid
        upper_half.lower[widest_dim] = lower_half.upper[widest_dim]

        return lower_half, upper_half

    @staticmethod
    def refine_input_bounds_after_split(pre_branch_bounds: dict, nn: SequentialNetwork,
                                        target: RefinementTarget, status: NeuronSplit, fixed_neurons: dict) \
            -> HyperRectangleBounds:
        """
        Given an unstable neuron y that we are going to constrain
        to be negative or positive according to the status,
        we recompute tighter input bounds of x=(x1,...,xn)
        for the solution space induced by this split.

        If y is set to be negative, we take its lower bound equation from the input variables:
            y >= c * x + b
        If y is set to be positive, we take its upper bound equation from the input variables:
            y <= c * x + b

        for all x coming from the hyperrectangle [l,u] (i.e., li <= xi <=ui).

        If we are constraining y to be negative, we have the constraint
            c * x + b <= 0.
        If we are constraining y to be positive, we have the constraint
            c * x + b >= 0 or, in the normal form, - c * x - b <= 0.

        We recompute the bounds for x using the constraint.

        Parameters
        ----------
        pre_branch_bounds : dict
            The bounds before the split
        nn : SequentialNetwork
            The neural network
        target : RefinementTarget
            The neuron to be split
        fixed_neurons : dict
            The dictionary of fixed neurons so far

        Returns
        -------
        Tighter input bounds induced by the split

        """
        # the bounds for the input layer that we try to refine
        input_bounds = pre_branch_bounds['numeric_pre'][nn.get_first_node().identifier]

        # If the bounds have not been refined,
        # try to use constraints from all the fixed neurons
        # fixed_neurons = compute_fixed_but_unstable_wrt_bounds(pre_branch_bounds, fixed_neurons)
        if len(fixed_neurons) > 0:
            refined_bounds = BoundsManager.optimise_input_bounds_for_branch(
                fixed_neurons | {target.to_pair(): status.value}, pre_branch_bounds, nn
            )
            # refined_bounds = BoundsManager._refine_input_bounds_for_branch(
            #     fixed_neurons, target, status, input_bounds, nn, pre_branch_bounds
            # )
        else:
            coef, shift = BoundsManager._get_equation_from_fixed_neuron(target, status.value, pre_branch_bounds, nn)
            refined_bounds = BoundsManager._refine_input_bounds_for_equation(coef, shift, input_bounds)

        return refined_bounds

    @staticmethod
    def _refine_input_bounds_for_branch(fixed_neurons: dict, target: RefinementTarget, value: NeuronSplit,
                                        input_bounds: HyperRectangleBounds,
                                        nn: SequentialNetwork, pre_branch_bounds: dict) -> HyperRectangleBounds | None:
        """
        We assume that the refinement is done when setting the equations to be <= 0
        """

        # Collecting the equations in normal form (<= 0) from all the fixes, including the latest
        # If value is 0, we take the lower bound.
        # Otherwise, we take the negation of the upper bound.
        equations = BoundsManager._get_equations_from_fixed_neurons(fixed_neurons, pre_branch_bounds, nn)
        coef, shift = BoundsManager._get_equation_from_fixed_neuron(target, value.value, pre_branch_bounds, nn)

        input_bounds = BoundsManager._refine_input_bounds_for_equation(coef, shift, input_bounds)
        if input_bounds is None:
            return None

        # The rest is similar to _refine_input_bounds,
        # but we get two different equations for each input dimension i,
        # obtained as the sum of the equations where i appears with the same sign
        n_input_dimensions = len(coef)

        dimensions_to_consider = np.array(range(n_input_dimensions))
        # An optimisation for very high-dimensional inputs
        if n_input_dimensions > BoundsManager.INPUT_DIMENSIONS_TO_REFINE:
            # we will only consider the dimensions
            # with the coefficient that is large enough in absolute terms
            # and at most BoundsManager.INPUT_DIMENSIONS_TO_REFINE
            percentage = 1 - BoundsManager.INPUT_DIMENSIONS_TO_REFINE / n_input_dimensions
            cutoff_c = np.quantile(abs(coef), percentage)
            mask = (abs(coef) > cutoff_c)
            dimensions_to_consider = dimensions_to_consider[mask]

        refined_input_bounds = input_bounds
        for i in dimensions_to_consider:
            i_bounds = BoundsManager._refine_input_dimension_for_neuron_and_branch(input_bounds, equations, coef, shift,
                                                                                   i)

            if i_bounds is None:
                LOGGER.info("!! Split is infeasible !!")
                # The split is infeasible
                return None

            elif i_bounds == 0:
                # No changes
                pass

            else:
                LOGGER.info(f"!! Bounds refined for branch !!")
                # Bounds have been refined
                if refined_input_bounds == input_bounds:
                    # Only create a new copy of the bounds if there was a change
                    refined_input_bounds = input_bounds.clone()
                # Update the bounds
                refined_input_bounds.get_lower()[i] = i_bounds[0]
                refined_input_bounds.get_upper()[i] = i_bounds[1]

        return refined_input_bounds

    @staticmethod
    def _refine_input_dimension_for_neuron_and_branch(input_bounds, equations, coef, shift, i):

        coefs = equations.matrix
        shifts = equations.offset

        # Find equations where the coefficient is the same sign as coef[i]
        if coef[i] > 0:
            mask = (coefs[:, i] > 0)
        elif coef[i] < 0:
            mask = (coefs[:, i] < 0)

        coefs1 = coefs[mask, :]
        shifts1 = shifts[mask]

        # If no equations have been selected, still can try to refine for the new equation
        if len(coefs1) == 0:
            return 0

        best_i_bounds = input_bounds.get_dimension_bounds(i)

        # For every other dimension j, choose an equation eq2 where
        # coefficient j is the opposite sign of coef[j].
        # The idea is to combine eq1 and eq2 so that
        # coefficient i is 1 and
        # coefficient j is 0.
        n_input_dimensions = len(coef)
        for j in [h for h in range(n_input_dimensions) if h != i]:
            if coef[j] > 0:
                mask1 = (coefs1[:, j] < 0)
            elif coef[j] < 0:
                mask1 = (coefs1[:, j] > 0)
            else:
                # Maybe try to refine using the equation eq1?
                continue

            coefs2 = coefs1[mask1, :]
            shifts2 = shifts1[mask1]

            for n in range(len(coefs2)):
                eq2_coef = coefs2[n]
                eq2_shift = shifts2[n]

                k = -coef[j] / eq2_coef[j]

                # in this equation coefficient j is 0
                combined_coef = coef + k * eq2_coef
                combined_shift = shift + k * eq2_shift

                i_bounds = BoundsManager._refine_input_dimension(input_bounds, combined_coef, combined_shift, i)
                if i_bounds is None:
                    # The split is infeasible
                    return None

                elif i_bounds != 0:
                    best_i_bounds = max(best_i_bounds[0], i_bounds[0]), min(best_i_bounds[1], i_bounds[1])

        if best_i_bounds != input_bounds.get_dimension_bounds(i):
            return best_i_bounds
        return 0

    @staticmethod
    def _refine_input_bounds_for_branch_naive(branch: dict, input_bounds: HyperRectangleBounds, nn: SequentialNetwork,
                                              pre_branch_bounds: dict) -> HyperRectangleBounds | None:
        """
        We assume that the refinement is done when setting the equations to be <= 0
        """

        # Collecting the equations in normal form (<= 0) from all the fixes, including the latest
        # If value is 0, we take the lower bound.
        # Otherwise, we take the negation of the upper bound.
        equations = BoundsManager._get_equations_from_fixed_neurons(branch, pre_branch_bounds, nn)
        coefs = equations.matrix
        shifts = equations.offset

        # The rest is similar to _refine_input_bounds,
        # but we get two different equations for each input dimension i,
        # obtained as the sum of the equations where i appears with the same sign
        n_input_dimensions = len(coefs[0])

        all_dimensions = np.array(range(n_input_dimensions))
        dimensions_to_consider = []
        # An optimisation for very high-dimensional inputs
        if n_input_dimensions > BoundsManager.INPUT_DIMENSIONS_TO_REFINE:
            # we will only consider the dimensions
            # with the aggregated coefficient that is large enough in absolute terms
            # and at most BoundsManager.INPUT_DIMENSIONS_TO_REFINE
            percentage = 1 - BoundsManager.INPUT_DIMENSIONS_TO_REFINE / n_input_dimensions

            filtering_pos_coefs = abs(np.array([coefs[(coefs[:, i] > 0), i].sum() for i in range(n_input_dimensions)]))
            cutoff_c = np.quantile(filtering_pos_coefs, percentage)
            dimensions_to_consider.append(all_dimensions[(filtering_pos_coefs > cutoff_c)])

            filtering_neg_coefs = abs(np.array([coefs[(coefs[:, i] < 0), i].sum() for i in range(n_input_dimensions)]))
            cutoff_c = np.quantile(filtering_neg_coefs, percentage)
            dimensions_to_consider.append(all_dimensions[(filtering_neg_coefs > cutoff_c)])

        else:
            dimensions_to_consider.extend([all_dimensions, all_dimensions])

        refined_input_bounds = input_bounds
        for dimensions, sign in zip(dimensions_to_consider, ["pos", "neg"]):
            for i in dimensions:
                # For each input dimension i, we select the subset of the equations
                # with the same coefficient sign for i and optimise for the sum of those equations
                if sign == "pos":
                    mask = (coefs[:, i] > 0)
                else:
                    mask = (coefs[:, i] < 0)

                coef_i = coefs[mask, :]
                shift_i = shifts[mask].sum()

                if len(coef_i) <= 1:
                    # none or one equation have been selected. We want to combine at least two equations.
                    # Nothing to be done
                    continue

                coef_i = coef_i.sum(axis=0)
                i_bounds = BoundsManager._refine_input_dimension(refined_input_bounds, coef_i, shift_i, i)

                if i_bounds is None:
                    LOGGER.info(f"!! Split is infeasible !! {coef_i[i]}")
                    # The split is infeasible
                    return None

                elif i_bounds == 0:
                    # No changes
                    pass

                else:
                    LOGGER.info(f"!! Bounds refined for branch !! {coef_i[i]}")
                    # Bounds have been refined
                    if refined_input_bounds == input_bounds:
                        # Only create a new copy of the bounds if there was a change
                        refined_input_bounds = input_bounds.clone()
                    # Update the bounds
                    refined_input_bounds.get_lower()[i] = i_bounds[0]
                    refined_input_bounds.get_upper()[i] = i_bounds[1]

        return refined_input_bounds

    @staticmethod
    def _get_equations_from_fixed_neurons(fixed_neurons: dict, bounds: dict, nn) -> LinearFunctions:
        """
        Extract the constraints in the normal from
            equation <= 0
        imposed by fixing neurons, given their symbolic preactivation bounds.

        The assumption is that if a neuron y is constrained to be negative, then
            lower_bound <= y <= 0
        gives as the constraint
            lower_bound <= 0.
        Conversely, if y is constrained to be positive, then we have that upper_bound >= y >= 0.
        In the normal form it gives us
            -upper_bound <= 0.
        """
        coefs = []
        shifts = []
        for ((layer_id, neuron_n), value) in fixed_neurons.items():
            coef, shift = BoundsManager._get_equation_from_fixed_neuron(
                RefinementTarget(layer_id, neuron_n), value, bounds, nn
            )
            coefs.append(coef)
            shifts.append(shift)

        return LinearFunctions(np.array(coefs), np.array(shifts))

    @staticmethod
    def _get_equation_from_fixed_neuron(target: RefinementTarget, value: int, bounds: dict, nn):
        """
        See _get_equations_from_fixed_neurons
        """
        symbolic_preact_bounds = BoundsManager.get_symbolic_preact_bounds_at(bounds, target.layer_id, nn)

        if value == 0:
            # The linear equation for the upper bound of the target neuron
            coef = symbolic_preact_bounds.get_lower().get_matrix()[target.neuron_idx]
            shift = symbolic_preact_bounds.get_lower().get_offset()[target.neuron_idx]
        else:  # sign == NeuronSplit.Positive:
            # The negated linear equation for the lower bound of the target neuron
            coef = -symbolic_preact_bounds.get_upper().get_matrix()[target.neuron_idx]
            shift = -symbolic_preact_bounds.get_upper().get_offset()[target.neuron_idx]

        return coef, shift

    @staticmethod
    def _refine_input_bounds_for_equation(coef: Tensor, shift: Tensor, input_bounds: HyperRectangleBounds) \
            -> HyperRectangleBounds | None:
        """
        We have a constraint from the input variables
            c1 * x1 + ... + cn * xn + b <= 0

        for x1,...,xn coming from the hyper-rectangle [l,u] (i.e., li <= xi <=ui).

        We refine the bounds for x to the imposed solution space.
        For instance, for x1 we have

            x1 <= (-c2 * x2 - ... -cn * xn - b)/c1 if c1 is positive
            x1 >= (-c2 * x2 - ... -cn * xn - b)/c1 if c1 is negative

        Thus, when c1 > 0, we can compute a new upper bound of x1,
        and when c1 < 0, we can compute a new lower bound of x1.
        We do it using the standard interval arithmetics.

        If the new bound is inconsistent, e.g., the new upper bound is below the existing lower bound,
        it means the corresponding split/branch is not feasible. We return None.

        We only update the bound if it improves the previous one.
        """

        n_input_dimensions = len(coef)

        dimensions_to_consider = np.array(range(n_input_dimensions))
        # An optimisation for very high-dimensional inputs
        if n_input_dimensions > BoundsManager.INPUT_DIMENSIONS_TO_REFINE:
            # we will only consider the dimensions
            # with the coefficient that is large enough in absolute terms
            # and at most BoundsManager.INPUT_DIMENSIONS_TO_REFINE
            percentage = 1 - BoundsManager.INPUT_DIMENSIONS_TO_REFINE / n_input_dimensions
            cutoff_c = np.quantile(abs(coef), percentage)
            mask = (abs(coef) > cutoff_c)
            dimensions_to_consider = dimensions_to_consider[mask]

        refined_input_bounds = input_bounds
        for i in dimensions_to_consider:
            # Refine the bounds for each input dimension
            i_bounds = BoundsManager._refine_input_dimension(refined_input_bounds, coef, shift, i)

            if i_bounds is None:
                # The split is infeasible

                # from pynever.strategies.verification.ssbp.intersection import \
                #     check_input_refining_one_equation_feasible_with_lp
                # feasible = check_input_refining_one_equation_feasible_with_lp(coef, shift, refined_input_bounds)
                # if feasible:
                #     print("Input refinement is infeasible but the LP is feasible")
                return None

            elif i_bounds == 0:
                # No changes
                pass

            else:
                # Bounds have been refined
                # LOGGER.debug(f"?? Dim {i} bounds were refined ??")
                if refined_input_bounds == input_bounds:
                    # Only create a new copy of the bounds if there was a change
                    refined_input_bounds = input_bounds.clone()
                # Update the bounds
                refined_input_bounds.get_lower()[i] = i_bounds[0]
                refined_input_bounds.get_upper()[i] = i_bounds[1]

        return refined_input_bounds

    @staticmethod
    def _refine_input_dimension(input_bounds: HyperRectangleBounds, coef: Tensor, shift: Tensor,
                                i: int) -> tuple[float, float] | int | None:
        """
        We are given the constraint
            coef * (x1,...,xn) + shift <= 0

        See _refine_input_bounds for more information.

        We are refining the bounds for the input dimension i:

            xi <= (-c1 * x1 - ... -cn * xn - b)/ci if ci is positive -- we refine the upper bound
            xi >= (-c1 * x1 - ... -cn * xn - b)/ci if ci is negative -- we refine the lower bound

        Returns
        -------
        None    if the constraint is infeasible
        0       if no changes
        (l, u)  the new bounds for input dimension i

        """

        c = coef[i]

        if c == 0:
            return None

        # the rest is moved to the other side, so we have the minus and divided by c
        negated_rem_coef = - np.array(list(coef[:i]) + list(coef[i + 1:])) / c
        shift_div_c = - shift / c
        pos_rem_coef = np.maximum(np.zeros(len(coef) - 1), negated_rem_coef)
        neg_rem_coef = np.minimum(np.zeros(len(coef) - 1), negated_rem_coef)

        rem_lower_input_bounds = np.array(
            list(input_bounds.get_lower()[:i]) + list(input_bounds.get_lower()[i + 1:]))
        rem_upper_input_bounds = np.array(
            list(input_bounds.get_upper()[:i]) + list(input_bounds.get_upper()[i + 1:]))

        if c > 0:
            # compute maximum of xi, xi <= (-coefi * rem_xi - b)/c
            new_upper_i = pos_rem_coef.dot(rem_upper_input_bounds) + \
                          neg_rem_coef.dot(rem_lower_input_bounds) + shift_div_c

            if new_upper_i < input_bounds.get_lower()[i]:
                # infeasible branch
                return None

            elif new_upper_i < input_bounds.get_upper()[i]:
                # from pynever.strategies.verification.ssbp.intersection import compute_input_new_max
                # new_upper = compute_input_new_max(coef, shift, input_bounds, i)
                # if new_upper != new_upper_i:
                #     print("Different new upper", new_upper_i, new_upper)
                return input_bounds.get_lower()[i], new_upper_i

        elif c < 0:
            # compute minimum of xi, xi >= (-coefi * rem_xi - b)/c
            new_lower_i = pos_rem_coef.dot(rem_lower_input_bounds) + \
                          neg_rem_coef.dot(rem_upper_input_bounds) + shift_div_c

            if new_lower_i > input_bounds.get_upper()[i]:
                # infeasible branch
                return None

            elif new_lower_i > input_bounds.get_lower()[i]:
                # from pynever.strategies.verification.ssbp.intersection import compute_input_new_min
                # new_lower = compute_input_new_min(coef, shift, input_bounds, i)
                # if new_lower != new_lower_i:
                #     print("Different new lower", new_lower_i, new_lower)
                return new_lower_i, input_bounds.get_upper()[i]

        return 0

    @staticmethod
    def optimise_input_bounds_for_branch(fixed_neurons: dict, bounds: dict, nn) -> dict:
        """
        Optimises input bounds by building a MILP that has
        input variables and, for each fixed neuron, a constraint using its symbolic lower or upper bound.
        The solves for each input variable two optimisation problems: minimising and maximising it.
        """

        input_bounds = bounds['numeric_pre'][nn.get_first_node().identifier]
        n_input_dimensions = input_bounds.get_size()

        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver("", pywraplp.Solver.CLP_LINEAR_PROGRAMMING)

        import numpy as np
        input_vars = np.array([
            solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
            for j in range(n_input_dimensions)])

        # The constraints from fixing the neurons
        equations = BoundsManager._get_equations_from_fixed_neurons(fixed_neurons, bounds, nn)

        # This way of encoding allows to access the dual solution
        worker_constraints = {}
        infinity = solver.infinity()
        for constr_n in range(len(equations.matrix)):
            # solver.Add(input_vars.dot(equations.matrix[i]) + equations.offset[i] <= 0)
            # -infinity <= eq <= 0
            worker_constraints[constr_n] = solver.Constraint(-infinity, -equations.offset[constr_n], 'c[%i]' % constr_n)
            for input_var_n in range(n_input_dimensions):
                worker_constraints[constr_n].SetCoefficient(input_vars[input_var_n],
                                                            equations.matrix[constr_n][input_var_n])

        ## The actual optimisation part
        new_input_bounds = input_bounds.clone()
        bounds_improved = False

        dimensions_to_consider = np.array(range(n_input_dimensions))
        # An optimisation for very high-dimensional inputs
        if n_input_dimensions > BoundsManager.INPUT_DIMENSIONS_TO_REFINE:
            # we will only consider the dimensions
            # with the coefficient that is large enough in absolute terms
            # and at most BoundsManager.INPUT_DIMENSIONS_TO_REFINE

            # This part needs checking
            percentage = 1 - BoundsManager.INPUT_DIMENSIONS_TO_REFINE / n_input_dimensions
            max_coefs = abs(equations.matrix).max(axis=0)
            cutoff_c = np.quantile(max_coefs, percentage)
            all_dimensions = np.array(range(n_input_dimensions))
            dimensions_to_consider = all_dimensions[(max_coefs > cutoff_c)]

        for i_dim in dimensions_to_consider:
            solver.Maximize(input_vars[i_dim])
            status = solver.Solve()

            new_lower, new_upper = input_bounds.get_dimension_bounds(i_dim)
            if status == pywraplp.Solver.INFEASIBLE:
                return None

            elif status == pywraplp.Solver.OPTIMAL:
                if input_vars[i_dim].solution_value() < new_upper:
                    # dual_sol = [worker_constraints[i].dual_value() for i in worker_constraints]
                    # LOGGER.debug(f"Dual solution: {dual_sol}")

                    # eq_mult = np.array([worker_constraints[i].dual_value() for i in worker_constraints])
                    # coef = -(eq_mult.reshape(-1, 1) * equations.matrix).sum(axis=0)
                    # shift = -(eq_mult * equations.offset).sum()
                    # print("Equation", list(coef), shift)

                    new_upper = input_vars[i_dim].solution_value()
                    bounds_improved = True

            solver.Minimize(input_vars[i_dim])
            status = solver.Solve()

            if status == pywraplp.Solver.INFEASIBLE:
                return None

            elif status == pywraplp.Solver.OPTIMAL:
                if input_vars[i_dim].solution_value() > new_lower:
                    # dual_sol = [worker_constraints[i].dual_value() for i in worker_constraints]
                    # LOGGER.debug(f"Dual solution: {dual_sol}")

                    # eq_mult = np.array([worker_constraints[i].dual_value() for i in worker_constraints])
                    # coef = -(eq_mult.reshape(-1, 1) * equations.matrix).sum(axis=0)
                    # shift = -(eq_mult * equations.offset).sum()
                    # print("Equation", list(coef), shift)

                    new_lower = input_vars[i_dim].solution_value()
                    bounds_improved = True

            new_input_bounds.get_lower()[i_dim] = new_lower
            new_input_bounds.get_upper()[i_dim] = new_upper

        if bounds_improved:
            return new_input_bounds
        return input_bounds

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


def extract_layer_inactive_from_bounds(bounds, layer_id):
    return {neuron_n for lay_n, neuron_n in bounds['stability_info'][StabilityInfo.UNSTABLE]
            if lay_n == layer_id}


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


def compute_fixed_but_unstable_wrt_bounds(bounds: dict, fixed_neurons: dict) -> dict:
    return {(layer_id, neuron_n): value for (layer_id, neuron_n), value in fixed_neurons.items()
            if (layer_id, neuron_n) in bounds['stability_info'][StabilityInfo.UNSTABLE]}


def compute_stable_from_bounds_and_fixed_neurons(bounds: dict, fixed_neurons: dict) -> set:
    """
    Utility method

    """
    stable = (
        {(layer_id, neuron_n)
         for (layer_id, neurons) in bounds['stability_info'][StabilityInfo.INACTIVE].items()
         for neuron_n in neurons}
        .union(
            {(layer_id, neuron_n)
             for (layer_id, neurons) in bounds['stability_info'][StabilityInfo.ACTIVE].items()
             for neuron_n in neurons}
        ).union(
            fixed_neurons.items())
    )
    return stable


def compute_overapproximation_volume(areas_map: dict) -> float:
    import numpy
    return numpy.prod(list(areas_map.values()))
