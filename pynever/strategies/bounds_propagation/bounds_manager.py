from collections import OrderedDict
from enum import Enum

from pynever import nodes
from pynever.networks import SequentialNetwork, NeuralNetwork
from pynever.strategies.bounds_propagation import LOGGER
from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds, AbstractBounds
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions
from pynever.strategies.bounds_propagation.utils.property_converter import *
from pynever.strategies.bounds_propagation.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper
from pynever.strategies.verification.parameters import NeuronState
from pynever.strategies.verification.properties import NeverProperty
from pynever.strategies.verification.search import RefinementTarget


class RefiningBound(Enum):
    LowerBound = 1
    UpperBound = -1


class BoundsManager:
    def __init__(self):
        self.numeric_bounds = None
        self.logger = LOGGER

    def __repr__(self):
        return str(self.numeric_bounds)

    @staticmethod
    def check_stable(var_index: int, bounds: AbstractBounds) -> NeuronState:
        precision_guard = 10e-15

        lb = bounds.get_lower()[var_index]
        ub = bounds.get_upper()[var_index]

        # Positive stable
        if lb >= precision_guard:
            return NeuronState.POSITIVE_STABLE

        # Negative stable
        elif ub <= -precision_guard:
            return NeuronState.NEGATIVE_STABLE

        # Unstable
        else:
            return NeuronState.UNSTABLE

    @staticmethod
    def get_input_bounds(prop: NeverProperty) -> HyperRectangleBounds:
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
    def get_symbolic_bounds_at(bounds: dict, target: RefinementTarget, nn: SequentialNetwork) -> SymbolicLinearBounds:
        """
        This method retrieves the symbolic bounds in the bounds dictionary
        at the specified target

        """

        return bounds['symbolic'][nn.get_identifier_from_index(target.layer_idx - 1)]

    def compute_bounds_from_property(self, network: NeuralNetwork, prop: NeverProperty) -> dict:
        """
        Precomputes bounds for all nodes using symbolic linear propagation

        """

        # HyperRectBounds input bounds
        input_hyper_rect = BoundsManager.get_input_bounds(prop)

        # Get layers
        if not isinstance(network, SequentialNetwork):
            raise NotImplementedError

        return self.compute_bounds(input_hyper_rect, network)

    def compute_bounds(self, input_hyper_rect: HyperRectangleBounds, network: SequentialNetwork) -> dict:
        """
        Given input hyper rectangle bounds, propagates them through the NN given as layers
        using forward symbolic bound propagation and
        returns a dictionary with the symbolic and numeric bounds

        """

        # We are collecting the bounds, symbolic and numeric, in these dictionaries
        symbolic_bounds = dict()
        numeric_preactivation_bounds = dict()
        numeric_postactivation_bounds = OrderedDict()

        # Initialising the current equations
        input_size = input_hyper_rect.get_size()
        lower_equation = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        upper_equation = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        current_layer_input_equation = SymbolicLinearBounds(lower_equation, upper_equation)
        current_layer_input_numeric_bounds = input_hyper_rect

        stable = 0

        # Iterate through the layers
        for layer in network.layers_iterator():

            if isinstance(layer, nodes.FullyConnectedNode):
                current_layer_output_equation = self.compute_dense_output_bounds(layer,
                                                                                 current_layer_input_equation)
                current_layer_output_numeric_bounds = current_layer_output_equation.to_hyper_rectangle_bounds(
                    input_hyper_rect)

            elif isinstance(layer, nodes.ReLUNode):
                current_layer_output_equation = self.compute_relu_output_bounds(current_layer_input_equation,
                                                                                input_hyper_rect)
                for neuron_n in range(current_layer_input_numeric_bounds.size):
                    l, u = current_layer_input_numeric_bounds.get_dimension_bounds(neuron_n)
                    if u <= 0 or l >= 0:
                        stable += 1

                # TODO: these bounds are somewhat useless. Perhaps copying input numeric bounds?
                # For instance, if the last layer is fully connected identity,
                # then the output bounds for the last layer are going to be different (non-clipped)
                # from these clipped bounds
                current_layer_output_numeric_bounds = HyperRectangleBounds(
                    np.maximum(current_layer_input_numeric_bounds.get_lower(), 0),
                    np.maximum(current_layer_input_numeric_bounds.get_upper(), 0))

            elif isinstance(layer, nodes.FlattenNode):
                current_layer_output_equation = current_layer_input_equation
                current_layer_output_numeric_bounds = current_layer_input_numeric_bounds
            else:
                raise Exception(
                    "Currently supporting bounds computation only for FullyConnected, Relu and Flatten layers. \n"
                    "Instead got {}".format(layer.__class__))

            # Store the current equations and numeric bounds
            symbolic_bounds[layer.identifier] = current_layer_output_equation
            numeric_preactivation_bounds[layer.identifier] = current_layer_input_numeric_bounds
            numeric_postactivation_bounds[layer.identifier] = current_layer_output_numeric_bounds

            # Update the current input equation and numeric bounds
            current_layer_input_equation = current_layer_output_equation
            current_layer_input_numeric_bounds = current_layer_output_numeric_bounds

        # Put all the collected bounds in a dictionary and return it
        return {
            'symbolic': symbolic_bounds,
            'numeric_pre': numeric_preactivation_bounds,
            'numeric_post': numeric_postactivation_bounds,
            'stable_count': stable
        }

    def branch_update_bounds(self, pre_branch_bounds: dict, nn: SequentialNetwork, target: RefinementTarget) \
            -> tuple[dict, dict]:
        """
        Create input bounds from the layer target layer and use the bounds [lb, 0] and [0, ub]
        for neuron target neuron to init a new shot of bounds propagation as if the input layer
        was target layer

        """

        self.logger.info(f"======================================================================\nTarget {target}")
        input_bounds = pre_branch_bounds['numeric_pre'][nn.get_first_node().identifier]
        self.logger.info(f"--- Input bounds\n{input_bounds} --- stable count {pre_branch_bounds['stable_count']}")

        negative_branch_input = self.refine_input_bounds_for_negative_branch(pre_branch_bounds, nn, target)
        negative_bounds = None if negative_branch_input is None else (
            pre_branch_bounds if negative_branch_input == input_bounds else
            self.compute_bounds(negative_branch_input, nn))
        self.logger.info(
            f"--- Updated bounds for negative branch: \n{negative_branch_input} --- stable count "
            f"{None if negative_bounds is None else negative_bounds['stable_count']}")

        positive_branch_input = self.refine_input_bounds_for_positive_branch(pre_branch_bounds, nn, target)
        positive_bounds = None if positive_branch_input is None else (
            pre_branch_bounds if positive_branch_input == input_bounds else
            self.compute_bounds(positive_branch_input, nn))
        self.logger.info(
            f"--- Updated bounds for positive branch: \n{positive_branch_input} --- stable count "
            f"{None if positive_bounds is None else positive_bounds['stable_count']}")
        self.logger.info('')

        return negative_bounds, positive_bounds

    def refine_input_bounds_for_positive_branch(self, pre_branch_bounds: dict, nn: SequentialNetwork,
                                                target: RefinementTarget) -> HyperRectangleBounds:
        """
        Given an unstable neuron y that we are going to constrain to be positive,
        we recompute tighter input bounds of x=(x1,...,xn)
        for the solution space induced by this positive branch.

        We have a lower bound equation for y from the input variables. Namely,
            y >= c * x + b

        for all x coming from the hyperrectangle [l,u] (i.e., li <= xi <=ui).

        Since we are constraining y to be positive, it means that c * x + b should be positive as well.
        We therefore need to recompute the bounds for x using its lower bound.


        Parameters
        ----------
        pre_branch_bounds: the bounds before the split
        nn: the neural network
        target: the neuron to be split

        Returns
        -------
        tighter input bounds induced by the branch where the target neuron is constrained to be **positive**
        """

        # the bounds for the input layer that we try to refine
        input_bounds = pre_branch_bounds['numeric_pre'][nn.get_first_node().identifier]

        try:
            symbolic_preact_bounds = BoundsManager.get_symbolic_bounds_at(pre_branch_bounds, target, nn)
        except KeyError:
            self.logger.info('KeyError in branching, no update was performed.')
            return input_bounds

        # The linear equation for the lower bound of the target neuron
        coef = symbolic_preact_bounds.get_lower().get_matrix()[target.neuron_idx]
        shift = symbolic_preact_bounds.get_lower().get_offset()[target.neuron_idx]

        refined_bounds = BoundsManager._refine_input_bounds(coef, shift, input_bounds, RefiningBound.LowerBound)

        return refined_bounds

    def refine_input_bounds_for_negative_branch(self, pre_branch_bounds: dict, nn: SequentialNetwork,
                                                target: RefinementTarget) -> HyperRectangleBounds:
        """
        Given an unstable neuron y that we are going to constrain to be negative,
        we recompute tighter input bounds of x=(x1,...,xn)
        for the solution space induced by this negative branch.

        We have an upper bound equation for y from the input variables. Namely,
            y <= c * x + b

        for all x coming from the hyperrectangle [l,u] (i.e., li <= xi <=ui).

        Since we are constraining y to be negative, it means that c * x + b should be negative as well.
        We therefore need to recompute the bounds for x using its upper bound.


        Parameters
        ----------
        pre_branch_bounds: the bounds before the split
        nn: the neural network
        target: the neuron to be split

        Returns
        -------
        tighter input bounds induced by the branch where the target neuron is constrained to be **positive**
        """

        # the bounds for the input layer that we try to refine
        input_bounds = pre_branch_bounds['numeric_pre'][nn.get_first_node().identifier]

        try:
            symbolic_preact_bounds = BoundsManager.get_symbolic_bounds_at(pre_branch_bounds, target, nn)
        except KeyError:
            self.logger.info('KeyError in branching, no update was performed.')
            return input_bounds

        # The linear equation for the upper bound of the target neuron
        coef = symbolic_preact_bounds.get_upper().get_matrix()[target.neuron_idx]
        shift = symbolic_preact_bounds.get_upper().get_offset()[target.neuron_idx]

        refined_bounds = BoundsManager._refine_input_bounds(coef, shift, input_bounds, RefiningBound.UpperBound)

        return refined_bounds

    @staticmethod
    def _refine_input_bounds(coef, shift, input_bounds: HyperRectangleBounds, sign: RefiningBound) \
            -> HyperRectangleBounds | None:
        """
        We have an equation from the input variables
            c * x + b

        for x coming from the hyperrectangle [l,u] (i.e., li <= xi <=ui).

        If sign is RefininBound.LowerBound, then we are constraining the equation to be positive.
        If sign is RefininBound.UpperBound, we are constraining the equation to be negative.

        In both cases, we can refine the bounds for x to the imposed solution space.
        We do it as follows.

        Assuming sign is RefininBound.LowerBound, we have the following constraint c1 * x1 + ... + cn * xn + b >= 0

        Then
            x1 >= (-c2 * x2 - ... -cn * xn - b)/c1 if c1 is positive
            x1 <= (-c2 * x2 - ... -cn * xn - b)/c1 if c1 is negative

        Thus, when c1 > 0, we can compute a new lower bound of x1,
        and when c1 < 0, we can compute a new upper bound of x1.
        We do it using the standard interval arithmetics.
        We only update the bound if it improves the previous one.

        Following a similar logic, if sign is RefiningBound.UpperBound,
        when c1 > 0, we can compute a new upper bound of x1,
        and when c1 < 0, we can compute a new lower bound of x1.

        """
        refined_input_bounds = input_bounds

        n_input_dimensions = len(coef)
        changes = True

        # Continue updating the bounds until they stop improving
        while changes:
            changes = False
            for i in range(n_input_dimensions):
                c = coef[i]

                if c == 0:
                    continue

                # the rest is moved to the other side, so we have the minus and divided by c
                negated_rem_coef = - np.array(list(coef[:i]) + list(coef[i + 1:])) / c
                shift_div_c = shift / c
                pos_rem_coef = np.maximum(np.zeros(n_input_dimensions - 1), negated_rem_coef)
                neg_rem_coef = np.minimum(np.zeros(n_input_dimensions - 1), negated_rem_coef)
                rem_lower_input_bounds = np.array(
                    list(refined_input_bounds.get_lower()[:i]) + list(refined_input_bounds.get_lower()[i + 1:]))
                rem_upper_input_bounds = np.array(
                    list(refined_input_bounds.get_upper()[:i]) + list(refined_input_bounds.get_upper()[i + 1:]))

                if c * sign.value > 0:
                    "c > 0 and sign = 1 or c < 0 and sign = -1"
                    # compute minimum of xi, xi >= (-coefi * rem_xi - b)/c
                    new_lower_i = pos_rem_coef.dot(rem_lower_input_bounds) + \
                                  neg_rem_coef.dot(rem_upper_input_bounds) - shift_div_c
                    if new_lower_i > refined_input_bounds.get_upper()[i]:
                        # infeasible branch
                        return None

                    elif new_lower_i > refined_input_bounds.get_lower()[i]:
                        if refined_input_bounds == input_bounds:
                            refined_input_bounds = input_bounds.clone()

                        refined_input_bounds.get_lower()[i] = new_lower_i
                        changes = True

                elif c * sign.value < 0:
                    "c < 0 and sign = 1 or c > 0 and sign = -1"
                    # compute maximum of xi, xi <= (-coefi * rem_xi - b)/c
                    new_upper_i = pos_rem_coef.dot(rem_upper_input_bounds) + \
                                  neg_rem_coef.dot(rem_lower_input_bounds) - shift_div_c
                    if new_upper_i < refined_input_bounds.get_lower()[i]:
                        # infeasible branch
                        return None

                    elif new_upper_i < refined_input_bounds.get_upper()[i]:
                        if refined_input_bounds == input_bounds:
                            refined_input_bounds = input_bounds.clone()
                        refined_input_bounds.get_upper()[i] = new_upper_i
                        changes = True

        return refined_input_bounds

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