import copy
import logging
from collections import OrderedDict

from pynever import nodes
from pynever.networks import SequentialNetwork, NeuralNetwork
from pynever.strategies.bp import LOGGER
from pynever.strategies.bp.bounds import SymbolicLinearBounds
from pynever.strategies.bp.linearfunctions import LinearFunctions
from pynever.strategies.bp.utils.property_converter import *
from pynever.strategies.bp.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper


class BoundsManager:
    def __init__(self):
        self.numeric_bounds = None
        self.logger = logging.getLogger(LOGGER)

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

    def compute_bounds_from_property(self, net: NeuralNetwork, prop: 'NeverProperty') -> dict:
        """
        precomputes bounds for all nodes using symbolic linear propagation

        """

        # HyperRectBounds input bounds
        input_hyper_rect = BoundsManager.get_input_bounds(prop)

        # Get layers
        if isinstance(net, SequentialNetwork):
            layers = net2list(net)
        else:
            raise NotImplementedError

        return self.compute_bounds(input_hyper_rect, layers)

    def compute_bounds(self, input_hyper_rect: HyperRectangleBounds, layers: list) -> dict:
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

        # Iterate through the layers
        for i in range(0, len(layers)):

            if isinstance(layers[i], nodes.FullyConnectedNode):
                current_layer_output_equation = self.compute_dense_output_bounds(layers[i],
                                                                                 current_layer_input_equation)
                current_layer_output_numeric_bounds = current_layer_output_equation.to_hyper_rectangle_bounds(
                    input_hyper_rect)

            elif isinstance(layers[i], nodes.ReLUNode):
                current_layer_output_equation = self.compute_relu_output_bounds(current_layer_input_equation,
                                                                                input_hyper_rect)
                current_layer_output_numeric_bounds = HyperRectangleBounds(
                    np.maximum(current_layer_input_numeric_bounds.get_lower(), 0),
                    np.maximum(current_layer_input_numeric_bounds.get_upper(), 0))

            else:
                raise Exception("Currently supporting bounds computation only for Relu and Linear activation functions")

            # Store the current equations and numeric bounds
            symbolic_bounds[layers[i].identifier] = current_layer_output_equation
            numeric_preactivation_bounds[layers[i].identifier] = current_layer_input_numeric_bounds
            numeric_postactivation_bounds[layers[i].identifier] = current_layer_output_numeric_bounds

            # Update the current input equation and numeric bounds
            current_layer_input_equation = current_layer_output_equation
            current_layer_input_numeric_bounds = current_layer_output_numeric_bounds

        # Put all the collected bounds in a dictionary and return it
        return {
            'symbolic': symbolic_bounds,
            'numeric_pre': numeric_preactivation_bounds,
            'numeric_post': numeric_postactivation_bounds
        }

    def branch_update_bounds(self, pre_branch_bounds: dict, nn: list, target: 'RefinementTarget') -> tuple[dict, dict]:
        """
        Create input bounds from the layer target.layer and use the bounds [lb, 0] and [0, ub]
        for neuron target neuron to init a new shot of bounds propagation as if the input layer
        was target layer

        """

        self.logger.info(f'======================================================================\nTarget {target}')
        negative_branch_input = self.refine_input_bounds_for_branch(pre_branch_bounds, nn, target, False)
        positive_branch_input = self.refine_input_bounds_for_branch(pre_branch_bounds, nn, target, True)
        self.logger.info('\n')

        negative_branch_bounds = self.compute_bounds(negative_branch_input, nn)
        positive_branch_bounds = self.compute_bounds(positive_branch_input, nn)

        out_layer = nn[-1].identifier
        self.logger.info(f'Original branch output bounds: {pre_branch_bounds["numeric_post"][out_layer]}')
        self.logger.info(f'Negative branch output bounds: {negative_branch_bounds["numeric_post"][out_layer]}')
        self.logger.info(f'Positive branch output bounds: {positive_branch_bounds["numeric_post"][out_layer]}')

        self.logger.info('\n')

        return (self.compute_bounds(negative_branch_input, nn),
                self.compute_bounds(positive_branch_input, nn))

    def refine_input_bounds_for_branch(self, pre_branch_bounds: dict,
                                       nn: list,
                                       target: 'RefinementTarget',
                                       positive_branch: bool):
        """
        Given an unstable neuron y that we are going to constrain to be positive or negative,
        we recompute tighter input bounds of x=(x1,...,xn) for the solution space induced
        by this branch.

        If the branch is negative, then we have an upper bound equation for y from the input
        variables. Namely,
            y <= c * x + b
        Otherwise, if the branch is positive,
            y >= c * x + b

        for all x coming from the HyperRectangle [l,u] (i.e., li <= xi <=ui).

        Since we are constraining y to be positive or negative, it means that c * x + b should
        be positive or negative as well.
        We therefore need to recompute the bounds for x. We do it as follows.

        For the positive branch we have the following constraint c1 * x1 + ... + cn * xn + b >= 0
        Then
            x1 >= (-c2 * x2 - ... -cn * xn - b )/c1 if c1 is positive
            x1 <= (-c2 * x2 - ... -cn * xn - b )/c1 if c1 is negative
        Thus, when c1 > 0, we can compute a new lower bound of x1,
        and when c1 < 0, we can compute a new upper bound of x1.

        For the negative branch we have the following constraint c1 * x1 + ... + cn * xn + b <= 0
        Then
            x1 <= (-c2 * x2 - ... -cn * xn - b )/c1 if c1 is positive
            x1 >= (-c2 * x2 - ... -cn * xn - b )/c1 if c1 is negative
        Thus, when c1 > 0, we can compute a new upper bound of x1,
        and when c1 < 0, we can compute a new lower bound of x1.

        --------

        We do this using the standard interval arithmetics.
        We only update the bounds if it improves the previous ones.


        Parameters
        ----------
        pre_branch_bounds: the bounds before the split
        nn: the neural network
        target: the neuron to be split
        positive_branch: True if we compute the positive branch, False for the negative

        Returns
        -------
        tighter input bounds induced by the branch where the target neuron is constrained
        to be **positive** or **negative**

        """

        # the bounds for the input layer that we try to refine
        input_bounds = pre_branch_bounds['numeric_pre'][nn[0].identifier]

        try:
            # TODO: retrieve symbolic preactivation bounds properly (make a function for that, instead of hardcoding -1)
            symbolic_preact_bounds = pre_branch_bounds['symbolic'][nn[target.layer_idx - 1].identifier]
        except KeyError:
            print('KeyError in branching, no update was performed.')
            return input_bounds

        input_lower_bounds = copy.deepcopy(input_bounds.get_lower())
        input_upper_bounds = copy.deepcopy(input_bounds.get_upper())

        if positive_branch:
            self.logger.info('Positive branch')

            # The linear equation for the lower bound of the target neuron
            coef = symbolic_preact_bounds.get_lower().get_matrix()[target.neuron_idx]
            shift = symbolic_preact_bounds.get_lower().get_offset()[target.neuron_idx]
        else:
            self.logger.info('Negative branch')

            # The linear equation for the upper bound of the target neuron
            coef = symbolic_preact_bounds.get_upper().get_matrix()[target.neuron_idx]
            shift = symbolic_preact_bounds.get_upper().get_offset()[target.neuron_idx]

        self.logger.info(f'lower: {input_lower_bounds}')
        self.logger.info(f'upper: {input_upper_bounds}')

        n_input_dimensions = len(coef)
        changes = True

        # Continue updating the bounds until they stop improving
        while changes:
            changes = False
            for i in range(n_input_dimensions):
                c = coef[i]

                ## the rest is moved to the other side, so we have the minus
                negated_rem_coef = - np.array(list(coef[:i]) + list(coef[i + 1:])) / c
                pos_rem_coef = np.maximum(np.zeros(n_input_dimensions - 1), negated_rem_coef)
                neg_rem_coef = np.minimum(np.zeros(n_input_dimensions - 1), negated_rem_coef)
                rem_lower_input_bounds = np.array(list(input_lower_bounds[:i]) + list(input_lower_bounds[i + 1:]))
                rem_upper_input_bounds = np.array(list(input_upper_bounds[:i]) + list(input_upper_bounds[i + 1:]))

                if positive_branch:
                    if c > 0:
                        # compute minimum of xi, xi >= (-coefi * rem_xi - b)/c
                        new_lower_i = pos_rem_coef.dot(rem_lower_input_bounds) + neg_rem_coef.dot(
                            rem_upper_input_bounds) - shift
                        if new_lower_i > input_lower_bounds[i]:
                            input_lower_bounds[i] = new_lower_i
                            changes = True
                    else:
                        # compute maximum of xi, xi <= (-coefi * rem_xi - b)/c
                        new_upper_i = pos_rem_coef.dot(rem_upper_input_bounds) + neg_rem_coef.dot(
                            rem_lower_input_bounds) - shift
                        if new_upper_i < input_upper_bounds[i]:
                            input_upper_bounds[i] = new_upper_i
                            changes = True
                else:
                    if c > 0:
                        # compute maximum of xi, xi <= (-coefi * rem_xi - b)/c
                        new_upper_i = pos_rem_coef.dot(rem_upper_input_bounds) + neg_rem_coef.dot(
                            rem_lower_input_bounds) - shift
                        if new_upper_i < input_upper_bounds[i]:
                            input_upper_bounds[i] = new_upper_i
                            changes = True
                    else:
                        # compute minimum of xi, xi >= (-coefi * rem_xi - b)/c
                        new_lower_i = pos_rem_coef.dot(rem_lower_input_bounds) + neg_rem_coef.dot(
                            rem_upper_input_bounds) - shift
                        if new_lower_i > input_lower_bounds[i]:
                            input_lower_bounds[i] = new_lower_i
                            changes = True

        self.logger.info(f'Updated bounds for {"positive" if positive_branch else "negative"} branch:\n'
                         f'{input_lower_bounds}\n{input_upper_bounds}')

        return HyperRectangleBounds(input_lower_bounds, input_upper_bounds)

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
