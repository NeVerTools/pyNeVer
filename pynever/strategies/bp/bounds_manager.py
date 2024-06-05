import copy
from collections import OrderedDict

from pynever import nodes
from pynever.networks import SequentialNetwork, NeuralNetwork
from pynever.strategies.bp.bounds import SymbolicLinearBounds
from pynever.strategies.bp.linearfunctions import LinearFunctions
from pynever.strategies.bp.utils.property_converter import *
from pynever.strategies.bp.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper


class BoundsManager:
    def __init__(self):
        self.numeric_bounds = None

    def __repr__(self):
        return str(self.numeric_bounds)

    def compute_bounds_from_property(self, net: NeuralNetwork, prop: 'NeverProperty') -> dict:
        """
        precomputes bounds for all nodes using symbolic linear propagation
        """

        # Create HyperRectBounds from property
        property_converter = PropertyFormatConverter(prop)

        # HyperRectBounds input bounds
        input_hyper_rect = property_converter.get_vectors()

        # Get layers
        if isinstance(net, SequentialNetwork):
            layers = net2list(net)
        else:
            raise NotImplementedError

        return self.compute_bounds(input_hyper_rect, layers)

    def compute_bounds(self, input_hyper_rect: HyperRectangleBounds, layers: list) -> dict:

        input_size = input_hyper_rect.get_size()

        lower = LinearFunctions(np.identity(input_size), np.zeros(input_size))
        upper = LinearFunctions(np.identity(input_size), np.zeros(input_size))

        input_bounds = SymbolicLinearBounds(lower, upper)

        numeric_preactivation_bounds = dict()
        numeric_postactivation_bounds = OrderedDict()
        symbolic_bounds = dict()

        current_input_bounds = input_bounds
        symbolic_dense_output_bounds = current_input_bounds
        preactivation_bounds = symbolic_dense_output_bounds.to_hyper_rectangle_bounds(input_hyper_rect)

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

    def branch_update_bounds(self, pre_branch_bounds: dict, nn: list, target: 'RefinementTarget') -> tuple[dict, dict]:
        """
        Create input bounds from the layer target.layer and use the bounds [lb, 0] and [0, ub]
        for neuron target.neuron to init a new shot of bounds propagation as if the input layer
        was target.layer

        """

        try:
            split_bounds = pre_branch_bounds['numeric_pre'][nn[target.layer_idx].identifier]

        except KeyError:
            print('KeyError in branching, no update was performed.')
            return pre_branch_bounds, pre_branch_bounds

        self.refine_input_bounds_for_positive_branch(pre_branch_bounds, nn, target)
        self.refine_input_bounds_for_negative_branch(pre_branch_bounds, nn, target)

        # Lower branch
        lower_branch = copy.deepcopy(split_bounds)
        lower_branch.upper[target.neuron_idx] = 0
        lower_input_bounds = HyperRectangleBounds(lower_branch.lower,
                                                  lower_branch.upper)

        # Upper branch
        upper_branch = copy.deepcopy(split_bounds)
        upper_branch.lower[target.neuron_idx] = 0
        upper_input_bounds = HyperRectangleBounds(upper_branch.lower,
                                                  upper_branch.upper)

        return (self.compute_bounds(lower_input_bounds, nn[target.layer_idx:]),
                self.compute_bounds(upper_input_bounds, nn[target.layer_idx:]))

    def refine_input_bounds_for_positive_branch(self, pre_branch_bounds: dict, nn: list, target: 'RefinementTarget'):
        """
        Given an unstable neuron y that we are going to constrain to be positive,
        we recompute tighter input bounds of x=(x1,...,xn)
        for the solution space induced by this positive branch.

        We have a lower bound equation for y from the input variables. Namely,
            y >= c * x + b

        for all x coming from the hyperrectangle [l,u] (i.e., li <= xi <=ui).

        Since we are constraining y to be positive, it means that c * x + b should be positive as well.
        We therefore need to recompute the bounds for x. We do it as follows.

        We have the following constraint c1 * x1 + ... + cn * xn + b >= 0

        Then
            x1 >= (-c2 * x2 - ... -cn * xn - b )/c1 if c1 is positive
            x1 <= (-c2 * x2 - ... -cn * xn - b )/c1 if c1 is negative

        Thus, when c1 > 0, we can compute a new lower bound of x1,
        and when c1 < 0, we can compute a new upper bound of x1.
        We do it using the standard interval arithmetics.
        We only update the bound if it improves the previous one.


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
        input_bounds = pre_branch_bounds['numeric_pre'][nn[0].identifier]

        try:
            # TODO: retrieve symbolic preactivation bounds properly (make a function for that, instead of hardcoding -1)
            symbolic_preact_bounds = pre_branch_bounds['symbolic'][nn[target.layer_idx - 1].identifier]
        except KeyError:
            print('KeyError in branching, no update was performed.')
            return input_bounds

        # The linear equation for the lower bound of the target neuron
        coef = symbolic_preact_bounds.get_lower().get_matrix()[target.neuron_idx]
        shift = symbolic_preact_bounds.get_lower().get_offset()[target.neuron_idx]

        input_lower_bounds = copy.deepcopy(input_bounds.get_lower())
        input_upper_bounds = copy.deepcopy(input_bounds.get_upper())

        print("Positive branch")
        print("lower", input_lower_bounds)
        print("upper", input_upper_bounds)

        n_input_dimensions = len(coef)
        changes = True

        # continue updating the bounds until they stop improving
        while changes:
            changes = False
            for i in range(n_input_dimensions):
                c = coef[i]

                ## the rest is moved to the other side, so we have the minus and divided by c
                negated_rem_coef = - np.array(list(coef[:i]) + list(coef[i+1:])) / c
                pos_rem_coef = np.maximum(np.zeros(n_input_dimensions - 1), negated_rem_coef)
                neg_rem_coef = np.minimum(np.zeros(n_input_dimensions - 1), negated_rem_coef)
                rem_lower_input_bounds = np.array(list(input_lower_bounds[:i]) + list(input_lower_bounds[i + 1:]))
                rem_upper_input_bounds = np.array(list(input_upper_bounds[:i]) + list(input_upper_bounds[i + 1:]))

                if c > 0:
                    # compute minimum of xi, xi >= (-coefi * rem_xi - b)/c
                    new_lower_i = pos_rem_coef.dot(rem_lower_input_bounds) + neg_rem_coef.dot(rem_upper_input_bounds) - shift
                    if new_lower_i > input_lower_bounds[i]:
                        input_lower_bounds[i] = new_lower_i
                        changes = True
                elif c < 0:
                    # compute maximum of xi, xi <= (-coefi * rem_xi - b)/c
                    new_upper_i = pos_rem_coef.dot(rem_upper_input_bounds) + neg_rem_coef.dot(rem_lower_input_bounds) - shift
                    if new_upper_i < input_upper_bounds[i]:
                        input_upper_bounds[i] = new_upper_i
                        changes = True

        print("updated lower", input_lower_bounds)
        print("updated upper", input_upper_bounds)

    def refine_input_bounds_for_negative_branch(self, pre_branch_bounds: dict, nn: list, target: 'RefinementTarget'):
        """
        Given an unstable neuron y that we are going to constrain to be negative,
        we recompute tighter input bounds of x=(x1,...,xn)
        for the solution space induced by this negative branch.

        We have an upper bound equation for y from the input variables. Namely,
            y <= c * x + b

        for all x coming from the hyperrectangle [l,u] (i.e., li <= xi <=ui).

        Since we are constraining y to be negative, it means that c * x + b should be negative as well.
        We therefore need to recompute the bounds for x. We do it as follows.

        We have the following constraint c1 * x1 + ... + cn * xn + b <= 0

        Then
            x1 <= (-c2 * x2 - ... -cn * xn - b )/c1 if c1 is positive
            x1 >= (-c2 * x2 - ... -cn * xn - b )/c1 if c1 is negative

        Thus, when c1 > 0, we can compute a new upper bound of x1,
        and when c1 < 0, we can compute a new lower bound of x1.
        We do it using the standard interval arithmetics.
        We only update the bound if it improves the previous one.


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
        input_bounds = pre_branch_bounds['numeric_pre'][nn[0].identifier]

        try:
            # TODO: retrieve symbolic preactivation bounds properly (make a function for that, instead of hardcoding -1)
            symbolic_preact_bounds = pre_branch_bounds['symbolic'][nn[target.layer_idx - 1].identifier]
        except KeyError:
            print('KeyError in branching, no update was performed.')
            return input_bounds

        # The linear equation for the upper bound of the target neuron
        coef = symbolic_preact_bounds.get_upper().get_matrix()[target.neuron_idx]
        shift = symbolic_preact_bounds.get_upper().get_offset()[target.neuron_idx]

        input_lower_bounds = copy.deepcopy(input_bounds.get_lower())
        input_upper_bounds = copy.deepcopy(input_bounds.get_upper())

        print("Negative branch")
        print("lower", input_lower_bounds)
        print("upper", input_upper_bounds)

        n_input_dimensions = len(coef)
        changes = True

        # Continue updating the bounds until they stop improving
        while changes:
            changes = False
            for i in range(n_input_dimensions):
                c = coef[i]

                ## the rest is moved to the other side, so we have the minus
                negated_rem_coef = - np.array(list(coef[:i]) + list(coef[i+1:])) / c
                pos_rem_coef = np.maximum(np.zeros(n_input_dimensions - 1), negated_rem_coef)
                neg_rem_coef = np.minimum(np.zeros(n_input_dimensions - 1), negated_rem_coef)
                rem_lower_input_bounds = np.array(list(input_lower_bounds[:i]) + list(input_lower_bounds[i + 1:]))
                rem_upper_input_bounds = np.array(list(input_upper_bounds[:i]) + list(input_upper_bounds[i + 1:]))

                if c > 0:
                    # compute maximum of xi, xi <= (-coefi * rem_xi - b)/c
                    new_upper_i = pos_rem_coef.dot(rem_upper_input_bounds) + neg_rem_coef.dot(rem_lower_input_bounds) - shift
                    if new_upper_i < input_upper_bounds[i]:
                        input_upper_bounds[i] = new_upper_i
                        changes = True
                elif c < 0:
                    # compute minimum of xi, xi >= (-coefi * rem_xi - b)/c
                    new_lower_i = pos_rem_coef.dot(rem_lower_input_bounds) + neg_rem_coef.dot(rem_upper_input_bounds) - shift
                    if new_lower_i > input_lower_bounds[i]:
                        input_lower_bounds[i] = new_lower_i
                        changes = True

        print("updated lower", input_lower_bounds)
        print("updated upper", input_upper_bounds)

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
