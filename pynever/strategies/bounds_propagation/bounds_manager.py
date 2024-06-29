from enum import Enum

from pynever import nodes
from pynever.networks import SequentialNetwork, NeuralNetwork
from pynever.strategies.bounds_propagation import LOGGER
from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions
from pynever.strategies.bounds_propagation.utils.property_converter import *
from pynever.strategies.bounds_propagation.utils.utils import get_positive_part, get_negative_part, \
    compute_lin_lower_and_upper
from pynever.strategies.verification.ssbp.constants import NeuronState, RefinementTarget
from pynever.tensors import Tensor


class RefiningBound(Enum):
    LowerBound = 1
    UpperBound = -1


class NeuronSplit(Enum):
    Negative = 0
    Positive = 1


class StabilityInfo(Enum):
    INACTIVE = 0
    ACTIVE = 1
    UNSTABLE = 2


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
        self.logger = LOGGER

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

        stable = 0

        # Iterate through the layers
        for layer in network.layers_iterator():

            if isinstance(layer, nodes.FullyConnectedNode):
                """ Fully Connected layer """

                cur_layer_output_eq = BoundsManager.compute_dense_output_bounds(layer, cur_layer_input_eq)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(input_hyper_rect)

            elif isinstance(layer, nodes.ReLUNode):
                """ ReLU layer """

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
                current_layer_inactive = []
                if BoundsManager.USE_FIXED_NEURONS:
                    current_layer_inactive = extract_layer_inactive_from_fixed_neurons(fixed_neurons, layer_id)
                    if len(current_layer_inactive) > 0:
                        cur_layer_input_eq = SymbolicLinearBounds(
                            cur_layer_input_eq.get_lower().mask_zero_outputs(current_layer_inactive),
                            cur_layer_input_eq.get_upper().mask_zero_outputs(current_layer_inactive))

                cur_layer_output_eq = self.compute_relu_output_bounds(cur_layer_input_eq, input_hyper_rect)

                stability_info[StabilityInfo.INACTIVE][layer_id] = list()
                stability_info[StabilityInfo.ACTIVE][layer_id] = list()

                for neuron_n in range(cur_layer_input_num_bounds.size):
                    l, u = cur_layer_input_num_bounds.get_dimension_bounds(neuron_n)
                    if BoundsManager.USE_FIXED_NEURONS:
                        if neuron_n in current_layer_inactive:
                            l, u = -BoundsManager.PRECISION_GUARD, -BoundsManager.PRECISION_GUARD

                    stable_status = BoundsManager.check_stable(l, u)
                    if stable_status == NeuronState.NEGATIVE_STABLE:
                        stability_info[StabilityInfo.INACTIVE][layer_id].append(neuron_n)
                        stable += 1

                    elif stable_status == NeuronState.POSITIVE_STABLE:
                        stability_info[StabilityInfo.ACTIVE][layer_id].append(neuron_n)
                        stable += 1

                    else:  # stable_status == NeuronState.UNSTABLE
                        stability_info[StabilityInfo.UNSTABLE].append((layer_id, neuron_n))

                        # Compute approximation area
                        area = 0.5 * (u - l) * u
                        overapprox_area['sorted'].append(((layer_id, neuron_n), area))
                        overapprox_area['map'][(layer_id, neuron_n)] = area

                # TODO: these bounds are somewhat useless. Perhaps copying input numeric bounds?
                cur_layer_output_num_bounds = HyperRectangleBounds(
                    np.maximum(cur_layer_input_num_bounds.get_lower(), 0),
                    np.maximum(cur_layer_input_num_bounds.get_upper(), 0))

            elif isinstance(layer, nodes.FlattenNode):
                """ Flatten layer """

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

        # Put all the collected bounds in a dictionary and return it
        # TODO create data structure
        return {
            'symbolic': symbolic_bounds,
            'numeric_pre': num_preact_bounds,
            'numeric_post': num_postact_bounds,
            'stability_info': stability_info,
            'stable_count': stable,
            'overapproximation_area': overapprox_area
        }

    def branch_update_bounds(self, pre_branch_bounds: dict, nn: SequentialNetwork, target: RefinementTarget,
                             fixed_neurons: dict) -> tuple[dict, dict]:
        """
        Update the bounds for after splitting the target neuron.
        Attempts to refine the input bounds for each of the two splits.
        If the input bounds have been updated, recomputes the bounds.
        """

        self.logger.debug(f"======================================================================\n"
                          f"Target {target} "
                          f"Overapprox. area {pre_branch_bounds['overapproximation_area']['map'][target.to_pair()]}")

        input_bounds = pre_branch_bounds['numeric_pre'][nn.get_id_from_index(0)]

        LOGGER.debug(f"--- Input bounds\n"
                     f"{input_bounds} --- stable count {pre_branch_bounds['stable_count']}")

        negative_branch_input = BoundsManager.refine_input_bounds_after_split(
            pre_branch_bounds, nn, target, fixed_neurons, NeuronSplit.Negative)
        negative_bounds = None if negative_branch_input is None else (
            pre_branch_bounds if negative_branch_input == input_bounds else
            self.compute_bounds(negative_branch_input, nn, fixed_neurons=fixed_neurons | {target.to_pair(): 0}))
        LOGGER.debug(f"--- Updated bounds for negative branch:\n"
                     f"{negative_branch_input} --- stable count "
                     f"{None if negative_bounds is None else negative_bounds['stable_count']}")

        positive_branch_input = BoundsManager.refine_input_bounds_after_split(
            pre_branch_bounds, nn, target, fixed_neurons, NeuronSplit.Positive)
        positive_bounds = None if positive_branch_input is None else (
            pre_branch_bounds if positive_branch_input == input_bounds else
            self.compute_bounds(positive_branch_input, nn, fixed_neurons=fixed_neurons | {target.to_pair(): 1}))
        LOGGER.debug(f"--- Updated bounds for positive branch:\n"
                     f"{positive_branch_input} --- stable count "
                     f"{None if positive_bounds is None else positive_bounds['stable_count']}\n\n")

        return negative_bounds, positive_bounds

    @staticmethod
    def refine_input_bounds_after_split(pre_branch_bounds: dict, nn: SequentialNetwork,
                                        target: RefinementTarget, fixed_neurons: dict, status: NeuronSplit) \
            -> HyperRectangleBounds:
        """
        Given an unstable neuron y that we are going to constrain
        to be negative or positive according to the status,
        we recompute tighter input bounds of x=(x1,...,xn)
        for the solution space induced by this split.

        If y is set to be negative, we take its upper bound equation from the input variables:
            y <= c * x + b
        If y is set to be positive, we take its lower bound equation from the input variables:
            y >= c * x + b

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

        try:
            symbolic_preact_bounds = BoundsManager.get_symbolic_preact_bounds_at(pre_branch_bounds, target.layer_id, nn)
        except KeyError:
            LOGGER.info('KeyError in branching, no update was performed.')
            return input_bounds

        if status == NeuronSplit.Negative:
            # The linear equation for the upper bound of the target neuron
            coef = symbolic_preact_bounds.get_upper().get_matrix()[target.neuron_idx]
            shift = symbolic_preact_bounds.get_upper().get_offset()[target.neuron_idx]
        else:  # sign == NeuronSplit.Positive:
            # The negated linear equation for the lower bound of the target neuron
            coef = -symbolic_preact_bounds.get_lower().get_matrix()[target.neuron_idx]
            shift = -symbolic_preact_bounds.get_lower().get_offset()[target.neuron_idx]

        refined_bounds = BoundsManager._refine_input_bounds_for_equation(coef, shift, input_bounds)

        # If the bounds have not been refined,
        # try to use constraints from all the fixed neurons
        if refined_bounds == input_bounds and len(fixed_neurons) > 0:
            refined_bounds = BoundsManager._refine_input_bounds_for_branch(
                fixed_neurons, coef, shift, input_bounds, nn, pre_branch_bounds
            )

        return refined_bounds

    @staticmethod
    def _refine_input_bounds_for_branch(branch: dict, coef: Tensor, shift: Tensor,
                                        input_bounds: HyperRectangleBounds, nn: SequentialNetwork,
                                        pre_branch_bounds: dict) -> HyperRectangleBounds | None:
        """
        We assume that the refinement is done when setting the equations to be <= 0
        """

        # Collecting the equations in normal form (<= 0) from all the fixes, including the latest
        # If value is 0, we take the upper bound.
        # Otherwise, we take the negation of the lower bound.
        coefs = np.array(
            [BoundsManager.get_symbolic_preact_bounds_at(pre_branch_bounds, layer_id, nn).get_upper().get_matrix()[
                 neuron_n]
             if value == 0 else
             -BoundsManager.get_symbolic_preact_bounds_at(pre_branch_bounds, layer_id, nn).get_lower().get_matrix()[
                 neuron_n]
             for ((layer_id, neuron_n), value) in branch.items()] + [coef]
        )

        shifts = np.array(
            [BoundsManager.get_symbolic_preact_bounds_at(pre_branch_bounds, layer_id, nn).get_upper().get_offset()[
                 neuron_n]
             if value == 0 else
             -BoundsManager.get_symbolic_preact_bounds_at(pre_branch_bounds, layer_id, nn).get_lower().get_offset()[
                 neuron_n]
             for ((layer_id, neuron_n), value) in branch.items()] + [shift]
        )

        # The rest is similar to _refine_input_bounds,
        # but we get two different equations for each input dimension i,
        # obtained as the sum of the equations where i appears with the same sign
        n_input_dimensions = len(coef)

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
                return None

            elif i_bounds == 0:
                # No changes
                pass

            else:
                # Bounds have been refined
                LOGGER.debug(f"?? Dim {i} bounds were refined ??")
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
                return input_bounds.get_lower()[i], new_upper_i

        elif c < 0:
            # compute minimum of xi, xi >= (-coefi * rem_xi - b)/c
            new_lower_i = pos_rem_coef.dot(rem_lower_input_bounds) + \
                          neg_rem_coef.dot(rem_upper_input_bounds) + shift_div_c

            if new_lower_i > input_bounds.get_upper()[i]:
                # infeasible branch
                return None

            elif new_lower_i > input_bounds.get_lower()[i]:
                return new_lower_i, input_bounds.get_upper()[i]

        return 0

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


def extract_layer_inactive_from_fixed_neurons(fixed_neurons, layer_id):
    # TODO make this a util method somewhere else
    return [neuron_n for ((lay_n, neuron_n), value) in fixed_neurons.items()
            if lay_n == layer_id and value == 0]


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


def compute_overapproximation_volume(bounds: dict) -> float:
    import numpy
    return numpy.prod(list(bounds['overapproximation_area']['map'].values()))
