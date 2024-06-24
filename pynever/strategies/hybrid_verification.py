import copy
import datetime

import numpy as np

import pynever.strategies.abstraction as abst
from pynever import nodes, networks
from pynever.strategies.abstraction import Star
from pynever.strategies.bp.bounds import HyperRectangleBounds
from pynever.strategies.bp.bounds_manager import BoundsManager, StabilityInfo
from pynever.strategies.bp.utils.utils import get_positive_part, get_negative_part, compute_max, compute_min


class RefinementTarget:
    """

    """

    # TODO how to use with ResNets? Layer identifier?
    def __init__(self, layer: int, neuron: int):
        self.layer_idx = layer
        self.neuron_idx = neuron

    def __repr__(self):
        return f'({self.layer_idx}, {self.neuron_idx})'

    def to_pair(self):
        return self.layer_idx, self.neuron_idx


"""
##############################################################
###               Next target computation                  ###                   
##############################################################
"""
def get_target_sequential_optimized(star: Star, nn_bounds: dict, network: networks.SequentialNetwork) -> tuple[RefinementTarget | None, Star]:
    unstable = nn_bounds['stability_info'][StabilityInfo.UNSTABLE]
    unstable = [neuron for neuron in unstable if not neuron in star.fixed_neurons]

    if len(unstable) > 0:
        for layer_n, neuron_n in unstable:
                if layer_n != star.ref_layer:
                    # TODO: have the check as a method of Star? Or some other util
                    if not star.ref_unstable_neurons is None and len(star.ref_unstable_neurons) == 0:
                        # the current layer is complete, so we need to move to the next layer
                        # through the fully connected transformation
                        #
                        star = propagate_and_init_star_before_relu_layer(star, nn_bounds, network)

                star.ref_neuron = neuron_n

                return RefinementTarget(layer_n, neuron_n), star

    return None, star

def get_target_lowest_overapprox_current_layer(star: Star, nn_bounds: dict, network) -> tuple[RefinementTarget | None, Star]:
    # Compute what we believe to be unstable neurons wrt the bounds and what we have fixed so far
    unstable = nn_bounds['stability_info'][StabilityInfo.UNSTABLE]
    unstable = [neuron for neuron in unstable if not neuron in star.fixed_neurons]

    unstable_lowest_layer = sorted(list({layer_n for (layer_n, _) in unstable}))
    if len(unstable_lowest_layer) != 0:
        unstable_lowest_layer = unstable_lowest_layer[0]

    # There are still unstable neurons
    if len(unstable) > 0:
        if not star.ref_unstable_neurons is None and len(star.ref_unstable_neurons) == 0:
            # the current layer is complete, so we need to move to the next layer
            # through the fully connected transformation
            star = propagate_and_init_star_before_relu_layer(star, nn_bounds, network)

            # if star.ref_layer > unstable_lowest_layer:
            #     x = 5

            next_layers = sorted(list({layer_n for (layer_n, neuron_n) in nn_bounds['overapproximation_area']['map'].keys()
                                       if layer_n >= star.ref_layer}))
            if len(next_layers) == 0:
                return None, star

            star.ref_layer = next_layers[0]

        else:
            # stay in the current layer
            pass

        # select candidates from star.ref_layer
        candidates_sorted = [((layer_n, neuron_n), area)
                             for (layer_n, neuron_n), area in nn_bounds['overapproximation_area']['sorted']
                             if layer_n == star.ref_layer]

        # select candidate that has not been fixed yet
        for (layer_n, neuron_n), _ in candidates_sorted:
            if not (layer_n, neuron_n) in star.fixed_neurons:

                star.ref_neuron = neuron_n
                return RefinementTarget(layer_n, neuron_n), star

    # No unstable neurons
    return None, star


"""
##############################################################
###                   Split computation                    ###                   
##############################################################
"""

def split_star_opt(star: Star, target: RefinementTarget, nn_list, nn_bounds: dict) -> list:
    """
    Optimized split method

    target is known to be unstable wrt bounds.
    """
    # Update the bounds after the split
    negative_bounds, positive_bounds = BoundsManager().branch_update_bounds(nn_bounds, nn_list, target, star.fixed_neurons)

    stars = compute_star_after_fixing_to_negative(star, negative_bounds, target, nn_list) +\
        compute_star_after_fixing_to_positive(star, positive_bounds, target, nn_list)

    # if (5, 35) in star.fixed_neurons and len(stars) == 2 and (5, 35) not in stars[1][0].fixed_neurons:
    #     x = 5

    # stars = sorted(stars, key=lambda x: x[2])
    stars = [(s, bounds) for (s, bounds, _) in stars]

    return stars


def compute_star_after_fixing_to_negative(star, bounds, target, nn_list):

    if bounds is None:
        return []

    if target.layer_idx != star.ref_layer:
        #TODO: add the symbolic equation constraint to the predicate.
        # Could be useful when doing the intersection check with LP
        return [(star, bounds)]

    index = target.neuron_idx
    layer_inactive = bounds['stability_info'][StabilityInfo.INACTIVE][nn_list[target.layer_idx].identifier]

    new_basis_matrix, new_center = mask_transformation_for_inactive_neurons(
        layer_inactive + [index], star.basis_matrix, star.center
    )

    fixed_so_far = copy.deepcopy(star.fixed_neurons)
    fixed_so_far[target.to_pair()] = 0

    # Some of the neurons that were unstable at the beginning
    # could have become stable due to prior splitting.
    # So we intersect ref_unstable_neurons with the unstable neurons according to the bounds.
    layer_unstable_per_bounds = {neuron_n for layer_n, neuron_n in bounds['stability_info'][StabilityInfo.UNSTABLE]
                                 if layer_n == target.layer_idx}
    ref_layer_unstable = star.ref_unstable_neurons & layer_unstable_per_bounds

    # We have just fixed the target neuron, so we remove it from the set of unstable neurons.
    ref_layer_unstable.discard(index)

    # if (5, 35) in bounds['stability_info'][StabilityInfo.UNSTABLE] and 35 not in ref_layer_unstable and target.to_pair() != (5,35) and target.layer_idx == 5:
    #     print(star.fixed_neurons, target)


    # Update the predicate to include the constraint that the target neuron y is inactive
    lower_pred, lower_bias = add_to_predicate_inactive_constraint(star.predicate_matrix, star.predicate_bias,
                                                                  get_neuron_equation(star, index))

    lower_star = Star(lower_pred, lower_bias, new_center, new_basis_matrix,
                      ref_layer=target.layer_idx, ref_neuron=target.neuron_idx,
                      ref_unstable_neurons=ref_layer_unstable, fixed_neurons=fixed_so_far)

    return [(lower_star, bounds, bounds['stable_count'])]


def compute_star_after_fixing_to_positive(star, bounds, target, nn_list):

    if bounds is None:
        return []

    if target.layer_idx != star.ref_layer:
        #TODO: add the symbolic equation constraint to the predicate.
        # Could be useful when doing the intersection check with LP
        return [(star, bounds)]

    index = target.neuron_idx
    layer_inactive = bounds['stability_info'][StabilityInfo.INACTIVE][nn_list[target.layer_idx].identifier]

    new_basis_matrix, new_center = mask_transformation_for_inactive_neurons(
        layer_inactive, star.basis_matrix, star.center)

    fixed_so_far = copy.deepcopy(star.fixed_neurons)
    fixed_so_far[target.to_pair()] = 1

    # Some of the neurons that were unstable at the beginning
    # could have become stable due to prior splitting.
    # So we intersect ref_unstable_neurons with the unstable neurons according to the bounds.
    layer_unstable_per_bounds = {neuron_n for layer_n, neuron_n in bounds['stability_info'][StabilityInfo.UNSTABLE]
                                 if layer_n == target.layer_idx}
    ref_layer_unstable = star.ref_unstable_neurons & layer_unstable_per_bounds

    # We have just fixed the target neuron, so we remove it from the set of unstable neurons.
    ref_layer_unstable.discard(index)

    # Update the predicate to include the constraint that the target neuron is active
    upper_pred, upper_bias = add_to_predicate_active_constraint(star.predicate_matrix, star.predicate_bias,
                                                                get_neuron_equation(star, index))

    upper_star = Star(upper_pred, upper_bias, new_center, new_basis_matrix,
                      ref_layer=target.layer_idx, ref_neuron=target.neuron_idx,
                      ref_unstable_neurons=ref_layer_unstable, fixed_neurons=fixed_so_far)

    return [(upper_star, bounds, bounds['stable_count'])]


"""
##############################################################
###               Intersection computation                 ###                   
##############################################################
"""
def intersect_adaptive(star, nn, nn_bounds, prop: 'NeverProperty'):
    unstable = nn_bounds['stability_info'][StabilityInfo.UNSTABLE]
    unstable = [neuron for neuron in unstable if not neuron in star.fixed_neurons]

    if len(unstable) == 0:
        return intersect_bounds(star, nn, nn_bounds, prop)
    return intersect_lightweight_milp(star, nn, nn_bounds, prop)


def intersect_bounds(star, nn, nn_bounds, prop: 'NeverProperty'):
    """
    This method should be called when there are no unstable neurons
    """
    star = abs_propagation(star, nn_bounds, nn)

    input_bounds = nn_bounds['numeric_pre'][nn[0].identifier]

    # Since all relu neurons have been fixed,
    # we assume that the basis is the linear function describing the relation
    # between the input and the output neurons.
    # So we recompute the output bounds given the equation by the bases
    output_bounds = HyperRectangleBounds(compute_min(star.basis_matrix, input_bounds) + star.center.reshape(-1),
                                         compute_max(star.basis_matrix, input_bounds) + star.center.reshape(-1))

    n_disjunctions = len(prop.out_coef_mat)

    # For each disjunction in the output property, check none is satisfied
    for i in range(n_disjunctions):
        # Every condition
        conjunction_intersects = True
        for j in range(len(prop.out_coef_mat[i])):
            max_value = compute_max(prop.out_coef_mat[i][j], output_bounds) - prop.out_bias_mat[i][j][0]
            if max_value > 0:
                # this conjunct is not satisfied, as it should be <= 0
                conjunction_intersects = False
                break
        if conjunction_intersects:
            # only now use the method that calls an LP solver when we need a counter-example
            return intersect_abstract_star_milp(star, nn, nn_bounds, prop)
    return False, []


def intersect_abstract_star_milp(star, nn, nn_bounds, prop: 'NeverProperty'):
    """
    Another implementation of intersect_star_lp
    """
    star = abs_propagation(star, nn_bounds, nn)

    input_bounds = nn_bounds['numeric_pre'][nn[0].identifier]
    # there could be new input dimensions introduced
    # comparing to the original network inputs
    n_input_dimensions = star.basis_matrix.shape[1]

    output_bounds = nn_bounds['numeric_post'][nn[-1].identifier]
    n_output_dimensions = output_bounds.get_size()

    from ortools.linear_solver import pywraplp
    solver = pywraplp.Solver("", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    ## Create the input and the output variables
    input_vars = np.array([
        solver.NumVar(input_bounds.get_lower()[j] if j < input_bounds.get_size() else 0,
                      input_bounds.get_upper()[j] if j < input_bounds.get_size() else solver.infinity(),
                      f'alpha_{j}')
        for j in range(n_input_dimensions)])

    output_vars = np.array([
        solver.NumVar(output_bounds.get_lower()[j], output_bounds.get_upper()[j], f'beta_{j}')
        for j in range(n_output_dimensions)])

    ## The constraints relating input and output variables
    for j in range(n_output_dimensions):
        solver.Add(
            input_vars.dot(star.basis_matrix[j]) + star.center[j][0] - output_vars[j] == 0)

    ## The constraints from the predicate
    for j in range(star.predicate_matrix.shape[0]):
        solver.Add(
            input_vars.dot(star.predicate_matrix[j]) + star.predicate_bias[j][0] <= 0)

    ## The constraints for the property
    _encode_output_property_constraints(solver, prop, output_bounds, output_vars)

    solver.Maximize(0)
    status = solver.Solve()
    if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
        return False, []
    else:
        # [sat_disj_index] = [i for i in range(n_disjunctions) if deltas[i].solution_value() == 1]
        return True, [input_vars[i].solution_value() for i in range(n_input_dimensions)]


def intersect_lightweight_milp(star, nn, nn_bounds, prop: 'NeverProperty'):
    """
    Checks for an intersection by building a MILP that has
    * three kinds of variables:
        - input variables
        - output variables
        - binary variables for encoding the disjunction in the output property
    * three kinds of constraints:
        - for each fixed neuron, a constraint using its symbolic lower or upper bound
        - for each output variable, a lower and an upper bound constraint from the input variables
        - the constraints involving binary variables to encode the output property

    Returns
    -------
    a pair
        - True if there is a solution to the above program, and
          the values to the input variables in this solution in an array
        - False otherwise, and empty array []
    """
    from ortools.linear_solver import pywraplp

    input_bounds = nn_bounds['numeric_pre'][nn[0].identifier]
    n_input_dimensions = input_bounds.get_size()
    output_bounds = nn_bounds['numeric_post'][nn[-1].identifier]
    n_output_dimensions = output_bounds.get_size()

    solver = pywraplp.Solver("", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    input_vars = np.array([
        solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
        for j in range(n_input_dimensions)])

    output_vars = np.array([
        solver.NumVar(output_bounds.get_lower()[j], output_bounds.get_upper()[j], f'beta_{j}')
        for j in range(n_output_dimensions)])

    ## The constraints from the branching
    for (layer_n, neuron_n), value in star.fixed_neurons.items():
        if value == 0:
            solver.Add(
                input_vars.dot(nn_bounds['symbolic'][nn[layer_n - 1].identifier].get_upper().get_matrix()[neuron_n]) +
                nn_bounds['symbolic'][nn[layer_n - 1].identifier].get_upper().get_offset()[neuron_n] <= 0)
        else:
            solver.Add(
                input_vars.dot(nn_bounds['symbolic'][nn[layer_n - 1].identifier].get_lower().get_matrix()[neuron_n]) +
                nn_bounds['symbolic'][nn[layer_n - 1].identifier].get_lower().get_offset()[neuron_n] >= 0)

    ## The constraints relating input and output variables
    for j in range(n_output_dimensions):
        solver.Add(
            input_vars.dot(nn_bounds['symbolic'][nn[-1].identifier].get_upper().get_matrix()[j]) +
            nn_bounds['symbolic'][nn[-1].identifier].get_upper().get_offset()[j] - output_vars[j] >= 0)
        solver.Add(
            input_vars.dot(nn_bounds['symbolic'][nn[-1].identifier].get_lower().get_matrix()[j]) +
            nn_bounds['symbolic'][nn[-1].identifier].get_lower().get_offset()[j] - output_vars[j] <= 0)

    ## The constraints for the property
    _encode_output_property_constraints(solver, prop, output_bounds, output_vars)

    solver.Maximize(0)
    status = solver.Solve()
    if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
        return False, []
    else:
        # [sat_disj_index] = [i for i in range(n_disjunctions) if deltas[i].solution_value() == 1]
        return True, [input_vars[i].solution_value() for i in range(n_input_dimensions)]


def _encode_output_property_constraints(solver, prop, output_bounds, output_vars):
    """
    Encodes and adds to the solver the constraints from encoding the ouput property.

    Parameters
    ----------
    solver
    prop
    output_bounds
    output_vars

    Returns
    -------

    """
    n_disjunctions = len(prop.out_coef_mat)

    # binary variables for each of the disjunctions
    # delta_i = 1 means disjunct i is satisfied
    deltas = np.array([solver.IntVar(0, 1, f"delta{j}") for j in range(n_disjunctions)])
    solver.Add(deltas.dot(np.ones(n_disjunctions)) == 1)

    # For each disjunction in the output property, add the constraints conditioned by deltas
    for i in range(n_disjunctions):
        conjunction = []
        for j in range(len(prop.out_coef_mat[i])):
            # the big M constant as not clear how to do indicator constraints
            bigM = compute_max(prop.out_coef_mat[i][j], output_bounds) - prop.out_bias_mat[i][j][0]

            # when delta_i = 0, the constraint is automatically satisfied because of the bigM
            conjunction.append(solver.Add(
                output_vars.dot(prop.out_coef_mat[i][j])
                - (1 - deltas[i]) * bigM
                - prop.out_bias_mat[i][j][0] <= 0
            ))


def check_valid_counterexample(candidate_cex, nn, prop):
    # TODO: replace this with a normal forward pass
    input_bounds = HyperRectangleBounds(np.array(candidate_cex), np.array(candidate_cex))
    bounds = BoundsManager().compute_bounds(input_bounds, nn)
    output_bounds = bounds['numeric_post'][nn[-1].identifier]

    candidate_output = output_bounds.get_lower()

    n_disjunctions = len(prop.out_coef_mat)

    # For each disjunction in the output property, check at least one is satisfied
    for i in range(n_disjunctions):
        # Every condition
        satisfied = True
        for j in range(len(prop.out_coef_mat[i])):
            if prop.out_coef_mat[i][j].dot(candidate_output) - prop.out_bias_mat[i][j][0] > 0:
                # this conjunct is not satisfied, as it should be <= 0
                satisfied = False
                break
        if satisfied:
            return True
    return False


"""
##############################################################
###               Star propagation methods                 ###                   
##############################################################
"""

def abs_propagation(star: Star, bounds: dict, nn_list: list) -> Star:
    """
    This method performs the abstract propagation of a single star starting
    from a specific layer and neuron. The output is a single star that uses
    approximation in the next layers

    Parameters
    ----------
    star : Star
        The star to process
    bounds : dict
        The bounds of the network layers
    nn_list : list
        The neural network represented as a list

    Returns
    ----------
    Star
        The resulting star approximate with the abstract propagation

    """

    start_layer = star.ref_layer
    neuron_idx = star.ref_neuron

    for layer in nn_list[start_layer:]:
        i = nn_list.index(layer)

        # Propagate fully connected entirely
        if isinstance(layer, nodes.FullyConnectedNode):
            # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
            bias = get_layer_bias_as_two_dimensional(layer)
            star = abst.single_fc_forward(star, layer.weight, bias).pop()

        # Propagate ReLU starting from target
        elif isinstance(layer, nodes.ReLUNode):
            layer_bounds = bounds['numeric_pre'][layer.identifier]
            # layer_inactive = bounds['stability_info'][StabilityInfo.INACTIVE][layer.identifier]

            if i == start_layer:
                star = abst.approx_relu_forward(star, layer_bounds, layer.in_dim[0], start_idx=neuron_idx, layer_n=i)
            else:
                star = abst.approx_relu_forward(star, layer_bounds, layer.in_dim[0], layer_n=i)

        elif isinstance(layer, nodes.FlattenNode):
            continue

        else:
            raise NotImplementedError('Unsupported layer')

    return star


def propagate_and_init_star_before_relu_layer(star, bounds, network, skip=True):
    """
    Compute the initial star which will always start from the first layer and
    where we will use the bounds to determine the inactive nodes,
    so that we could set the transformation for them to 0.
    """
    new_star, relu_layer = propagate_until_relu(star, network, skip=skip)
    relu_layer_n = new_star.ref_layer

    if relu_layer is not None:
        layer_inactive = (bounds['stability_info'][StabilityInfo.INACTIVE][relu_layer.identifier] +
                          [i for (lay_n, i), value in new_star.fixed_neurons.items() if lay_n == relu_layer_n and value == 0])
        layer_unstable = {neuron_n for layer_n, neuron_n in bounds['stability_info'][StabilityInfo.UNSTABLE]
                          if layer_n == relu_layer_n and not (layer_n, neuron_n) in new_star.fixed_neurons}

        new_basis_matrix, new_center = mask_transformation_for_inactive_neurons(
            layer_inactive, new_star.basis_matrix, new_star.center)

        return abst.Star(new_star.predicate_matrix, new_star.predicate_bias, new_center, new_basis_matrix,
                         ref_layer=relu_layer_n, ref_unstable_neurons=layer_unstable, fixed_neurons=new_star.fixed_neurons)

    return new_star


def propagate_until_relu(star: Star, network: networks.SequentialNetwork, skip: bool) -> Star:
    """
    This function performs the star propagation throughout Fully Connected layers
    only, until a ReLU layer is encountered. This is used in order to process
    Fully Connected layers only once per cycle

    Parameters
    ----------
    star : Star
        The star to process
    nn_list : list
        The neural network represented as a list
    skip : bool
        Flag to signal end of propagation

    Returns
    ----------
    Star
        The resulting star before the next ReLU layer

    """

    if not skip:
        # not skip means we are starting from the beginning.
        # So the stars ref_layer might not be initialised.
        start_layer = 0
    else:
        start_layer = star.ref_layer

    i = 0

    relu_layer = None
    for layer in network.layers_iterator(start_layer):

        # Propagate fully connected entirely
        if isinstance(layer, nodes.FullyConnectedNode):
            # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
            bias = get_layer_bias_as_two_dimensional(layer)
            star = abst.single_fc_forward(star, layer.weight, bias).pop()
            i += 1

        elif isinstance(layer, nodes.ReLUNode):
            # If all the neurons have been processed...
            if skip:  # star.ref_neuron == star.center.shape[0] - 1 and skip:
                skip = False
                i += 1
                continue
            # Otherwise, stay on this layer and interrupt cycle
            else:
                relu_layer = layer
                break

        else:
            raise NotImplementedError(f'Currently supporting only FullyConnected and ReLU nodes. '
                                      f'Unsupported layer {layer.__class__}')

    # Set reference layer
    star.ref_layer = start_layer + i

    return star, relu_layer


"""
##############################################################
###                    Various utils                       ###                   
##############################################################
"""

def mask_transformation_for_inactive_neurons(inactive_neurons: list, matrix, offset):
    # The mask for all inactive neurons, to set the transformation of the corresponding neurons to 0
    mask = np.diag(
        [0 if neuron_n in inactive_neurons else 1 for neuron_n in range(matrix.shape[0])]
    )

    return np.matmul(mask, matrix), np.matmul(mask, offset)

def get_neuron_equation(star, neuron_idx):
    return star.basis_matrix[neuron_idx, :], star.center[neuron_idx]


def add_to_predicate_active_constraint(predicate_matrix, predicate_bias, equation):
    """
    Update the predicate to include the constraint that the neuron defined by equation is active,
    i.e., if the neuron y is defined as coeff * x + shift, where x are input neurons,
    then coeff * x + shift >= 0, or, equivalently, - coeff * x <= shift
    """
    coeff, shift = equation
    pred = np.vstack((predicate_matrix, -coeff))
    bias = np.vstack((predicate_bias, shift))
    return pred, bias


def add_to_predicate_inactive_constraint(predicate_matrix, predicate_bias, equation):
    """
    Update the predicate to include the constraint that the neuron defined by equation is inactive,
    i.e., if the neuron y is defined as coeff * x + shift, where x are input neurons,
    then coeff * x + shift <= 0, or, equivalently, coeff * x <= -shift
    """
    coeff, shift = equation
    pred = np.vstack((predicate_matrix, coeff))
    bias = np.vstack((predicate_bias, -shift))
    return pred, bias



def get_layer_bias_as_two_dimensional(layer):
    # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
    if layer.bias.shape != (layer.weight.shape[0], 1):
        bias = np.expand_dims(layer.bias, 1)
    else:
        bias = layer.bias
    return bias
