import datetime
from enum import Enum

import numpy as np
from ortools.linear_solver import pywraplp

from pynever import utilities
from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction.star import ExtendedStar
from pynever.strategies.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.bounds_propagation.bounds_manager import compute_unstable_from_bounds_and_fixed_neurons, \
    BoundsManager, compute_stable_from_bounds_and_fixed_neurons, compute_overapproximation_volume, StabilityInfo
from pynever.strategies.bounds_propagation.utils import utils as bounds_utils
from pynever.strategies.verification.properties import NeverProperty
from pynever.strategies.verification.ssbp import split, propagation
from pynever.tensors import Tensor


def intersect_star_lp(star: ExtendedStar, prop: NeverProperty, network: SequentialNetwork, nn_bounds: dict) \
        -> tuple[bool, Tensor | None]:
    """
    This method computes the intersection between a star and one or
    more hyper planes specified by a property using an LP

    """

    # Compute the output abstract star from star/bounds
    out_star = propagation.abs_propagation(star, network, nn_bounds)

    # Check intersection using an LP
    intersects, unsafe_stars = check_star_intersection(out_star, prop)

    if len(unsafe_stars) == 0:
        return intersects, None

    return intersects, unsafe_stars[0].get_samples(num_samples=1)[0]


def check_star_intersection(star: ExtendedStar, prop: NeverProperty) -> tuple[bool, list[ExtendedStar]]:
    """
    This function checks whether a star intersects with the output property
    using a linear program. Since the output property may contain disjunction
    the intersection is computed with all the half-spaces in the output property.
    If an intersection occurs, the resulting star is saved and returned for
    extracting a counterexample

    Parameters
    ----------
    star : Star
        The star to intersect with the output property
    prop : NeverProperty
        The property of interest

    Returns
    -------
    (bool, list)
        A tuple containing the result of the intersection check (True/False) and
        the list of unfeasible stars. If the result is True the list must contain
        at least one element, otherwise the list must be empty

    """

    intersects = False
    unsafe_stars = []

    # Loop possible disjunctions
    for i in range(len(prop.out_coef_mat)):
        intersection = star.intersect_with_halfspace(prop.out_coef_mat[i], prop.out_bias_mat[i])
        if not intersection.check_if_empty():
            intersects = True
            unsafe_stars.append(intersection)

    return intersects, unsafe_stars


def intersect_adaptive(star: ExtendedStar, nn: SequentialNetwork, nn_bounds: dict, prop: NeverProperty) \
        -> tuple[bool, list[float]]:
    """
    Control the intersection method based on the unstable neurons

    Returns
    ----------
    tuple[bool, list[float]]
        A tuple with the result (True or False) and the counterexample, if present

    """

    unstable = compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    if len(unstable) == 0:
        return intersect_bounds(star, nn, nn_bounds, prop)
    # elif overapprox_volume > 10e12:
    #     return True, []
    # elif len(unstable) <= 20 or nn_bounds['overapproximation_area']['volume'] < 1:
    #     return intersect_complete_milp(star, nn, nn_bounds, prop)
    elif len(unstable) <= 50:# or nn_bounds['overapproximation_area']['volume'] < 1:
        return intersect_abstract_milp(star, nn, nn_bounds, prop)

    # return intersect_abstract_milp(star, nn, nn_bounds, prop)
    return intersect_light_milp(star, nn, nn_bounds, prop)
    # if abstract[0] != light[0]:
    #     abs = intersect_abstract_milp(star, nn, nn_bounds, prop)
    #     lig = intersect_light_milp(star, nn, nn_bounds, prop)
    # return light


def intersect_bounds(star: ExtendedStar, nn: SequentialNetwork, nn_bounds: dict, prop: NeverProperty) \
        -> tuple[bool, list[float]]:
    """
    This intersection method should be called only when the bounds are thought to be precise enough,
    i.e., the over-approximation volume is small enough.
    It uses an LP only for finding a counterexample.

    Returns
    ----------
    tuple[bool, list[float]]
        A tuple with the result (True or False) and the counterexample, if present

    """

    output_bounds = nn_bounds['numeric_post'][nn.get_last_node().identifier]

    return check_bounds_satisfy_property(output_bounds, prop, nn, star, nn_bounds)


def intersect_abstract_milp(star: ExtendedStar, nn: SequentialNetwork, nn_bounds: dict, prop: NeverProperty) \
        -> tuple[bool, list[float]]:
    """
    This intersection method uses a MILP to retrieve a counterexample

    Returns
    ----------
    tuple[bool, list[float]]
        A tuple with the result (True or False) and the counterexample, if present

    """

    star = propagation.abs_propagation(star, nn, nn_bounds)

    input_bounds = nn_bounds['numeric_pre'][nn.get_id_from_index(0)]

    # there could be new input dimensions introduced
    # comparing to the original network inputs
    n_input_dimensions = star.basis_matrix.shape[1]

    output_bounds = nn_bounds['numeric_post'][nn.get_id_from_index(-1)]
    n_output_dimensions = output_bounds.get_size()

    solver = pywraplp.Solver("", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # Create the input and the output variables
    input_vars = np.array([
        solver.NumVar(input_bounds.get_lower()[j] if j < input_bounds.get_size() else 0,
                      input_bounds.get_upper()[j] if j < input_bounds.get_size() else solver.infinity(),
                      f'alpha_{j}')
        for j in range(n_input_dimensions)])

    output_vars = np.array([
        solver.NumVar(output_bounds.get_lower()[j], output_bounds.get_upper()[j], f'beta_{j}')
        for j in range(n_output_dimensions)])

    # The constraints relating input and output variables
    for j in range(n_output_dimensions):
        solver.Add(
            input_vars.dot(star.basis_matrix[j]) + star.center[j][0] - output_vars[j] == 0)

    # The constraints from the predicate
    for j in range(star.predicate_matrix.shape[0]):
        solver.Add(
            input_vars.dot(star.predicate_matrix[j]) - star.predicate_bias[j][0] <= 0)

    # The constraints for the property
    _encode_output_property_constraints(solver, prop, output_bounds, output_vars)

    solver.Maximize(0)
    status = solver.Solve()

    if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
        return False, []

    else:
        # Only return the values of the original input vars
        # cex = Tensor(np.array([input_vars[i].solution_value() for i in range(input_bounds.get_size())]))
        # if not check_valid_counterexample(cex, nn, prop):
        #     return intersect_complete_milp(star, nn, nn_bounds, prop)
        return True, [input_vars[i].solution_value() for i in range(input_bounds.get_size())]


def intersect_light_milp(star: ExtendedStar, nn: SequentialNetwork, nn_bounds: dict, prop: NeverProperty) \
        -> tuple[bool, list[float]]:
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

    input_bounds = nn_bounds['numeric_pre'][nn.get_first_node().identifier]
    n_input_dimensions = input_bounds.get_size()
    output_bounds = nn_bounds['numeric_post'][nn.get_last_node().identifier]
    n_output_dimensions = output_bounds.get_size()

    solver = pywraplp.Solver("", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    input_vars = np.array([
        solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
        for j in range(n_input_dimensions)])

    output_vars = np.array([
        solver.NumVar(output_bounds.get_lower()[j], output_bounds.get_upper()[j], f'beta_{j}')
        for j in range(n_output_dimensions)])

    if len(nn_bounds['stability_info'][StabilityInfo.UNSTABLE]) > 0:
        # The constraints from the branching only if there are unstable neurons according to the bounds,
        # hence there was some approximation and the output equations are not exact
        for (layer_id, neuron_n), value in star.fixed_neurons.items():
            if value == 0:
                solver.Add(
                    input_vars.dot(
                        BoundsManager.get_symbolic_preact_bounds_at(nn_bounds, layer_id, nn).get_lower().get_matrix()[neuron_n]) +
                    BoundsManager.get_symbolic_preact_bounds_at(nn_bounds, layer_id, nn).get_lower().get_offset()[neuron_n] <= 0)
            else:
                solver.Add(
                    input_vars.dot(
                        BoundsManager.get_symbolic_preact_bounds_at(nn_bounds, layer_id, nn).get_upper().get_matrix()[neuron_n]) +
                    BoundsManager.get_symbolic_preact_bounds_at(nn_bounds, layer_id, nn).get_upper().get_offset()[neuron_n] >= 0)

    # The constraints relating input and output variables
    for j in range(n_output_dimensions):
        solver.Add(
            input_vars.dot(nn_bounds['symbolic'][nn.get_last_node().identifier].get_upper().get_matrix()[j]) +
            nn_bounds['symbolic'][nn.get_last_node().identifier].get_upper().get_offset()[j] - output_vars[j] >= 0)
        solver.Add(
            input_vars.dot(nn_bounds['symbolic'][nn.get_last_node().identifier].get_lower().get_matrix()[j]) +
            nn_bounds['symbolic'][nn.get_last_node().identifier].get_lower().get_offset()[j] - output_vars[j] <= 0)

    # The constraints for the property
    _encode_output_property_constraints(solver, prop, output_bounds, output_vars)

    solver.Maximize(0)
    status = solver.Solve()

    if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
        return False, []

    else:
        # TODO check
        return True, [input_vars[i].solution_value() for i in range(n_input_dimensions)]


def intersect_complete_milp(star: ExtendedStar, nn: SequentialNetwork, nn_bounds: dict, prop: NeverProperty) \
        -> tuple[bool, list[float]]:
    """
    Checks for an intersection by building a complete MILP encoding

    Returns
    -------
    a pair
        - True if there is a solution to the above program, and
          the values to the input variables in this solution in an array
        - False otherwise, and empty array []

    """

    solver = pywraplp.Solver("", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    variables = _create_variables_and_constraints(solver, nn, nn_bounds)

    # The constraints for the property
    output_bounds = nn_bounds['numeric_post'][nn.get_last_node().identifier]
    _encode_output_property_constraints(solver, prop, output_bounds, variables[-1])

    solver.Maximize(0)
    status = solver.Solve()

    if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
        return False, []

    else:
        # TODO check
        return True, [variables[0][i].solution_value() for i in range(len(variables[0]))]


def _encode_output_property_constraints(solver: pywraplp.Solver, prop: NeverProperty,
                                        output_bounds: dict, output_vars: np.ndarray) -> None:
    """
    Encodes and adds to the solver the constraints from encoding the output property.

    Parameters
    ----------
    solver
    prop
    output_bounds
    output_vars

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
            bigM = bounds_utils.compute_max(prop.out_coef_mat[i][j], output_bounds) - prop.out_bias_mat[i][j][0]

            # when delta_i = 0, the constraint is automatically satisfied because of the bigM
            conjunction.append(solver.Add(
                output_vars.dot(prop.out_coef_mat[i][j])
                - (1 - deltas[i]) * bigM
                - prop.out_bias_mat[i][j][0] <= 0
            ))


def _create_variables_and_constraints(solver, nn, nn_bounds):

    variables = []

    input_bounds = nn_bounds['numeric_pre'][nn.get_first_node().identifier]
    input_vars = np.array([
        solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
        for j in range(input_bounds.get_size())])
    variables.append(input_vars)

    for layer in nn.layers_iterator():
        from pynever import nodes
        if isinstance(layer, nodes.ReLUNode):
            lower_bounds = nn_bounds['numeric_pre'][layer.identifier].get_lower()
            upper_bounds = nn_bounds['numeric_pre'][layer.identifier].get_upper()

            layer_vars = np.array([
                solver.NumVar(lower_bounds[node_n], upper_bounds[node_n], f'x_{layer.identifier}_{node_n}')
                for node_n in range(lower_bounds.size)])

            variables.append(np.array(layer_vars))

            prev_layer = nn.get_prev_node(layer)
            dot_product = np.array([
                prev_layer.weight[i].dot(variables[-2]) + prev_layer.bias[i]
                for i in range(prev_layer.weight.shape[0])
            ])

            for node_n in range(lower_bounds.size):
                node_var = layer_vars[node_n]

                if lower_bounds[node_n] >= 0:
                    solver.Add(node_var == dot_product[node_n])

                elif upper_bounds[node_n] <= 0:
                    solver.Add(node_var == 0)

                else:
                    """
                    Add the binary variable for the BIGM encoding
                    """
                    delta = solver.IntVar(0, 1, f"delta{layer.identifier}_{node_n}")

                    """
                    The BIG-M constraints
                    """
                    solver.Add(node_var >= dot_product[node_n])
                    solver.Add(node_var <= dot_product[node_n] - lower_bounds[node_n] * (1 - delta))
                    solver.Add(node_var <= upper_bounds[node_n] * delta)

    last_layer = nn.get_last_node()
    output_bounds = nn_bounds['numeric_post'][last_layer.identifier]
    output_vars = np.array([
        solver.NumVar(output_bounds.get_lower()[j], output_bounds.get_upper()[j], f'beta_{j}')
        for j in range(output_bounds.get_size())])
    variables.append(output_vars)

    for node_n in range(output_bounds.size):
        solver.Add(output_vars[node_n] == last_layer.weight[node_n].dot(variables[-2]) + last_layer.bias[node_n])

    return variables


def check_bounds_satisfy_property(output_bounds, prop, nn, star, nn_bounds):
    n_disjunctions = len(prop.out_coef_mat)

    possible_counter_example = False

    # For each disjunction in the output property, check none is satisfied by output_bounds.
    # If one disjunction is satisfied, then it represents a potential counter-example.
    for i in range(n_disjunctions):
        disj_res = check_disjunct_satisfied(output_bounds, prop.out_coef_mat[i], prop.out_bias_mat[i])
        if disj_res == PropertySatisfied.Yes:
            # We are 100% sure there is a counter-example.
            # It can be any point from the input space.
            # Return anything from the input bounds
            input_bounds = nn_bounds['numeric_pre'][nn.get_first_node().identifier]
            return True, list(input_bounds.get_lower())
        elif disj_res == PropertySatisfied.Maybe:
            # We are not 100% sure there is a counter-example.
            # Call an LP solver when we need a counter-example
            possible_counter_example = True
        else: # disj_res == PropertySatisfied.No
            # nothing to be done. Maybe other disjuncts will be satisfied
            pass

    # At least for one disjunct there is a possibility of a counter-example.
    # Do a more powerful check with an LP solver
    if possible_counter_example:
        return intersect_abstract_milp(star, nn, nn_bounds, prop)

    # Every disjunction is definitely not satisfied.
    # So we return False.
    return False, []


class PropertySatisfied(Enum):
    No = 0
    Yes = 1
    Maybe = 2


PRECISION_GUARD = 1e-06

def check_disjunct_satisfied(bounds, matrix, bias):
    """
    Checks if the bounds satisfy the conjunction of constraints given by

        matrix * x - bias <= 0

    Returns
    -------
    Yes if definitely satisfied
    No if definitely not satisfied
    Maybe when unsure
    """
    a_conjunct_possibly_not_satisfied = False

    # Check every conjunct in the disjunction
    for j in range(len(matrix)):
        max_value = bounds_utils.compute_max(matrix[j], bounds) - bias[j][0]
        min_value = bounds_utils.compute_min(matrix[j], bounds) - bias[j][0]

        if min_value > PRECISION_GUARD:
            # the constraint j is definitely not satisfied, as it should be <= 0
            return PropertySatisfied.No
        if max_value > PRECISION_GUARD:
            # the constraint j might not be satisfied, but we are not sure
            a_conjunct_possibly_not_satisfied = True

    if a_conjunct_possibly_not_satisfied:
        return PropertySatisfied.Maybe

    # if we reached here, means that all max values were below 0
    # so we now for sure that the property was satisfied
    # and there is a counter-example (any point from the input bounds)
    return PropertySatisfied.Yes


def check_valid_counterexample(candidate_cex: Tensor, nn: SequentialNetwork, prop: NeverProperty) -> bool:
    """
    This functions checks if a candidate counterexample is a true counterexample for the property

    """
    if candidate_cex is None:
        return False

    candidate_output = utilities.execute_network(nn, candidate_cex)
    n_disjunctions = len(prop.out_coef_mat)

    # For each disjunction in the output property, check at least one is satisfied
    for i in range(n_disjunctions):
        # Every condition
        satisfied = True
        for j in range(len(prop.out_coef_mat[i])):
            if prop.out_coef_mat[i][j].dot(candidate_output) - prop.out_bias_mat[i][j][0] > PRECISION_GUARD:
                # this conjunct is not satisfied, as it should be <= 0
                satisfied = False
                break

        if satisfied:
            return True

    return False


# def check_input_refining_one_equation_feasible_with_lp(coef, shift, input_bounds) -> bool:
#     """
#     Check if equation <= 0 is satisfiable with respect to the input bounds using an LP
#     """
#
#     n_input_dimensions = input_bounds.get_size()
#
#     solver = pywraplp.Solver("GLOP", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
#
#     input_vars = np.array([
#         solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
#         for j in range(n_input_dimensions)])
#
#     solver.Add(input_vars.dot(coef) + shift <= 0)
#
#     solver.Maximize(0)
#     status = solver.Solve()
#
#     if status == pywraplp.Solver.INFEASIBLE:
#         return False
#
#     elif status == pywraplp.Solver.OPTIMAL:
#         return True
#
#     raise Exception("Neither infeasible, not optimal")
#
#
# def compute_input_new_max(coef, shift, input_bounds, i) -> bool:
#     """
#     Given equation <= 0, compute the max value of the input dimension i
#     """
#
#     n_input_dimensions = input_bounds.get_size()
#
#     solver = pywraplp.Solver("GLOP", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
#
#     input_vars = np.array([
#         solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
#         for j in range(n_input_dimensions)])
#
#     solver.Add(input_vars.dot(coef) + shift <= 0)
#
#     solver.Maximize(input_vars[i])
#     status = solver.Solve()
#
#     if status == pywraplp.Solver.OPTIMAL:
#         return input_vars[i].solution_value()
#
#     raise Exception("Cannot compute new max")
#
#
# def compute_input_new_min(coef, shift, input_bounds, i) -> bool:
#     """
#     Given equation <= 0, compute the max value of the input dimension i
#     """
#
#     n_input_dimensions = input_bounds.get_size()
#
#     solver = pywraplp.Solver("GLOP", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
#
#     input_vars = np.array([
#         solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
#         for j in range(n_input_dimensions)])
#
#     solver.Add(input_vars.dot(coef) + shift <= 0)
#
#     solver.Minimize(input_vars[i])
#     status = solver.Solve()
#
#     if status == pywraplp.Solver.OPTIMAL:
#         return input_vars[i].solution_value()
#
#     raise Exception("Cannot compute new max")
