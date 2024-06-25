import numpy as np
from ortools.linear_solver import pywraplp

from pynever import utilities
from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction.star import ExtendedStar
from pynever.strategies.bounds_propagation.bounds import HyperRectangleBounds
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

    unstable = split.get_unstable_fixed_neurons(star, nn_bounds)

    if len(unstable) == 0:
        return intersect_bounds(star, nn, nn_bounds, prop)

    return intersect_light_milp(star, nn, nn_bounds, prop)


def intersect_bounds(star: ExtendedStar, nn: SequentialNetwork, nn_bounds: dict, prop: NeverProperty) \
        -> tuple[bool, list[float]]:
    """
    This intersection method uses an LP and should be called only
    when there are no unstable neurons

    Returns
    ----------
    tuple[bool, list[float]]
        A tuple with the result (True or False) and the counterexample, if present

    """

    star = propagation.abs_propagation(star, nn, nn_bounds)

    input_bounds = nn_bounds['numeric_pre'][nn.get_id_from_index(0)]

    # Since all relu neurons have been fixed,
    # we assume that the basis is the linear function describing the relation
    # between the input and the output neurons.
    # So we recompute the output bounds given the equation by the bases
    output_bounds = HyperRectangleBounds(
        bounds_utils.compute_min(star.basis_matrix, input_bounds) + star.center.reshape(-1),
        bounds_utils.compute_max(star.basis_matrix, input_bounds) + star.center.reshape(-1))

    n_disjunctions = len(prop.out_coef_mat)

    # For each disjunction in the output property, check none is satisfied
    for i in range(n_disjunctions):
        # Every condition
        conjunction_intersects = True

        for j in range(len(prop.out_coef_mat[i])):
            max_value = bounds_utils.compute_max(prop.out_coef_mat[i][j], output_bounds) - prop.out_bias_mat[i][j][0]

            if max_value > 0:
                # this conjunct is not satisfied, as it should be <= 0
                conjunction_intersects = False
                break

        if conjunction_intersects:
            # only now use the method that calls an LP solver when we need a counter-example
            return intersect_abstract_milp(star, nn, nn_bounds, prop)

    return False, []


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
            input_vars.dot(star.predicate_matrix[j]) + star.predicate_bias[j][0] <= 0)

    # The constraints for the property
    _encode_output_property_constraints(solver, prop, output_bounds, output_vars)

    solver.Maximize(0)
    status = solver.Solve()

    if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
        return False, []

    else:
        # TODO check
        return True, [input_vars[i].solution_value() for i in range(n_input_dimensions)]


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

    input_bounds = nn_bounds['numeric_pre'][nn.get_id_from_index(0)]
    n_input_dimensions = input_bounds.get_size()
    output_bounds = nn_bounds['numeric_post'][nn.get_id_from_index(-1)]
    n_output_dimensions = output_bounds.get_size()

    solver = pywraplp.Solver("", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    input_vars = np.array([
        solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
        for j in range(n_input_dimensions)])

    output_vars = np.array([
        solver.NumVar(output_bounds.get_lower()[j], output_bounds.get_upper()[j], f'beta_{j}')
        for j in range(n_output_dimensions)])

    # The constraints from the branching
    for (layer_n, neuron_n), value in star.fixed_neurons.items():
        if value == 0:
            solver.Add(
                input_vars.dot(
                    nn_bounds['symbolic'][nn.get_id_from_index(layer_n - 1)].get_upper().get_matrix()[neuron_n]) +
                nn_bounds['symbolic'][nn.get_id_from_index(layer_n - 1)].get_upper().get_offset()[neuron_n] <= 0)
        else:
            solver.Add(
                input_vars.dot(
                    nn_bounds['symbolic'][nn.get_id_from_index(layer_n - 1)].get_lower().get_matrix()[neuron_n]) +
                nn_bounds['symbolic'][nn.get_id_from_index(layer_n - 1)].get_lower().get_offset()[neuron_n] >= 0)

    # The constraints relating input and output variables
    for j in range(n_output_dimensions):
        solver.Add(
            input_vars.dot(nn_bounds['symbolic'][nn.get_id_from_index(-1)].get_upper().get_matrix()[j]) +
            nn_bounds['symbolic'][nn.get_id_from_index(-1)].get_upper().get_offset()[j] - output_vars[j] >= 0)
        solver.Add(
            input_vars.dot(nn_bounds['symbolic'][nn.get_id_from_index(-1)].get_lower().get_matrix()[j]) +
            nn_bounds['symbolic'][nn.get_id_from_index(-1)].get_lower().get_offset()[j] - output_vars[j] <= 0)

    # The constraints for the property
    _encode_output_property_constraints(solver, prop, output_bounds, output_vars)

    solver.Maximize(0)
    status = solver.Solve()

    if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
        return False, []

    else:
        # TODO check
        return True, [input_vars[i].solution_value() for i in range(n_input_dimensions)]


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


def check_valid_counterexample(candidate_cex: Tensor, nn: SequentialNetwork, prop: NeverProperty) -> bool:
    """
    This functions checks if a candidate counterexample is a true counterexample for the property

    """

    candidate_output = utilities.execute_network(nn, candidate_cex)
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
