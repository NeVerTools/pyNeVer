import numpy as np
import torch
from ortools.linear_solver import pywraplp

from pynever import networks
from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction.bounds_propagation import util
from pynever.strategies.verification.statistics import VerboseBounds
from pynever.strategies.abstraction.bounds_propagation.manager import BoundsManager
from pynever.strategies.abstraction.star import ExtendedStar
from pynever.strategies.verification.parameters import SSBPVerificationParameters
from pynever.strategies.verification.ssbp import propagation
from pynever.strategies.verification.ssbp.constants import RefinementTarget, NeuronSplit
from pynever.strategies.verification.ssbp.refinement import BoundsRefinement


def get_target_sequential(star: ExtendedStar, nn_bounds: VerboseBounds, network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """
    This function selects the next refinement target in sequential order
    """
    unstable = util.compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    if len(unstable) > 0:
        for layer_id, neuron_n in unstable:
            if layer_id != star.ref_layer:

                layer_unstable = util.compute_layer_unstable_from_bounds_and_fixed_neurons(nn_bounds,
                                                                                           star.fixed_neurons,
                                                                                           star.ref_layer)
                # TODO: have the check as a method of Star? Or some other util?
                if len(layer_unstable) == 0:
                    # the current layer is complete, so we need to move to the next layer
                    # through the fully connected transformation
                    #
                    star = propagation.propagate_and_init_star_before_relu_layer(star, nn_bounds, network)

            star.ref_neuron = neuron_n

            return RefinementTarget(layer_id, neuron_n), star

    return None, star


def get_target_lowest_overapprox_current_layer(star: ExtendedStar, nn_bounds: VerboseBounds,
                                               network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """
    """
    # Compute what we believe to be unstable neurons wrt the bounds and what we have fixed so far
    unstable = util.compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    # There are still unstable neurons
    if len(unstable) > 0:
        layer_unstable = util.compute_layer_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons,
                                                                                   star.ref_layer)
        if len(layer_unstable) == 0:
            # The current layer is complete, so we need to move to the next layer

            # Optimise the input bounds using an LP. Recompute all the bounds
            # nn_bounds = optimise_input_bounds_before_moving_to_next_layer(star, nn_bounds, network)
            # if nn_bounds is None:
            #     # Detected that the current branch is infeasible
            #     return None, star

            # Propagate through the fully connected transformation
            star = propagation.propagate_and_init_star_before_relu_layer(star, nn_bounds, network)

            if star.ref_layer == network.get_last_node().identifier:
                return None, star

            # TODO: sort using correct comparator that takes into account the NN structure
            next_layers = sorted(
                list({layer_id for (layer_id, neuron_n) in nn_bounds.statistics.overapprox_area['map'].keys()
                      if layer_id == star.ref_layer or network.layer_precedes(star.ref_layer, layer_id)}),

            )
            if len(next_layers) == 0:
                return None, star

            star.ref_layer = next_layers[0]

        else:
            # stay in the current layer
            pass

        # select candidates from star.ref_layer
        candidates_sorted = [((layer_id, neuron_n), area)
                             for (layer_id, neuron_n), area in nn_bounds.statistics.overapprox_area['sorted']
                             if layer_id == star.ref_layer]

        # select candidate that has not been fixed yet
        for (layer_id, neuron_n), _ in candidates_sorted:
            if not (layer_id, neuron_n) in star.fixed_neurons:
                star.ref_neuron = neuron_n
                return RefinementTarget(layer_id, neuron_n), star

    # No unstable neurons
    return None, star


def optimise_input_bounds_before_moving_to_next_layer(star: ExtendedStar, nn_bounds: VerboseBounds,
                                                      nn: networks.SequentialNetwork) -> VerboseBounds | None:
    """
    Optimises input bounds by building a MILP that has
    input variables and, for each fixed neuron, a constraint using its symbolic lower or upper bound.
    The solves for each input variable two optimisation problems: minimising and maximising it.
    """
    input_bounds = nn_bounds.numeric_pre_bounds[nn.get_first_node().identifier]
    n_input_dimensions = input_bounds.get_size()

    solver = pywraplp.Solver("", pywraplp.Solver.CLP_LINEAR_PROGRAMMING)

    input_vars = np.array([
        solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
        for j in range(n_input_dimensions)])

    # The constraints from fixing the neurons
    equations = BoundsRefinement.get_equations_from_fixed_neurons(star.fixed_neurons, nn_bounds, nn)
    worker_constraints = {}
    infinity = solver.infinity()
    for i in range(len(equations.matrix)):
        # solver.Add(input_vars.dot(equations.matrix[i]) + equations.offset[i] <= 0)
        # -infinity <= eq <= 0
        worker_constraints[i] = solver.Constraint(-infinity, -equations.offset[i].item(), 'c[%i]' % i)
        for j in range(n_input_dimensions):
            worker_constraints[i].SetCoefficient(input_vars[j], equations.matrix[i][j].item())

    new_input_bounds = input_bounds.clone()
    bounds_improved = False

    for i_dim in range(n_input_dimensions):
        solver.Maximize(input_vars[i_dim])
        status = solver.Solve()

        new_lower, new_upper = input_bounds.get_dimension_bounds(i_dim)
        if status == pywraplp.Solver.INFEASIBLE:
            return None

        elif status == pywraplp.Solver.OPTIMAL:
            if input_vars[i_dim].solution_value() < new_upper:
                new_upper = input_vars[i_dim].solution_value()
                bounds_improved = True

        solver.Minimize(input_vars[i_dim])
        status = solver.Solve()

        if status == pywraplp.Solver.INFEASIBLE:
            return None

        elif status == pywraplp.Solver.OPTIMAL:
            if input_vars[i_dim].solution_value() > new_lower:
                eq_mult = torch.Tensor([worker_constraints[i].dual_value() for i in worker_constraints])
                print("Dual solution", list(eq_mult))

                # the equation that optimises the bound found my LP
                coef = -(eq_mult.reshape(-1, 1) * equations.matrix).sum(dim=0)
                shift = -(eq_mult * equations.offset).sum()
                print("Selected equations", equations.matrix[(eq_mult != 0), :])
                print("Equation", list(coef), shift)

                new_bounds = BoundsRefinement.refine_input_dimension(input_bounds, coef, shift, i_dim)
                print("New bounds", new_bounds)
                print()

                new_lower = input_vars[i_dim].solution_value()
                bounds_improved = True

        new_input_bounds.get_lower()[i_dim] = new_lower
        new_input_bounds.get_upper()[i_dim] = new_upper

    if bounds_improved:
        new_bounds, _ = BoundsManager(nn, input_bounds=new_input_bounds).compute_bounds()
        return new_bounds

    return nn_bounds


def get_target_lowest_overapprox(star: ExtendedStar, nn_bounds: VerboseBounds) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """
    """
    # Compute what we believe to be unstable neurons wrt the bounds and what we have fixed so far
    unstable = util.compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    # There are still unstable neurons
    if len(unstable) > 0:

        for (layer_id, neuron_n), area in nn_bounds.statistics.overapprox_area['sorted']:
            if (layer_id, neuron_n) in unstable:
                return RefinementTarget(layer_id, neuron_n), star

    # No unstable neurons
    return None, star


def get_target_most_input_change(star: ExtendedStar, nn_bounds: VerboseBounds, network: networks.SequentialNetwork,
                                 params: SSBPVerificationParameters) -> tuple[RefinementTarget | None, ExtendedStar]:
    """
    """
    # Compute what we believe to be unstable neurons wrt the bounds and what we have fixed so far
    unstable = util.compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    # There are still unstable neurons
    if len(unstable) > 0:
        # initialise input_differences.
        if star.input_differences is None:
            star.input_differences = BoundsRefinement(params.bounds_direction).compute_refines_input_by(unstable,
                                                                                                        star.fixed_neurons,
                                                                                                        nn_bounds,
                                                                                                        network)

        candidates = [((layer_id, neuron_n), diff) for (layer_id, neuron_n), diff in star.input_differences
                      if (layer_id, neuron_n) in unstable]

        if len(candidates) > 0 and candidates[0][1] != 0:
            return RefinementTarget(candidates[0][0][0], candidates[0][0][1]), star

        # No candidates. Recompute input_differences again for all unstable neurons
        candidates = BoundsRefinement(params.bounds_direction).compute_refines_input_by(unstable,
                                                                                        star.fixed_neurons,
                                                                                        nn_bounds,
                                                                                        network)
        star.input_differences = candidates

        if len(candidates) > 0 and candidates[0][1] != 0:
            return RefinementTarget(candidates[0][0][0], candidates[0][0][1]), star

        # No candidates. Revert to lowest overapprox area heuristic.
        for (layer_id, neuron_n), area in nn_bounds.statistics.overapprox_area['sorted']:
            if (layer_id, neuron_n) in unstable:
                return RefinementTarget(layer_id, neuron_n), star

    # No unstable neurons
    return None, star


def split_star_opt(star: ExtendedStar, target: RefinementTarget, network: networks.SequentialNetwork,
                   nn_bounds: VerboseBounds, params: SSBPVerificationParameters) \
        -> list[tuple[ExtendedStar, VerboseBounds]]:
    """
    Optimized split method

    target is known to be unstable wrt bounds.
    """
    # Update the bounds after the split
    negative_bounds, positive_bounds = BoundsRefinement(params.bounds_direction).branch_update_bounds(nn_bounds,
                                                                                                      network, target,
                                                                                                      star.fixed_neurons)

    stars = compute_star_after_fixing_target_to_value(star, negative_bounds, target, NeuronSplit.NEGATIVE, nn_bounds,
                                                      network, params) + \
            compute_star_after_fixing_target_to_value(star, positive_bounds, target, NeuronSplit.POSITIVE, nn_bounds,
                                                      network, params)

    stars = sorted(stars, key=lambda x: x[1].statistics.stability_info['stable_count'])

    return stars


def compute_star_after_fixing_target_to_value(star: ExtendedStar, bounds: VerboseBounds, target: RefinementTarget,
                                              split: NeuronSplit, pre_split_bounds: VerboseBounds,
                                              network: SequentialNetwork, params: SSBPVerificationParameters) \
        -> list[tuple[ExtendedStar, VerboseBounds]]:
    """
    This function creates the star after fixing target according to the split
    with the new constraints and updated bounds
    """
    if bounds is None:
        return []

    fixed_so_far = star.fixed_neurons | {target.to_pair(): split.value}

    # if True or target.layer_id != star.ref_layer:
    # Only update fixed_neurons, as we cannot update the basis.
    # In principle, we could update the predicate, but we do not need it
    # as we have the information about the split in fixed_neurons,
    # so later when doing the intersection check we can recover the required constraints.
    star_after_split = ExtendedStar(star.get_predicate_equation(), star.get_transformation_equation(),
                                    ref_layer=star.ref_layer, fixed_neurons=fixed_so_far,
                                    enforced_constraints=star.enforced_constraints,
                                    input_differences=star.input_differences)

    if bounds.statistics.stability_info['stable_count'] - pre_split_bounds.statistics.stability_info['stable_count'] <= 2:
        negative_bounds, positive_bounds = BoundsRefinement(params.bounds_direction).branch_bisect_input(bounds,
                                                                                                         network,
                                                                                                         fixed_so_far)

        return compute_star_after_input_split(star_after_split, negative_bounds) + \
            compute_star_after_input_split(star_after_split, positive_bounds)

    return [(star_after_split, bounds)]


def compute_star_after_input_split(star: ExtendedStar, bounds: VerboseBounds) \
        -> list[tuple[ExtendedStar, VerboseBounds]]:
    """
    This function creates the star after splitting an input dimension
    """
    if bounds is None:
        return []

    star_after_split = ExtendedStar(star.get_predicate_equation(), star.get_transformation_equation(),
                                    ref_layer=star.ref_layer, fixed_neurons=star.fixed_neurons,
                                    enforced_constraints=star.enforced_constraints,
                                    input_differences=star.input_differences)

    return [(star_after_split, bounds)]
