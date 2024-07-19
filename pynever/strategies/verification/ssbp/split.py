import copy

from pynever import networks
from pynever.strategies.abstraction.star import ExtendedStar
from pynever.strategies.bounds_propagation.bounds import AbstractBounds, HyperRectangleBounds
from pynever.strategies.bounds_propagation.bounds_manager import BoundsManager, StabilityInfo, NeuronSplit, \
    compute_layer_unstable_from_bounds_and_fixed_neurons, compute_unstable_from_bounds_and_fixed_neurons, \
    compute_layer_inactive_from_bounds_and_fixed_neurons
from pynever.strategies.verification.ssbp import propagation
from pynever.strategies.verification.ssbp.constants import RefinementTarget




def get_target_sequential(star: ExtendedStar, nn_bounds: dict, network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """
    This function selects the next refinement target in sequential order

    """

    unstable = compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    if len(unstable) > 0:
        for layer_id, neuron_n in unstable:
            if layer_id != star.ref_layer:

                layer_unstable = compute_layer_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons,
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


def get_target_lowest_overapprox_current_layer(star: ExtendedStar, nn_bounds: dict,
                                               network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """

    """

    # Compute what we believe to be unstable neurons wrt the bounds and what we have fixed so far
    unstable = compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    # There are still unstable neurons
    if len(unstable) > 0:
        layer_unstable = compute_layer_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons, star.ref_layer)
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
                list({layer_id for (layer_id, neuron_n) in nn_bounds['overapproximation_area']['map'].keys()
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
                             for (layer_id, neuron_n), area in nn_bounds['overapproximation_area']['sorted']
                             if layer_id == star.ref_layer]

        # select candidate that has not been fixed yet
        for (layer_id, neuron_n), _ in candidates_sorted:
            if not (layer_id, neuron_n) in star.fixed_neurons:
                star.ref_neuron = neuron_n
                return RefinementTarget(layer_id, neuron_n), star

    # No unstable neurons
    return None, star


def optimise_input_bounds_before_moving_to_next_layer(star: ExtendedStar, nn_bounds: dict, nn: networks.SequentialNetwork) \
        -> dict:
    """
    Optimises input bounds by building a MILP that has
    input variables and, for each fixed neuron, a constraint using its symbolic lower or upper bound.
    The solves for each input variable two optimisation problems: minimising and maximising it.
    """

    input_bounds = nn_bounds['numeric_pre'][nn.get_first_node().identifier]
    n_input_dimensions = input_bounds.get_size()

    from ortools.linear_solver import pywraplp
    solver = pywraplp.Solver("", pywraplp.Solver.CLP_LINEAR_PROGRAMMING)

    import numpy as np
    input_vars = np.array([
        solver.NumVar(input_bounds.get_lower()[j], input_bounds.get_upper()[j], f'alpha_{j}')
        for j in range(n_input_dimensions)])

    # The constraints from fixing the neurons
    equations = BoundsManager._get_equations_from_fixed_neurons(star.fixed_neurons, nn_bounds, nn)
    worker_constraints = {}
    infinity = solver.infinity()
    for i in range(len(equations.matrix)):
        # solver.Add(input_vars.dot(equations.matrix[i]) + equations.offset[i] <= 0)
        worker_constraints[i] = solver.Constraint(-infinity, -equations.offset[i], 'c[%i]' % i) # -infinity <= eq <= 0
        for j in range(n_input_dimensions):
            worker_constraints[i].SetCoefficient(input_vars[j], equations.matrix[i][j])

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
                eq_mult = np.array([worker_constraints[i].dual_value() for i in worker_constraints])
                print("Dual solution", list(eq_mult))

                # the equation that optimises the bound found my LP
                coef = -(eq_mult.reshape(-1, 1) * equations.matrix).sum(axis=0)
                shift = -(eq_mult * equations.offset).sum()
                print("Selected equations", equations.matrix[(eq_mult!=0),:])
                print("Equation", list(coef), shift)

                new_bounds = BoundsManager._refine_input_dimension(input_bounds, coef, shift, i_dim)
                print("New bounds", new_bounds)
                print()


                new_lower = input_vars[i_dim].solution_value()
                bounds_improved = True

        new_input_bounds.get_lower()[i_dim] = new_lower
        new_input_bounds.get_upper()[i_dim] = new_upper

    if bounds_improved:
        return BoundsManager().compute_bounds(new_input_bounds, nn, star.fixed_neurons)

    return nn_bounds


def get_target_lowest_overapprox(star: ExtendedStar, nn_bounds: dict,
                                 network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """

    """
    # Compute what we believe to be unstable neurons wrt the bounds and what we have fixed so far
    unstable = compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    # There are still unstable neurons
    if len(unstable) > 0:

        for (layer_id, neuron_n), area in nn_bounds['overapproximation_area']['sorted']:
            if (layer_id, neuron_n) in unstable:
                return RefinementTarget(layer_id, neuron_n), star

    # No unstable neurons
    return None, star


def get_target_most_input_change(star: ExtendedStar, nn_bounds: dict,
                                 network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """

    """
    # Compute what we believe to be unstable neurons wrt the bounds and what we have fixed so far
    unstable = compute_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons)

    # There are still unstable neurons
    if len(unstable) > 0:
        # initialise input_differences.
        if star.input_differences is None:
            star.input_differences = BoundsManager.compute_refines_input_by(unstable, star.fixed_neurons, nn_bounds, network)

        candidates = [((layer_id, neuron_n), diff) for (layer_id, neuron_n), diff in star.input_differences
                      if (layer_id, neuron_n) in unstable]

        if len(candidates) > 0 and candidates[0][1] != 0:
            return RefinementTarget(candidates[0][0][0], candidates[0][0][1]), star

        # No candidates. Recompute input_differences again for all unstable neurons
        candidates = BoundsManager.compute_refines_input_by(unstable, star.fixed_neurons, nn_bounds, network)
        star.input_differences = candidates

        if len(candidates) > 0 and candidates[0][1] != 0:
            return RefinementTarget(candidates[0][0][0], candidates[0][0][1]), star

        # No candidates. Revert to lowest overapprox area heuristic.
        for (layer_id, neuron_n), area in nn_bounds['overapproximation_area']['sorted']:
            if (layer_id, neuron_n) in unstable:
                return RefinementTarget(layer_id, neuron_n), star

    # No unstable neurons
    return None, star


def split_star_opt(star: ExtendedStar, target: RefinementTarget, network: networks.SequentialNetwork, nn_bounds: dict) \
        -> list[tuple[ExtendedStar, dict]]:
    """
    Optimized split method

    target is known to be unstable wrt bounds.

    """

    # Update the bounds after the split
    negative_bounds, positive_bounds = BoundsManager().branch_update_bounds(nn_bounds, network, target, star.fixed_neurons)

    stars = compute_star_after_fixing_target_to_value(star, negative_bounds, target, NeuronSplit.Negative, nn_bounds, network) + \
            compute_star_after_fixing_target_to_value(star, positive_bounds, target, NeuronSplit.Positive, nn_bounds, network)

    stars = sorted(stars, key=lambda x: x[2])
    stars = [(s, bounds) for (s, bounds, _) in stars]

    return stars


def compute_star_after_fixing_target_to_value(star: ExtendedStar, bounds: dict, target: RefinementTarget,
                                              split: NeuronSplit, pre_split_bounds, network) \
        -> list[tuple[ExtendedStar, dict, AbstractBounds | None]]:
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
                                    ref_layer=star.ref_layer, ref_neuron=star.ref_neuron,
                                    fixed_neurons=fixed_so_far, enforced_constraints=star.enforced_constraints,
                                    input_differences=star.input_differences)

    if bounds['stable_count'] - pre_split_bounds['stable_count'] <= 2:
        negative_bounds, positive_bounds = BoundsManager().branch_bisect_input(bounds, network, fixed_so_far)

        return compute_star_after_input_split(star_after_split, negative_bounds) + \
               compute_star_after_input_split(star_after_split, positive_bounds)

    return [(star_after_split, bounds, bounds['stable_count'])]

    # # Compute new transformation
    # layer_inactive = compute_layer_inactive_from_bounds_and_fixed_neurons(bounds, fixed_so_far, target.layer_id)
    # new_transformation = star.mask_for_inactive_neurons(layer_inactive)
    #
    # # Compute new predicate
    # if split == NeuronSplit.Negative:
    #     # Update the predicate to include the constraint that the target neuron y is inactive
    #     new_predicate = star.add_to_predicate_inactive_constraint(target.neuron_idx)
    # else:
    #     # Update the predicate to include the constraint that the target neuron is active
    #     new_predicate = star.add_to_predicate_active_constraint(target.neuron_idx)
    #
    # # The constraints from splitting that have been encoded in the predicate.
    # # If we split the neuron not in the ref_layer, we do not encode them,
    # # so we need to add them during abs_propagation.
    # enforced_so_far = star.enforced_constraints | {target.to_pair(): split.value}
    #
    # star_after_split = ExtendedStar(new_predicate, new_transformation,
    #                                 ref_layer=target.layer_id, ref_neuron=target.neuron_idx,
    #                                 fixed_neurons=fixed_so_far, enforced_constraints=enforced_so_far,
    #                                 input_differences=star.input_differences)
    #
    # return [(star_after_split, bounds, bounds['stable_count'])]


def compute_star_after_input_split(star: ExtendedStar, bounds: dict) \
        -> list[tuple[ExtendedStar, dict, AbstractBounds | None]]:
    """
    This function creates the star after splitting an input dimension
    """

    if bounds is None:
        return []

    star_after_split = ExtendedStar(star.get_predicate_equation(), star.get_transformation_equation(),
                                    ref_layer=star.ref_layer, ref_neuron=star.ref_neuron,
                                    fixed_neurons=star.fixed_neurons, enforced_constraints=star.enforced_constraints,
                                    input_differences=star.input_differences)
    return [(star_after_split, bounds, bounds['stable_count'])]

