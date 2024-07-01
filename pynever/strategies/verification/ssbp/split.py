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
            nn_bounds = optimise_input_bounds_before_moving_to_next_layer(star, nn_bounds, network)
            if nn_bounds is None:
                # Detected that the current branch is infeasible
                return None, star

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
                # coefs = equations.matrix
                # shifts = equations.offset
                #
                # with_i_neg_coefs = coefs[(coefs[:, i_dim] < 0)]
                # with_i_neg_shifts = shifts[(coefs[:, i_dim] < 0)]
                #
                from pynever.strategies.bounds_propagation.utils.utils import compute_min
                # lowers = []
                # for mask in [[True, True, True], [True, True, False], [True, False, True], [False, True, True], [True, False, False], [False, True, False], [False, False, True]]:
                #
                #     neg_coef = with_i_neg_coefs[mask].sum(axis=0)
                #     neg_shift = with_i_neg_shifts[mask].sum()
                #
                #     c = neg_coef[i_dim]
                #     negated_rem_coef = - np.array(list(neg_coef[:i_dim]) + list(neg_coef[i_dim + 1:])) / c
                #     shift_div_c = - neg_shift / c
                #
                #     rem_lower_input_bounds = np.array(
                #         list(input_bounds.get_lower()[:i_dim]) + list(input_bounds.get_lower()[i_dim + 1:]))
                #     rem_upper_input_bounds = np.array(
                #         list(input_bounds.get_upper()[:i_dim]) + list(input_bounds.get_upper()[i_dim + 1:]))
                #
                #     new_lower_i = compute_min(negated_rem_coef, HyperRectangleBounds(rem_lower_input_bounds, rem_upper_input_bounds)) + shift_div_c
                #     lowers.append(new_lower_i)

                for i in worker_constraints:
                    print(worker_constraints[i].dual_value())
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
        # layer_unstable = compute_layer_unstable_from_bounds_and_fixed_neurons(nn_bounds, star.fixed_neurons, star.ref_layer)
        # if len(layer_unstable) == 0:
        #     # the current layer is complete, so we need to move to the next layer
        #     # through the fully connected transformation
        #     star = propagation.propagate_and_init_star_before_relu_layer(star, nn_bounds, network)
        #
        #     if star.ref_layer == network.get_last_node().identifier:
        #         return None, star

        # select candidate that has not been fixed yet
        # candidates = [((layer_id, neuron_n), area)
        #               for ((layer_id, neuron_n), area) in nn_bounds['overapproximation_area']['sorted']
        #               if (layer_id, neuron_n) not in star.fixed_neurons]
        # if len(candidates) > 0:
        #     (layer_id, neuron_n), _ = candidates[0]
        #     return RefinementTarget(layer_id, neuron_n), star
        for (layer_id, neuron_n), _ in nn_bounds['overapproximation_area']['sorted']:
            if not (layer_id, neuron_n) in star.fixed_neurons:
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
    mgr = BoundsManager()
    negative_bounds, positive_bounds = mgr.branch_update_bounds(nn_bounds, network, target, star.fixed_neurons)

    stars = (compute_star_after_fixing_target_to_value(star, negative_bounds, target, NeuronSplit.Negative) +
             compute_star_after_fixing_target_to_value(star, positive_bounds, target, NeuronSplit.Positive))

    stars = [(s, bounds) for (s, bounds, _) in stars]

    return stars


def compute_star_after_fixing_target_to_value(star: ExtendedStar, bounds: dict, target: RefinementTarget, split: NeuronSplit) \
        -> list[tuple[ExtendedStar, dict, AbstractBounds | None]]:
    """
    This function creates the star after fixing target according to the split
    with the new constraints and updated bounds
    """

    if bounds is None:
        return []

    fixed_so_far = star.fixed_neurons | {target.to_pair(): split.value}

    if target.layer_id != star.ref_layer:
        # Only update fixed_neurons, as we cannot update the basis.
        # In principle, we could update the predicate, but we do not need it
        # as we have the information about the split in fixed_neurons,
        # so later when doing the intersection check we can recover the required constraints.
        star_after_split = ExtendedStar(star.get_predicate_equation(), star.get_transformation_equation(),
                                        ref_layer=star.ref_layer, ref_neuron=star.ref_neuron,
                                        fixed_neurons=fixed_so_far, enforced_constraints=star.enforced_constraints)
        return [(star_after_split, bounds, bounds['stable_count'])]

    # Compute new transformation
    layer_inactive = (compute_layer_inactive_from_bounds_and_fixed_neurons(bounds, star.fixed_neurons, target.layer_id)
                      + ([target.neuron_idx] if split == NeuronSplit.Negative else []))

    new_transformation = star.mask_for_inactive_neurons(layer_inactive)

    # Compute new predicate
    if split == NeuronSplit.Negative:
        # Update the predicate to include the constraint that the target neuron y is inactive
        new_predicate = star.add_to_predicate_inactive_constraint(target.neuron_idx)
    else:
        # Update the predicate to include the constraint that the target neuron is active
        new_predicate = star.add_to_predicate_active_constraint(target.neuron_idx)

    # The constraints from splitting that have been encoded in the predicate.
    # If we split the neuron not in the ref_layer, we do not encode them,
    # so we need to add them during abs_propagation.
    enforced_so_far = star.enforced_constraints | {target.to_pair(): split.value}

    star_after_split = ExtendedStar(new_predicate, new_transformation,
                                    ref_layer=target.layer_id, ref_neuron=target.neuron_idx,
                                    fixed_neurons=fixed_so_far, enforced_constraints=enforced_so_far)

    return [(star_after_split, bounds, bounds['stable_count'])]
