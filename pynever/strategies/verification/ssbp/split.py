import copy

from pynever import networks
from pynever.strategies.abstraction.star import ExtendedStar
from pynever.strategies.bounds_propagation.bounds import AbstractBounds
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

                # TODO: have the check as a method of Star? Or some other util?
                if star.ref_unstable_neurons is not None and len(star.ref_unstable_neurons) == 0:
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
        if star.ref_unstable_neurons is not None and len(star.ref_unstable_neurons) == 0:
            # the current layer is complete, so we need to move to the next layer
            # through the fully connected transformation
            star = propagation.propagate_and_init_star_before_relu_layer(star, nn_bounds, network)

            next_layers = sorted(
                list({layer_id for (layer_id, neuron_n) in nn_bounds['overapproximation_area']['map'].keys()
                      if layer_id >= star.ref_layer})
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


def split_star_opt(star: ExtendedStar, target: RefinementTarget, network: networks.SequentialNetwork, nn_bounds: dict) \
        -> list[tuple[ExtendedStar, dict]]:
    """
    Optimized split method

    target is known to be unstable wrt bounds.

    """

    # Update the bounds after the split
    mgr = BoundsManager()
    negative_bounds, positive_bounds = mgr.branch_update_bounds(nn_bounds, network, target, star.fixed_neurons)

    stars = compute_star_after_fixing_target_to_value(star, negative_bounds, target, NeuronSplit.Negative) + \
            compute_star_after_fixing_target_to_value(star, positive_bounds, target, NeuronSplit.Positive)

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

    if target.layer_id != star.ref_layer:
        # TODO: add the symbolic equation constraint to the predicate.
        # Could be useful when doing the intersection check with LP
        return [(star, bounds, None)]

    layer_inactive = (compute_layer_inactive_from_bounds_and_fixed_neurons(bounds, star.fixed_neurons, target.layer_id)
                      + ([target.neuron_idx] if split == NeuronSplit.Negative else []))

    new_transformation = star.mask_for_inactive_neurons(layer_inactive)

    fixed_so_far = star.fixed_neurons | {target.to_pair(): split.value}

    # Some of the neurons that were unstable at the beginning
    # could have become stable due to prior splitting.
    # So we intersect ref_unstable_neurons with the unstable neurons according to the bounds and fixed_so_far.
    ref_layer_unstable = (star.ref_unstable_neurons &
                          compute_layer_unstable_from_bounds_and_fixed_neurons(bounds, fixed_so_far, target.layer_id))

    if split == NeuronSplit.Negative:
        # Update the predicate to include the constraint that the target neuron y is inactive
        new_predicate = star.add_to_predicate_inactive_constraint(target.neuron_idx)
    else:
        # Update the predicate to include the constraint that the target neuron is active
        new_predicate = star.add_to_predicate_active_constraint(target.neuron_idx)

    star_after_split = ExtendedStar(new_predicate, new_transformation,
                                    ref_layer=target.layer_id, ref_neuron=target.neuron_idx,
                                    ref_unstable_neurons=ref_layer_unstable, fixed_neurons=fixed_so_far)

    return [(star_after_split, bounds, bounds['stable_count'])]
