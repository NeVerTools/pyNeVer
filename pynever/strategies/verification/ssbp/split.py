import copy

from pynever import networks
from pynever.strategies.abstraction.star import ExtendedStar
from pynever.strategies.bounds_propagation.bounds import AbstractBounds
from pynever.strategies.bounds_propagation.bounds_manager import BoundsManager, StabilityInfo, NeuronSplit
from pynever.strategies.verification.ssbp import propagation
from pynever.strategies.verification.ssbp.constants import RefinementTarget


def get_unstable_fixed_neurons(star: ExtendedStar, bounds: dict) -> list:
    """
    Utility method

    """

    unstable = bounds['stability_info'][StabilityInfo.UNSTABLE]
    return [neuron for neuron in unstable if neuron not in star.fixed_neurons]


def get_target_sequential(star: ExtendedStar, nn_bounds: dict, network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """
    This function selects the next refinement target in sequential order

    """

    unstable = get_unstable_fixed_neurons(star, nn_bounds)

    if len(unstable) > 0:
        for layer_n, neuron_n in unstable:
            if layer_n != star.ref_layer:

                # TODO: have the check as a method of Star? Or some other util?
                if star.ref_unstable_neurons is not None and len(star.ref_unstable_neurons) == 0:
                    # the current layer is complete, so we need to move to the next layer
                    # through the fully connected transformation
                    #
                    star = propagation.propagate_and_init_star_before_relu_layer(star, nn_bounds, network)

            star.ref_neuron = neuron_n

            return RefinementTarget(layer_n, neuron_n), star

    return None, star


def get_target_lowest_overapprox_current_layer(star: ExtendedStar, nn_bounds: dict,
                                               network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """

    """

    # Compute what we believe to be unstable neurons wrt the bounds and what we have fixed so far
    unstable = get_unstable_fixed_neurons(star, nn_bounds)

    unstable_lowest_layer = sorted(list({layer_n for (layer_n, _) in unstable}))
    if len(unstable_lowest_layer) != 0:
        unstable_lowest_layer = unstable_lowest_layer[0]

    # There are still unstable neurons
    if len(unstable) > 0:
        if star.ref_unstable_neurons is not None and len(star.ref_unstable_neurons) == 0:
            # the current layer is complete, so we need to move to the next layer
            # through the fully connected transformation
            star = propagation.propagate_and_init_star_before_relu_layer(star, nn_bounds, network)

            next_layers = sorted(
                list({layer_n for (layer_n, neuron_n) in nn_bounds['overapproximation_area']['map'].keys()
                      if layer_n >= star.ref_layer})
            )
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


def split_star_opt(star: ExtendedStar, target: RefinementTarget, network: networks.SequentialNetwork, nn_bounds: dict) \
        -> list[tuple[ExtendedStar, dict]]:
    """
    Optimized split method

    target is known to be unstable wrt bounds.

    """

    # Update the bounds after the split
    mgr = BoundsManager()
    negative_bounds, positive_bounds = mgr.branch_update_bounds(nn_bounds, network, target, star.fixed_neurons)

    stars = compute_star_after_fixing_to_value(star, negative_bounds, target, NeuronSplit.Negative, network) + \
            compute_star_after_fixing_to_value(star, positive_bounds, target, NeuronSplit.Positive, network)

    stars = [(s, bounds) for (s, bounds, _) in stars]

    return stars


def compute_star_after_fixing_to_value(star: ExtendedStar, bounds: dict, target: RefinementTarget, split: NeuronSplit,
                                       network: networks.SequentialNetwork) \
        -> list[tuple[ExtendedStar, dict, AbstractBounds | None]]:
    """
    This function creates the star after fixing target to the value
    with the new constraints and updated bounds
    """

    if bounds is None:
        return []

    if target.layer_idx != star.ref_layer:
        # TODO: add the symbolic equation constraint to the predicate.
        # Could be useful when doing the intersection check with LP
        return [(star, bounds, None)]

    index = target.neuron_idx
    layer_inactive = (bounds['stability_info'][StabilityInfo.INACTIVE][network.get_id_from_index(target.layer_idx)] +
                      ([index] if split == NeuronSplit.Negative else []))

    new_transformation = star.mask_for_inactive_neurons(layer_inactive)

    fixed_so_far = copy.deepcopy(star.fixed_neurons)
    fixed_so_far[target.to_pair()] = split.value

    # Some of the neurons that were unstable at the beginning
    # could have become stable due to prior splitting.
    # So we intersect ref_unstable_neurons with the unstable neurons according to the bounds.
    layer_unstable_per_bounds = {neuron_n for layer_n, neuron_n in bounds['stability_info'][StabilityInfo.UNSTABLE]
                                 if layer_n == target.layer_idx}
    ref_layer_unstable = star.ref_unstable_neurons & layer_unstable_per_bounds

    # We have just fixed the target neuron, so we remove it from the set of unstable neurons.
    ref_layer_unstable.discard(index)

    if split == NeuronSplit.Negative:
        # Update the predicate to include the constraint that the target neuron y is inactive
        new_predicate = star.add_to_predicate_inactive_constraint(index)
    else:
        # Update the predicate to include the constraint that the target neuron is active
        new_predicate = star.add_to_predicate_active_constraint(index)

    star_after_split = ExtendedStar(new_predicate, new_transformation,
                                    ref_layer=target.layer_idx, ref_neuron=target.neuron_idx,
                                    ref_unstable_neurons=ref_layer_unstable, fixed_neurons=fixed_so_far)

    return [(star_after_split, bounds, bounds['stable_count'])]
