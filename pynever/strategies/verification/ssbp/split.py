from pynever import networks

from pynever.strategies.abstraction.star import ExtendedStar
from pynever.strategies.verification.parameters import RefinementTarget


def get_target_sequential(star: ExtendedStar, nn_bounds: dict, network: networks.SequentialNetwork) \
        -> tuple[RefinementTarget | None, ExtendedStar]:
    """
    This function selects the next refinement target in sequential order

    """

    unstable = nn_bounds['stability_info'][StabilityInfo.UNSTABLE]
    unstable = [neuron for neuron in unstable if neuron not in star.fixed_neurons]

    if len(unstable) > 0:
        for layer_n, neuron_n in unstable:
            if layer_n != star.ref_layer:

                # TODO: have the check as a method of Star? Or some other util
                if star.ref_unstable_neurons is not None and len(star.ref_unstable_neurons) == 0:
                    # the current layer is complete, so we need to move to the next layer
                    # through the fully connected transformation
                    #
                    star = propagate_and_init_star_before_relu_layer(star, nn_bounds, network)

            star.ref_neuron = neuron_n

            return RefinementTarget(layer_n, neuron_n), star

    return None, star


def get_target_lowest_overapprox_current_layer(star: Star, nn_bounds: dict, network) -> tuple[
    RefinementTarget | None, Star]:
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

            next_layers = sorted(
                list({layer_n for (layer_n, neuron_n) in nn_bounds['overapproximation_area']['map'].keys()
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
