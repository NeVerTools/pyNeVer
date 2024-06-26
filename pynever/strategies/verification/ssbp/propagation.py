from pynever import networks, nodes

from pynever.strategies.abstraction.star import ExtendedStar
from pynever.strategies.bounds_propagation.bounds_manager import StabilityInfo, \
    compute_layer_unstable_from_bounds_and_fixed_neurons, compute_layer_inactive_from_bounds_and_fixed_neurons


def abs_propagation(star: ExtendedStar, network: networks.SequentialNetwork, bounds: dict) -> ExtendedStar:
    """
    This method performs the abstract propagation of a single star starting
    from a specific layer and neuron. The output is a single star that uses
    approximation in the next layers

    Parameters
    ----------
    star : ExtendedStar
        The star to process
    network : networks.SequentialNetwork
        The neural network to propagate through
    bounds : dict
        The bounds of the network layers

    Returns
    ----------
    ExtendedStar
        The resulting star approximate with the abstract propagation

    """

    start_layer = star.ref_layer

    if start_layer is None:
        return star

    skip = True

    for layer in network.layers_iterator():
        # skip until we reach the reference layer
        if skip:
            if layer.identifier == star.ref_layer:
                skip = False
            else:
                continue

        # Propagate fully connected entirely
        if isinstance(layer, nodes.FullyConnectedNode):
            # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
            bias = layer.get_layer_bias_as_two_dimensional()
            star = star.single_fc_forward(layer.weight, bias)

        # Propagate ReLU starting from target
        elif isinstance(layer, nodes.ReLUNode):
            star = star.approx_relu_forward(bounds, layer.identifier)

        elif isinstance(layer, nodes.FlattenNode):
            # Do nothing
            continue

        else:
            raise NotImplementedError('Unsupported layer')

    return star


def propagate_and_init_star_before_relu_layer(star: ExtendedStar, bounds: dict, network: networks.SequentialNetwork,
                                              skip: bool = True) -> ExtendedStar:
    """
    Compute the initial star which will always start from the first layer and
    where we will use the bounds to determine the inactive nodes,
    so that we could set the transformation for them to 0.

    """

    new_star, relu_layer = propagate_until_relu(star, network, skip=skip)
    relu_layer_id = new_star.ref_layer

    if relu_layer is not None:
        layer_inactive = compute_layer_inactive_from_bounds_and_fixed_neurons(bounds, new_star.fixed_neurons, relu_layer_id)
        layer_unstable = compute_layer_unstable_from_bounds_and_fixed_neurons(bounds, new_star.fixed_neurons, relu_layer_id)

        new_transformation = new_star.mask_for_inactive_neurons(layer_inactive)

        return ExtendedStar(new_star.get_predicate_equation(), new_transformation, ref_layer=relu_layer_id,
                            ref_unstable_neurons=layer_unstable, fixed_neurons=new_star.fixed_neurons)

    return new_star


def propagate_until_relu(star: ExtendedStar, network: networks.SequentialNetwork, skip: bool) \
        -> tuple[ExtendedStar, nodes.ReLUNode | None]:
    """
    This function performs the star propagation throughout Fully Connected layers
    only, until a ReLU layer is encountered. This is used in order to process
    Fully Connected layers only once per cycle

    Parameters
    ----------
    star : ExtendedStar
        The star to process
    network : networks.SequentialNetwork
        The neural network
    skip : bool
        Flag to signal end of propagation

    Returns
    ----------
    tuple[ExtendedStar, nodes.ReLUNode]
        The resulting star before the next ReLU layer and the ReLU layer

    """

    relu_layer = None
    for layer in network.layers_iterator():
        if skip:
            if layer.identifier == star.ref_layer:
                skip = False

        else:
            # Propagate fully connected entirely
            if isinstance(layer, nodes.FullyConnectedNode):
                # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
                bias = layer.get_layer_bias_as_two_dimensional()
                star = star.single_fc_forward(layer.weight, bias)

            elif isinstance(layer, nodes.ReLUNode):
                relu_layer = layer
                break

            elif isinstance(layer, nodes.FlattenNode):
                # Do nothing
                pass
            else:
                raise NotImplementedError(f'Currently supporting only FullyConnected and ReLU nodes. '
                                          f'Unsupported layer {layer.__class__}')

    # Set reference layer
    star.ref_layer = relu_layer.identifier

    return star, relu_layer
