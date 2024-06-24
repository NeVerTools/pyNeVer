from pynever.strategies.abstraction.star import ExtendedStar


def abs_propagation(star: ExtendedStar, bounds: dict, nn_list: list) -> ExtendedStar:
    """
    This method performs the abstract propagation of a single star starting
    from a specific layer and neuron. The output is a single star that uses
    approximation in the next layers

    Parameters
    ----------
    star : ExtendedStar
        The star to process
    bounds : dict
        The bounds of the network layers
    nn_list : list
        The neural network represented as a list

    Returns
    ----------
    ExtendedStar
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
                          [i for (lay_n, i), value in new_star.fixed_neurons.items() if
                           lay_n == relu_layer_n and value == 0])
        layer_unstable = {neuron_n for layer_n, neuron_n in bounds['stability_info'][StabilityInfo.UNSTABLE]
                          if layer_n == relu_layer_n and not (layer_n, neuron_n) in new_star.fixed_neurons}

        new_basis_matrix, new_center = mask_transformation_for_inactive_neurons(
            layer_inactive, new_star.basis_matrix, new_star.center)

        return abst.ExtendedStar(new_star.predicate_matrix, new_star.predicate_bias, new_center, new_basis_matrix,
                                 ref_layer=relu_layer_n, ref_unstable_neurons=layer_unstable,
                                 fixed_neurons=new_star.fixed_neurons)

    return new_star


def propagate_until_relu(star: ExtendedStar, network: networks.SequentialNetwork, skip: bool) -> ExtendedStar:
    """
    This function performs the star propagation throughout Fully Connected layers
    only, until a ReLU layer is encountered. This is used in order to process
    Fully Connected layers only once per cycle

    Parameters
    ----------
    star : ExtendedStar
        The star to process
    nn_list : list
        The neural network represented as a list
    skip : bool
        Flag to signal end of propagation

    Returns
    ----------
    ExtendedStar
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
