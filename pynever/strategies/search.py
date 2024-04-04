import numpy as np

import pynever.strategies.abstraction as abst
from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction import Star
from pynever.strategies.bp.bounds_manager import BoundsManager


def get_bounds(nn: SequentialNetwork, prop: 'NeVerProperty', strategy: str) -> dict:
    """
    This function gets the bounds of the neural network for the given property
    of interest. The bounds are computed based on a strategy that allows to
    plug and play different bound propagation algorithms

    Parameters
    ----------
    nn : SequentialNetwork
        The neural network of interest in the internal representation
    prop : NeVerProperty
        The property of interest
    strategy : str
        The strategy to use for computing the bounds [symbolic, lirpa, ...]

    Returns
    ----------
    dict
        The dictionary of the bounds wrapped in an AbstractBounds object for each layer

    """

    if strategy == 'symbolic':
        # Return the pre-activation bounds for ReLU layers
        return BoundsManager(nn, prop).compute_bounds()  # [1]
    elif strategy == 'lirpa':
        # return something...
        pass
    # TODO add more strategies


def abs_propagation(star: Star, bounds: dict, target: tuple, nn_list: list) -> Star:
    """
    This method performs the abstract propagation of a single star starting
    from a specific layer and neuron. The output is a single star that uses
    approximation in the next layers

    Parameters
    ----------
    star : Star
        The star to process
    bounds : dict
        The bounds of the network layers
    target : tuple
        The target layer and neuron to start from wrapped in a tuple
    nn_list : list
        The neural network represented as a list

    Returns
    ----------
    Star
        The resulting star approximate with the abstract propagation

    """

    start_layer = star.ref_layer

    for layer in nn_list[start_layer:]:
        i = nn_list.index(layer)

        # Propagate fully connected entirely
        if isinstance(layer, nodes.FullyConnectedNode):
            # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
            if layer.bias.shape != (layer.weight.shape[0], 1):
                bias = np.expand_dims(layer.bias, 1)
            else:
                bias = layer.bias
            star = abst.single_fc_forward(star, layer.weight, bias).pop()

        # Propagate ReLU starting from target
        elif isinstance(layer, nodes.ReLUNode):
            l_bounds = bounds[layer.identifier]
            if i == target[0]:
                star = abst.approx_relu_forward(star, l_bounds, layer.in_dim[0], start_idx=target[1])
            else:
                star = abst.approx_relu_forward(star, l_bounds, layer.in_dim[0])

        else:
            raise NotImplementedError('Unsupported layer')

    return star


def propagate_until_relu(star: Star, nn_list: list) -> Star:
    """
    This function performs the star propagation throughout Fully Connected layers
    only, until a ReLU layer is encountered. This is used in order to process
    Fully Connected layers only once per cycle

    Parameters
    ----------
    star : Star
        The star to process
    nn_list : list
        The neural network represented as a list

    Returns
    ----------
    Star
        The resulting star before the next ReLU layer

    """

    start_layer = star.ref_layer
    i = 0
    skip = True

    for layer in nn_list[start_layer:]:

        # Propagate fully connected entirely
        if isinstance(layer, nodes.FullyConnectedNode):
            # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
            if layer.bias.shape != (layer.weight.shape[0], 1):
                bias = np.expand_dims(layer.bias, 1)
            else:
                bias = layer.bias
            star = abst.single_fc_forward(star, layer.weight, bias).pop()
            i += 1

        elif isinstance(layer, nodes.ReLUNode):
            # If all the neurons have been processed...
            if star.predicate_matrix.shape[0] - star.initial_pred == star.center.shape[0] and skip:
                skip = False
                i += 1
                continue
            # Otherwise, stay on this layer and interrupt cycle
            else:
                break

        else:
            raise NotImplementedError('Unsupported layer')

    # Set reference layer
    star.ref_layer = start_layer + i

    return star


def check_intersection(star: Star, prop: 'NeVerProperty') -> (bool, list):
    """
    This function checks whether a star intersects with the output property
    using a linear program. Since the output property may contain disjunction
    the intersection is computed with all the halfspaces in the output property.
    If an intersection occurs, the resulting star is saved and returned for
    extracting a counterexample

    Parameters
    ----------
    star : Star
        The star to intersect with the output property
    prop : NeVerProperty
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
        intersection = abst.intersect_with_halfspace(star,
                                                     prop.out_coef_mat[i],
                                                     prop.out_bias_mat[i])
        if not intersection.check_if_empty():
            intersects = True
            unsafe_stars.append(intersection)

    return intersects, unsafe_stars


def intersect_star_lp(current_star, net_list, nn_bounds, prop, target):
    # Compute the output abstract star from current_star/bounds
    out_star = abs_propagation(current_star, nn_bounds, target, net_list)

    # Check intersection using a LP
    # TODO is it possible to check whether it is fully inside?
    intersects, unsafe_stars = check_intersection(out_star, prop)

    return intersects, unsafe_stars


def intersect_symb_lp(current_star, net_list, nn_bounds, prop, target):
    # Compute the output abstract star from current_star/bounds
    out_star = abs_propagation(current_star, nn_bounds, target, net_list)

    # output >= nn_bounds[0]['model_out'][1].lower.matrix * x_input + lower.offset
    # output <= nn_bounds[0]['model_out'][1].upper.matrix * x_input + upper.offset

    # Check intersection using a LP
    # TODO is it possible to check whether it is fully inside?
    intersects, unsafe_stars = check_intersection(out_star, prop)
    return intersects, unsafe_stars


def get_target_sequential(star: Star, current_target: tuple, nn_list: list) -> (tuple, Star):
    """
    This function updates the target for the refinement of the star using
    a sequential approach. For each ReLU layer all neurons are refined
    sequentially.

    Parameters
    ----------
    star : Star
        The star to refine
    current_target : tuple
        The current target to update
    nn_list : list
        The list of the network layers

    Returns
    ----------
    tuple, Star
        The new target for the refinement, it is None when there is no more
        refinement to do for this star, and the propagated star

    """

    def get_last_relu(net_list: list):
        last_relu_idx = 0
        for net_layer in net_list[::-1]:
            if isinstance(net_layer, nodes.ReLUNode):
                last_relu_idx = net_list.index(net_layer)
                break

        return last_relu_idx

    new_target = None

    # Propagate current star to the next ReLU layer
    star = propagate_until_relu(star, nn_list)
    target_layer = star.ref_layer

    # Check if the target refers to a previous layer
    if target_layer != current_target[0] and target_layer < len(nn_list):
        new_target = (target_layer, 0)

    # Check if the neurons in the layer have been all processed
    elif current_target[1] == star.center.shape[0] - 1:

        # Check if all the layers have been processed
        if target_layer < get_last_relu(nn_list):

            # Go to the first neuron of next ReLU layer
            next_relu = target_layer
            for layer in nn_list[target_layer + 1:]:
                if isinstance(layer, nodes.ReLUNode):
                    next_relu = nn_list.index(layer)
                    break

            new_target = (next_relu, 0)
            star.ref_layer += 1
            star = propagate_until_relu(star, nn_list)
    else:
        # Increment the neuron
        new_target = (target_layer, current_target[1] + 1)

    return new_target, star


def split_star(star: Star, target: tuple, nn_list: list, bounds_dict: dict) -> list:
    """
    For a star we only need the var_index to target a specific neuron.
    The index relative to this neuron is determined by the heuristic that
    also takes into account what layer the star comes from.

    When splitting I also need to update the bounds and return them

    Parameters
    ----------
    star : Star
        The star object to split
    target : tuple
        The target layer and neuron to refine
    nn_list : list
        The neural network as a list of layers
    bounds_dict : dict
        The bounds of the network layers

    Returns
    ----------
    list
        A list of one or two 'Star' elements, depending on the stability
        of the current neuron

    """

    index = target[1]

    mask = np.identity(star.center.shape[0])
    mask[index, index] = 0

    cur_bounds = bounds_dict[nn_list[star.ref_layer].identifier]
    stable = abst.check_stable(index, cur_bounds)

    # Positive stable
    if stable == 1:
        star.ref_layer = target[0]
        return [(star, bounds_dict)]

    # Negative stable
    elif stable == -1:
        new_c = np.matmul(mask, star.center)
        new_b = np.matmul(mask, star.basis_matrix)
        new_pred = star.predicate_matrix
        new_bias = star.predicate_bias
        new_star = Star(new_pred, new_bias, new_c, new_b)

        new_star.ref_layer = target[0]

        return [(new_star, bounds_dict)]

    # Unstable
    else:
        # Lower star
        lower_c = np.matmul(mask, star.center)
        lower_b = np.matmul(mask, star.basis_matrix)
        lower_pred = np.vstack((star.predicate_matrix, star.basis_matrix[index, :]))
        lower_bias = np.vstack((star.predicate_bias, -star.center[index]))
        lower_star = Star(lower_pred, lower_bias, lower_c, lower_b)

        lower_star.ref_layer = target[0]
        lower_star.initial_pred = star.initial_pred

        # Upper star
        upper_c = star.center
        upper_b = star.basis_matrix
        upper_pred = np.vstack((star.predicate_matrix, -star.basis_matrix[index, :]))
        upper_bias = np.vstack((star.predicate_bias, star.center[index]))
        upper_star = Star(upper_pred, upper_bias, upper_c, upper_b)

        upper_star.ref_layer = target[0]
        upper_star.initial_pred = star.initial_pred

        # TODO update bounds
        return [
            (lower_star, bounds_dict),
            (upper_star, bounds_dict)
        ]
