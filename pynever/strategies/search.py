import copy

import numpy as np

import pynever.strategies.abstraction as abst
from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction import Star
from pynever.strategies.bp.bounds import HyperRectangleBounds
from pynever.strategies.bp.bounds_manager import BoundsManager
from pynever.strategies.bp.utils.property_converter import PropertyFormatConverter
from pynever.strategies.bp.utils.utils import get_positive_part, get_negative_part
from pynever.tensors import Tensor


class RefinementTarget:
    """

    """

    # TODO how to use with ResNets? Layer identifier?
    def __init__(self, layer: int, neuron: int):
        self.layer_idx = layer
        self.neuron_idx = neuron


def get_input_bounds(prop: 'NeverProperty') -> HyperRectangleBounds:
    """
    This method computes the numeric bounds of the input layer

    Parameters
    ----------
    prop : NeverProperty
        The property to verify

    Returns
    ----------
    HyperRectangleBounds
        The numeric bounds of the input layer

    """

    # HyperRectBounds input bounds
    return PropertyFormatConverter(prop).get_vectors()


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
        The dictionary of the bounds computed by the Bounds Manager

    """

    if strategy == 'symbolic':
        return BoundsManager().compute_bounds_from_property(nn, prop)

    elif strategy == 'LiRPA':
        # return something...
        pass
    # TODO add more strategies


def abs_propagation(star: Star, bounds: dict, nn_list: list) -> Star:
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
    nn_list : list
        The neural network represented as a list

    Returns
    ----------
    Star
        The resulting star approximate with the abstract propagation

    """

    start_layer = star.ref_layer
    neuron_idx = star.ref_neuron

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
            l_bounds = bounds['numeric_pre'][layer.identifier]
            if i == start_layer:
                star = abst.approx_relu_forward(star, l_bounds, layer.in_dim[0], start_idx=neuron_idx)
            else:
                star = abst.approx_relu_forward(star, l_bounds, layer.in_dim[0])

        else:
            raise NotImplementedError('Unsupported layer')

    return star


def propagate_until_relu(star: Star, nn_list: list, skip: bool) -> Star:
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
    skip : bool
        Flag to signal end of propagation

    Returns
    ----------
    Star
        The resulting star before the next ReLU layer

    """

    start_layer = star.ref_layer
    i = 0

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
            if skip:  # star.ref_neuron == star.center.shape[0] - 1 and skip:
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


def intersect_star_lp(current_star, net_list, nn_bounds, prop):
    # Compute the output abstract star from current_star/bounds
    out_star = abs_propagation(current_star, nn_bounds, net_list)

    # Check intersection using a LP
    intersects, unsafe_stars = check_intersection(out_star, prop)

    return intersects, unsafe_stars


def intersect_symb_lp(input_bounds, nn_bounds, prop):
    nn_bounds = nn_bounds['symbolic']

    out_id = list(nn_bounds.keys())[-1]
    out_neurons = nn_bounds[out_id].lower.matrix.shape[0]

    basis = np.eye(out_neurons, out_neurons)
    center = np.zeros((out_neurons, 1))

    predicate_matrix = np.zeros((2 * out_neurons, out_neurons))
    predicate_bias = np.zeros((2 * out_neurons, 1))

    # Compute positive and negative weights for the lower bounds
    lower_weights_plus = get_positive_part(nn_bounds[out_id].lower.matrix)
    lower_weights_minus = get_negative_part(nn_bounds[out_id].lower.matrix)

    # Compute positive and negative weights for the upper bounds
    upper_weights_plus = get_positive_part(nn_bounds[out_id].upper.matrix)
    upper_weights_minus = get_negative_part(nn_bounds[out_id].upper.matrix)

    # Get input lower and upper bounds
    input_lbs = input_bounds.lower
    input_ubs = input_bounds.upper

    for i in range(center.shape[0]):
        # For each i add two rows
        lb_row_idx = 2 * i
        ub_row_idx = 2 * i + 1

        # Structure Cx <= d
        predicate_matrix[lb_row_idx] = nn_bounds[out_id].lower.matrix[i]
        predicate_matrix[ub_row_idx] = nn_bounds[out_id].upper.matrix[i]

        predicate_bias[lb_row_idx] = (
                -lower_weights_plus[i].dot(input_lbs) -
                lower_weights_minus[i].dot(input_ubs) -
                nn_bounds[out_id].lower.offset[i]
        )
        predicate_bias[ub_row_idx] = (
                upper_weights_plus[i].dot(input_ubs) +
                upper_weights_minus[i].dot(input_lbs) +
                nn_bounds[out_id].upper.offset[i]
        )

    output = Star(predicate_matrix, predicate_bias, center, basis)
    intersects, unsafe_stars = check_intersection(output, prop)

    return intersects, unsafe_stars


def get_next_target(ref_heur: str,
                    star: Star,
                    nn_list: list) -> (RefinementTarget, Star):
    if ref_heur == 'sequential':
        return get_target_sequential(star, nn_list)
    else:
        raise NotImplementedError('Only sequential refinement supported')


def get_target_sequential(star: Star, nn_list: list) -> (RefinementTarget, Star):
    """
    This function updates the target for the refinement of the star using
    a sequential approach. For each ReLU layer all neurons are refined
    sequentially.

    Parameters
    ----------
    star : Star
        The star to refine
    nn_list : list
        The list of the network layers

    Returns
    ----------
    tuple, Star
        The new target for the refinement, which is None when there is no more
        refinement to do for this star, and the propagated star

    """

    def get_last_relu(net_list: list):
        last_relu_idx = 0
        for net_layer in net_list[::-1]:
            if isinstance(net_layer, nodes.ReLUNode):
                last_relu_idx = net_list.index(net_layer)
                break

        return last_relu_idx

    star = propagate_until_relu(star, nn_list, False)
    current_neuron = star.ref_neuron

    if current_neuron < star.center.shape[0]:
        # There are more neurons in the layer: increment the neuron count
        new_target = RefinementTarget(star.ref_layer, current_neuron)

    else:
        if star.ref_layer == get_last_relu(nn_list):
            # There are no more neurons and no more layers
            new_target = None

        else:
            # There is another ReLU layer: propagate the star to that layer and reset the neuron
            star = propagate_until_relu(star, nn_list, True)
            star.ref_neuron = 0
            next_layer = star.ref_layer
            new_target = RefinementTarget(next_layer, 0)

    return new_target, star


def split_star(star: Star, target: RefinementTarget, nn_list: list, bounds_dict: dict) -> list:
    """
    For a star we only need the var_index to target a specific neuron.
    The index relative to this neuron is determined by the heuristic that
    also takes into account what layer the star comes from.

    When splitting I also need to update the bounds and return them

    Parameters
    ----------
    star : Star
        The star object to split
    target : RefinementTarget
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

    index = target.neuron_idx
    cur_bounds = bounds_dict['numeric_pre'][nn_list[star.ref_layer].identifier]

    # Loop to filter positive stable neurons
    while index < star.center.shape[0]:

        mask = np.identity(star.center.shape[0])
        mask[index, index] = 0

        stable = abst.check_stable(index, cur_bounds)

        # Positive stable
        if stable == 1:
            star.ref_layer = target.layer_idx
            star.ref_neuron += 1
            index += 1

        # Negative stable
        elif stable == -1:
            new_c = np.matmul(mask, star.center)
            new_b = np.matmul(mask, star.basis_matrix)
            new_pred = star.predicate_matrix
            new_bias = star.predicate_bias
            new_star = Star(new_pred, new_bias, new_c, new_b)

            new_star.ref_layer = target.layer_idx
            new_star.ref_neuron = star.ref_neuron + 1

            return [(new_star, bounds_dict)]

        # Unstable
        else:
            # Lower star
            lower_c = np.matmul(mask, star.center)
            lower_b = np.matmul(mask, star.basis_matrix)
            lower_pred = np.vstack((star.predicate_matrix, star.basis_matrix[index, :]))
            lower_bias = np.vstack((star.predicate_bias, -star.center[index]))
            lower_star = Star(lower_pred, lower_bias, lower_c, lower_b)

            lower_star.ref_layer = target.layer_idx
            lower_star.ref_neuron = star.ref_neuron + 1

            # Upper star
            upper_c = star.center
            upper_b = star.basis_matrix
            upper_pred = np.vstack((star.predicate_matrix, -star.basis_matrix[index, :]))
            upper_bias = np.vstack((star.predicate_bias, star.center[index]))
            upper_star = Star(upper_pred, upper_bias, upper_c, upper_b)

            upper_star.ref_layer = target.layer_idx
            upper_star.ref_neuron = star.ref_neuron + 1

            # Update the bounds after the split
            lower_bounds, upper_bounds = BoundsManager().branch_update_bounds(bounds_dict, nn_list, target)

            return [
                (lower_star, lower_bounds),
                (upper_star, upper_bounds)
            ]

    # I get here only if I complete the while loop
    return [(star, bounds_dict)]


def get_counterexample(unsafe_stars: list, prop: 'NeverProperty') -> Tensor:
    """
    This function is used to extract a counterexample from a star.
    The counterexample that we are interested into is the witness, i.e.,
    the input assignment that resulted in an intersection.

    Parameters
    ----------
    unsafe_stars : list
        Unsafe stars in output
    prop : NeverProperty
        The property of interest

    Returns
    ----------
    Tensor
        The counterexample input Tensor

    """

    # Extract counterexample stars
    counterexample_stars = []

    for unsafe_star in unsafe_stars:
        temp_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
        temp_star.predicate_matrix = copy.deepcopy(unsafe_star.predicate_matrix)
        temp_star.predicate_bias = copy.deepcopy(unsafe_star.predicate_bias)
        counterexample_stars.append(temp_star)

    return counterexample_stars[0].get_samples(num_samples=1)[0]
