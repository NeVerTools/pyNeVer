import copy
import datetime

import numpy as np

import pynever.strategies.abstraction as abst
from pynever import nodes, networks
from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction import Star
from pynever.strategies.bp.bounds import HyperRectangleBounds
from pynever.strategies.bp.bounds_manager import BoundsManager, StabilityInfo
from pynever.strategies.bp.utils.property_converter import PropertyFormatConverter
from pynever.strategies.bp.utils.utils import get_positive_part, get_negative_part, compute_max, compute_min
from pynever.tensors import Tensor



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

    elif strategy == 'lirpa':
        return BoundsManager.compute_lirpa_bounds(nn, prop)
        pass
    # TODO add more strategies





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
        # print(f"{datetime.datetime.now()} Intersected with half space")

        if not intersection.check_if_empty():
            intersects = True
            unsafe_stars.append(intersection)
        # print(f"{datetime.datetime.now()} Checked if empty")

    return intersects, unsafe_stars


def intersect_star_lp(current_star, net_list, nn_bounds, prop):
    # Compute the output abstract star from current_star/bounds
    out_star = abs_propagation(current_star, nn_bounds, net_list)

    # print(f"Abstract propagation computed {datetime.datetime.now()}")
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





def get_target_sequential(star: Star, network: networks.SequentialNetwork) -> tuple[RefinementTarget | None, Star]:
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

    star, _ = propagate_until_relu(star, network, False)
    current_neuron = star.ref_neuron

    if current_neuron < star.center.shape[0]:
        # There are more neurons in the layer: increment the neuron count
        new_target = RefinementTarget(star.ref_layer, current_neuron)

    else:
        if star.ref_layer == network.get_last_relu_index():
            # There are no more neurons and no more layers
            new_target = None

        else:
            # There is another ReLU layer: propagate the star to that layer and reset the neuron
            star, _ = propagate_until_relu(star, network, True)
            star.ref_neuron = 0
            next_layer = star.ref_layer
            new_target = RefinementTarget(next_layer, 0)

    return new_target, star


def split_star(star: Star, target: RefinementTarget, nn_list: list, bounds_dict: dict, update_bounds: bool) -> list:
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
    update_bounds : bool
        Flag to update the bounds after the split

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
            star = Star(new_pred, new_bias, new_c, new_b)

            star.ref_layer = target.layer_idx
            star.ref_neuron = star.ref_neuron + 1
            index += 1

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

            # Update the target neuron based on the index increment if the branch was stable (@392)
            target.neuron_idx = index

            # Update the bounds after the split
            if update_bounds:
                lower_bounds, upper_bounds = BoundsManager().branch_update_bounds(bounds_dict, nn_list, target)
            else:
                lower_bounds, upper_bounds = bounds_dict, bounds_dict

            return \
                    ([] if lower_bounds is None else [(lower_star, lower_bounds)]) + \
                    ([] if upper_bounds is None else [(upper_star, upper_bounds)])

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
