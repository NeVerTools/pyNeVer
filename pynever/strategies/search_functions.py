import numpy as np

import pynever.strategies.abstraction as abst
from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction import Star
from pynever.strategies.bp.bounds import HyperRectangleBounds, AbstractBounds
from pynever.strategies.bp.bounds_manager import BoundsManager
from pynever.strategies.verification import NeVerProperty
from pynever.tensor import Tensor


def get_bounds(nn: SequentialNetwork, prop: NeVerProperty, strategy: str) -> AbstractBounds:
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
    AbstractBounds
        The bounds wrapped in an AbstractBounds object

    """

    if strategy == 'symbolic':
        return BoundsManager(nn, prop).compute_bounds()[0]
    elif strategy == 'lirpa':
        # return something...
        pass
    # TODO add more strategies


def abs_propagation(star: Star, bounds: AbstractBounds, target: tuple, network_list: list) -> Star:
    """
    This method performs the abstract propagation of a single star starting
    from a specific layer and neuron. The output is a single star that uses
    approximation in the next layers

    Parameters
    ----------
    star : Star
        The star to process
    bounds : AbstractBounds
        The bounds for the layers
    target : tuple
        The target layer and neuron to start from wrapped in a tuple
    network_list : list
        The neural network represented as a list

    Returns
    ----------
    Star
        The resulting star approximate with the abstract propagation

    """

    start_layer = star.ref_layer

    for layer in network_list[start_layer:]:
        i = network_list.index(layer)
        if isinstance(layer, nodes.FullyConnectedNode):
            # Propagate fully connected entirely
            star = abst.single_fc_forward(star, layer.weight, layer.bias)

        elif isinstance(layer, nodes.ReLUNode):
            # Propagate ReLU starting from target
            # TODO assign correct bound values
            if i == target[0]:
                star = abst.approx_relu_forward(star, bounds, layer.in_dim[0], start_idx=target[1])
            else:
                star = abst.approx_relu_forward(star, bounds, layer.in_dim[0])

        else:
            raise NotImplementedError('Unsupported layer')

    return star


def check_intersection(star: Star, property: NeVerProperty) -> (bool, list):
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
    property : NeVerProperty
        The property of interest

    Returns
    -------
    (bool, list)
        A tuple containing the result of the intersection check (True/False) and
        the list of unfeasible stars. If the result is True the list must contain
        at least one element, otherwise the list must be empty

    """

    safe = True
    unsafe_stars = []

    # Loop possible disjunctions
    for i in range(len(property.out_coef_mat)):
        intersection = abst.intersect_with_halfspace(star,
                                                     property.out_coef_mat[i],
                                                     property.out_bias_mat[i])
        if not intersection.check_if_empty():
            safe = False
            unsafe_stars.append(intersection)

    return safe, unsafe_stars


def split_star(star: Star, index: int, cur_bounds: AbstractBounds) -> list:
    """
    For a star we only need the var_index to target a specific neuron.
    The index relative to this neuron is determined by the heuristic that
    also takes into account what layer the star comes from.

    When splitting I also need to update the bounds and return them

    """

    mask = np.identity(star.center.shape[0])
    mask[index, index] = 0

    # TODO link bounds to layer
    stable = abst.check_stable(index, cur_bounds)

    # Positive stable
    if stable == 1:
        star.ref_layer += 1
        return [(star, cur_bounds)]

    # Negative stable
    elif stable == -1:
        new_c = np.matmul(mask, star.center)
        new_b = np.matmul(mask, star.basis_matrix)
        new_pred = star.predicate_matrix
        new_bias = star.predicate_bias
        new_star = Star(new_pred, new_bias, new_c, new_b)
        new_star.ref_layer = star.ref_layer + 1

        return [(new_star, cur_bounds)]

    # Unstable
    else:
        # Lower star
        lower_c = np.matmul(mask, star.center)
        lower_b = np.matmul(mask, star.basis_matrix)
        lower_pred = np.vstack((star.predicate_matrix, star.basis_matrix[index, :]))
        lower_bias = np.vstack((star.predicate_bias, -star.center[index]))
        lower_star = Star(lower_pred, lower_bias, lower_c, lower_b)
        lower_star.ref_layer = star.ref_layer + 1

        # Upper star
        upper_c = star.center
        upper_b = star.basis_matrix
        upper_pred = np.vstack((star.predicate_matrix, -star.basis_matrix[index, :]))
        upper_bias = np.vstack((star.predicate_bias, star.center[index]))
        upper_star = Star(upper_pred, upper_bias, upper_c, upper_b)
        upper_star.ref_layer = star.ref_layer + 1

        # TODO update bounds
        return [
            (lower_star, cur_bounds),
            (upper_star, cur_bounds)
        ]


def get_counterexample(star: Star) -> Tensor:
    return star.get_samples(num_samples=1)[0]


def get_target(star: Star, out_bounds: HyperRectangleBounds, nn: SequentialNetwork) -> 'TargetNeuron':
    # TODO select next neuron
    # Assignee: Stefano
    print('Hi')
