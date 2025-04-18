from enum import Enum

import numpy

from pynever import tensors
from pynever.strategies.abstraction.bounds_propagation import BOUNDS_PRECISION_GUARD
from pynever.strategies.abstraction.bounds_propagation.bounds import VerboseBounds, AbstractBounds
from pynever.tensors import Tensor


class ReLUStatus(Enum):
    """This enumerator registers the status of a ReLU neuron

    ACTIVE means that the input is positive, i.e., ReLU acts as identity
    INACTIVE means that the input is negative, i.e., ReLU outputs zero
    UNSTABLE means that the input is both positive and negative

    """
    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2


def check_stable(lb: float, ub: float) -> ReLUStatus:
    """Return the status of a ReLU neuron given the values of the lower and upper pre-activation bounds"""

    # Positive stable
    if lb >= BOUNDS_PRECISION_GUARD:
        return ReLUStatus.ACTIVE

    # Negative stable
    elif ub <= -BOUNDS_PRECISION_GUARD:
        return ReLUStatus.INACTIVE

    # Unstable
    else:
        return ReLUStatus.UNSTABLE


def compute_lower(weights_minus: Tensor, weights_plus: Tensor, input_lower: Tensor, input_upper: Tensor) -> Tensor:
    """Procedure that computes the matrix of coefficients for a lower bounds linear function.

    Parameters
    ----------
    weights_minus : Tensor
        The negative part of the weights
    weights_plus : Tensor
        The positive part of the weights
    input_lower : Tensor
        The lower input bounds
    input_upper : Tensor
        The upper input bounds

    Returns
    -----------
    Tensor
        The lower bounds matrix
    """
    return tensors.dot(weights_plus, input_lower) - tensors.dot(weights_minus, input_upper)


def compute_upper(weights_minus: Tensor, weights_plus: Tensor, input_lower: Tensor, input_upper: Tensor) -> Tensor:
    """Procedure that computes the matrix of coefficients for an upper bounds linear function.

    Parameters
    ----------
    weights_minus : Tensor
        The negative part of the weights
    weights_plus : Tensor
        The positive part of the weights
    input_lower : Tensor
        The lower input bounds
    input_upper : Tensor
        The upper input bounds

    Returns
    -----------
    Tensor
        The upper bounds matrix
    """
    return tensors.dot(weights_plus, input_upper) + tensors.dot(weights_minus, input_lower)


def compute_max(weights: Tensor, input_bounds: AbstractBounds) -> Tensor:
    """Procedure that computes the maximum value of a weights matrix given some input bounds

    Parameters
    ----------
    weights : Tensor
        The weights matrix
    input_bounds : AbstractBounds
        The input bounds

    Returns
    ----------
    Tensor
        The computed output
    """
    return compute_upper(tensors.get_negative(weights), tensors.get_positive(weights),
                         input_bounds.get_lower(), input_bounds.get_upper())


def compute_min(weights: Tensor, input_bounds: AbstractBounds) -> Tensor:
    """Procedure that computes the minimum value of a weights matrix given some input bounds

    Parameters
    ----------
    weights : Tensor
        The weights matrix
    input_bounds : AbstractBounds
        The input bounds

    Returns
    ----------
    Tensor
        The computed output
    """
    return compute_lower(tensors.get_negative(weights), tensors.get_positive(weights),
                         input_bounds.get_lower(), input_bounds.get_upper())


def compute_overapproximation_volume(areas_map: dict) -> float:
    """Procedure that computes the volume of the approximation as the product of the areas.
    For some reason raises a warning on data types
    """
    return numpy.prod(list(areas_map.values()))


def compute_layer_inactive_from_bounds_and_fixed_neurons(bounds: VerboseBounds,
                                                         fixed_neurons: dict,
                                                         layer_id: str) -> list[int]:
    """Procedure to obtain the inactive neurons for a given layer based on bounds and information about neurons.

    Parameters
    ----------
    bounds : VerboseBounds
        The bounds information
    fixed_neurons : dict
        The fixed neurons so far
    layer_id : str
        The layer id

    Returns
    ----------
    list[int]
        The list of computed inactive neurons and fixed inactive neurons in the layer
    """
    return (bounds.statistics.stability_info[ReLUStatus.INACTIVE][layer_id] +
            [i for (lay_id, i), value in fixed_neurons.items() if lay_id == layer_id and value == 0])


def compute_layer_unstable_from_bounds_and_fixed_neurons(bounds: VerboseBounds,
                                                         fixed_neurons: dict,
                                                         layer_id: str) -> list[int]:
    """Procedure to obtain the unstable neurons for a given layer based on bounds and information about neurons.

    Parameters
    ----------
    bounds : VerboseBounds
        The bounds information
    fixed_neurons : dict
        The fixed neurons so far
    layer_id : str
        The layer id

    Returns
    ----------
    list[int]
        The list of computed unstable neurons and fixed unstable neurons in the layer
    """
    layer_unstable = {
        neuron_n
        for lay_id, neuron_n in bounds.statistics.stability_info[ReLUStatus.UNSTABLE]
        if lay_id == layer_id
    }
    return [neuron_n for neuron_n in layer_unstable if (layer_id, neuron_n) not in fixed_neurons]


def compute_unstable_from_bounds_and_fixed_neurons(bounds: VerboseBounds, fixed_neurons: dict) -> list[int]:
    """Procedure to obtain the overall unstable neurons based on bounds and information about neurons.

    Parameters
    ----------
    bounds : VerboseBounds
        The bounds information
    fixed_neurons : dict
        The fixed neurons so far

    Returns
    ----------
    list[int]
        The list of overall computed unstable neurons and fixed unstable neurons
    """
    unstable = bounds.statistics.stability_info[ReLUStatus.UNSTABLE]
    return [neuron for neuron in unstable if neuron not in fixed_neurons]
