import torch
from torch import Tensor

from pynever.strategies.abstraction import ABSTRACTION_PRECISION_GUARD
from pynever.strategies.abstraction.bounds_propagation import ReLUStatus
from pynever.strategies.abstraction.bounds_propagation.bounds import AbstractBounds
from pynever.strategies.verification.statistics import VerboseBounds


def check_stable(lb: float, ub: float) -> ReLUStatus:
    """Return the status of a ReLU neuron given the values of the lower and upper pre-activation bounds"""

    # Positive stable
    if lb >= ABSTRACTION_PRECISION_GUARD:
        return ReLUStatus.ACTIVE

    # Negative stable
    elif ub <= -ABSTRACTION_PRECISION_GUARD:
        return ReLUStatus.INACTIVE

    # Unstable
    else:
        return ReLUStatus.UNSTABLE


def compute_lower(weights_minus: Tensor, weights_plus: Tensor, input_lower: Tensor, input_upper: Tensor) -> Tensor:
    """Procedure that computes the matrix of coefficients for a lower bounds linear function.

    Parameters
    ----------
    weights_minus: Tensor
        The negative part of the weights
    weights_plus: Tensor
        The positive part of the weights
    input_lower: Tensor
        The lower input bounds
    input_upper: Tensor
        The upper input bounds

    Returns
    -----------
    Tensor
        The lower bounds matrix
    """
    return torch.matmul(weights_plus, input_lower) + torch.matmul(weights_minus, input_upper)


def compute_upper(weights_minus: Tensor, weights_plus: Tensor, input_lower: Tensor, input_upper: Tensor) -> Tensor:
    """Procedure that computes the matrix of coefficients for an upper bounds linear function.

    Parameters
    ----------
    weights_minus: Tensor
        The negative part of the weights
    weights_plus: Tensor
        The positive part of the weights
    input_lower: Tensor
        The lower input bounds
    input_upper: Tensor
        The upper input bounds

    Returns
    -----------
    Tensor
        The upper bounds matrix
    """
    return torch.matmul(weights_plus, input_upper) + torch.matmul(weights_minus, input_lower)


def compute_max(weights: Tensor, input_bounds: AbstractBounds) -> Tensor:
    """Procedure that computes the maximum value of a weights matrix given some input bounds

    Parameters
    ----------
    weights: Tensor
        The weights matrix
    input_bounds: AbstractBounds
        The input bounds

    Returns
    ----------
    Tensor
        The computed output
    """
    return compute_upper(torch.clamp(weights, max=0), torch.clamp(weights, min=0),
                         input_bounds.get_lower(), input_bounds.get_upper())


def compute_min(weights: Tensor, input_bounds: AbstractBounds) -> Tensor:
    """Procedure that computes the minimum value of a weights matrix given some input bounds

    Parameters
    ----------
    weights: Tensor
        The weights matrix
    input_bounds: AbstractBounds
        The input bounds

    Returns
    ----------
    Tensor
        The computed output
    """
    return compute_lower(torch.clamp(weights, max=0), torch.clamp(weights, min=0),
                         input_bounds.get_lower(), input_bounds.get_upper())


def compute_overapproximation_volume(areas_map: dict) -> float:
    """Procedure that computes the volume of the approximation as the product of the areas.
    """
    return torch.prod(Tensor(list(areas_map.values()))).item()


def compute_layer_inactive_from_bounds_and_fixed_neurons(bounds: VerboseBounds,
                                                         fixed_neurons: dict,
                                                         layer_id: str) -> list[int]:
    """Procedure to obtain the inactive neurons for a given layer based on bounds and information about neurons.

    Parameters
    ----------
    bounds: VerboseBounds
        The bounds information
    fixed_neurons: dict
        The fixed neurons so far
    layer_id: str
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
    bounds: VerboseBounds
        The bounds information
    fixed_neurons: dict
        The fixed neurons so far
    layer_id: str
        The layer id

    Returns
    ----------
    list[int]
        The list of computed unstable neurons and fixed unstable neurons in the layer
    """
    layer_unstable = []
    for lay_id, neurons in bounds.statistics.stability_info[ReLUStatus.UNSTABLE].items():
        if lay_id == layer_id:
            layer_unstable.extend(neurons)

    return [neuron_n for neuron_n in layer_unstable if (layer_id, neuron_n) not in fixed_neurons]


def compute_unstable_from_bounds_and_fixed_neurons(bounds: VerboseBounds, fixed_neurons: dict) -> list[tuple]:
    """Procedure to obtain the overall unstable neurons based on bounds and information about neurons.

    Parameters
    ----------
    bounds: VerboseBounds
        The bounds information
    fixed_neurons: dict
        The fixed neurons so far

    Returns
    ----------
    list[int]
        The list of overall computed unstable neurons and fixed unstable neurons
    """
    unstable = bounds.statistics.stability_info[ReLUStatus.UNSTABLE]
    return [(layer, neuron) for layer, neurons in unstable.items() for neuron in neurons if (layer, neuron) not in fixed_neurons]
