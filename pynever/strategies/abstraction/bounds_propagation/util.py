from enum import Enum

import torch

from pynever import tensors
from pynever.strategies.abstraction.bounds_propagation.bounds import VerboseBounds


class StabilityInfo(Enum):
    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2


def compute_lower(weights_minus, weights_plus, input_lower, input_upper):
    return tensors.dot(weights_plus, input_lower) - tensors.dot(weights_minus, input_upper)


def compute_upper(weights_minus, weights_plus, input_lower, input_upper):
    return tensors.dot(weights_plus, input_upper) + tensors.dot(weights_minus, input_lower)


def compute_max(weights, input_bounds):
    return compute_upper(tensors.get_negative(weights), tensors.get_positive(weights),
                         input_bounds.get_lower(), input_bounds.get_upper())


def compute_min(weights, input_bounds):
    return compute_lower(tensors.get_negative(weights), tensors.get_positive(weights),
                         input_bounds.get_lower(), input_bounds.get_upper())


# TODO check data type
def compute_overapproximation_volume(areas_map: dict) -> float:
    return tensors.prod(list(areas_map.values()))


def compute_layer_inactive_from_bounds_and_fixed_neurons(bounds: VerboseBounds, fixed_neurons: dict, layer_id: str):
    return (bounds.statistics.stability_info[StabilityInfo.INACTIVE][layer_id] +
            [i for (lay_id, i), value in fixed_neurons.items() if lay_id == layer_id and value == 0])


def extract_layer_unstable_from_bounds(bounds: VerboseBounds, layer_id: str):
    return {neuron_n for lay_id, neuron_n in bounds.statistics.stability_info[StabilityInfo.UNSTABLE]
            if lay_id == layer_id}


def compute_layer_unstable_from_bounds_and_fixed_neurons(bounds: VerboseBounds, fixed_neurons, layer_id):
    return [neuron_n for neuron_n in extract_layer_unstable_from_bounds(bounds, layer_id)
            if (layer_id, neuron_n) not in fixed_neurons]


def compute_unstable_from_bounds_and_fixed_neurons(bounds: VerboseBounds, fixed_neurons: dict) -> list:
    """
    Utility method

    """

    unstable = bounds.statistics.stability_info[StabilityInfo.UNSTABLE]
    return [neuron for neuron in unstable if neuron not in fixed_neurons]
