from enum import Enum

import numpy as np

from pynever.strategies.bounds_propagation.bounds import VerboseBounds


class StabilityInfo(Enum):
    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2


def get_positive_part(weights):
    return np.maximum(weights, np.zeros(weights.shape))


def get_negative_part(weights):
    return np.minimum(weights, np.zeros(weights.shape))


def compute_lower(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_lower) + weights_minus.dot(input_upper)


def compute_upper(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_upper) + weights_minus.dot(input_lower)


def compute_max(weights, input_bounds):
    return compute_upper(get_negative_part(weights), get_positive_part(weights),
                         input_bounds.get_lower(), input_bounds.get_upper())


def compute_min(weights, input_bounds):
    return compute_lower(get_negative_part(weights), get_positive_part(weights),
                         input_bounds.get_lower(), input_bounds.get_upper())


def compute_lin_lower_and_upper(weights_minus, weights_plus, bias, lower_matrix, upper_matrix,
                                lower_offset, upper_offset):
    return compute_lower(weights_minus, weights_plus, lower_matrix, upper_matrix), \
        compute_lower(weights_minus, weights_plus, lower_offset, upper_offset) + bias, \
        compute_upper(weights_minus, weights_plus, lower_matrix, upper_matrix), \
        compute_upper(weights_minus, weights_plus, lower_offset, upper_offset) + bias


def compute_overapproximation_volume(areas_map: dict) -> float:
    return np.prod(list(areas_map.values()))


def compute_layer_inactive_from_bounds_and_fixed_neurons(bounds: VerboseBounds, fixed_neurons, layer_id):
    return (bounds.statistics.stability_info[StabilityInfo.INACTIVE][layer_id] +
            [i for (lay_id, i), value in fixed_neurons.items() if lay_id == layer_id and value == 0])


def extract_layer_unstable_from_bounds(bounds: VerboseBounds, layer_id):
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
