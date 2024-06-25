import numpy as np


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
