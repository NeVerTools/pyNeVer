import numpy as np


def get_positive_part(weights):
    return np.maximum(weights, np.zeros(weights.shape))


def get_negative_part(weights):
    return np.minimum(weights, np.zeros(weights.shape))


def get_positive_flags(weights):
    output = np.array(weights)
    output[output > 0] = 1
    # output = np.array(output, dtype=int)
    return output


def get_negative_flags(weights):
    output = np.array(weights)
    output[output < 0] = 1
    # output = np.array(output, dtype=int)
    return output


def compute_lower(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_lower) + weights_minus.dot(input_upper)


def compute_upper(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_upper) + weights_minus.dot(input_lower)


def compute_lin_lower_and_upper(weights_minus, weights_plus, bias, lower_matrix, upper_matrix,
                                lower_offset, upper_offset):

    return compute_lower(weights_minus, weights_plus, lower_matrix, upper_matrix), \
           compute_lower(weights_minus, weights_plus, lower_offset, upper_offset) + bias, \
           compute_upper(weights_minus, weights_plus, lower_matrix, upper_matrix), \
           compute_upper(weights_minus, weights_plus, lower_offset, upper_offset) + bias


def compute_lower_hammard(weights_minus, weights_plus, lower_b, upper_b):
    return weights_plus * lower_b + weights_minus * upper_b


def compute_upper_hammard(weights_minus, weights_plus, lower_b, upper_b):
    return weights_minus * lower_b + weights_plus * upper_b


def epsilon_neighbourhood(input, epsilon, minimal=0, maximal=1):
    epsilon_array = np.full(input.shape, epsilon)
    lower = input - epsilon_array
    lower[lower<minimal] = minimal
    upper = input + epsilon_array
    upper[upper>maximal] = maximal

    return (lower,upper)


