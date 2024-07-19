from pynever import tensors
from pynever.strategies.bounds_propagation.utils.utils import *


class LinearFunctions:
    """
    matrix is an (n x m) np array
    offset is an (n) np array

    An object represents n linear functions f(i) of m input variables x

    f(i) = matrix[i]*x + offset[i]

    """

    def __init__(self, matrix, offset):
        self.size = matrix.shape[0]
        self.matrix = matrix
        self.offset = offset

    def clone(self):
        return LinearFunctions(self.matrix.copy(), self.offset.copy())

    def mask_zero_outputs(self, zero_outputs):
        mask = np.diag(
            [0 if neuron_n in zero_outputs else 1 for neuron_n in range(self.size)]
        )

        return LinearFunctions(tensors.matmul(mask, self.matrix), tensors.matmul(mask, self.offset))

    def get_size(self):
        return self.size

    def get_matrix(self):
        return self.matrix

    def get_offset(self):
        return self.offset

    def compute_value(self, row_number, input_values):
        return self.matrix[row_number].dot(input_values) + self.offset[row_number]

    def compute_values(self, input_values):
        return self.matrix.dot(input_values) + self.offset

    def compute_max_value(self, row_number, input_bounds):
        row_coeff = self.matrix[row_number]
        return get_positive_part(row_coeff).dot(input_bounds.get_upper()) + \
            get_negative_part(row_coeff).dot(input_bounds.get_lower()) + self.offset[row_number]

    def compute_min_value(self, row_number, input_bounds):
        row_coeff = self.matrix[row_number]
        return get_positive_part(row_coeff).dot(input_bounds.get_lower()) + \
            get_negative_part(row_coeff).dot(input_bounds.get_upper()) + self.offset[row_number]

    def compute_max_values(self, input_bounds):
        return get_positive_part(self.matrix).dot(input_bounds.get_upper()) + \
            get_negative_part(self.matrix).dot(input_bounds.get_lower()) + \
            self.offset

    def compute_min_values(self, input_bounds):
        return get_positive_part(self.matrix).dot(input_bounds.get_lower()) + \
            get_negative_part(self.matrix).dot(input_bounds.get_upper()) + \
            self.offset
