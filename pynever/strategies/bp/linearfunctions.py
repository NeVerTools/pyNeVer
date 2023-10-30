from pynever.strategies.bp.utils.utils import *


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

    def get_input_for_max(self, input_bounds):
        positive_mask = get_positive_flags(self.matrix)

        (input_lower_bounds, input_upper_bounds) = input_bounds.get_lower(), input_bounds.get_upper()
        values = np.zeros((self.size, len(input_lower_bounds)))
        for j in range(self.size):
            for i in range(len(input_lower_bounds)):
                if positive_mask[j][i] == 1:
                    values[j][i] = input_upper_bounds[i]
                else:
                    values[j][i] = input_lower_bounds[i]
        return values

    def get_input_for_min(self, input_bounds):
        positive_mask = get_positive_flags(self.matrix)

        (input_lower_bounds, input_upper_bounds) = input_bounds.get_lower(), input_bounds.get_upper()
        values = np.zeros((self.size, len(input_lower_bounds)))
        for j in range(self.size):
            for i in range(len(input_lower_bounds)):
                if positive_mask[j][i] == 1:
                    values[j][i] = input_lower_bounds[i]
                else:
                    values[j][i] = input_upper_bounds[i]
        return values
