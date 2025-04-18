import copy

from pynever import tensors
from pynever.tensors import Tensor


class LinearFunctions:
    """
    A class representing a set of n linear functions f(i) of m input variables x

    matrix is an (n x m) Tensor
    offset is an (n) Tensor

    f(i) = matrix[i]*x + offset[i]

    """

    def __init__(self, matrix: Tensor, offset: Tensor):
        self.size = matrix.shape[0]
        self.matrix = matrix
        self.offset = offset

    def clone(self):
        return LinearFunctions(copy.deepcopy(self.matrix), copy.deepcopy(self.offset))

    def mask_zero_outputs(self, zero_outputs):
        mask = tensors.diag(
            Tensor([0 if neuron_n in zero_outputs else 1 for neuron_n in range(self.size)])
        )
        return LinearFunctions(tensors.matmul(mask, self.matrix), tensors.matmul(mask, self.offset))

    def get_size(self) -> int:
        return self.size

    def get_matrix(self) -> Tensor:
        return self.matrix

    def get_offset(self) -> Tensor:
        return self.offset

    def compute_max_values(self, input_bounds) -> Tensor:
        return tensors.dot(tensors.get_positive(self.matrix), input_bounds.get_upper()) + \
            tensors.dot(tensors.get_negative(self.matrix), input_bounds.get_lower()) + \
            self.offset

    def compute_min_values(self, input_bounds) -> Tensor:
        return tensors.dot(tensors.get_positive(self.matrix), input_bounds.get_lower()) + \
            tensors.dot(tensors.get_negative(self.matrix), input_bounds.get_upper()) + \
            self.offset
