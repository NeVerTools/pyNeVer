import copy

import torch


class LinearFunctions:
    """
    A class representing a set of n linear functions f(i) of m input variables x

    matrix is an (n x m) Tensor
    offset is an (n) Tensor

    f(i) = matrix[i]*x + offset[i]

    """

    def __init__(self, matrix: torch.Tensor, offset: torch.Tensor):
        self.size = matrix.shape[0]
        self.matrix = matrix
        self.offset = offset

    def __repr__(self):
        return "LinearFunctions({})".format(self.size)

    def clone(self):
        return LinearFunctions(copy.deepcopy(self.matrix), copy.deepcopy(self.offset))

    def mask_zero_outputs(self, zero_outputs):
        mask = torch.diag(
            torch.Tensor([0 if neuron_n in zero_outputs else 1 for neuron_n in range(self.size)])
        )
        return LinearFunctions(torch.matmul(mask, self.matrix), torch.matmul(mask, self.offset))

    def get_size(self) -> int:
        return self.size

    def get_matrix(self) -> torch.Tensor:
        return self.matrix

    def get_offset(self) -> torch.Tensor:
        return self.offset

    def compute_max_values(self, input_bounds) -> torch.Tensor:
        return torch.matmul(torch.clamp(self.matrix, min=0), input_bounds.get_upper()) + \
            torch.matmul(torch.clamp(self.matrix, max=0), input_bounds.get_lower()) + \
            self.offset

    def compute_min_values(self, input_bounds) -> torch.Tensor:
        return torch.matmul(torch.clamp(self.matrix, min=0), input_bounds.get_lower()) + \
            torch.matmul(torch.clamp(self.matrix, max=0), input_bounds.get_upper()) + \
            self.offset
