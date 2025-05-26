"""
This module defines the behavior of an interval arithmetic-based representation
of a neural network linear layer

"""

import numpy as np

from strategies.verification.interval.core import ops


class IntervalLayer:
    pass


class LinearIntervalLayer(IntervalLayer):
    def __init__(self, weight: list, bias: list):
        super().__init__()

        self.weight = weight
        self.bias = bias

    def predict(self, x: list) -> int:
        """Procedure to execute the matrix multiplication"""
        # TODO argmax
        return np.argmax(ops.add(ops.matmul_left(self.weight, x), self.bias))
