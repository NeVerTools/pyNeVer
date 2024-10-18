"""
This file contains specialized methods that provide
the linearization of sigmoid and hyperbolic tangent activation functions

"""

from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds
from pynever.strategies.bounds_propagation.linearfunctions import LinearizeActivation


class LinearizeSLikeActivation(LinearizeActivation):

    def __init__(self):
        pass

    def compute_output_linear_bounds(self, input_eq: SymbolicLinearBounds) -> SymbolicLinearBounds:
        pass


class LinearizeSigmoid(LinearizeSLikeActivation):
    def __init__(self):
        super().__init__()

    def compute_output_linear_bounds(self, input_eq: SymbolicLinearBounds) -> SymbolicLinearBounds:
        pass


class LinearizeTanh(LinearizeSLikeActivation):
    def __init__(self):
        super().__init__()

    def compute_output_linear_bounds(self, input_eq: SymbolicLinearBounds) -> SymbolicLinearBounds:
        pass
