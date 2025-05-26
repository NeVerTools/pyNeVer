"""
This module defines the behavior of an interval arithmetic-based representation
of a neural network model

"""

from strategies.verification.interval.parser import onnx
from strategies.verification.interval.parser import vnnlib

from strategies.verification.interval.core import ops
from strategies.verification.interval.core.layer import LinearIntervalLayer


class ModelOptions:
    pass


class IntervalModel:
    def __init__(self, onnx_path: str, work_precision: int = 3, options: ModelOptions = None):
        self.onnx_path = onnx_path
        self.epsilon = 10 ** -work_precision
        self.options = options

        self.layer = self.parse_layer()

    def check_num_robust(self, lbs: list, ubs: list, label: int) -> bool:
        """Procedure to check whether the robustness specification holds using numeric bounds"""

        # Create the matrices of the disjunctions
        out_props = ops.create_disjunction_matrix(len(lbs), label)
        bounds = {
            'lower': lbs,
            'upper': ubs
        }

        # For each disjunction in the output property, check none is satisfied by output_bounds.
        # If one disjunction is satisfied, then it represents a potential counter-example.
        for i in range(len(out_props)):
            if ops.check_unsafe(bounds, out_props[i], self.epsilon):
                return False

        return True

    def check_sym_robust(self, in_lbs: list, in_ubs: list, sym_bounds: dict, label: int) -> bool:
        """Procedure to check whether the robustness specification holds using symbolic bounds"""

        out_props = ops.create_disjunction_matrix(len(sym_bounds['matrix']), label)

        for i in range(len(out_props)):
            if ops.check_unsafe_symbolic(sym_bounds, out_props[i], in_lbs, in_ubs, self.epsilon):
                return False

        return True

    def parse_layer(self) -> LinearIntervalLayer:
        """Procedure to read the first layer of a ONNX network"""
        nn = onnx.to_nn(self.onnx_path, self.epsilon)
        return nn[0]

    def propagate(self, lbs: list, ubs: list) -> tuple:
        """Procedure to compute the numeric interval bounds of a linear layer"""
        weights_plus = ops.get_positive(self.layer.weight)
        weights_minus = ops.get_negative(self.layer.weight)

        low = ops.add(ops.matmul_left(weights_plus, lbs), ops.matmul_left(weights_minus, ubs), self.layer.bias)
        upp = ops.add(ops.matmul_left(weights_plus, ubs), ops.matmul_left(weights_minus, lbs), self.layer.bias)

        return low, upp

    def verify(self, vnnlib_path: str) -> bool:
        # 1: Read VNNLIB bounds
        in_lbs, in_ubs, label = vnnlib.read_vnnlib(vnnlib_path)

        # 2: Get interval input lbs and ubs
        in_lbs = [ops.interval_from_value(v, self.epsilon) for v in in_lbs]
        in_ubs = [ops.interval_from_value(v, self.epsilon) for v in in_ubs]

        # 3: Check output intersection
        bounds = {
            'matrix': self.layer.weight,
            'offset': self.layer.bias
        }
        return self.check_sym_robust(in_lbs, in_ubs, bounds, label)
