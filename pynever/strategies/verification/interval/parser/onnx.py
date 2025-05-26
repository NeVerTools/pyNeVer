"""
This module reads ONNX files representing neural networks
"""

import numpy as np
import onnx

from strategies.verification.interval.core import ops
from strategies.verification.interval.core.layer import LinearIntervalLayer


def interval_convert(matrix: np.ndarray, epsilon: float) -> list:
    """Procedure to convert a Tensor to an interval matrix"""
    if len(matrix.shape) < 2:
        matrix = np.expand_dims(matrix, 1)

    result = [[0 for _ in range(matrix.shape[1])] for _ in range(matrix.shape[0])]

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result[i][j] = ops.interval_from_value(matrix[i, j], epsilon)

    return result


def to_nn(onnx_path: str, epsilon: float):
    """Procedure to read a ONNX network as a list of IntervalLayer objects"""

    # Open model
    onnx_net = onnx.load(onnx_path)

    # Read data
    parameters = {}
    for initializer in onnx_net.graph.initializer:
        parameters[initializer.name] = onnx.numpy_helper.to_array(initializer)

    # Create layers
    net = []

    for node in onnx_net.graph.node:
        if node.op_type == 'Gemm':

            weight = parameters[node.input[1]]
            for att in node.attribute:
                if (att.name == 'transA' or att.name == 'transB') and att.i == 0:
                    weight = parameters[node.input[1]].T

            neurons = weight.shape[0]
            weight = interval_convert(weight, epsilon)

            if len(node.input) <= 2:
                bias = interval_convert(np.zeros((neurons, 1)), epsilon)
            else:
                bias = interval_convert(parameters[node.input[2]], epsilon)

            net.append(LinearIntervalLayer(weight, bias))

    assert (len(net)) == 1
    return net
