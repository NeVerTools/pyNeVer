from pynever.networks import SequentialNetwork
from pynever.strategies import conversion, smt_reading, verification
from pynever.strategies.bp.utils.property_converter import PropertyFormatConverter

import numpy as np
import torch

np.random.seed(0)


def load_network(network_path: str) -> SequentialNetwork:
    # Load the network
    # TODO better error handling
    nn = conversion.load_network_path(network_path)
    if not isinstance(nn, conversion.ONNXNetwork):
        raise Exception

    nn = conversion.ONNXConverter().to_neural_network(nn)
    if not isinstance(nn, SequentialNetwork):
        raise Exception

    return nn


def sampling(network: SequentialNetwork, property_path: str, n_points: int = 1000, subset_percentage: float = 0.1):
    in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias = smt_reading. \
        SmtPropertyParser(property_path, "X", "Y").parse_property()

    prop = verification.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)

    bounds = PropertyFormatConverter(prop).get_vectors()
    lower_bounds, upper_bounds = bounds.get_lower(), bounds.get_upper()
    sample_points = []

    starting_point = np.array([np.random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))])
    sample_points.append(starting_point)
    for i in range(1, n_points):
        v = np.random.rand(starting_point.size)
        new_direction = v / np.linalg.norm(v)
        max_l = min([(upper_bounds[i] - starting_point[i]) / new_direction[i] for i in range(starting_point.size)])
        min_l = max([(lower_bounds[i] - starting_point[i]) / new_direction[i] for i in range(starting_point.size)])
        scale = np.random.uniform(low=min_l, high=max_l)
        new_point = starting_point + scale * new_direction
        sample_points.append(new_point)
        starting_point = new_point

    # Check validity of all inputs
    for p in sample_points:
        if any(np.greater(prop.in_coef_mat @ p, prop.in_bias_mat.flatten())):
            print(f"{p} is an invalid sample point")
            raise Exception

    # Calculate outputs
    pytorch_nn = conversion.PyTorchConverter().from_neural_network(network)

    sample_outputs = []
    for p in sample_points:
        sample_outputs.append(pytorch_nn.pytorch_network(torch.from_numpy(p.astype(dtype=np.float32))).detach().numpy())

    for idx, o in enumerate(sample_outputs):
        in_safe_region = True
        for C, d in zip(prop.out_coef_mat, prop.out_bias_mat):
            if np.less_equal(C @ o, d.flatten()).all():
                in_safe_region = False
                break
        if not in_safe_region:
            print(f"output {o} of input {sample_points[idx]} is not in the safe region")
            return None

    nearest_points = []
    for C, d in zip(prop.out_coef_mat, prop.out_bias_mat):
        min_distances = []
        for _ in C:
            min_distances.append([])
        for idx, o in enumerate(sample_outputs):
            distances = [float(abs(w @ o - b) / np.linalg.norm(w)) for w, b in zip(C, d)]
            for index, distance in enumerate(distances):
                min_distances[index].append((distance, idx))
        for hp in min_distances:
            hp.sort()
            nearest_points.append(hp[:int(subset_percentage * n_points)])

    avg_points = []
    for hp in nearest_points:
        points = []
        distances = []
        for distance, idx in hp:
            points.append(sample_points[idx])
            distances.append(distance)
        avg_point = np.average(points, axis=0)
        avg_distance = np.average(distances)
        avg_points.append((avg_distance, avg_point))

    avg_points.sort()
    return avg_points


if __name__ == "__main__":
    starting_points = sampling(load_network("networks/cartpole.onnx"),
                               "properties/cartpole/cartpole_case_safe_30.vnnlib")
    print(starting_points)
