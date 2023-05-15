import argparse
import time
import pynever.strategies.conversion as conv
import pynever.strategies.verification as ver
import pynever.nodes as nodes
from gym import spaces
import numpy as np
import onnx
import logging
import torch
from datetime import datetime
from test.DRONE_tests import shared_constants
from master_thesis.csv_handler.violation_tester import ViolationsManager


# path test manager
path_1 = 'test_results/pynever_tests/'
path_2 = 'test_results/gimelli_tests/'
path_3 = 'test_results/elena_tests/'

in_low_b = np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
in_high_b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
input_space = spaces.Box(low=in_low_b, high=in_high_b, dtype=np.float32)

num_neuron_mixed = 1

if __name__ == "__main__":

    for i in range(7):

        net_path = f"onnx_nets/AC{i + 1}.onnx"
        heuristic = "best_n_neurons"
        epsilon = 0.1

        random_input = input_space.sample()
        tensor_input = torch.from_numpy(random_input)

        net_id = net_path.replace("onnx_nets/", "").replace(".onnx", "")

        onnx_net = conv.ONNXNetwork(net_id, onnx.load(net_path))
        net = conv.ONNXConverter().to_neural_network(onnx_net)
        torch_net = conv.PyTorchConverter().from_neural_network(net).pytorch_network

        numpy_output = torch_net(tensor_input).detach().numpy()

        ver_param = [[1000] for _ in range(net.count_relu_layers())]

        # INPUT CONSTRAINTS DEFINITION

        in_pred_mat = []
        in_pred_bias = []
        data_size = len(random_input)
        for i in range(len(random_input)):

            lb_constraint = np.zeros(data_size)
            ub_constraint = np.zeros(data_size)
            lb_constraint[i] = -1
            ub_constraint[i] = 1
            in_pred_mat.append(lb_constraint)
            in_pred_mat.append(ub_constraint)
            # Errata Corrige: we wrongly used -1 and 1 as upper and lower bounds for all the variables. Now it is correct.
            if random_input[i] - epsilon < in_low_b[i]:
                in_pred_bias.append([-in_low_b[i]])
            else:
                in_pred_bias.append([-(random_input[i] - epsilon)])

            if random_input[i] + epsilon > in_high_b[i]:
                in_pred_bias.append([in_high_b[i]])
            else:
                in_pred_bias.append([random_input[i] + epsilon])

        in_pred_bias = np.array(in_pred_bias)
        in_pred_mat = np.array(in_pred_mat)

        in_prop = ver.NeVerProperty(in_pred_mat, in_pred_bias, [], [])

        verifier = ver.NeverVerification(heuristic="best_n_neurons", params=ver_param)
        safe = not verifier.verify(net, in_prop)

        # stars for each layers
        stars_dict = verifier.stars_dict

        path_1 = path_1 + f"AC{i + 1}.csv"
        path_2 = path_2 + f"AC{i + 1}.csv"
        path_3 = path_3 + f"AC{i + 1}.csv"

        violations_manager = ViolationsManager(path_1,
                                               path_2, path_3, net, in_prop, stars_dict)

        error = 0.0000001
        violations_manager.check(error, True)
