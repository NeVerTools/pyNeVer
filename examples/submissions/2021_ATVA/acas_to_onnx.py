import numpy as np
import onnx
import torch

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.conversion as conv
import pynever.strategies.verification as ver
import pynever.utilities as utilities

input_lb = [[1500, -0.06, 3.1, 980, 960], [1500, -0.06, 3.1, 1000, 700]]
input_ub = [[1800, 0.06, 3.14, 1200, 1000], [1800, 0.06, 3.14, 1200, 800]]

unsafe_mats = [[[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]],
               [[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]]]

unsafe_vecs = [[[0], [0], [0], [0]], [[0], [0], [0], [0]]]

p_id = ["P3", "P4"]

prop_set = False

for i in range(1, 6):
    for j in range(1, 10):

        weights, biases, inputMeans, inputRanges, outputMean, outputRange = \
            utilities.parse_nnet(f"nnet/ACASXU_experimental_v2a_{i}_{j}.nnet")

        network = networks.SequentialNetwork(f"ACAS_XU_{i}_{j}", "X")

        for k in range(len(weights)):

            new_fc_node = nodes.FullyConnectedNode(f"FC{k}", (weights[k].shape[1],), weights[k].shape[0], weights[k],
                                                   biases[k], True)
            network.add_node(new_fc_node)

            if k < len(weights) - 1:
                new_relu_node = nodes.ReLUNode(f"ReLU{k}", (weights[k].shape[0],))
                network.add_node(new_relu_node)

        if not prop_set:

            for k in range(len(input_lb)):

                # Normalization of the lb and ub.
                norm_input_lb = []
                norm_input_ub = []
                for m in range(len(input_lb[k])):
                    norm_input_lb.append((input_lb[k][m] - inputMeans[m]) / inputRanges[m])
                    norm_input_ub.append((input_ub[k][m] - inputMeans[m]) / inputRanges[m])

                # Matrixes Creation.
                in_pred_mat = []
                in_pred_bias = []
                for m in range(len(norm_input_lb)):
                    lb_constraint = np.zeros(len(norm_input_lb))
                    ub_constraint = np.zeros(len(norm_input_ub))
                    lb_constraint[m] = -1
                    ub_constraint[m] = 1
                    in_pred_mat.append(lb_constraint)
                    in_pred_mat.append(ub_constraint)
                    in_pred_bias.append([-norm_input_lb[m]])
                    in_pred_bias.append([norm_input_ub[m]])

                in_pred_bias = np.array(in_pred_bias)
                in_pred_mat = np.array(in_pred_mat)
                out_pred_mat = np.array(unsafe_mats[k])
                out_pred_bias = np.array(unsafe_vecs[k])

                prop = ver.NeVerProperty(in_pred_mat, in_pred_bias, [out_pred_mat], [out_pred_bias])

                input_prefix = network.input_id
                output_prefix = network.get_last_node().identifier
                prop.to_smt_file(input_prefix, output_prefix, f"smt_property/SMT_{p_id[k]}.smt2")

                p2 = ver.NeVerProperty()
                p2.from_smt_file(f"smt_property/SMT_{p_id[k]}.smt2", input_prefix, output_prefix)

            prop_set = True

        onnx_net = conv.ONNXConverter().from_neural_network(network)
        pytorch_net = conv.PyTorchConverter().from_neural_network(network)
        torch.save(pytorch_net.pytorch_network, f"acas_pytorch/{network.identifier}.pt")
        onnx.save(onnx_net.onnx_network, f"acas_onnx/{network.identifier}.onnx")
