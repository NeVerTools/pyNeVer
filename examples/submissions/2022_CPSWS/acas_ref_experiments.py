import pynever.utilities as utilities
import numpy as np
import pynever.strategies.verification as ver
import pynever.nodes as nodes
import pynever.networks as networks
import time
import logging


property_ids = ["P3_no_prop", "P4_no_prop"]

unsafe_mats = [[[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]],
               [[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]]]
unsafe_vecs = [[[0], [0], [0], [0]], [[0], [0], [0], [0]]]

input_lb = [[1500, -0.06, 3.1, 980, 960], [1500, -0.06, 3.1, 1000, 700]]
input_ub = [[1800, 0.06, 3.14, 1200, 1000], [1800, 0.06, 3.14, 1200, 800]]

networks_ids = [["1_1", "1_3", "2_3", "4_3", "5_1"], ["1_1", "1_3", "3_2", "4_2"]]

max_iter_search = 50
scale_coeff = 10
iter_change_scale = 5
iter_early_stop = 20
adjustment_rate = 0.5
num_samples = 100
threshold = 1e-5

max_neurons = 1
rel_ref = True
precision = 0.00005

search_params = {"max_iter": max_iter_search, "scale_coeff": scale_coeff, "iter_change_scale": iter_change_scale,
                 "iter_early_stop": iter_early_stop, "adjustment_rate": adjustment_rate, "num_samples": num_samples,
                 "threshold": threshold}

# Loggers and Handler definition

logger_acas_stream = logging.getLogger("pynever.strategies.verification")
logger_acas_file = logging.getLogger("acas_file")

acas_file_handler = logging.FileHandler("logs/ACASXUExperimentLog.txt")
acas_stream_handler = logging.StreamHandler()

acas_file_handler.setLevel(logging.INFO)
acas_stream_handler.setLevel(logging.INFO)

logger_acas_file.addHandler(acas_file_handler)
logger_acas_stream.addHandler(acas_stream_handler)

logger_acas_file.setLevel(logging.INFO)
logger_acas_stream.setLevel(logging.INFO)


if __name__ == '__main__':
    # Begin Experiment
    logger_acas_file.info(f"Dataset,NetworkID,PropertyID,Methodology,Safety,Time\n")
    exp_params = [(True, 1), (False, 6)]
    for exp_p in exp_params:

        only_rel, max_n = exp_p
        if only_rel:
            significance = "RO"
        else:
            significance = "PS"

        for i in range(0, len(property_ids)):

            for j in range(len(networks_ids[i])):

                logger_acas_stream.info(f"Verifying {property_ids[i]} on Network {networks_ids[i][j]}.\n")

                # Loading of the values of interest of the corresponding ACAS XU network.
                weights, biases, inputMeans, inputRanges, outputMean, outputRange = \
                    utilities.parse_nnet(f"nnet/ACASXU_experimental_v2a_{networks_ids[i][j]}.nnet")

                # Creation of the matrixes defining the input set (i.e., in_pred_mat * x <= in_pred_bias).

                # Normalization of the lb and ub.
                norm_input_lb = []
                norm_input_ub = []
                for k in range(len(input_lb[i])):
                    norm_input_lb.append((input_lb[i][k] - inputMeans[k]) / inputRanges[k])
                    norm_input_ub.append((input_ub[i][k] - inputMeans[k]) / inputRanges[k])

                # Matrixes Creation.
                in_pred_mat = []
                in_pred_bias = []
                for k in range(len(norm_input_lb)):
                    lb_constraint = np.zeros(len(norm_input_lb))
                    ub_constraint = np.zeros(len(norm_input_ub))
                    lb_constraint[k] = -1
                    ub_constraint[k] = 1
                    in_pred_mat.append(lb_constraint)
                    in_pred_mat.append(ub_constraint)
                    in_pred_bias.append([-norm_input_lb[k]])
                    in_pred_bias.append([norm_input_ub[k]])

                in_pred_bias = np.array(in_pred_bias)
                in_pred_mat = np.array(in_pred_mat)

                # Creation of the matrixes defining the negation of the wanted property (i.e., unsafe region)
                # (i.e., out_pred_mat * y <= out_pred_bias).
                out_pred_mat = np.array(unsafe_mats[i])
                if property_ids[i] == "Property 1":
                    out_pred_bias = (np.array(unsafe_vecs[i]) - outputMean) / outputRange
                else:
                    out_pred_bias = np.array(unsafe_vecs[i])

                # Construction of our internal representation for the ACAS network.

                network = networks.SequentialNetwork(f"ACAS_XU_{networks_ids[i][j]}", "X")

                for k in range(len(weights)):

                    new_fc_node = nodes.FullyConnectedNode(f"FC_{k}", (weights[k].shape[1],), weights[k].shape[0], weights[k],
                                                           biases[k], True)
                    network.add_node(new_fc_node)

                    if k < len(weights) - 1:
                        new_relu_node = nodes.ReLUNode(f"ReLU_{k}", (weights[k].shape[0],))
                        network.add_node(new_relu_node)

                # Verification of the network of interest for the property of interest
                prop = ver.NeVerProperty(in_pred_mat, in_pred_bias, [out_pred_mat], [out_pred_bias])

                net_id = networks_ids[i][j]
                p_id = property_ids[i]

                logger_acas_stream.info(f"Verification Methodology: CEGAR")

                verifier = ver.NeverVerificationRef(max_n, search_params, precision, only_rel)

                time_start = time.perf_counter()
                safe, counter_example = verifier.verify(network, prop)
                time_end = time.perf_counter()
                logger_acas_file.info(f"ACASXU,{net_id},{property_ids[i]},CEGAR,SIG={significance},MaxN={max_n},{safe},"
                                      f"{time_end - time_start}")