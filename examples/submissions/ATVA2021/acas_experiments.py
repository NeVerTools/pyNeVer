import logging
import time

import numpy as np

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.verification as ver
import pynever.utilities as utilities

if __name__ == "__main__":

    property_ids = ["P3_no_prop", "P4_no_prop"]
    # property_ids = ["P3"]
    # It should be noted that we are checking the negation of the property of interest: in the case of P3
    # we are verifying if there exist a point for which the COC is indeed minimal
    # (see https://arxiv.org/pdf/1702.01135.pdf for details on the properties)
    unsafe_mats = [[[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]],
                   [[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]]]
    unsafe_vecs = [[[0], [0], [0], [0]], [[0], [0], [0], [0]]]

    input_lb = [[1500, -0.06, 3.1, 980, 960], [1500, -0.06, 3.1, 1000, 700]]
    input_ub = [[1800, 0.06, 3.14, 1200, 1000], [1800, 0.06, 3.14, 1200, 800]]

    networks_ids = [["1_1", "1_3", "2_3", "4_3", "5_1"], ["1_1", "1_3", "3_2", "4_2"]]

    # networks_ids = [["1_1"]]

    # verification_parameters = [[False, 0, False, 0], [True, 1, False, 0], [False, 0.1, False, 0]]
    # param_set_id = ["Over-Approx", "Complete", "Mixed"]

    """verification_parameters = [["given_flags", [[False for i in range(50)]]], ["best_n_neurons", [1]],
                               ["given_flags", [[True for i in range(50)]]]]"""
    """verification_parameters = [["best_n_neurons", [[0], [0], [0], [0], [0], [0]]],
                               ["best_n_neurons", [[1], [1], [1], [1], [1], [1]]],
                               ["best_n_neurons", [[100], [100], [100], [100], [100], [100]]]]"""

    """verification_parameters = [["best_n_neurons", [[0], [0], [0], [0], [0], [0]]],
                               ["best_n_neurons", [[1], [1], [1], [1], [1], [1]]]]
    
    param_set_id = ["Over-Approx", "Mixed", "Complete"]"""

    verification_parameters = [["mixed", [1]]]
    param_set_id = ["Mixed"]

    # Loggers and Handler definition

    logger_empty = logging.getLogger("pynever.strategies.abstraction.empty_times")
    logger_lp = logging.getLogger("pynever.strategies.abstraction.lp_times")
    logger_lb = logging.getLogger("pynever.strategies.abstraction.lb_times")
    logger_ub = logging.getLogger("pynever.strategies.abstraction.ub_times")
    logger_acas_stream = logging.getLogger("pynever.strategies.verification")
    logger_acas_file = logging.getLogger("acas_file")

    empty_handler = logging.FileHandler("logs/empty_times.txt")
    lp_handler = logging.FileHandler("logs/lp_times.txt")
    lb_handler = logging.FileHandler("logs/lb_times.txt")
    ub_handler = logging.FileHandler("logs/ub_times.txt")
    acas_file_handler = logging.FileHandler("logs/ACASXUExperimentLog.txt")
    acas_stream_handler = logging.StreamHandler()

    acas_file_handler.setLevel(logging.INFO)
    acas_stream_handler.setLevel(logging.INFO)
    empty_handler.setLevel(logging.DEBUG)
    lp_handler.setLevel(logging.DEBUG)
    lb_handler.setLevel(logging.DEBUG)
    ub_handler.setLevel(logging.DEBUG)

    logger_empty.addHandler(empty_handler)
    logger_lp.addHandler(lp_handler)
    logger_ub.addHandler(ub_handler)
    logger_lb.addHandler(lb_handler)
    logger_acas_file.addHandler(acas_file_handler)
    logger_acas_stream.addHandler(acas_stream_handler)

    logger_empty.setLevel(logging.DEBUG)
    logger_lp.setLevel(logging.DEBUG)
    logger_ub.setLevel(logging.DEBUG)
    logger_lb.setLevel(logging.DEBUG)
    logger_acas_file.setLevel(logging.INFO)
    logger_acas_stream.setLevel(logging.INFO)

    # Begin Experiment
    logger_acas_file.info(f"Dataset,NetworkID,PropertyID,Methodology,Safety,Time\n")

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

                new_fc_node = nodes.FullyConnectedNode(f"FC_{k}", (weights[k].shape[1],), weights[k].shape[0],
                                                       weights[k],
                                                       biases[k], True)
                network.add_node(new_fc_node)

                if k < len(weights) - 1:
                    new_relu_node = nodes.ReLUNode(f"ReLU_{k}", (weights[k].shape[0],))
                    network.add_node(new_relu_node)

            # Verification of the network of interest for the property of interest
            prop = ver.NeVerProperty(in_pred_mat, in_pred_bias, [out_pred_mat], [out_pred_bias])

            for k in range(len(verification_parameters)):
                heuristic = verification_parameters[k][0]
                params = verification_parameters[k][1]
                net_id = networks_ids[i][j]
                p_id = property_ids[i]

                logger_acas_stream.info(f"Verification Methodology: {param_set_id[k]}")
                logger_empty.debug(f"\nACASXU_{net_id}_P={property_ids[i]}_{param_set_id[k]}\n")
                logger_lp.debug(f"\nACASXU_{net_id}_P={property_ids[i]}_{param_set_id[k]}\n")
                logger_lb.debug(f"\nACASXU_{net_id}_P={property_ids[i]}_{param_set_id[k]}\n")
                logger_ub.debug(f"\nACASXU_{net_id}_P={property_ids[i]}_{param_set_id[k]}\n")

                verifier = ver.NeverVerification(heuristic, params)

                time_start = time.perf_counter()
                # Remember: we are verifying the negation of the original property
                safe = not verifier.verify(network, prop)
                time_end = time.perf_counter()
                logger_acas_file.info(f"ACASXU,{net_id},{property_ids[i]},{param_set_id[k]},{safe},"
                                      f"{time_end - time_start}")
