import pynever.utilities as utilities
import numpy as np
import pynever.strategies.verification as ver
import pynever.nodes as nodes
import pynever.networks as networks
import time
import logging
from trash.master_thesis.csv_handler.violation_tester import ViolationsManager

if __name__ == "__main__":

    # logger for violation of the bound propagation
    violations_logger = logging.getLogger('pynever/master_thesis/csv_handler/violation_tester')
    fh_1 = logging.FileHandler("test_results/violations.txt")
    violations_logger.setLevel(logging.DEBUG)
    fh_1.setLevel(logging.DEBUG)
    violations_logger.addHandler(fh_1)

    # logger for saving the run time
    time_logger = logging.getLogger('test/ACAS_tests/acas_experiments')
    fh_2 = logging.FileHandler("test_results/time_records.txt")
    violations_logger.setLevel(logging.DEBUG)
    fh_2.setLevel(logging.DEBUG)
    violations_logger.addHandler(fh_2)

    # path test manager
    path_1 = 'test_results/pynever_tests/'
    path_2 = 'test_results/gimelli_tests/'
    path_3 = 'test_results/elena_tests/'

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

    param_set_id = ["Mixed"]

    for i in range(0, len(property_ids)):

        for j in range(len(networks_ids[i])):
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

            heuristic = "best_n_neurons"
            params = [[1000] for _ in range(network.count_relu_layers())]

            net_id = networks_ids[i][j]
            p_id = property_ids[i]

            verifier = ver.NeverVerification(heuristic, params)

            time_start = time.perf_counter()
            # Remember: we are verifying the negation of the original property
            safe = not verifier.verify(network, prop)
            time_end = time.perf_counter()

            time_logger.debug(f"{networks_ids[i][j]}" + "\n" + str(time_end - time_start))

            # stars for each layers
            stars_dict = verifier.stars_dict

            path_1t = path_1 + f"{networks_ids[i][j]}.csv"
            path_2t = path_2 + f"{networks_ids[i][j]}.csv"
            path_3t = path_3 + f"{networks_ids[i][j]}.csv"

            violations_manager = ViolationsManager(path_1t,
                                                   path_2t, path_3t, network, prop, stars_dict)

            error = 0.5
            violations_manager.check(error, True)
