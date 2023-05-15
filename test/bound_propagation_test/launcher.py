import numpy as np
import scipy.spatial
import torch
import logging

from pynever.networks import SequentialNetwork as SeqNetwork
import pynever.nodes as nodes
import pynever.strategies.verification as ver
import pynever.networks as networks
import pynever.strategies.abstraction as abst
import pynever.strategies.conversion as conv
from pynever.strategies.bound_propagation_gimelli.bounds_menager import *
from pynever.strategies.bound_propagation_elena.verification.bounds.boundsmanagerelena import *
from pynever.strategies import smt_reading, verification
from master_thesis.csv_handler.csvhandler import print_to_csv_pynever_bounds
from master_thesis.csv_handler.violation_tester import ViolationsManager
import logging
import os
import time


def getAbstractNetwork(net):
    fully_connected_counter = 0
    relu_counter = 0

    # ABSTRACT SEQUENTIAL NETWORK
    abs_net = abst.AbsSeqNetwork("SmallAbstractNetwork")

    node = net.get_first_node()

    if isinstance(node, nodes.FullyConnectedNode):
        title = "AbsFC_" + str(fully_connected_counter)
        abs_net.add_node(abst.AbsFullyConnectedNode(title, node))
        fully_connected_counter += 1
    elif isinstance(node, nodes.ReLUNode):
        title = "AbsReLU_" + str(relu_counter)
        abs_net.add_node(abst.AbsReLUNode(title, node, "best_n_neurons", [10]))
        relu_counter += 1

    while node is not net.get_last_node():

        node = net.get_next_node(node)

        if isinstance(node, nodes.FullyConnectedNode):
            title = "AbsFC_" + str(fully_connected_counter)
            abs_net.add_node(abst.AbsFullyConnectedNode(title, node))
            fully_connected_counter += 1
        elif isinstance(node, nodes.ReLUNode):
            title = "AbsReLU_" + str(relu_counter)
            abs_net.add_node(abst.AbsReLUNode(title, node, "best_n_neurons", [1000]))
            relu_counter += 1

    # create layers list of abstract nodes
    layers = []

    abs_node = abs_net.get_first_node()
    layers.append(abs_node)

    while abs_node is not abs_net.get_last_node():
        abs_node = abs_net.get_next_node(abs_node)
        layers.append(abs_node)

    return abs_net


def path_manager(fc_dim, absolute_path):
    obj = time.strftime("%H:%M:%S", time.localtime())
    path = absolute_path + pynever_bounds_folder + str(fc_dim) + '.csv'
    path2 = absolute_path + gimelli_bounds_folder + str(fc_dim) + '.csv'
    path3 = absolute_path + elena_bounds_folder + str(fc_dim) + '.csv'
    generic_data_file = absolute_path + 'test_results/generic_data.txt'
    print(path)
    print(path2)
    print(path3)
    return path, path2, path3, generic_data_file


def BigNetwork(network: SeqNetwork, fc_dim, logger=None):
    input_dim = 2
    output_dim = 2
    fc_dim = int(fc_dim)
    weight_matrix_1 = np.random.randn(fc_dim, input_dim)
    weight_matrix_2 = np.random.randn(fc_dim, fc_dim)
    weight_matrix_3 = np.random.randn(input_dim, fc_dim)

    if logger is not None:
        logger.debug("weight_matrix_1:" + str(weight_matrix_1))
        logger.debug("weight_matrix_2:" + str(weight_matrix_2))
        logger.debug("weight_matrix_3:" + str(weight_matrix_3))

    bias_1 = np.zeros(fc_dim)
    bias_2 = np.zeros(fc_dim)
    bias_3 = np.zeros(input_dim)

    network.add_node(
        nodes.FullyConnectedNode("FC_1", (input_dim,), fc_dim, weight_matrix_1, bias_1))

    network.add_node(
        nodes.ReLUNode("ReLU_1", (fc_dim,)))

    network.add_node(
        nodes.FullyConnectedNode("FC_2", (fc_dim,), fc_dim, weight_matrix_2, bias_2))

    network.add_node(
        nodes.ReLUNode("ReLU_2", (fc_dim,)))

    network.add_node(
        nodes.FullyConnectedNode("FC_3", (fc_dim,), output_dim, weight_matrix_3, bias_3))


def verify_network(fc_connected_layers_dim: list, property_path='property.smt2'):
    violations_logger = logging.getLogger('pynever/master_thesis/csv_handler/violation_tester')
    fh = logging.FileHandler("test_results/violations.txt")
    violations_logger.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
    violations_logger.addHandler(fh)

    for fc_dim in fc_connected_layers_dim:
        # path test manager
        path_1 = 'test_results/pynever_tests/first_test.csv'
        path_2 = 'test_results/gimelli_tests/first_test.csv'
        path_3 = 'test_results/elena_tests/first_test.csv'

        # network init
        net = SeqNetwork("network", "IMP")
        BigNetwork(net, fc_dim)

        # property init
        parser = smt_reading.SmtPropertyParser(property_path, 'X', 'Y')
        prop = verification.NeVerProperty(*parser.parse_property())

        # network verification init
        heuristic = "best_n_neurons"
        params = [[1000] for _ in range(net.count_relu_layers())]
        verifier = ver.NeverVerification(heuristic, params)

        # start time
        time_start = time.perf_counter()

        # verify
        safe = not verifier.verify(net, prop)

        # stop timer
        time_end = time.perf_counter()
        delta_time = time_end - time_start

        # write data
        time_results = open("test_results/time_data.txt", 'a')
        time_results.write(str(fc_dim) + '\n' + str(delta_time) + '\n')
        time_results.close()

        violations_logger.debug("Iterazione fc_dim: " + str(fc_dim))

        # stars for each layers
        stars_dict = verifier.stars_dict

        violations_manager = ViolationsManager(path_1,
                                               path_2, path_3, net, prop, stars_dict)

        error = 0.000005
        violations_manager.check(error, True)


if __name__ == '__main__':
    verify_network([7])
