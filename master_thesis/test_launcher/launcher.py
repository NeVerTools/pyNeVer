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

DEBUG = True
fc_bounds_path = '/Users/andrea/Documents/PYCHARM/pyNeVer/master_thesis/test_launcher/test_results/pynever_bounds.txt'

pynever_bounds_folder = 'test_results/pynever_results/ts'
gimelli_bounds_folder = 'test_results/elena_results/ts'
elena_bounds_folder = 'test_results/gimelli_results/ts'


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


def verify_network(fc_connected_layers_dim: list, property_path, absolute_path):
    violations_logger = logging.getLogger('pynever/master_thesis/csv_handler/violation_tester')
    print(absolute_path)
    fh = logging.FileHandler(absolute_path + "test_results/violations.txt")
    violations_logger.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
    violations_logger.addHandler(fh)

    for fc_dim in fc_connected_layers_dim:
        try:
            os.remove(absolute_path+'test_results/pynever_bounds.txt')
        except:
            pass

        # path test manager
        path, path2, path3, generic_data_file = path_manager(fc_dim, absolute_path)

        net = SeqNetwork("SmallNetwork", "IMP")
        BigNetwork(net, fc_dim)
        parser = smt_reading.SmtPropertyParser(property_path, 'X', 'Y')
        prop = verification.NeVerProperty(*parser.parse_property())

        bound_manager_gimelli = MyBoundsManager(getAbstractNetwork(net), prop)
        bound_manager_gimelli.compute_bounds()

        bound_manager_elena = BoundsManagerElena(getAbstractNetwork(net), prop)
        bound_manager_elena.compute_bounds()

        heuristic = "best_n_neurons"
        params = [[1000] for _ in range(20)]
        verifier = ver.NeverVerification(heuristic, params)

        # start time
        time_start = time.perf_counter()

        # verify
        safe = not verifier.verify(net, prop, 3, path2, path3)

        #print(verifier.get_output_starset(net, prop))
        # stop timer
        time_end = time.perf_counter()
        delta_time = time_end - time_start

        # write data
        time_results = open(generic_data_file, 'a')
        time_results.write(str(fc_dim) + '\n' + str(delta_time) + '\n')
        time_results.close()

        print_to_csv_pynever_bounds(fc_bounds_path, path)

        violations_logger.debug("Iterazione fc_dim: " + str(fc_dim))

        violations_manager = ViolationsManager(path,
                                               path2, path3)
        violations_manager.check()
