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

pynever_bounds_folder = 'test_results/pynever_results/'
gimelli_bounds_folder = 'test_results/elena_results/'
elena_bounds_folder = 'test_results/gimelli_results/'
obj = time.strftime("%H:%M:%S", time.localtime())
time = str(obj)
path = pynever_bounds_folder + time + '.csv'
path2 = gimelli_bounds_folder + time + '.csv'
path3 = elena_bounds_folder + time + '.csv'


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


def BigFullyConnectedLayers(network: SeqNetwork, logger=None):
    fc_dim = 30
    weight_matrix_1 = np.random.randn(fc_dim, 2)
    weight_matrix_2 = np.random.randn(fc_dim, fc_dim)
    weight_matrix_3 = np.random.randn(2, fc_dim)

    if logger is not None:
        logger.debug("weight_matrix_1:" + str(weight_matrix_1))
        logger.debug("weight_matrix_2:" + str(weight_matrix_2))
        logger.debug("weight_matrix_3:" + str(weight_matrix_3))

    bias_1 = np.zeros(fc_dim)
    bias_2 = np.zeros(fc_dim)
    bias_3 = np.zeros(2)

    network.add_node(
        nodes.FullyConnectedNode("FC_1", (2,), fc_dim, weight_matrix_1, bias_1))

    network.add_node(
        nodes.FullyConnectedNode("FC_2", (fc_dim,), fc_dim, weight_matrix_2, bias_2))

    network.add_node(
        nodes.FullyConnectedNode("FC_3", (fc_dim,), 2, weight_matrix_3, bias_3))


def BigNetwork(network: SeqNetwork, fcdim, logger=None):
    input_dim = 2
    fc_dim = fcdim
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
        nodes.FullyConnectedNode("FC_3", (fc_dim,), input_dim, weight_matrix_3, bias_3))


def FiveInputNetwork(network: SeqNetwork):
    weight_matrix_1 = np.ones((2, 2))
    weight_matrix_2 = np.ones((2, 2))
    weight_matrix_3 = np.ones((2, 2))

    weight_matrix_1[1, 1] = -1
    weight_matrix_2[1, 1] = -1
    weight_matrix_3[1, 0] = 0

    bias_1 = np.zeros(2)
    bias_2 = np.zeros(2)
    bias_3 = np.array([1, 0])

    network.add_node(
        nodes.FullyConnectedNode("FC_1", (2,), 2, weight_matrix_1, bias_1))

    network.add_node(
        nodes.ReLUNode("ReLU_1", (2,)))

    network.add_node(
        nodes.FullyConnectedNode("FC_2", (2,), 2, weight_matrix_2, bias_2))

    network.add_node(
        nodes.ReLUNode("ReLU_2", (2,)))

    network.add_node(
        nodes.FullyConnectedNode("FC_3", (2,), 2, weight_matrix_3, bias_3))


def FiveInputNetwork2(network: SeqNetwork):
    weight_matrix_1 = np.ones((2, 2))
    weight_matrix_2 = np.ones((2, 2))
    weight_matrix_3 = np.ones((2, 2))

    weight_matrix_1[1, 1] = -1
    weight_matrix_2[1, 1] = -1
    weight_matrix_3[0, 0] = -1

    bias_1 = np.zeros(2)
    bias_2 = np.array([-1 / 2, 0])
    bias_3 = np.array([3, 0])

    network.add_node(
        nodes.FullyConnectedNode("FC_1", (2,), 2, weight_matrix_1, bias_1))

    network.add_node(
        nodes.ReLUNode("ReLU_1", (2,)))

    network.add_node(
        nodes.FullyConnectedNode("FC_2", (2,), 2, weight_matrix_2, bias_2))

    network.add_node(
        nodes.ReLUNode("ReLU_2", (2,)))

    network.add_node(
        nodes.FullyConnectedNode("FC_3", (2,), 2, weight_matrix_3, bias_3))


if __name__ == '__main__':
    # # logger
    # logger = logging.getLogger("bound_propagation.bounds_menager")
    # # os.remove("../logs/numeric_bounds_mine.txt")
    # fh = logging.FileHandler("../logs/numeric_bounds_mine.txt")
    # logger.setLevel(logging.DEBUG)
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    try:
        os.remove('test_results/pynever_bounds.txt')
    except:
        pass

    # clear all file in test_results
    fc_dim = 7

    net = SeqNetwork("SmallNetwork", "IMP")
    BigNetwork(net, fc_dim)
    property_path = "property.smt2"
    parser = smt_reading.SmtPropertyParser(property_path, 'X', 'Y')
    prop = verification.NeVerProperty(*parser.parse_property())

    bound_manager_gimelli = MyBoundsManager(getAbstractNetwork(net), prop)
    bound_manager_gimelli.compute_bounds()

    bound_manager_elena = BoundsManagerElena(getAbstractNetwork(net), prop)
    bound_manager_elena.compute_bounds()

    heuristic = "best_n_neurons"
    params = [[1000] for _ in range(20)]
    verifier = ver.NeverVerification(heuristic, params)
    safe = not verifier.verify(net, prop, 3, path2, path3)

    print(safe)

    print_to_csv_pynever_bounds('test_results/pynever_bounds.txt', path)

    violations_manager = ViolationsManager(path,
                                           path2, path3)
    violations_manager.check()
