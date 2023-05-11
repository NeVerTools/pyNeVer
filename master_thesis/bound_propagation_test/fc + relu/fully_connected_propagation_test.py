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
from bound_propagation.bounds_menager import *
from pynever.strategies import smt_reading, verification


def OnlyFullyConnectedLayers(network: SeqNetwork):
    fc_dim = 5
    weight_matrix_1 = np.ones((fc_dim, 2))
    weight_matrix_2 = np.ones((fc_dim, fc_dim))
    weight_matrix_3 = np.ones((2, fc_dim))

    weight_matrix_1[1, 1] = -1
    weight_matrix_2[1, 1] = -1
    weight_matrix_3[1, 0] = 0

    bias_1 = np.zeros(fc_dim)
    bias_2 = np.zeros(fc_dim)
    bias_3 = np.zeros(2)

    network.add_node(
        nodes.FullyConnectedNode("FC_1", (2,), fc_dim, weight_matrix_1, bias_1))

    network.add_node(
        nodes.FullyConnectedNode("FC_2", (fc_dim,), fc_dim, weight_matrix_2, bias_2))

    network.add_node(
        nodes.FullyConnectedNode("FC_3", (fc_dim,), 2, weight_matrix_3, bias_3))


def BigFullyConnectedLayers(network: SeqNetwork, logger=None):
    fc_dim = 15
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


def BaseInputNetwork(net: SeqNetwork):
    weight_matrix_1 = np.array(([2, 3], [1, 1]))
    weight_matrix_2 = np.array([[1, -1]])
    # print(weight_matrix_2.shape)

    bias_1 = np.array([0, 1])
    bias_2 = np.zeros(1)

    net.add_node(
        nodes.FullyConnectedNode("FC_1", (2,), 2, weight_matrix_1, bias_1))

    net.add_node(
        nodes.FullyConnectedNode("FC_2", (2,), 1, weight_matrix_2, bias_2))


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
        abs_net.add_node(abst.AbsReLUNode(title, node, "best_n_neurons", [10000]))
        relu_counter += 1

    while node is not net.get_last_node():

        node = net.get_next_node(node)

        if isinstance(node, nodes.FullyConnectedNode):
            title = "AbsFC_" + str(fully_connected_counter)
            abs_net.add_node(abst.AbsFullyConnectedNode(title, node))
            fully_connected_counter += 1
        elif isinstance(node, nodes.ReLUNode):
            title = "AbsReLU_" + str(relu_counter)
            abs_net.add_node(abst.AbsReLUNode(title, node, "best_n_neurons", [10]))
            relu_counter += 1

    # create layers list of abstract nodes
    layers = []

    abs_node = abs_net.get_first_node()
    layers.append(abs_node)

    while abs_node is not abs_net.get_last_node():
        abs_node = abs_net.get_next_node(abs_node)
        layers.append(abs_node)

    return abs_net


if __name__ == '__main__':
    net = SeqNetwork("SmallNetwork", "IMP")
    BigFullyConnectedLayers(net)
    path = "../../property.smt2"
    parser = smt_reading.SmtPropertyParser(path, 'X', 'Y')
    prop = verification.NeVerProperty(*parser.parse_property())

    bound_manager = BoundsManager(getAbstractNetwork(net), prop)
    bound_manager.compute_bounds()
    print(bound_manager)
