from unittest import TestCase

import pynever.strategies.conversion as conv
from pynever.networks import SequentialNetwork
from pynever.nodes import FullyConnectedNode, ReLUNode
from pynever.strategies.conversion import DirectKerasConverter


class KerasConversionTest(TestCase):
    def setUp(self):
        print('Testing')

    def test_save(self):
        network = SequentialNetwork('ex_net', 'X')
        network.add_node(FullyConnectedNode('FC_0', (2,), 10))
        network.add_node(ReLUNode('ReLU_0', (10,)))
        network.add_node(FullyConnectedNode('FC_1', (10,), 5))
        network.add_node(ReLUNode('ReLU_1', (5,)))
        network.add_node(FullyConnectedNode('FC_2', (5,), 1))

        # Conversion
        tf_net = DirectKerasConverter().from_neural_network(network)
        conv.save_network_path(tf_net, 'converted.h5')

    def test_open(self):
        load_tf_net = conv.load_network_path('vcas_1.h5')
        if isinstance(load_tf_net, conv.KerasNetwork):
            pynever_net = DirectKerasConverter().to_neural_network(load_tf_net)
            print(pynever_net)
