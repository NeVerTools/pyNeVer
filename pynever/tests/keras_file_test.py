from unittest import TestCase

import pynever.strategies.conversion as conv
from pynever.networks import SequentialNetwork
from pynever.nodes import FullyConnectedNode, ReLUNode


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
        tf_net = conv.TensorflowConverter().from_neural_network(network)
        conv.save_network_path(tf_net, 'converted.h5')

    def test_open(self):
        load_tf_net = conv.load_network_path('nets/vcas_1.h5')
        if isinstance(load_tf_net, conv.TensorflowNetwork):
            pynever_net = conv.TensorflowConverter().to_neural_network(load_tf_net)
            conv.save_network_path(conv.ONNXConverter().from_neural_network(pynever_net), 'vcas-converted.onnx')

    def test_onnx2keras2pt(self):
        onnx_net = conv.load_network_path('nets/AC4.onnx')

        assert isinstance(onnx_net, conv.ONNXNetwork)

        pynever_net = conv.ONNXConverter().to_neural_network(onnx_net)
        tf_net = conv.TensorflowConverter().from_neural_network(pynever_net)

        assert isinstance(tf_net, conv.TensorflowNetwork)

        conv.save_network_path(tf_net, f'{tf_net.identifier}-converted.h5')

        pynever_net = conv.TensorflowConverter().to_neural_network(tf_net)
        pytorch_net = conv.PyTorchConverter().from_neural_network(pynever_net)

        assert isinstance(pytorch_net, conv.PyTorchNetwork)

        conv.save_network_path(pytorch_net, f'{pytorch_net.identifier}-pt.pth')
