import numpy as np
from pynever import nodes, networks
from pynever.strategies import conversion

# Create a network named 'ex' and with input name 'X'
nn = networks.SequentialNetwork('ex', 'X')

W1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.double)
B1 = np.array([0.0, 0.0], dtype=np.double)
nn.add_node(nodes.FullyConnectedNode('FC_0', (2,), 2, W1, B1))
nn.add_node(nodes.ReLUNode('ReLU_0', (2,)))

W2 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.double)
B2 = np.array([0.0, 0.0], dtype=np.double)
nn.add_node(nodes.FullyConnectedNode("FC_1", (2,), 2, W2, B2))
nn.add_node(nodes.ReLUNode("Y", (2,)))


# Convert network to ONNX
converter = conversion.ONNXConverter()
onnx_net = converter.from_neural_network(nn)
conversion.save_network_path(onnx_net, 'ex.onnx')
