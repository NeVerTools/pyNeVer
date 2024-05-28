import numpy as np

from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies import verification

# NETWORK DEFINITION
# W = np.ones((2, 2))
# W[1, 1] = -1
W = np.array([[0.6, -0.4], [0.25, 0.75]])

fc_1 = nodes.FullyConnectedNode('FC_1', (2,), 2, W, np.zeros(2))
rl_1 = nodes.ReLUNode('ReLU_1', (2,))
fc_2 = nodes.FullyConnectedNode('FC_2', (2,), 2, W, np.zeros(2))
rl_2 = nodes.ReLUNode('ReLU_2', (2,))
fc_3 = nodes.FullyConnectedNode('Y', (2,), 2, np.eye(2, 2), np.zeros(2))

nn = SequentialNetwork('NN', 'X')
nn.add_node(fc_1)
nn.add_node(rl_1)
nn.add_node(fc_2)
nn.add_node(rl_2)
nn.add_node(fc_3)

prop = verification.NeVerProperty()
prop.from_smt_file('2d_prop.vnnlib')

print(verification.SearchVerification().verify(nn, prop))
