import torch

from pynever import nodes
from pynever.networks import SequentialNetwork
from pynever.strategies.verification.algorithms import SSBPVerification
from pynever.strategies.verification.parameters import SSBPVerificationParameters
from pynever.strategies.verification.properties import VnnLibProperty

# NETWORK DEFINITION
# W = np.array([[1, 1], [1, -1]])
# W2 = np.array([[1, 1], [0, 1]])
# b2 = np.array([1, 0])
W = torch.Tensor([[1, 1], [-1, 1]])
W2 = torch.eye(2, 2)
b2 = torch.zeros(2)

fc_1 = nodes.FullyConnectedNode('FC_1', (2,), 2, W, torch.zeros(2))
rl_1 = nodes.ReLUNode('ReLU_1', (2,))
fc_2 = nodes.FullyConnectedNode('FC_2', (2,), 2, W, torch.zeros(2))
rl_2 = nodes.ReLUNode('ReLU_2', (2,))
fc_3 = nodes.FullyConnectedNode('Y', (2,), 2, W2, b2)

nn = SequentialNetwork('NN', 'X')
nn.append_node(fc_1)
nn.append_node(rl_1)
# nn.append_node(fc_2)
# nn.append_node(rl_2)
# nn.append_node(fc_3)

print(nn)
print(nn.get_topological_order())
print(nn.get_roots())
print(nn.get_leaves())

prop = VnnLibProperty('2d_prop.vnnlib')

print(SSBPVerification(SSBPVerificationParameters()).verify(nn, prop))
