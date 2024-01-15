import pandas as pd

from pynever import networks, nodes
from pynever.strategies import abstraction

# Read weights from file (w25 = 0)
df = pd.read_excel('data/SI_weights.xlsx')
weights = df['linear_regression_weights']

# Create FC layer
nn = networks.SequentialNetwork('movendo_net', 'X')
fc_1 = nodes.FullyConnectedNode('FC_1', (31,), 1, weights.to_numpy().reshape((1, 31)))
nn.add_node(fc_1)

# Create logistic layer
nn.add_node(nodes.SigmoidNode('SIG_1', (1,)))

print(nn)

# Create abstract network
abst_nn = abstraction.AbsSeqNetwork('movendo_abs_net')
abst_nn.add_node(abstraction.AbsFullyConnectedNode('ABS_FC_1', fc_1))
abst_nn.add_node(abstraction.AbsSigmoidNode('ABS_SIG_1', nodes.SigmoidNode('SIG', (1,)), B=3, M=0.42))

print(abst_nn)

# Create input star (center in normal values)
