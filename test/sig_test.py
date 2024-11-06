# NETWORK DEFINITION
import numpy as np
import matplotlib.pyplot as plt

from pynever import nodes, tensors, utilities
from pynever.networks import SequentialNetwork
from pynever.strategies.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.bounds_propagation.bounds_manager import BoundsManager
from pynever.strategies.verification.ssbp.constants import BoundsDirection

W = tensors.Tensor(np.array([[1.0, -1.0], [0.0, 1.0]]))
b = tensors.Tensor(np.array([0.0, 0.0]))

fc_1 = nodes.FullyConnectedNode('FC', (2,), 2, W, b)
sig_1 = nodes.SigmoidNode('Sig_0', (2,))
fc_2 = nodes.FullyConnectedNode('FC_1', (2,), 2, W, b)
sig_2 = nodes.SigmoidNode('Sig', (2,))

nn = SequentialNetwork('NN', 'X')
nn.append_node(fc_1)
# nn.append_node(sig_1)
nn.append_node(fc_2)
nn.append_node(sig_2)

N = 1000

x_points = np.random.uniform(-2, 2, N)
y_points = np.random.uniform(-2, 2, N)
lbs = np.array([-2, -2])
ubs = np.array([2, 2])

out_points = []

for i in range(N):
    point = tensors.Tensor(np.array([x_points[i], y_points[i]]))
    out_points.append(utilities.execute_network(nn, point))

plt.scatter(*zip(*out_points), s=10)
plt.xlim([0, 1])
plt.ylim([0, 1])

mgr = BoundsManager(direction=BoundsDirection.FORWARDS)
out_bounds = mgr.compute_bounds(HyperRectangleBounds(lbs, ubs), nn)
print(out_bounds.numeric_post_bounds)

plt.axvline(x=out_bounds.numeric_post_bounds['Sig'].lower[0], color='r', linestyle='-')
plt.axvline(x=out_bounds.numeric_post_bounds['Sig'].upper[0], color='r', linestyle='-')
plt.axhline(y=out_bounds.numeric_post_bounds['Sig'].lower[1], color='r', linestyle='-')
plt.axhline(y=out_bounds.numeric_post_bounds['Sig'].upper[1], color='r', linestyle='-')

plt.show()
