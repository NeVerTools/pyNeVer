# NETWORK DEFINITION
import matplotlib.pyplot as plt
import numpy as np

from pynever import nodes, tensors, utilities
from pynever.networks import SequentialNetwork
from pynever.strategies.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.bounds_propagation.bounds_manager import BoundsManager
from pynever.strategies.verification.ssbp.constants import BoundsDirection


def plot_output_bounds(input_bounds: HyperRectangleBounds, network: SequentialNetwork, samples: int = 1000):
    # Sample and propagate concrete points
    x_points = np.random.uniform(input_bounds.get_lower()[0], input_bounds.get_upper()[0], samples)
    y_points = np.random.uniform(input_bounds.get_lower()[1], input_bounds.get_upper()[1], samples)

    out_points = []

    for i in range(samples):
        point = tensors.Tensor(np.array([x_points[i], y_points[i]]))
        out_points.append(utilities.execute_network(network, point))

    # Plot the concrete outputs
    plt.scatter(*zip(*out_points), s=10)

    # Propagate the bounds
    mgr = BoundsManager(direction=BoundsDirection.FORWARDS)
    out_bounds = mgr.compute_bounds(input_bounds, network)

    # Plot the output numeric bounds
    last_layer_id = network.get_last_node().identifier
    plt.axvline(x=out_bounds.numeric_post_bounds[last_layer_id].lower[0], color='r', linestyle='-')
    plt.axvline(x=out_bounds.numeric_post_bounds[last_layer_id].upper[0], color='r', linestyle='-')
    plt.axhline(y=out_bounds.numeric_post_bounds[last_layer_id].lower[1], color='r', linestyle='-')
    plt.axhline(y=out_bounds.numeric_post_bounds[last_layer_id].upper[1], color='r', linestyle='-')

    plt.show()


W = tensors.Tensor(np.array([[1.0, 1.0], [-1.0, 1.0]]))
b = tensors.Tensor(np.array([1.0, 1.0]))

fc_1 = nodes.FullyConnectedNode('FC', (2,), 2, W, b)
sig_1 = nodes.SigmoidNode('Sig_0', (2,))
fc_2 = nodes.FullyConnectedNode('FC_1', (2,), 2, W, b)
sig_2 = nodes.SigmoidNode('Sig', (2,))
relu1 = nodes.ReLUNode('ReLU', (2,))
relu2 = nodes.ReLUNode('ReLU2', (2,))

nn = SequentialNetwork('NN', 'X')
nn.append_node(fc_1)
nn.append_node(sig_1)
nn.append_node(fc_2)
nn.append_node(sig_2)

plot_output_bounds(HyperRectangleBounds(np.array([-1, -1]), np.array([1, 1])), nn)
