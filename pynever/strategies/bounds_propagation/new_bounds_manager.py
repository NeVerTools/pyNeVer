"""
This module controls the bounds propagation for neural networks

"""
from pynever.networks import NeuralNetwork
from pynever.nodes import LayerNode
from pynever.strategies.bounds_propagation.bounds import HyperRectangleBounds, SymbolicLinearBounds
from pynever.strategies.verification.properties import NeverProperty


class NewBoundsManager:
    def __init__(self, network: NeuralNetwork, prop: NeverProperty = None, input_bounds: HyperRectangleBounds = None):
        if prop is None and input_bounds is None:
            raise Exception('Please initialize with either a property or input bounds')

        self.network = network
        self.topological_stack = self.network.get_topological_order()
        self.prop = prop
        self.input_bounds = input_bounds


    def propagate_bounds(self, in_num_bounds: HyperRectangleBounds,
                         in_sym_bounds: SymbolicLinearBounds = None,
                         in_layer: LayerNode = None) -> tuple[LayerNode, SymbolicLinearBounds]:
        if in_layer is None:
            in_layer = self.network.get_roots()[0]
            in_sym_bounds = self.init_symb_bounds(in_num_bounds)  # Already in init?
            bounds_dict[cur_layer.identifier] = cur_bounds
            next_layer = self.get_next_layer(in_layer)  # Usa la topological sort

            return self.propagate_bounds(in_num_bounds, in_sym_bounds, next_layer)

        cur_layer = in_layer

        next_layers = self.network.get_children(cur_layer)
        parents = self.network.get_parents(cur_layer)

        # get input symbolic bounds for the current layer
        ...

        # compute the output bounds for the layer
        cur_bounds = compute_bounds(in_sym_bounds, cur_layer)
        bounds_dict[cur_layer.identifier] = cur_bounds

        if len(self.topological_stack) == 0:
            return bounds_dict, cur_bounds.concretize()

        else:
            return self.propagate_bounds(in_num_bounds, in_sym_bounds, next_layer)  # ?
