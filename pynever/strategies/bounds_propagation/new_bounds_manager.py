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

        if input_bounds is None:
            self.input_bounds = self.prop.to_numeric_bounds()
        else:
            self.input_bounds = input_bounds

    def init_symb_bounds(self) -> SymbolicLinearBounds:
        # Usa self.input_bounds
        pass

    def get_next_layer(self, layer: LayerNode) -> LayerNode:
        # Usa self.topological_stack
        pass

    def propagate_bounds(self, in_num_bounds: HyperRectangleBounds, in_sym_bounds: SymbolicLinearBounds = None,
                         in_layer: LayerNode = None):

        if in_layer is None:
            in_layer = self.network.get_roots()[0]

            # Dev'essere così per forza - succede solo una volta
            # TODO remove after debugging
            assert in_layer.identifier == self.topological_stack.pop()

            if in_sym_bounds is None:
                in_sym_bounds = self.init_symb_bounds()

            # return self.propagate_bounds(in_num_bounds, in_sym_bounds, next_layer)

        cur_layer = in_layer
        next_layer = self.get_next_layer(in_layer)
        bounds_dict[cur_layer.identifier] = cur_bounds
        parents = self.network.get_parents(cur_layer)

        assert len(self.network.get_children(cur_layer)) == 0 XNOR len(self.topological_stack) == 0

        # get input symbolic bounds for the current layer
        """
        if len(parents) > 1:
            cur_input_bounds = list...
        else:
            cur_input_bounds = bounds...
        """

        # Non mi serve un for figli... perché ho l'ordine topologico.

        # compute the output bounds for the layer
        cur_bounds = compute_bounds(cur_input_bounds, cur_layer)
        bounds_dict[cur_layer.identifier] = cur_bounds

        if len(self.topological_stack) == 0:
            return bounds_dict, cur_bounds.concretize()

        else:
            return self.propagate_bounds(in_num_bounds, in_sym_bounds, next_layer)  # next layer dipende dallo stack
