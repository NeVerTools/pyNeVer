"""
This module controls the bounds propagation for neural networks

"""
from pynever.networks import NeuralNetwork
from pynever.nodes import LayerNode
from pynever.strategies.bounds_propagation.bounds import HyperRectangleBounds, SymbolicLinearBounds, VerboseBounds
from pynever.strategies.verification.properties import NeverProperty
from pynever.utilities import xnor


class NewBoundsManager:
    def __init__(self, network: NeuralNetwork, prop: NeverProperty = None, input_bounds: HyperRectangleBounds = None):
        if prop is None and input_bounds is None:
            raise Exception('Please initialize with either a property or input bounds')

        self.network: NeuralNetwork = network
        self.topological_stack: list[str] = self.network.get_topological_order()
        self.prop: NeverProperty = prop

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

    def compute_bounds(self, layer: LayerNode, symbolic_bounds: SymbolicLinearBounds | list[SymbolicLinearBounds],
                       numeric_bounds: HyperRectangleBounds | list[HyperRectangleBounds]) -> tuple[
        SymbolicLinearBounds, HyperRectangleBounds]:

        # Next layer data
        next_layer = self.get_next_layer(layer)

        # Parent data
        parents = self.network.get_parents(layer)

        raise NotImplementedError

    def propagate_bounds(self, in_num_bounds: HyperRectangleBounds | list[HyperRectangleBounds],
                         in_sym_bounds: SymbolicLinearBounds | list[SymbolicLinearBounds] | None,
                         in_layer: LayerNode = None):

        if in_layer is None:
            in_layer = self.network.get_roots()[0]

            # Dev'essere così per forza - succede solo una volta
            # TODO remove after debugging
            assert in_layer.identifier == self.topological_stack.pop()

            if in_sym_bounds is None:
                in_sym_bounds = self.init_symb_bounds()

        # Initialize the bounds data structure
        bounds_dict = VerboseBounds()

        # Current layer data
        cur_layer = in_layer
        cur_sym_bounds = in_sym_bounds
        cur_num_bounds = in_num_bounds

        # TODO remove after debugging
        assert xnor(len(self.network.get_children(cur_layer)) == 0, len(self.topological_stack) == 0)

        # Compute bounds for this layer
        out_sym_bounds, out_num_bounds = self.compute_bounds(cur_layer, cur_sym_bounds, cur_num_bounds)

        # Fill the bounds dictionary for this layer
        bounds_dict.numeric_pre_bounds[cur_layer.identifier] = cur_num_bounds
        bounds_dict.symbolic_bounds[cur_layer.identifier] = out_sym_bounds
        bounds_dict.numeric_post_bounds[cur_layer.identifier] = out_num_bounds

        if len(self.topological_stack) == 0:
            return bounds_dict, cur_bounds.concretize()

        else:
            next_layer = self.network.nodes[self.topological_stack.pop()]
            return self.propagate_bounds(in_num_bounds, in_sym_bounds, next_layer)
