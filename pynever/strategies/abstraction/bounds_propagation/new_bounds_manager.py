"""
This module controls the bounds propagation for neural networks

"""

from pynever import nodes, tensors
from pynever.networks import NeuralNetwork
from pynever.nodes import LayerNode
from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds, SymbolicLinearBounds, VerboseBounds
from pynever.strategies.abstraction.bounds_propagation.layers.affine_layer import compute_dense_output_bounds
from pynever.strategies.abstraction.bounds_propagation.layers.convolution import LinearizeConv
from pynever.strategies.abstraction.bounds_propagation.layers.relu import LinearizeReLU
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.strategies.verification.properties import NeverProperty
from pynever.strategies.verification.ssbp.constants import BoundsDirection
from pynever.utilities import xnor


class NewBoundsManager:
    def __init__(self, network: NeuralNetwork, prop: NeverProperty = None, input_bounds: HyperRectangleBounds = None,
                 direction: BoundsDirection = BoundsDirection.FORWARDS):
        if prop is None and input_bounds is None:
            raise Exception('Please initialize with either a property or input bounds')

        # Initialize the parameters
        self.network: NeuralNetwork = network
        self.topological_stack: list[str] = self.network.get_topological_order()
        self.direction: BoundsDirection = direction

        # Initialize the bounds data structure
        self.bounds_dict = VerboseBounds()

        # Initialize the bounds
        self.input_bounds = prop.to_numeric_bounds() if prop else input_bounds

    def init_symbolic_bounds(self) -> SymbolicLinearBounds:
        """
        Initialize the input symbolic linear bounds
        """
        input_size = self.input_bounds.get_size()
        lower_equation = LinearFunctions(tensors.identity(input_size), tensors.zeros(input_size))
        upper_equation = LinearFunctions(tensors.identity(input_size), tensors.zeros(input_size))

        return SymbolicLinearBounds(lower_equation, upper_equation)

    def compute_layer(self, layer: LayerNode, layer_in_symbolic: SymbolicLinearBounds | list[SymbolicLinearBounds],
                      layer_in_numeric: HyperRectangleBounds | list[HyperRectangleBounds]) \
            -> tuple[SymbolicLinearBounds, HyperRectangleBounds]:

        if isinstance(layer, nodes.FullyConnectedNode):
            """ Fully Connected layer """

            if self.direction == BoundsDirection.FORWARDS:
                cur_layer_output_eq = compute_dense_output_bounds(layer, layer_in_symbolic)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(self.input_bounds)

            else:
                raise NotImplementedError('Backwards bounds propagation not yet implemented for fully connected layers')

        elif isinstance(layer, nodes.ConvNode):
            """ Convolutional layer """

            if self.direction == BoundsDirection.FORWARDS:
                cur_layer_output_eq = LinearizeConv().compute_output_equations(layer, layer_in_symbolic)
                cur_layer_output_num_bounds = cur_layer_output_eq.to_hyper_rectangle_bounds(self.input_bounds)

            else:
                raise NotImplementedError('Backwards bounds propagation not yet implemented for convolutional layers')

        elif isinstance(layer, nodes.ReLUNode):
            """ ReLU layer """

            relu_lin = LinearizeReLU(fixed_neurons={}, input_hyper_rect=self.input_bounds)

            if self.direction == BoundsDirection.FORWARDS:
                cur_layer_output_eq = relu_lin.compute_output_linear_bounds(layer_in_symbolic)
                cur_layer_output_num_bounds = relu_lin.compute_output_numeric_bounds(layer, layer_in_numeric,
                                                                                     layer_in_symbolic)

            else:
                raise NotImplementedError('Backwards bounds propagation not yet implemented for convolutional layers')

        elif isinstance(layer, nodes.FlattenNode):
            """ Flatten layer """

            # self.layer2layer_equations[layer.identifier] = layer_in_eq
            cur_layer_output_eq = layer_in_symbolic
            cur_layer_output_num_bounds = layer_in_numeric

        elif isinstance(layer, nodes.ReshapeNode):
            """ Reshape layer """

            # self.layer2layer_equations[layer.identifier] = layer_in_eq
            cur_layer_output_eq = layer_in_symbolic
            cur_layer_output_num_bounds = layer_in_numeric

        else:
            raise Exception(
                "Currently supporting bounds computation only for FullyConnected, Convolutional, ReLU "
                "and Flatten layers.\n Instead got {}".format(layer.__class__))

        return cur_layer_output_eq, cur_layer_output_num_bounds

    def propagate_bounds(self, in_num_bounds: HyperRectangleBounds | list[HyperRectangleBounds] | None = None,
                         in_sym_bounds: SymbolicLinearBounds | list[SymbolicLinearBounds] | None = None,
                         start_layer: LayerNode = None):

        if start_layer is None:
            start_layer = self.network.get_roots()[0]

            # TODO remove after debugging
            assert start_layer.identifier == self.topological_stack.pop()

            if in_sym_bounds is None:
                in_sym_bounds = self.init_symbolic_bounds()

            if in_num_bounds is None:
                in_num_bounds = self.input_bounds

        # Current layer data
        cur_layer = start_layer
        cur_sym_bounds = in_sym_bounds
        cur_num_bounds = in_num_bounds

        # TODO remove after debugging
        assert xnor(len(self.network.get_children(cur_layer)) == 0, len(self.topological_stack) == 0)

        # Compute bounds for this layer
        out_sym_bounds, out_num_bounds = self.compute_layer(cur_layer, cur_sym_bounds, cur_num_bounds)

        # Fill the bounds dictionary for this layer
        self.bounds_dict.identifiers.append(cur_layer.identifier)
        self.bounds_dict.numeric_pre_bounds[cur_layer.identifier] = cur_num_bounds
        self.bounds_dict.symbolic_bounds[cur_layer.identifier] = out_sym_bounds
        self.bounds_dict.numeric_post_bounds[cur_layer.identifier] = out_num_bounds

        if len(self.topological_stack) == 0:
            return self.bounds_dict, out_num_bounds

        else:
            next_layer = self.network.nodes[self.topological_stack.pop()]
            return self.propagate_bounds(out_num_bounds, out_sym_bounds, next_layer)
