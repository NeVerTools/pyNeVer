"""
This module controls the bounds propagation for neural networks

"""
from pynever import tensors
from pynever.networks import NeuralNetwork, SequentialNetwork, AcyclicNetwork
from pynever.nodes import LayerNode
from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds, SymbolicLinearBounds, \
    VerboseBounds
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.strategies.abstraction.networks import AbsSeqNetwork, AbsAcyclicNetwork
from pynever.strategies.verification.parameters import SSBPVerificationParameters
from pynever.strategies.verification.properties import NeverProperty
from pynever.strategies.verification.ssbp.constants import BoundsDirection
from pynever.utilities import xnor


class NewBoundsManager:
    def __init__(self, network: NeuralNetwork, prop: NeverProperty = None, input_bounds: HyperRectangleBounds = None,
                 parameters: SSBPVerificationParameters = None):
        if prop is None and input_bounds is None:
            raise Exception('Please initialize with either a property or input bounds')

        # Initialize the parameters
        self.network: NeuralNetwork = network

        if isinstance(self.network, SequentialNetwork):
            self.abs_network = AbsSeqNetwork(self.network, parameters)
        elif isinstance(self.network, AcyclicNetwork):
            self.abs_network = AbsAcyclicNetwork(self.network, parameters)
        else:
            raise NotImplementedError

        self.topological_stack: list[str] = self.network.get_topological_order()
        self.direction: BoundsDirection = parameters.bounds_direction

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

    def propagate_bounds(self, in_num_bounds: HyperRectangleBounds | list[HyperRectangleBounds] | None = None,
                         in_sym_bounds: SymbolicLinearBounds | list[SymbolicLinearBounds] | None = None,
                         start_layer: LayerNode = None):

        if start_layer is None:
            start_layer = self.network.get_roots()[0]

            # TODO remove after debugging
            assert start_layer.identifier == self.topological_stack.pop()
            start_layer = self.abs_network.get_abstract(start_layer)

            if in_sym_bounds is None:
                in_sym_bounds = self.init_symbolic_bounds()
            if in_num_bounds is None:
                in_num_bounds = self.input_bounds

        # Current layer data
        cur_layer = start_layer
        cur_sym_bounds = in_sym_bounds
        cur_num_bounds = in_num_bounds

        # TODO remove after debugging
        assert xnor(len(self.network.get_children(self.abs_network.get_concrete(cur_layer))) == 0,
                    len(self.topological_stack) == 0)

        # Compute bounds for this layer
        out_sym_bounds, out_num_bounds = cur_layer.forward_bounds(cur_sym_bounds, cur_num_bounds, self.input_bounds)

        # Fill the bounds dictionary for this layer
        self.bounds_dict.identifiers.append(cur_layer.identifier)
        self.bounds_dict.numeric_pre_bounds[cur_layer.identifier] = cur_num_bounds
        self.bounds_dict.symbolic_bounds[cur_layer.identifier] = out_sym_bounds
        self.bounds_dict.numeric_post_bounds[cur_layer.identifier] = out_num_bounds

        if len(self.topological_stack) == 0:
            return self.bounds_dict, out_num_bounds

        else:
            next_layer = self.abs_network.get_abstract(self.network.nodes[self.topological_stack.pop()])
            return self.propagate_bounds(out_num_bounds, out_sym_bounds, next_layer)
