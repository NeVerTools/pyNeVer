"""
This module controls the bounds propagation for neural networks

"""
import torch

from pynever.networks import NeuralNetwork, SequentialNetwork, AcyclicNetwork
from pynever.nodes import LayerNode, ConcreteLayerNode
from pynever.strategies.abstraction.bounds_propagation import ReLUStatus
from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds, SymbolicLinearBounds
from pynever.strategies.abstraction.bounds_propagation.util import check_stable
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.strategies.abstraction.networks import AbsSeqNetwork, AbsAcyclicNetwork, AbsNeuralNetwork
from pynever.strategies.abstraction.nodes import AbsReLUNode
from pynever.strategies.verification.parameters import SSBPVerificationParameters
from pynever.strategies.verification.properties import NeverProperty
from pynever.strategies.verification.ssbp.constants import BoundsDirection
from pynever.strategies.verification.statistics import VerboseBounds, BoundsStats
from pynever.utilities import xnor


class BoundsManager:
    """
    This class manages the symbolic bounds propagation framework for NeVer2.
    It is designed to handle feed-forward neural networks as computational graphs and can be instantiated
    either with fixed lower and upper bounds or with a structured verification property.

    Attributes
    ----------
    ref_nn: NeuralNetwork
        The reference NN that defines the structure of the graph
    abs_nn: AbsNeuralNetwork
        The abstract NN that contains the abstraction of the layers
    topological_stack: list[str]
        The topological sort of the layers in the NN used for the propagation
    direction: BoundsDirection
        The direction in which the bounds are computed, either forwards or backwards
    bounds_dict: VerboseBounds
        The data structure storing all bounds information
    input_bounds: HyperRectangleBounds
        The input bounds to propagate
    statistics: BoundsStats
        Statistics about neurons stability

    Methods
    ----------
    init_symbolic_bounds()
        Procedure to set up the initial symbolic bounds
    propagate_bounds(HyperRectangleBounds | None, SymbolicLinearBounds | None, LayerNode | None)
        Recursive procedure to propagate the bounds. When invoked as a root level, all parameters are None
    update_stats(AbsLayerNode, HyperRectangleBounds)
        Procedure to update statistics
    """

    def __init__(self, network: NeuralNetwork, prop: NeverProperty = None, input_bounds: HyperRectangleBounds = None,
                 parameters: SSBPVerificationParameters = None):
        if prop is None and input_bounds is None:
            raise Exception('Please initialize with either a property or input bounds')

        # Initialize the parameters
        self.ref_nn: NeuralNetwork = network

        if isinstance(self.ref_nn, SequentialNetwork):
            self.abs_nn: AbsNeuralNetwork = AbsSeqNetwork(self.ref_nn, parameters)
        elif isinstance(self.ref_nn, AcyclicNetwork):
            self.abs_nn: AbsNeuralNetwork = AbsAcyclicNetwork(self.ref_nn, parameters)
        else:
            raise NotImplementedError

        self.topological_stack: list[str] = self.ref_nn.get_topological_order(reverse=True)
        self.direction: BoundsDirection = parameters.bounds_direction if parameters else BoundsDirection.FORWARDS

        # Initialize the bounds data structure
        self.bounds_dict = VerboseBounds()

        # Initialize the bounds
        self.input_bounds: HyperRectangleBounds = prop.to_numeric_bounds() if prop else input_bounds

        # Initialize the statistics
        self.statistics = BoundsStats()

    def init_symbolic_bounds(self) -> SymbolicLinearBounds:
        """Initialize the input symbolic linear bounds"""
        input_size = self.input_bounds.get_size()
        lower_equation = LinearFunctions(torch.eye(input_size), torch.zeros(input_size))
        upper_equation = LinearFunctions(torch.eye(input_size), torch.zeros(input_size))

        return SymbolicLinearBounds(lower_equation, upper_equation)

    def compute_bounds(self, in_num_bounds: HyperRectangleBounds | list[HyperRectangleBounds] | None = None,
                       in_sym_bounds: SymbolicLinearBounds | list[SymbolicLinearBounds] | None = None,
                       start_layer: LayerNode = None) -> VerboseBounds:
        """
        Entry point
        
        N.B. inside the propagation we use abstract layers but with their concrete counterpart identifier
        """
        if start_layer is None:
            # Set the identifier for the abstract layer equal to the concrete
            start_layer = self.abs_nn.get_abstract(self.ref_nn.get_first_node(), abs_id=False)

            if in_sym_bounds is None:
                in_sym_bounds = self.init_symbolic_bounds()
            if in_num_bounds is None:
                in_num_bounds = self.input_bounds

        # Pop the current layer
        self.topological_stack.pop()

        # Current layer data
        cur_layer = start_layer
        cur_sym_bounds = in_sym_bounds
        cur_num_bounds = in_num_bounds

        # TODO remove after debugging
        assert xnor(len(self.ref_nn.get_children(self.abs_nn.get_concrete(cur_layer))) == 0,
                    len(self.topological_stack) == 0)

        # Compute bounds for this layer
        out_sym_bounds, out_num_bounds = cur_layer.forward_bounds(cur_sym_bounds, cur_num_bounds, self.input_bounds)
        self.update_stats(cur_layer, cur_num_bounds)

        # Fill the bounds dictionary for this layer
        self.bounds_dict.identifiers.append(cur_layer.identifier)
        self.bounds_dict.numeric_pre_bounds[cur_layer.identifier] = cur_num_bounds
        self.bounds_dict.symbolic_bounds[cur_layer.identifier] = out_sym_bounds
        self.bounds_dict.numeric_post_bounds[cur_layer.identifier] = out_num_bounds
        self.bounds_dict.statistics = self.statistics

        if len(self.topological_stack) == 0:
            return self.bounds_dict

        else:
            next_layer = self.abs_nn.get_abstract(self.ref_nn.nodes[self.topological_stack[-1]], abs_id=False)
            return self.compute_bounds(out_num_bounds, out_sym_bounds, start_layer=next_layer)

    def update_stats(self, layer: LayerNode, num_bounds: HyperRectangleBounds) -> None:
        """Update the statistics for this layer

        Parameters
        ----------
        layer: LayerNode
            The current layer
        num_bounds: HyperRectangleBounds
            The numeric pre-activation bounds
        """

        # Update statistics for ReLU layers only
        if not isinstance(layer, AbsReLUNode):
            return

        layer_id = layer.identifier

        for neuron in range(num_bounds.get_size()):
            l, u = num_bounds.get_dimension_bounds(neuron)
            status = check_stable(l, u)

            for relu in [ReLUStatus.ACTIVE, ReLUStatus.INACTIVE, ReLUStatus.UNSTABLE]:
                if layer_id not in self.statistics.stability_info[relu].keys():
                    self.statistics.stability_info[relu][layer_id] = list()

            match status:
                case ReLUStatus.ACTIVE:
                    self.statistics.stability_info[ReLUStatus.ACTIVE][layer_id].append(neuron)
                    self.statistics.stability_info['stable_count'] += 1

                case ReLUStatus.INACTIVE:
                    self.statistics.stability_info[ReLUStatus.INACTIVE][layer_id].append(neuron)
                    self.statistics.stability_info['stable_count'] += 1

                case ReLUStatus.UNSTABLE:
                    self.statistics.stability_info[ReLUStatus.UNSTABLE][layer_id].append(neuron)

                    # Compute approximation area
                    area = 0.5 * (u - l) * u
                    self.statistics.approximation_info[(layer_id, neuron)] = area

                case _:
                    raise NotImplementedError

    @staticmethod
    def get_symbolic_preactivation_bounds_at(bounds: VerboseBounds, layer: ConcreteLayerNode,
                                             nn: NeuralNetwork) -> list[SymbolicLinearBounds]:
        """Retrieve the preactivation symbolic bounds for the given layer"""
        return [bounds.symbolic_bounds[identifier]
                for identifier in
                [parent.identifier
                 for parent in nn.get_parents(layer)
                 ]
                ]
