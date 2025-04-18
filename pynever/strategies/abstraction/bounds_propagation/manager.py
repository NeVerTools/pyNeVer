"""
This module controls the bounds propagation for neural networks

"""
from pynever import tensors
from pynever.networks import NeuralNetwork, SequentialNetwork, AcyclicNetwork
from pynever.nodes import LayerNode, ConcreteLayerNode
from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds, SymbolicLinearBounds, \
    VerboseBounds
from pynever.strategies.abstraction.bounds_propagation.util import ReLUStatus, check_stable
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.strategies.abstraction.networks import AbsSeqNetwork, AbsAcyclicNetwork, AbsNeuralNetwork
from pynever.strategies.verification.parameters import SSBPVerificationParameters
from pynever.strategies.verification.properties import NeverProperty
from pynever.strategies.verification.ssbp.constants import BoundsDirection
from pynever.utilities import xnor


class BoundsManager:
    """
    This class manages the symbolic bounds propagation framework for NeVer2.
    It is designed to handle feed-forward neural networks as computational graphs and can be instantiated
    either with fixed lower and upper bounds or with a structured verification property.

    Attributes
    ----------
    ref_network : NeuralNetwork
        The reference NN that defines the structure of the graph
    abs_network : AbsNeuralNetwork
        The abstract NN that contains the abstraction of the layers
    topological_stack : list[str]
        The topological sort of the layers in the NN used for the propagation
    direction : BoundsDirection
        The direction in which the bounds are computed, either forwards or backwards
    bounds_dict : VerboseBounds
        The data structure storing all bounds information
    input_bounds : HyperRectangleBounds
        The input bounds to propagate
    statistics : dict
        Statistics about neurons stability

    Methods
    ----------
    init_symbolic_bounds()
        Procedure to set up the initial symbolic bounds
    propagate_bounds(HyperRectangleBounds | None, SymbolicLinearBounds | None, LayerNode | None)
        Recursive procedure to propagate the bounds. When invoked as a root level, all parameters are None
    reset_stats()
        Procedure to reset statistics
    update_stats(AbsLayerNode, HyperRectangleBounds)
        Procedure to update statistics

    """

    def __init__(self, network: NeuralNetwork, prop: NeverProperty = None, input_bounds: HyperRectangleBounds = None,
                 parameters: SSBPVerificationParameters = None):
        if prop is None and input_bounds is None:
            raise Exception('Please initialize with either a property or input bounds')

        # Initialize the parameters
        self.ref_network: NeuralNetwork = network

        if isinstance(self.ref_network, SequentialNetwork):
            self.abs_network: AbsNeuralNetwork = AbsSeqNetwork(self.ref_network, parameters)
        elif isinstance(self.ref_network, AcyclicNetwork):
            self.abs_network: AbsNeuralNetwork = AbsAcyclicNetwork(self.ref_network, parameters)
        else:
            raise NotImplementedError

        self.topological_stack: list[str] = self.ref_network.get_topological_order()
        self.direction: BoundsDirection = parameters.bounds_direction

        # Initialize the bounds data structure
        self.bounds_dict = VerboseBounds()

        # Initialize the bounds
        self.input_bounds = prop.to_numeric_bounds() if prop else input_bounds

        # Initialize the statistics
        self.statistics = BoundsManager.reset_stats()

    def init_symbolic_bounds(self) -> SymbolicLinearBounds:
        """
        Initialize the input symbolic linear bounds
        """
        input_size = self.input_bounds.get_size()
        lower_equation = LinearFunctions(tensors.identity(input_size), tensors.zeros(input_size))
        upper_equation = LinearFunctions(tensors.identity(input_size), tensors.zeros(input_size))

        return SymbolicLinearBounds(lower_equation, upper_equation)

    def compute_bounds(self, in_num_bounds: HyperRectangleBounds | list[HyperRectangleBounds] | None = None,
                       in_sym_bounds: SymbolicLinearBounds | list[SymbolicLinearBounds] | None = None,
                       start_layer: LayerNode = None):
        """
        Entry point
        
        N.B. inside the propagation we use abstract layers but with their concrete counterpart identifier
        """
        if start_layer is None:
            start_layer = self.ref_network.get_roots()[0]

            # TODO remove after debugging
            assert start_layer.identifier == self.topological_stack.pop()
            # Set the identifier for the abstract layer equal to the concrete
            start_layer = self.abs_network.get_abstract(start_layer, abs_id=False)

            if in_sym_bounds is None:
                in_sym_bounds = self.init_symbolic_bounds()
            if in_num_bounds is None:
                in_num_bounds = self.input_bounds

        # Current layer data
        cur_layer = start_layer
        cur_sym_bounds = in_sym_bounds
        cur_num_bounds = in_num_bounds

        # TODO remove after debugging
        assert xnor(len(self.ref_network.get_children(self.abs_network.get_concrete(cur_layer))) == 0,
                    len(self.topological_stack) == 0)

        # Compute bounds for this layer
        out_sym_bounds, out_num_bounds = cur_layer.forward_bounds(cur_sym_bounds, cur_num_bounds, self.input_bounds)
        self.update_stats(cur_layer.identifier, cur_num_bounds)

        # Fill the bounds dictionary for this layer
        self.bounds_dict.identifiers.append(cur_layer.identifier)
        self.bounds_dict.numeric_pre_bounds[cur_layer.identifier] = cur_num_bounds
        self.bounds_dict.symbolic_bounds[cur_layer.identifier] = out_sym_bounds
        self.bounds_dict.numeric_post_bounds[cur_layer.identifier] = out_num_bounds

        if len(self.topological_stack) == 0:
            return self.bounds_dict, out_num_bounds

        else:
            next_layer = self.abs_network.get_abstract(self.ref_network.nodes[self.topological_stack.pop()],
                                                       abs_id=False)
            return self.compute_bounds(out_num_bounds, out_sym_bounds, start_layer=next_layer)

    def update_stats(self, layer_id: str, num_bounds: HyperRectangleBounds) -> None:
        """Update the statistics for this layer

        Parameters
        ----------
        layer_id : str
            The identifier of the layer
        num_bounds : HyperRectangleBounds
            The numeric pre-activation bounds
        """
        for neuron in range(num_bounds.get_size()):
            l, u = num_bounds.get_dimension_bounds(neuron)
            status = check_stable(l, u)

            match status:
                case ReLUStatus.ACTIVE:
                    self.statistics['relu'][ReLUStatus.ACTIVE][layer_id].append(neuron)
                    self.statistics['relu']['stable_count'] += 1

                case ReLUStatus.INACTIVE:
                    self.statistics['relu'][ReLUStatus.INACTIVE][layer_id].append(neuron)
                    self.statistics['relu']['stable_count'] += 1

                case ReLUStatus.UNSTABLE:
                    self.statistics['relu'][ReLUStatus.UNSTABLE][layer_id].append(neuron)

                    # Compute approximation area
                    area = 0.5 * (u - l) * u
                    self.statistics['approximation'][(layer_id, neuron)] = area

                case _:
                    raise NotImplementedError

    @staticmethod
    def reset_stats() -> dict:
        """Reset the dictionary of statistics about neurons"""
        return {
            'relu': {
                # These dictionaries are structured as
                # <layer_id: str> -> list[neuron: int]
                ReLUStatus.ACTIVE: dict(),
                ReLUStatus.INACTIVE: dict(),
                ReLUStatus.UNSTABLE: dict(),
                'stable_count': 0
            },
            # This dictionary is structured as
            # <(layer_id: str, neuron: int)> -> area: float
            'approximation': dict()
        }

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
