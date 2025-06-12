import abc
import copy
import time

import pynever.strategies.abstraction.nodes as absnodes
from pynever import networks, nodes
from pynever.strategies.abstraction import LOGGER_LAYER
from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.abstraction.star import AbsElement
from pynever.strategies.verification.parameters import VerificationParameters


# TODO update documentation


class AbsNeuralNetwork(abc.ABC):
    """
    An abstract class used for our internal representation of a generic NeuralNetwork for Abstract Interpretation.
    It consists of a graph of AbsLayerNodes. The properties of the computational graph are specialized in the
    concrete classes. The method forward and backward calls the corresponding methods in the AbsLayerNodes following the
    correct order to compute the output AbsElement.

    Attributes
    ----------
    nodes : dict <str, AbsLayerNode>
        Dictionary containing str keys and AbsLayerNodes values. It contains the nodes of the graph,
        the identifier of the node of interest is used as a key in the nodes dictionary.

    Methods
    ----------
    forward(AbsElement)
        Procedure which takes an AbsElement and computes the corresponding output AbsElement based on the AbsLayerNode
        of the network.

    """

    corresponding_classes = {
        'FullyConnectedNode': absnodes.AbsFullyConnectedNode,
        'ConvNode': absnodes.AbsConvNode,
        'ReshapeNode': absnodes.AbsReshapeNode,
        'FlattenNode': absnodes.AbsFlattenNode,
        'ReLUNode': absnodes.AbsReLUNode,
        'ConcatNode': absnodes.AbsConcatNode,
        'SumNode': absnodes.AbsSumNode,
    }

    def __init__(self, ref_network: networks.NeuralNetwork, parameters: VerificationParameters):
        self.nodes: dict[str, absnodes.AbsLayerNode] = {}
        self.ref_network = ref_network

        for node_id, node in ref_network.nodes.items():
            self.nodes[f'ABS_{node_id}'] = AbsNeuralNetwork.__get_abstract_node_class(node)(f'ABS_{node_id}',
                                                                                            node, parameters)

    @staticmethod
    def __get_abstract_node_class(node: nodes.ConcreteLayerNode) -> absnodes.AbsLayerNode.__class__:
        try:
            return AbsNeuralNetwork.corresponding_classes[type(node).__name__]
        except KeyError:
            raise Exception(f'Node {type(node).__name__} is not supported')

    @abc.abstractmethod
    def forward(self, abs_input: AbsElement | list[AbsElement]) -> AbsElement | list[AbsElement]:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformers.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        raise NotImplementedError

    def get_abstract(self, node: nodes.ConcreteLayerNode, abs_id: bool = True) -> absnodes.AbsLayerNode:
        node = self.nodes[f'ABS_{node.identifier}']
        node.identifier = node.identifier.replace('ABS_', '') if not abs_id else node.identifier

        return node

    def get_concrete(self, absnode: absnodes.AbsLayerNode) -> nodes.ConcreteLayerNode:
        return self.ref_network.nodes[absnode.identifier.replace('ABS_', '', 1)]


class AbsSeqNetwork(AbsNeuralNetwork):
    """
    Concrete children of AbsNeuralNetwork representing a sequential AbsNeuralNetwork.
    It consists of a graph of LayerNodes. The computational graph of a SequentialNetwork must
    correspond to a standard list. The method forward and backward calls the corresponding methods
    in the AbsLayerNodes following the correct order to compute the output AbsElement.

    Methods
    -------
    forward(AbsElement)
        Procedure which takes an AbsElement and computes the corresponding output AbsElement based on the AbsLayerNode
        of the network.

    """

    def __init__(self, ref_network: networks.SequentialNetwork, parameters: VerificationParameters):
        super().__init__(ref_network, parameters)
        self.ref_network = ref_network
        self.bounds = None

    def set_bounds(self, layers_bounds: dict[str, HyperRectangleBounds]):
        self.bounds = layers_bounds

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformers.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """

        current_node = self.get_abstract(self.ref_network.get_first_node())

        while current_node is not None:
            time_start = time.perf_counter()

            LOGGER_LAYER.info(f"Computing starset for layer {current_node.identifier}...")

            if self.bounds:
                identifier = current_node.identifier.replace('ABS_', '')
                abs_input = current_node.forward_star(abs_input, self.bounds[identifier])
            else:
                abs_input = current_node.forward_star(abs_input)

            time_end = time.perf_counter()

            LOGGER_LAYER.info(f"Current starset has dimension {len(abs_input.stars)}\n"
                              f"Time to compute: {time_end - time_start:.4f} s\n")

            next_node = self.ref_network.get_next_node(self.get_concrete(current_node))
            current_node = self.get_abstract(next_node) if next_node is not None else None

        return abs_input


class AbsAcyclicNetwork(AbsNeuralNetwork):

    def __init__(self, ref_network: networks.AcyclicNetwork, parameters: VerificationParameters):
        super().__init__(ref_network, parameters)
        self.ref_network = ref_network
        self.input_ids: dict[str, str | None] = {k: self.get_abstract(v).identifier
                                                 for k, v in self.ref_network.input_ids}

    def get_node_inputs(self, node: absnodes.AbsLayerNode) -> list[str]:
        c_node = self.get_concrete(node)

        if not self.ref_network.has_parents(c_node):
            return [k for k, v in self.input_ids.items() if node.identifier in v]

        return [self.get_abstract(parent).identifier for parent in self.ref_network.get_parents(c_node)]

    def forward(self, abs_inputs: list[AbsElement]) -> list[AbsElement]:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformers.

        Parameters
        ----------
        abs_inputs : List[AbsElement]
            The input abstract element.

        Returns
        ----------
        List[AbsElement]
            The AbsElements resulting from the computation corresponding to the abstract transformer.
        """

        abs_input_ids = [abs_elem.identifier for abs_elem in abs_inputs]

        if set(abs_input_ids) != set(self.input_ids.keys()):
            raise Exception("The IDs of the Abstract Elements do not corresponds to the Keys of the Input Edges Dict!")

        if None in self.input_ids.values():
            raise Exception("Every Input in the should be associated to a Node")

        node_queue: list[absnodes.AbsLayerNode] = [self.get_abstract(n) for n in self.ref_network.get_roots()]
        temp_abs_inputs = copy.deepcopy(abs_inputs)

        while len(node_queue) != 0:

            current_node = node_queue.pop(0)
            input_ids = self.get_node_inputs(current_node)

            current_node_inputs = filter(lambda i: i.identifier in input_ids, temp_abs_inputs)

            current_abs_output = current_node.forward_star(current_node_inputs)

            current_abs_output.identifier = current_node.identifier
            temp_abs_inputs.append(current_abs_output)

            current_children = [self.get_abstract(n) for n in
                                self.ref_network.get_children(self.get_concrete(current_node))]
            for child in current_children:
                if child not in node_queue:
                    node_queue.append(child)

        leaves_ids = [self.get_abstract(leaf).identifier for leaf in self.ref_network.get_leaves()]
        final_outputs = filter(lambda fo: fo.identifier in leaves_ids, temp_abs_inputs)

        return final_outputs
