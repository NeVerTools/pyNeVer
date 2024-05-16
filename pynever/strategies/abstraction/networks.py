import abc
import copy

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.abstraction.nodes as absnodes
from pynever.strategies.abstraction.star import AbsElement
from pynever.strategies.verification.parameters import VerificationParameters


# TODO update documentation

corresponding_classes = {
    'FullyConnectedNode': absnodes.AbsFullyConnectedNode,
    'ReLUNode': absnodes.AbsReLUNode,
    'SigmoidNode': absnodes.AbsSigmoidNode,
    'ConcatNode': absnodes.AbsConcatNode,
    'SumNode': absnodes.AbsSumNode,
}


def get_abstract_node_class(node: nodes.ConcreteLayerNode) -> absnodes.AbsLayerNode.__class__:
    try:
        return corresponding_classes[type(node).__name__]
    except KeyError:
        raise Exception(f'Node {type(node).__name__} is not supported')


class AbsNeuralNetwork(networks.NeuralNetwork):
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

    edges : dict <str, list <str>>
        Dictionary of identifiers of AbsLayerNodes, it contains for each nodes identified by the keys, the list of nodes
        connected to it.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the AbsLayerNode
        of the network.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the AbsLayerNodes
        to control the refinement component of the abstraction. At present the function is just a placeholder
        for future implementations.

    """
    def __init__(self, ref_network: networks.NeuralNetwork, parameters: VerificationParameters):
        super().__init__(f'ABS_{ref_network.identifier}', [])
        self.input_ids = ref_network.input_ids

        for node_id, node in ref_network.nodes.items():
            self.nodes[f'ABS_{node_id}'] = get_abstract_node_class(node)(f'ABS_{node_id}', node, parameters)

        for node_id, children in ref_network.edges.items():
            self.edges[f'ABS_{node_id}'] = [f'ABS_{child_id}' for child_id in children]

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, ref_state: absnodes.RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        raise NotImplementedError


class AbsSeqNetwork(AbsNeuralNetwork, networks.SequentialNetwork):
    """
    Concrete children of AbsNeuralNetwork representing a sequential AbsNeuralNetwork.
    It consists of a graph of LayerNodes. The computational graph of a SequentialNetwork must
    correspond to a standard list. The method forward and backward calls the corresponding methods
    in the AbsLayerNodes following the correct order to compute the output AbsElement.

    Attributes
    ----------
    identifier : str
        Identifier of the Sequential AbsNeuralNetwork.

    Methods
    -------
    add_node(SingleInputLayerNode)
        Procedure to add a new AbsLayerNode to the sequential AbsNeuralNetwork.

    get_first_node()
        Procedure to extract the first AbsLayerNode of the sequential AbsNeuralNetwork.

    get_next_node(SingleInputLayerNode)
        Procedure to get the next AbsLayerNode of the AbsNeuralNetwork given an input AbsLayerNode

    get_last_node()
        Procedure to extract the last AbsLayerNode of the sequential AbsNeuralNetwork.

    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the AbsLayerNode
        of the network.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the AbsLayerNodes
        to control the refinement component of the abstraction. At present the function is just a placeholder for
        future implementations.

    """

    def __init__(self, ref_network: networks.SequentialNetwork, parameters: VerificationParameters):
        super().__init__(ref_network, parameters)

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

        current_node = self.get_first_node()
        while current_node is not None:
            abs_input = current_node.forward(abs_input)
            current_node = self.get_next_node(current_node)

        return abs_input

    def backward(self, ref_state: absnodes.RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        raise NotImplementedError


class AbsAcyclicNetwork(AbsNeuralNetwork, networks.AcyclicNetwork):

    def __init__(self, ref_network: networks.AcyclicNetwork, parameters: VerificationParameters):
        super().__init__(ref_network, parameters)
        self.input_edges = ref_network.input_edges

    def get_node_inputs(self, node: absnodes.AbsLayerNode):

        if not self.has_parents(node):
            input_ids = [key for key, value in self.input_edges.items() if node.identifier in value]
        else:
            input_ids = [parent.identifier for parent in self.get_parents(node)]

        return input_ids

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
        if set(abs_input_ids) != set(self.input_ids):
            raise Exception("The IDs of the Abstract Elements do not corresponds to the expected Input IDs!")

        if set(abs_input_ids) != set(self.input_edges.keys()):
            raise Exception("The IDs of the Abstract Elements do not corresponds to the Keys of the Input Edges Dict!")

        if [] in self.input_edges.values():
            raise Exception("Every Input in the Input Edges Dictionary should have at least an Edge!")

        node_queue = self.get_roots()
        temp_abs_inputs = copy.deepcopy(abs_inputs)

        while node_queue.__len__() != 0:

            current_node = node_queue.pop(0)
            input_ids = self.get_node_inputs(current_node)

            current_node_inputs = [a_input for a_input in temp_abs_inputs if a_input.identifier in input_ids]
            # TODO simplify
            # TODO: At this time we need to check the difference between the inputs for multinputlayernodes and single
            # input layer nodes. Once nodes refactor is done it can be simplified.
            if isinstance(current_node, AbsSingleInputLayerNode):

                if len(current_node_inputs) > 1:
                    raise Exception(f"{current_node.__class__} should have a single input!")
                else:
                    current_abs_output = current_node.forward(current_node_inputs[0])

            elif isinstance(current_node, AbsMultiInputLayerNode):
                current_abs_output = current_node.forward(current_node_inputs)
            else:
                raise NotImplementedError

            current_abs_output.identifier = current_node.identifier
            temp_abs_inputs.append(current_abs_output)

            current_children = self.get_children(current_node)
            for child in current_children:
                if child not in node_queue:
                    node_queue.append(child)

        leaves_ids = [leaf.identifier for leaf in self.get_leaves()]
        final_outputs = [final_output for final_output in temp_abs_inputs if final_output.identifier in leaves_ids]

        return final_outputs

    def backward(self, ref_state: absnodes.RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        raise NotImplementedError
