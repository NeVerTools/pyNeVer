from typing import List, Optional, Tuple
import copy

import pynever.nodes as nodes

from pynever.exceptions import EmptyNetworkError, InvalidNodeError


class NeuralNetwork:
    """
    An abstract class used for our internal representation of a generic NeuralNetwork. It consists of a graph of LayerNodes
    and a list of AlternativeRepresentations. It should be noted that this data structure it is not able
    to compute the input-output relation defined by the network. The properties of the computational graph are
    specialized in the concrete classes.

    Attributes
    ----------
    nodes : dict <str, LayerNode>
        Dictionary containing str keys and LayerNodes values. It contains the nodes of the graph,
        the identifier of the node of interest is used as a key in the nodes' dictionary.
    edges : dict <str, list <str>>
        Dictionary of identifiers of LayerNodes, it contains for each node identified by the keys, the list of nodes
        connected to it.
    identifier : str
        Identifier of the Sequential Neural Network.

    """

    def __init__(self, identifier: str = ''):
        self.nodes = {}
        self.edges = {}
        self.identifier = identifier

    def get_children(self, node: nodes.LayerNode) -> List[nodes.LayerNode]:

        child_nodes = [self.nodes[child_node_id] for child_node_id in self.edges[node.identifier]]
        return child_nodes

    def get_parents(self, node: nodes.LayerNode) -> List[nodes.LayerNode]:

        parent_nodes = [self.nodes[parent_node_id] for parent_node_id, end_nodes_ids in self.edges.items() if
                        node.identifier in end_nodes_ids]

        return parent_nodes

    def has_parents(self, node: nodes.LayerNode) -> bool:
        return len(self.get_parents(node)) != 0

    def has_children(self, node: nodes.LayerNode) -> bool:
        return len(self.get_children(node)) != 0

    def get_roots(self) -> List[nodes.LayerNode]:

        root_nodes = [root_node for root_node_id, root_node in self.nodes.items() if not self.has_parents(root_node)]
        return root_nodes

    def get_leaves(self) -> List[nodes.LayerNode]:

        leaf_nodes = [leaf_node for leaf_node_id, leaf_node in self.nodes.items() if not self.has_children(leaf_node)]
        return leaf_nodes

    def remove_node(self, node: nodes.LayerNode):

        for parent_node in self.get_parents(node):
            self.edges[parent_node.identifier].remove(node.identifier)

        self.edges.pop(node.identifier)
        self.nodes.pop(node.identifier)

        return

    def generic_add_node(self, node: nodes.LayerNode, parents: Optional[List[nodes.LayerNode]] = None,
                         children: Optional[List[nodes.LayerNode]] = None):

        if parents is None:
            parents = []

        if children is None:
            children = []

        for parent_node in parents:

            if parent_node.identifier not in self.nodes.keys():
                raise Exception(f"Parent Node {parent_node.identifier} is not a node of the Network.")

        for child_node in children:

            if child_node.identifier not in self.nodes.keys():
                raise Exception(f"Child Node {child_node.identifier} is not a node of the Network.")

        self.nodes[node.identifier] = node
        self.edges[node.identifier] = [c_node.identifier for c_node in children]

        for parent in parents:
            self.edges[parent.identifier].append(node.identifier)

    def is_acyclic(self):

        aux_network = copy.deepcopy(self)
        root_nodes = aux_network.get_roots()
        topologically_sorted = []

        while len(root_nodes) > 0:

            temp_node = root_nodes[0]
            root_nodes.remove(temp_node)

            topologically_sorted.append(temp_node)
            for child_node in aux_network.get_children(temp_node):
                aux_network.edges[temp_node.identifier].remove(child_node.identifier)
                if not aux_network.has_parents(child_node):
                    root_nodes.append(child_node)

        has_edges = False
        for start_node_id, end_nodes_ids in aux_network.edges.items():
            if len(end_nodes_ids) > 0:
                has_edges = True

        return not has_edges


class SequentialNetwork(NeuralNetwork):
    """
    Concrete children of NeuralNetwork representing a sequential NeuralNetwork. It consists of a graph of
    ConcreteLayerNodes and a list of AlternativeRepresentations. It should be noted that this data structure it is not
    able to compute the input-output relation defined by the network. The computational graph of a SequentialNetwork
    must correspond to a standard list.

    Attributes
    ----------
    input_id : str
        Identifier for the input of the Sequential Neural Network.

    Methods
    -------
    is_empty()
        Procedure to check whether the network is empty.
    add_node(LayerNode)
        Procedure to add a new LayerNode to the sequential Neural Network.
    get_first_node()
        Procedure to extract the first node of the sequential Neural Network.
    get_next_node(ConcreteLayerNode)
        Procedure to get the next node of the network given an input LayerNode.
    get_last_node()
        Procedure to extract the last node of the sequential Neural Network.
    delete_last_node()
        Procedure to delete the last node of the sequential Neural Network.
    get_input_len()
        Procedure to count the number of single inputs
    get_output_len()
        Procedure to count the number of single outputs
    count_relu_layers()
        Procedure to extract the number of layers of the sequential Neural Network.

    """

    def __init__(self, identifier: str, input_id: str):

        super().__init__(identifier)
        self.input_id = input_id

    @staticmethod
    def __is_single_concrete(node: nodes.LayerNode) -> bool:
        """
        Procedure to check whether the LayerNode given as a parameter is a ConcreteLayerNode with a single input.
        Parameters
        ----------
        node: LayerNode
            The node to be checked.
        Returns
        -------
        bool
            True if the LayerNode is a ConcreteLayerNode with a single input, False otherwise.

        """
        return ((node is not None) and
                isinstance(node, nodes.ConcreteLayerNode) and
                isinstance(node.get_input_dim(), Tuple))

    def is_empty(self) -> bool:
        """
        Procedure to check whether the network is empty.

        Returns
        -------
        bool
            True if there are no nodes in the network, False otherwise.

        """

        return len(self.nodes) == 0

    def add_node(self, node: nodes.ConcreteLayerNode):
        """
        Procedure to add a new ConcreteLayerNode. In sequential network the new node must be connected directly to the
        previous node forming a list.

        Parameters
        ----------
        node : ConcreteLayerNode
            New node to add to the Sequential network.

        """
        if not SequentialNetwork.__is_single_concrete(node):
            raise InvalidNodeError(f'{node.identifier} is not a ConcreteLayerNode with a single input!')

        if self.is_empty():
            self.generic_add_node(node)
        else:
            parents = [self.get_last_node()]
            self.generic_add_node(node, parents=parents)

    def get_first_node(self) -> nodes.ConcreteLayerNode:
        """
        Procedure to get the first ConcreteLayerNode of the network.

        Return
        ---------
        ConcreteLayerNode
            The first node of the network.

        """

        if self.is_empty():
            raise EmptyNetworkError()

        first_node = self.get_roots()[0]
        if not SequentialNetwork.__is_single_concrete(first_node):
            raise InvalidNodeError(f'{first_node.identifier} is not a ConcreteLayerNode with a single input!')

        return first_node

    def get_next_node(self, node: nodes.ConcreteLayerNode) -> nodes.ConcreteLayerNode:
        """
        Procedure to get the next ConcreteLayerNode of the network given an input ConcreteLayerNode.

        Return
        ---------
        ConcreteLayerNode
            The next node of the network.

        """

        if self.is_empty():
            raise EmptyNetworkError()

        children = self.get_children(node)
        if len(children) == 0:
            next_node = None
        else:
            next_node = children[0]

        if next_node is not None and not SequentialNetwork.__is_single_concrete(next_node):
            raise Exception(f'{next_node.identifier} is not a ConcreteLayerNode with a single input!')

        return next_node

    def get_last_node(self) -> nodes.ConcreteLayerNode:
        """
        Procedure to get the last ConcreteLayerNode of the network.

        Return
        ---------
        ConcreteLayerNode
            The last node of the network.

        """

        if self.is_empty():
            raise EmptyNetworkError()

        last_node = self.get_leaves()[0]

        if not SequentialNetwork.__is_single_concrete(last_node):
            raise Exception(f'{last_node.identifier} is not a ConcreteLayerNode with a single input!')

        return last_node

    def delete_last_node(self) -> nodes.ConcreteLayerNode:
        """
        Procedure to remove the last ConcreteLayerNode from the network.

        Returns
        ---------
        ConcreteLayerNode
            The last node of the network.

        """

        if self.is_empty():
            raise EmptyNetworkError()

        last_node = self.get_last_node()
        self.remove_node(last_node)
        return last_node

    def get_input_len(self) -> int:
        """
        Count the number of inputs in in_dim

        Returns
        -------
        int
            The number of single inputs

        """

        if self.is_empty():
            raise EmptyNetworkError()

        count = 0
        for d in range(len(self.get_first_node().get_input_dim())):
            count += self.get_first_node().get_input_dim()[d]

        return count

    def get_output_len(self) -> int:
        """
        Count the number of outputs in out_dim

        Returns
        -------
        int
            The number of single outputs

        """

        if self.is_empty():
            raise EmptyNetworkError()

        count = 0
        for d in range(len(self.get_last_node().get_output_dim())):
            count += self.get_last_node().get_output_dim()[d]

        return count

    def count_relu_layers(self) -> int:
        """
        Count the number of ReLU layers of the NN.

        Returns
        -------
        int
            The number of ReLU layers

        """

        if self.nodes:
            current_node = self.get_first_node()
            relu_count = 0

            while self.get_next_node(current_node) is not None:
                current_node = self.get_next_node(current_node)
                if isinstance(current_node, nodes.ReLUNode):
                    relu_count += 1

            return relu_count
        else:
            return 0

    def __repr__(self):
        body = [node.__str__() for node in self.nodes.values()]
        return f"{self.identifier} : {body}"


class AcyclicNetwork(NeuralNetwork):

    def __init__(self, identifier: str, input_ids: List[str], input_edges: dict):
        super().__init__(identifier)
        self.input_ids = input_ids
        self.input_edges = input_edges

    def add_node(self, node: nodes.LayerNode, parents: Optional[List[nodes.LayerNode]] = None,
                 children: Optional[List[nodes.LayerNode]] = None):

        self.generic_add_node(node, parents, children)
        if not self.is_acyclic():
            self.remove_node(node)
            raise Exception(f"Adding {node.identifier} with the provided parents and children would create a cycle"
                            f" in the Network!")


        







