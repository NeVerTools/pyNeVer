import abc
import collections
import copy

import pynever.nodes as nodes
from pynever.exceptions import EmptyNetworkError, InvalidNodeError, NotInNetworkError


class NeuralNetwork(abc.ABC):
    """
    An abstract class used for our internal representation of a generic NeuralNetwork. It consists of a graph of
    LayerNodes and a list of AlternativeRepresentations. It should be noted that this data structure it is not able
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
    input_ids : dict <str, str | None>
        Dictionary containing the inputs of the networks as keys and the corresponding layer identifier of the Node of
        which they are the input.

    """

    def __init__(self, identifier: str, input_ids: list[str]):
        self.nodes: dict[str, nodes.ConcreteLayerNode] = {}
        self.edges: dict[str, list[str]] = {}
        self.identifier = identifier
        self.input_ids: dict[str, str | None] = {i: None for i in input_ids}

    def get_children(self, node: nodes.ConcreteLayerNode) -> list[nodes.ConcreteLayerNode]:
        """
        Procedure to return the children of a node as a list of ConcreteLayerNodes.

        Parameters
        ----------
        node: ConcreteLayerNode
            The node whose children should be returned.

        Returns
        -------
        list[ConcreteLayerNodes]
            The children of the node passed as argument.

        """
        child_nodes = [self.nodes[child_node_id] for child_node_id in self.edges[node.identifier]]
        return child_nodes

    def get_parents(self, node: nodes.ConcreteLayerNode) -> list[nodes.ConcreteLayerNode]:
        """
        Procedure to return the parents of a node as a list of ConcreteLayerNodes.

        Parameters
        ----------
        node: ConcreteLayerNode
            The node whose parents should be returned.

        Returns
        -------
        list[ConcreteLayerNodes]
            The parents of the node passed as argument.

        """
        parent_nodes = [self.nodes[parent_node_id] for parent_node_id, end_nodes_ids in self.edges.items() if
                        node.identifier in end_nodes_ids]

        return parent_nodes

    def has_parents(self, node: nodes.ConcreteLayerNode) -> bool:
        """
        Procedure to check if a node has parents.
        Parameters
        ----------
        node: ConcreteLayerNode
            The node of which the existence of its parents should be checked.
        Returns
        -------
        bool
            True if the node has parents, False otherwise.
        """
        return len(self.get_parents(node)) != 0

    def has_children(self, node: nodes.ConcreteLayerNode) -> bool:
        """
        Procedure to check if a node has children.
        Parameters
        ----------
        node: ConcreteLayerNode
            The node of which the existence of its children should be checked.
        Returns
        -------
        bool
            True if the node has children, False otherwise.
        """
        return len(self.get_children(node)) != 0

    def get_roots(self) -> list[nodes.ConcreteLayerNode]:
        """
        Procedure to return the roots of the network as a list of ConcreteLayerNodes.
        Returns
        -------
        list[ConcreteLayerNodes]
            The roots of the network as a list of ConcreteLayerNodes.
        """
        root_nodes = [root_node for root_node_id, root_node in self.nodes.items() if not self.has_parents(root_node)]
        return root_nodes

    def get_leaves(self) -> list[nodes.ConcreteLayerNode]:
        """
        Procedure to return the leaves of the network as a list of ConcreteLayerNodes.
        Returns
        -------
        list[ConcreteLayerNodes]
            The leaves of the network as a list of ConcreteLayerNodes.
        """
        leaf_nodes = [leaf_node for leaf_node_id, leaf_node in self.nodes.items() if not self.has_children(leaf_node)]
        return leaf_nodes

    def remove_node(self, node: nodes.ConcreteLayerNode):
        """
        Procedure to remove a node from the network.
        Parameters
        ----------
        node: ConcreteLayerNode
            The node to be removed.

        """
        for parent_node in self.get_parents(node):
            self.edges[parent_node.identifier].remove(node.identifier)

        self.edges.pop(node.identifier)
        self.nodes.pop(node.identifier)

        for i, n in self.input_ids.items():
            if n == node.identifier:
                self.input_ids[i] = None

    def generic_add_node(self, node: nodes.ConcreteLayerNode, parents: list[nodes.ConcreteLayerNode] | None = None,
                         children: list[nodes.ConcreteLayerNode] | None = None,
                         input_ids: list[str] | None = None):
        """
        Procedure to add a node to the network. A node cannot have both parents and inputs.
        Parameters
        ----------
        node: ConcreteLayerNode
            The node to be added to the network.
        parents: list[ConcreteLayerNode] | None
            The parents of the node. (Optional)
        children: list[ConcreteLayerNode] | None
            The children of the node. (Optional)
        input_ids: list[ConcreteLayerNode] | None
            The inputs of the node. (Optional)

        """

        if input_ids is not None and parents is not None:
            raise Exception("A node cannot have both a parent and an input!")

        if input_ids is None:
            input_ids = []

        if parents is None:
            parents = []

        if children is None:
            children = []

        for parent_node in parents:

            if parent_node.identifier not in self.nodes.keys():
                raise NotInNetworkError(parent_node)

        for child_node in children:

            if child_node.identifier not in self.nodes.keys():
                raise NotInNetworkError(child_node)

        for input_id in input_ids:

            if input_id not in self.input_ids.keys():
                raise NotInNetworkError(f'{input_id} is not an input of the network')

        self.nodes[node.identifier] = node
        self.edges[node.identifier] = [c_node.identifier for c_node in children]

        for parent in parents:
            self.edges[parent.identifier].append(node.identifier)

        for input_id in input_ids:
            if self.input_ids[input_id] is not None:
                raise Exception(f'{input_id} has already been set as input to node {self.input_ids[input_id]}')
            self.input_ids[input_id] = node.identifier

    def is_acyclic(self) -> bool:
        """
        Procedure to check whether the network is acyclic.
        Returns
        -------
        bool
            True if network is acyclic, False otherwise.
        """

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
        super().__init__(identifier, [input_id])

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
                isinstance(node.get_input_dim(), tuple))

    def get_input_id(self) -> str:
        """
        Procedure to return the input_id of the network.
        Returns
        -------
        str
            The input_id of the network.
        """
        return list(self.input_ids.keys())[0]

    def is_empty(self) -> bool:
        """
        Procedure to check whether the network is empty.

        Returns
        -------
        bool
            True if there are no nodes in the network, False otherwise.

        """

        return len(self.nodes) == 0

    def append_node(self, node: nodes.ConcreteLayerNode):
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
            self.generic_add_node(node, input_ids=[self.get_input_id()])
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

    def get_next_node(self, node: nodes.ConcreteLayerNode) -> nodes.ConcreteLayerNode | None:
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
            raise InvalidNodeError(f'{next_node.identifier} is not a ConcreteLayerNode with a single input!')

        return next_node

    def get_prev_node(self, node: nodes.ConcreteLayerNode) -> nodes.ConcreteLayerNode | None:
        """
        Procedure to get the previous ConcreteLayerNode of the network given an input ConcreteLayerNode.

        Return
        ----------
        ConcreteLaterNode
            The previous node in the network.

        """

        if node == self.get_first_node():
            return None

        prev_node = self.get_parents(node)[0]

        if prev_node is not None and not SequentialNetwork.__is_single_concrete(prev_node):
            raise InvalidNodeError(f'{prev_node.identifier} is not a ConcreteLayerNode with a single input!')

        return prev_node

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
            raise InvalidNodeError(f'{last_node.identifier} is not a ConcreteLayerNode with a single input!')

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

    def layers_iterator(self, offset: int = 0) -> collections.abc.Generator[nodes.ConcreteLayerNode | None, None, None]:
        """
        This method builds a generator for the layers of the network in sequential order.
        It allows to have an iterable interface when needed

        offset: int
            Offset to start the generation

        """

        if self.is_empty():
            return
        else:
            node = self.get_first_node()

            counter = 0
            while node is not None:
                if counter >= offset:
                    yield node

                node = self.get_next_node(node)
                counter += 1

    def get_id_from_index(self, index: int) -> str:
        """
        This method returns the identifier of the layer at the given index

        Parameters
        ----------

        index : int
            Index of the layer to return

        Returns
        ----------
        str
            The identifier of the layer at the given index

        """

        if abs(index) > len(self.nodes):
            raise IndexError
        else:

            # If index is negative flip it
            if index < 0:
                index = len(self.nodes) + index

            counter = 0
            for layer in self.layers_iterator():
                if counter == index:
                    return layer.identifier

                counter += 1

    def get_index_from_id(self, identifier: str) -> int:
        """
        This method returns the index of the layer with the given
        identifier

        Parameters
        ----------
        identifier : str
            Identifier of the layer to return

        Returns
        ----------
        int
            The index of the layer with the given identifier

        """

        counter = 0

        for layer in self.layers_iterator():
            if layer.identifier == identifier:
                return counter

            counter += 1

        raise NotInNetworkError(f'There is no layer with identifier {identifier}')

    def get_previous_id(self, layer_identifier: str) -> str | None:
        """
        Procedure to get the identifier of the previous layer given another layer identifier

        """

        prev_layer = None

        for layer in self.layers_iterator():
            if layer.identifier == layer_identifier:
                return None if prev_layer is None else prev_layer.identifier
            prev_layer = layer

        return None

    def layer_precedes(self, layer_id1: str, layer_id2: str) -> bool:
        """
        Procedure to check whether a given layer precedes another or not

        """

        found_id1 = False
        for layer in self.layers_iterator():
            if not found_id1:
                if layer.identifier == layer_id1:
                    found_id1 = True

            else:
                if layer.identifier == layer_id2:
                    return True

        return False

    def __repr__(self):
        body = [node.__str__() for node in self.nodes.values()]
        return f"{self.identifier} : {body}"


class AcyclicNetwork(NeuralNetwork):

    def __init__(self, identifier: str, input_ids: list[str], input_edges: dict):
        super().__init__(identifier, input_ids)
        self.input_edges = input_edges

    def add_node(self, node: nodes.ConcreteLayerNode, parents: list[nodes.ConcreteLayerNode] | None = None,
                 children: list[nodes.LayerNode] | None = None):
        self.generic_add_node(node, parents, children)
        if not self.is_acyclic():
            self.remove_node(node)
            raise Exception(f"Adding {node.identifier} with the provided parents and children would create a cycle"
                            f" in the Network!")
