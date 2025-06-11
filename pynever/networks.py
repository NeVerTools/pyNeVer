"""
This module contains the classes to define and create neural network objects.
The abstract class ``NeuralNetwork`` represents a generic computational graph with layers as nodes and connections as edges.
The children of ``NeuralNetwork`` are ``SequentialNetwork`` and ``AcyclicNetwork``, to represent standard feed-forward NNs and residual NNs, respectively.
"""
import abc
import collections
import copy

import pynever.nodes as nodes
from pynever.exceptions import EmptyNetworkError, InvalidNodeError, NotInNetworkError
from pynever.nodes import ConcreteLayerNode


class NeuralNetwork(abc.ABC):
    """
    An abstract class used for our internal representation of a generic Neural Network. It consists of a graph of
    :class:`~pynever.nodes.LayerNode` objects and edges connecting them. It should be noted that this data structure is
    not able to compute the input-output relation defined by the network.
    The properties of the computational graph are specialized in the concrete classes.

    Attributes
    ----------
    nodes: dict[str, LayerNode]
        Dictionary containing string keys and :class:`~pynever.nodes.LayerNode` values.
        It contains the nodes of the graph, the identifier of the node is used as a key in the nodes' dictionary.
    edges: dict[str, list[str]]
        Dictionary of :class:`~pynever.nodes.LayerNode` identifiers.
        It contains for each node identified by the keys, the list of nodes connected to it.
    identifier: str
        Identifier of the Neural Network.
    input_ids: dict[str, str | None]
        Dictionary containing the inputs of the networks as keys and the corresponding layer identifier of the node of
        which they are the input.
    """

    def __init__(self, identifier: str, input_ids: list[str]):
        self.nodes: dict[str, nodes.ConcreteLayerNode] = {}
        self.edges: dict[str, list[str]] = {}
        self.identifier = identifier
        self.input_ids: dict[str, str | None] = {i: None for i in input_ids}

    def __repr__(self):
        body = '{}' if self.is_empty() else [self.nodes[k].__str__() for k in self.get_topological_order()]
        return f"{self.identifier} : {body}"

    def is_empty(self) -> bool:
        """Procedure to check whether the network is empty.

        Returns
        -------
        bool
            True if there are no nodes in the network, False otherwise.
        """
        return len(self.nodes) == 0

    def is_acyclic(self) -> bool:
        """Procedure to check whether the network is acyclic.

        Returns
        -------
        bool
            True if the network is acyclic, False otherwise.
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

    def has_children(self, node: nodes.ConcreteLayerNode) -> bool:
        """Procedure to check if a node has children.

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

    def get_children(self, node: nodes.ConcreteLayerNode) -> list[nodes.ConcreteLayerNode]:
        """Procedure to return the children of a node as a list of ConcreteLayerNodes.

        Parameters
        ----------
        node: ConcreteLayerNode
            The node whose children should be returned.

        Returns
        -------
        list[ConcreteLayerNode]
            The children of the node passed as argument.
        """
        return [self.nodes[child_node_id] for child_node_id in self.edges[node.identifier]]

    def has_parents(self, node: nodes.ConcreteLayerNode) -> bool:
        """Procedure to check if a node has parents.

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

    def get_parents(self, node: nodes.ConcreteLayerNode) -> list[nodes.ConcreteLayerNode]:
        """Procedure to return the parents of a node as a list of ConcreteLayerNodes.

        Parameters
        ----------
        node: ConcreteLayerNode
            The node whose parents should be returned

        Returns
        -------
        list[ConcreteLayerNode]
            The parents of the node passed as argument.
        """
        return [self.nodes[parent_node_id]
                for parent_node_id, end_nodes_ids in self.edges.items()
                if node.identifier in end_nodes_ids]

    def get_input_id(self) -> str:
        """Procedure to return the input_id of the network, assuming there is a single input layer.

        Returns
        -------
        str
            The input_id of the network.
        """
        return list(self.input_ids.keys())[0]

    def get_roots(self) -> list[nodes.ConcreteLayerNode]:
        """Procedure to return the roots of the network as a list of ConcreteLayerNodes.

        Returns
        -------
        list[ConcreteLayerNode]
            The roots of the network as a list of :class:`~pynever.nodes.ConcreteLayerNode` objects.
        """
        return [root_node
                for root_node_id, root_node in self.nodes.items()
                if not self.has_parents(root_node)]

    def get_leaves(self) -> list[nodes.ConcreteLayerNode]:
        """Procedure to return the leaves of the network as a list of ConcreteLayerNodes.

        Returns
        -------
        list[ConcreteLayerNode]
            The leaves of the network as a list of :class:`~pynever.nodes.ConcreteLayerNode` objects.
        """
        return [leaf_node
                for leaf_node_id, leaf_node in self.nodes.items()
                if not self.has_children(leaf_node)]

    def get_topological_order(self, reverse: bool = False) -> list[str]:
        """Procedure to walk the network with a DFS and build the topological sort.

        Parameters
        ----------
        reverse: bool
            Flag to reverse the order

        Returns
        ----------
        list[str]
            The topological sort of the network nodes identifiers as a stack.
        """

        def recursive_dfs(node_id: str, visited: set[str], order: list[str]) -> None:
            """Procedure to recursively read the graph"""
            for child in self.get_children(self.nodes[node_id]):
                if child.identifier not in visited:
                    visited.add(child.identifier)
                    recursive_dfs(child.identifier, visited, order)

            order.insert(0, node_id)

        result = []
        seen = set()
        recursive_dfs(self.get_roots()[0].identifier, seen, result)

        if reverse:
            result.reverse()

        return result

    def layers_iterator(self, offset: int = 0) -> collections.abc.Generator[nodes.ConcreteLayerNode | None, None, None]:
        """Procedure to build a generator for the layers of the network in sequential order.
        It allows having an iterable interface when needed

        Parameters
        ----------
        offset: int
            Offset to start the generation

        Returns
        ----------
        Generator[ConcreteLayerNode | None, None]
            The generator object
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

    def get_first_node(self) -> nodes.ConcreteLayerNode:
        """Procedure to get the first ConcreteLayerNode of the network.

        Returns
        -------
        ConcreteLayerNode
            The first node of the network.
        """
        if self.is_empty():
            raise EmptyNetworkError()

        return self.nodes[self.get_topological_order()[0]]

    def get_next_node(self, node: nodes.ConcreteLayerNode) -> nodes.ConcreteLayerNode | None:
        """Procedure to get the next ConcreteLayerNode of the network given an input ConcreteLayerNode.

        Returns
        -------
        ConcreteLayerNode
            The next node of the network.
        """
        if self.is_empty():
            raise EmptyNetworkError()

        order = self.get_topological_order()
        this_idx = order.index(node.identifier)
        return self.nodes[order[this_idx + 1]] if this_idx + 1 < len(order) else None

    def get_previous_node(self, node: nodes.ConcreteLayerNode) -> nodes.ConcreteLayerNode | None:
        """Procedure to get the previous ConcreteLayerNode of the network given an input ConcreteLayerNode.

        Returns
        -------
        ConcreteLaterNode
            The previous node in the network.
        """
        if node == self.get_first_node():
            return None

        order = self.get_topological_order()
        this_idx = order.index(node.identifier)
        return self.nodes[order[this_idx - 1]]

    def get_last_node(self) -> nodes.ConcreteLayerNode:
        """Procedure to get the last ConcreteLayerNode of the network.

        Returns
        -------
        ConcreteLayerNode
            The last node of the network.
        """
        if self.is_empty():
            raise EmptyNetworkError()

        return self.nodes[self.get_topological_order()[-1]]

    def get_input_len(self) -> int:
        """Procedure to count the number of inputs in in_dim

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
        """Procedure to count the number of outputs in out_dim

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

    def get_id_from_index(self, index: int) -> str:
        """This method returns the identifier of the layer at the given index

        Parameters
        ----------
        index: int
            Index of the layer to return

        Returns
        ----------
        str
            The identifier of the layer at the given index
        """
        if abs(index) > len(self.nodes):
            raise IndexError

        # If index is negative flip it
        if index < 0:
            index = len(self.nodes) + index

        counter = 0
        for layer in self.layers_iterator():
            if counter == index:
                return layer.identifier
            counter += 1

        # Should never get here
        raise IndexError

    def get_index_from_id(self, identifier: str) -> int:
        """This method returns the index of the layer with the given identifier

        Parameters
        ----------
        identifier: str
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

    def get_previous_id(self, identifier: str) -> str | None:
        """Procedure to get the identifier of the previous layer given another layer identifier

        Parameters
        ----------
        identifier: str
            Identifier of the layer

        Returns
        ----------
        str
            The identifier of the previous layer
        """
        prev_layer = None

        for layer in self.layers_iterator():
            if layer.identifier == identifier:
                return None if prev_layer is None else prev_layer.identifier
            prev_layer = layer

        raise NotInNetworkError(f'There is no layer with identifier {identifier}')

    def layer_precedes(self, layer_id1: str, layer_id2: str) -> bool:
        """Procedure to check whether a given layer precedes another or not

        Parameters
        ----------
        layer_id1: str
            Identifier of the first layer
        layer_id2: str
            Identifier of the second layer

        Returns
        ----------
        bool
            True if layer1 precedes layer2, False otherwise
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

    def generic_add_node(self, node: nodes.ConcreteLayerNode, parents: list[nodes.ConcreteLayerNode] | None = None,
                         children: list[nodes.ConcreteLayerNode] | None = None,
                         input_ids: list[str] | None = None):
        """Procedure to add a node to the network. A node cannot have both parents and inputs.

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

        input_ids = [] if input_ids is None else input_ids
        parents = [] if parents is None else parents
        children = [] if children is None else children

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

    def remove_node(self, node: nodes.ConcreteLayerNode):
        """Procedure to remove a node from the network.

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

    def delete_last_node(self) -> nodes.ConcreteLayerNode:
        """Procedure to remove the last ConcreteLayerNode from the network.

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

    def count_relu_layers(self) -> int:
        """Procedure to count the number of ReLU layers.

        Returns
        -------
        int
            The number of ReLU layers
        """
        relu_count = 0

        for layer in self.layers_iterator():
            if isinstance(layer, nodes.ReLUNode):
                relu_count += 1

        return relu_count


class SequentialNetwork(NeuralNetwork):
    """
    Concrete child of NeuralNetwork representing a sequential NeuralNetwork.
    It consists of a graph of ConcreteLayerNodes. It should be noted that this data structure
    is not able to compute the input-output relation defined by the network.
    The computational graph of a SequentialNetwork must correspond to a standard list.
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
            True if the :class:`~pynever.nodes.LayerNode` is a :class:`~pynever.nodes.ConcreteLayerNode`
            with a single input, False otherwise.
        """
        return ((node is not None) and
                isinstance(node, nodes.ConcreteLayerNode) and
                isinstance(node.get_input_dim(), tuple))

    def append_node(self, node: nodes.ConcreteLayerNode):
        """Procedure to add a new ConcreteLayerNode.
        In a sequential network the new node must be connected directly to the previous node forming a list.

        Parameters
        ----------
        node: ConcreteLayerNode
            New node to add to the :class:`~pynever.networks.SequentialNetwork`.
        """
        if not SequentialNetwork.__is_single_concrete(node):
            raise InvalidNodeError(f'{node.identifier} is not a ConcreteLayerNode with a single input!')

        if self.is_empty():
            self.generic_add_node(node, input_ids=[self.get_input_id()])
        else:
            parents = [self.get_last_node()]
            self.generic_add_node(node, parents=parents)


class AcyclicNetwork(NeuralNetwork):
    """
    Concrete child of NeuralNetwork representing an acyclic NeuralNetwork.
    The computational graph of an AcyclicNetwork must correspond to a standard list.
    """

    def __init__(self, identifier: str, input_ids: list[str]):
        super().__init__(identifier, input_ids)

    def add_node(self, node: nodes.ConcreteLayerNode, parents: list[nodes.ConcreteLayerNode] | None = None,
                 children: list[nodes.LayerNode] | None = None):
        """Procedure to add a node to the network only if it preserves the acyclic property

        Parameters
        ----------
        node: ConcreteLayerNode
            The :class:`~pynever.nodes.LayerNode` to add
        parents: list[ConcreteLayerNode] | None
            The new node parents, if existing
        children: list[ConcreteLayerNode] | None
            The new node children, if existing
        """
        self.generic_add_node(node, parents, children)
        if not self.is_acyclic():
            self.remove_node(node)
            raise Exception(f"Adding {node.identifier} with the provided parents and children would create a cycle"
                            f" in the Network!")
