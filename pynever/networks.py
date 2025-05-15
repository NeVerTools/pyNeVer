"""
This module contains the definition and structure of our Neural Network representation

"""

import abc
import collections
import copy

import pynever.nodes as nodes
from pynever.exceptions import EmptyNetworkError, InvalidNodeError, NotInNetworkError
from pynever.nodes import ConcreteLayerNode


class NeuralNetwork(abc.ABC):
    """An abstract class used for our internal representation of a generic NeuralNetwork.
    
    It consists of a graph of LayerNodes and a list of AlternativeRepresentations. 
    It should be noted that this data structure it is not able to compute the input-output 
    relation defined by the network. The properties of the computational graph are
    specialized in the concrete classes.

    :param nodes: Dictionary containing str keys and LayerNodes values. It contains the nodes of the graph,
                the identifier of the node of interest is used as a key in the nodes' dictionary.
    :type nodes: dict[str, LayerNode]
    :param edges: Dictionary of identifiers of LayerNodes, it contains for each node identified by the keys,
                the list of nodes connected to it.
    :type edges: dict[str, list[str]]
    :param identifier: Identifier of the Sequential Neural Network.
    :type identifier: str
    :param input_ids: Dictionary containing the inputs of the networks as keys and the corresponding layer
                    identifier of the Node of which they are the input.
    :type input_ids: dict[str, str | None]
    """

    def __init__(self, identifier: str, input_ids: list[str]):
        self.nodes: dict[str, nodes.ConcreteLayerNode] = {}
        self.edges: dict[str, list[str]] = {}
        self.identifier = identifier
        self.input_ids: dict[str, str | None] = {i: None for i in input_ids}

    def __repr__(self):
        body = [self.nodes[k].__str__() for k in self.get_topological_order()]
        return f"{self.identifier} : {body}"

    def is_empty(self) -> bool:
        """Check whether the network is empty.

        :return: True if there are no nodes in the network, False otherwise.
        :rtype: bool
        """
        return len(self.nodes) == 0

    def is_acyclic(self) -> bool:
        """Check whether the network is acyclic.

        :return: True if network is acyclic, False otherwise. 
        :rtype: bool
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
        """Check if a node has children.

        :param node: The node of which the existence of its children should be checked.
        :type node: ConcreteLayerNode
        :return: True if the node has children, False otherwise.
        :rtype: bool
        """
        return len(self.get_children(node)) != 0

    def get_children(self, node: nodes.ConcreteLayerNode) -> list[nodes.ConcreteLayerNode]:
        """Return the children of a node as a list of ConcreteLayerNodes.

        :param node: The node whose children should be returned.
        :type node: ConcreteLayerNode
        :return: The children of the node passed as argument.
        :rtype: list[ConcreteLayerNode]
        """
        return [self.nodes[child_node_id] for child_node_id in self.edges[node.identifier]]

    def has_parents(self, node: nodes.ConcreteLayerNode) -> bool:
        """Check if a node has parents.

        :param node: The node of which the existence of its parents should be checked.
        :type node: ConcreteLayerNode
        :return: True if the node has parents, False otherwise.
        :rtype: bool
        """
        return len(self.get_parents(node)) != 0

    def get_parents(self, node: nodes.ConcreteLayerNode) -> list[nodes.ConcreteLayerNode]:
        """Return the parents of a node as a list of ConcreteLayerNodes.

        :param node: The node whose parents should be returned.
        :type node: ConcreteLayerNode
        :return: The parents of the node passed as argument.
        :rtype: list[ConcreteLayerNode]
        """
        return [self.nodes[parent_node_id]
                for parent_node_id, end_nodes_ids in self.edges.items()
                if node.identifier in end_nodes_ids]

    def get_input_id(self) -> str:
        """Return the input_id of the network, assuming there is a single input layer.

        :return: The input_id of the network.
        :rtype: str
        """
        return list(self.input_ids.keys())[0]

    def get_roots(self) -> list[nodes.ConcreteLayerNode]:
        """Return the roots of the network as a list of ConcreteLayerNodes.

        :return: The roots of the network as a list of ConcreteLayerNodes. 
        :rtype: list[ConcreteLayerNode]
        """
        return [root_node
                for root_node_id, root_node in self.nodes.items()
                if not self.has_parents(root_node)]

    def get_leaves(self) -> list[nodes.ConcreteLayerNode]:
        """Return the leaves of the network as a list of ConcreteLayerNodes.

        :return: The leaves of the network as a list of ConcreteLayerNodes.
        :rtype: list[ConcreteLayerNode]
        """
        return [leaf_node
                for leaf_node_id, leaf_node in self.nodes.items()
                if not self.has_children(leaf_node)]

    def get_topological_order(self, reverse: bool = False) -> list[str]:
        """Walk the network with a DFS and build the topological sort.
 
        :param reverse: Flag to reverse the order.
        :type reverse: bool
        :return: The topological sort of the network nodes identifiers as a stack.
        :rtype: list[str]
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
        """Build a generator for the layers of the network in sequential order.

        It allows to have an iterable interface when needed.

        :param offset: Offset to start the generation.
        :type offset: int
        :return: The generator object.
        :rtype: Generator[ConcreteLayerNode | None, None]
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
        """Get the first ConcreteLayerNode of the network.

        :return: The first node of the network.
        :rtype: ConcreteLayerNode
        :raises: EmptyNetworkError if network is empty
        """
        if self.is_empty():
            raise EmptyNetworkError()

        return self.nodes[self.get_topological_order()[0]]

    def get_next_node(self, node: nodes.ConcreteLayerNode) -> nodes.ConcreteLayerNode | None:
        """Get the next ConcreteLayerNode of the network given an input ConcreteLayerNode.

        :param node: The input node.
        :type node: ConcreteLayerNode
        :return: The next node of the network.
        :rtype: ConcreteLayerNode | None
        :raises: EmptyNetworkError if network is empty
        """
        if self.is_empty():
            raise EmptyNetworkError()

        order = self.get_topological_order()
        this_idx = order.index(node.identifier)
        return self.nodes[order[this_idx + 1]] if this_idx + 1 < len(order) else None

    def get_previous_node(self, node: nodes.ConcreteLayerNode) -> nodes.ConcreteLayerNode | None:
        """Get the previous ConcreteLayerNode of the network given an input ConcreteLayerNode.

        :param node: The input node.
        :type node: ConcreteLayerNode  
        :return: The previous node in the network.
        :rtype: ConcreteLayerNode | None
        """
        if node == self.get_first_node():
            return None

        order = self.get_topological_order()
        this_idx = order.index(node.identifier)
        return self.nodes[order[this_idx - 1]]

    def get_last_node(self) -> nodes.ConcreteLayerNode:
        """Get the last ConcreteLayerNode of the network.

        :return: The last node of the network.
        :rtype: ConcreteLayerNode
        :raises: EmptyNetworkError if network is empty
        """
        if self.is_empty():
            raise EmptyNetworkError()

        return self.nodes[self.get_topological_order()[-1]]

    def get_input_len(self) -> int:
        """Count the number of inputs in in_dim.

        :return: The number of single inputs.
        :rtype: int
        :raises: EmptyNetworkError if network is empty
        """
        if self.is_empty():
            raise EmptyNetworkError()

        count = 0
        for d in range(len(self.get_first_node().get_input_dim())):
            count += self.get_first_node().get_input_dim()[d]

        return count

    def get_output_len(self) -> int:
        """Count the number of outputs in out_dim.

        :return: The number of single outputs.
        :rtype: int
        :raises: EmptyNetworkError if network is empty
        """
        if self.is_empty():
            raise EmptyNetworkError()

        count = 0
        for d in range(len(self.get_last_node().get_output_dim())):
            count += self.get_last_node().get_output_dim()[d]

        return count

    def get_id_from_index(self, index: int) -> str:
        """Return the identifier of the layer at the given index.

        :param index: Index of the layer to return.
        :type index: int
        :return: The identifier of the layer at the given index.
        :rtype: str
        :raises: IndexError if index out of range
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
        """Return the index of the layer with the given identifier.

        :param identifier: Identifier of the layer to return.
        :type identifier: str
        :return: The index of the layer with the given identifier.
        :rtype: int
        :raises: NotInNetworkError if layer not found
        """
        counter = 0

        for layer in self.layers_iterator():
            if layer.identifier == identifier:
                return counter
            counter += 1

        raise NotInNetworkError(f'There is no layer with identifier {identifier}')

    def get_previous_id(self, identifier: str) -> str | None:
        """Get the identifier of the previous layer given another layer identifier.

        :param identifier: Identifier of the layer.
        :type identifier: str  
        :return: The identifier of the previous layer.
        :rtype: str | None
        :raises: NotInNetworkError if layer not found
        """
        prev_layer = None

        for layer in self.layers_iterator():
            if layer.identifier == identifier:
                return None if prev_layer is None else prev_layer.identifier
            prev_layer = layer

        raise NotInNetworkError(f'There is no layer with identifier {identifier}')

    def layer_precedes(self, layer_id1: str, layer_id2: str) -> bool:
        """Check whether a given layer precedes another or not.

        :param layer_id1: Identifier of the first layer.
        :type layer_id1: str
        :param layer_id2: Identifier of the second layer.
        :type layer_id2: str
        :return: True if layer1 precedes layer2, False otherwise.
        :rtype: bool
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
        """Add a node to the network. A node cannot have both parents and inputs.

        :param node: The node to be added to the network.
        :type node: ConcreteLayerNode  
        :param parents: The parents of the node, defaults to None.
        :type parents: list[ConcreteLayerNode] | None
        :param children: The children of the node, defaults to None.
        :type children: list[ConcreteLayerNode] | None
        :param input_ids: The inputs of the node, defaults to None.
        :type input_ids: list[ConcreteLayerNode] | None
        :raises: Exception if node has both parents and inputs
        :raises: NotInNetworkError if parent/child doesn't exist
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
        """Remove a node from the network.

        :param node: The node to be removed.
        :type node: ConcreteLayerNode
        """
        for parent_node in self.get_parents(node):
            self.edges[parent_node.identifier].remove(node.identifier)

        self.edges.pop(node.identifier)
        self.nodes.pop(node.identifier)

        for i, n in self.input_ids.items():
            if n == node.identifier:
                self.input_ids[i] = None

    def delete_last_node(self) -> nodes.ConcreteLayerNode:
        """Remove the last ConcreteLayerNode from the network.

        :return: The last node of the network.
        :rtype: ConcreteLayerNode
        :raises: EmptyNetworkError if network is empty
        """
        if self.is_empty():
            raise EmptyNetworkError()

        last_node = self.get_last_node()
        self.remove_node(last_node)

        return last_node

    def count_relu_layers(self) -> int:
        """Count the number of ReLU layers of the NN.

        :return: The number of ReLU layers.
        :rtype: int
        """
        relu_count = 0

        for layer in self.layers_iterator():
            if isinstance(layer, nodes.ReLUNode):
                relu_count += 1

        return relu_count


class SequentialNetwork(NeuralNetwork):
    """Concrete child of NeuralNetwork representing a sequential NeuralNetwork.
    
    It consists of a graph of ConcreteLayerNodes. It should be noted that this data structure
    is not able to compute the input-output relation defined by the network.
    The computational graph of a SequentialNetwork must correspond to a standard list.
    """

    def __init__(self, identifier: str, input_id: str):
        super().__init__(identifier, [input_id])

    @staticmethod
    def __is_single_concrete(node: nodes.LayerNode) -> bool:
        """Check whether the LayerNode given as a parameter is a ConcreteLayerNode with a single input.

        :param node: The node to be checked.
        :type node: LayerNode
        :return: True if the LayerNode is a ConcreteLayerNode with a single input, False otherwise.
        :rtype: bool
        """

        return ((node is not None) and
                isinstance(node, nodes.ConcreteLayerNode) and
                isinstance(node.get_input_dim(), tuple))

    def append_node(self, node: nodes.ConcreteLayerNode):
        """Add a new ConcreteLayerNode.

        In a sequential network the new node must be connected directly to the previous node forming a list.

        :param node: New node to add to the Sequential network.
        :type node: ConcreteLayerNode
        :raises: InvalidNodeError if node is not valid
        """
        if not SequentialNetwork.__is_single_concrete(node):
            raise InvalidNodeError(f'{node.identifier} is not a ConcreteLayerNode with a single input!')

        if self.is_empty():
            self.generic_add_node(node, input_ids=[self.get_input_id()])
        else:
            parents = [self.get_last_node()]
            self.generic_add_node(node, parents=parents)


class AcyclicNetwork(NeuralNetwork):
    """Concrete child of NeuralNetwork representing an acyclic NeuralNetwork.
    
    The computational graph of a AcyclicNetwork must correspond to a standard list.
    """

    def __init__(self, identifier: str, input_ids: list[str]):
        super().__init__(identifier, input_ids)

    def add_node(self, node: nodes.ConcreteLayerNode, parents: list[nodes.ConcreteLayerNode] | None = None,
                 children: list[nodes.LayerNode] | None = None):
        """Add a node to the network only if it preserves the acyclic property.

        :param node: The node to add.
        :type node: ConcreteLayerNode
        :param parents: Parent nodes to connect to, defaults to None.
        :type parents: list[ConcreteLayerNode] | None 
        :param children: Child nodes to connect to, defaults to None.
        :type children: list[ConcreteLayerNode] | None
        :raises: Exception if adding the node creates a cycle
        """
        self.generic_add_node(node, parents, children)
        if not self.is_acyclic():
            self.remove_node(node)
            raise Exception(f"Adding {node.identifier} with the provided parents and children would create a cycle"
                            f" in the Network!")
