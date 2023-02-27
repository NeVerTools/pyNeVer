import abc

import pynever.nodes as nodes


class NeuralNetwork(abc.ABC):
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
    alt_rep_cache : List<AlternativeRepresentation>
        Dictionary of containing str keys and AlternativeRepresentation values, it contains the
        AlternativeRepresentations of out network.
    up_to_date : bool
        Boolean Flag which shows if the internal representation is up-to-date with respect to the
        AlternativeRepresentations.
    identifier : str
        Identifier of the Sequential Neural Network.

    """

    def __init__(self, identifier: str = ''):
        self.nodes = {}
        self.edges = {}
        self.alt_rep_cache = []
        self.up_to_date = True
        self.identifier = identifier


class SequentialNetwork(NeuralNetwork):
    """
    Concrete children of NeuralNetwork representing a sequential NeuralNetwork. It consists of a graph of LayerNodes
    and a list of AlternativeRepresentations. It should be noted that this data structure it is not able
    to compute the input-output relation defined by the network. The computational graph of a SequentialNetwork must
    correspond to a standard list.

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
    get_next_node(LayerNode)
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

    def is_empty(self) -> bool:
        """
        Procedure to check whether the network is empty.

        Returns
        -------
        bool
            True if there are no nodes in the network, False otherwise.

        """

        return not bool(self.nodes)

    def add_node(self, node: nodes.LayerNode):
        """
        Procedure to add a new LayerNode. In sequential network the new node must be connected directly to the
        previous node forming a list.

        Parameters
        ----------
        node : LayerNode
            New node to add to the Sequential network.

        """

        if len(self.nodes.keys()) == 0:
            self.nodes[node.identifier] = node
            self.edges[node.identifier] = []
        else:
            previous_node_key = self.get_last_node().identifier
            self.nodes[node.identifier] = node
            self.edges[previous_node_key].append(node.identifier)
            self.edges[node.identifier] = []

    def get_first_node(self) -> nodes.LayerNode:
        """
        Procedure to get the first LayerNode of the network.

        Return
        ---------
        LayerNode
            The first node of the network.

        """

        if self.is_empty():
            raise Exception('The network is empty')

        keys = [key for key in self.nodes.keys()]
        for key in self.nodes.keys():
            for sub_key in self.nodes.keys():
                if sub_key in self.edges[key]:
                    keys.remove(sub_key)

        return self.nodes[keys[0]]

    def get_next_node(self, node: nodes.LayerNode) -> nodes.LayerNode:
        """
        Procedure to get the next LayerNode of the network given an input LayerNode.

        Return
        ---------
        LayerNode
            The next node of the network.

        """

        if self.is_empty():
            raise Exception('The network is empty')

        next_node = None
        if node is not None:
            current_key = node.identifier
            if len(self.edges[current_key]) != 0:
                next_key = self.edges[current_key][0]
                next_node = self.nodes[next_key]
        else:
            next_node = self.get_first_node()

        return next_node

    def get_last_node(self) -> nodes.LayerNode:
        """
        Procedure to get the last LayerNode of the network.

        Return
        ---------
        LayerNode
            The last node of the network.

        """

        if self.is_empty():
            raise Exception('The network is empty')

        current_node = self.get_first_node()
        while self.get_next_node(current_node) is not None:
            current_node = self.get_next_node(current_node)

        return current_node

    def delete_last_node(self) -> nodes.LayerNode:
        """
        Procedure to remove the last LayerNode from the network.

        Returns
        ---------
        LayerNode
            The last node of the network.

        """

        if self.is_empty():
            raise Exception('The network is empty')

        last = self.nodes.pop(self.get_last_node().identifier)
        self.edges.pop(last.identifier)

        # Delete reference in edges dict
        if not self.is_empty():
            for k, v in self.edges.items():
                if last.identifier in v:
                    self.edges[k] = []
        return last

    def get_input_len(self) -> int:
        """
        Count the number of inputs in in_dim

        Returns
        -------
        int
            The number of single inputs

        """

        if self.is_empty():
            raise Exception('The network is empty')

        count = 0
        for d in range(len(self.get_first_node().in_dim)):
            count += self.get_first_node().in_dim[d]

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
            raise Exception('The network is empty')

        count = 0
        for d in range(len(self.get_last_node().out_dim)):
            count += self.get_last_node().out_dim[d]

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
