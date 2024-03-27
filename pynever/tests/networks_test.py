import pynever.networks as pyn_networks
import pynever.nodes as pyn_nodes
import networkx
import matplotlib.pyplot as plt


def test_neural_network():

    network = pyn_networks.NeuralNetwork("Test Neural Network")

    root_node_1 = pyn_nodes.FullyConnectedNode("FC_1", (2,), 16)
    root_node_2 = pyn_nodes.FullyConnectedNode("FC_2", (3,), 32)

    child_node_1 = pyn_nodes.ReLUNode("ReLU_1", (16,))
    child_node_2 = pyn_nodes.ReLUNode("ReLU_2", (32,))

    multi_input_node = pyn_nodes.ConcatNode("Concat_1", [(16,), (32,)])

    leaf_node_1 = pyn_nodes.FullyConnectedNode("FC_3", (48,), 8)
    leaf_node_2 = pyn_nodes.FullyConnectedNode("FC_4", (48,), 4)

    network.generic_add_node(root_node_1)
    network.generic_add_node(root_node_2)

    network.generic_add_node(child_node_1, [root_node_1])
    network.generic_add_node(child_node_2, [root_node_2])

    network.generic_add_node(leaf_node_1)

    network.generic_add_node(multi_input_node, [child_node_1, child_node_2], [leaf_node_1])

    network.generic_add_node(leaf_node_2, [multi_input_node])

    temp_g = networkx.DiGraph(network.edges)
    networkx.draw_networkx(temp_g)
    plt.show()

    print(network.get_roots())
    print(network.get_leaves())

    print(network.get_children(multi_input_node))
    print(network.get_parents(multi_input_node))

    print(network.get_children(leaf_node_1))
    print(network.get_parents(root_node_1))

    print(network.has_parents(multi_input_node))
    print(network.has_children(multi_input_node))
    print(network.has_parents(root_node_2))
    print(network.has_children(leaf_node_2))

    print(network.remove_node(multi_input_node))

    temp_g = networkx.DiGraph(network.edges)
    networkx.draw_networkx(temp_g)
    plt.show()

    print(network.get_roots())
    print(network.get_leaves())

    network.generic_add_node(multi_input_node, [child_node_1, child_node_2], [leaf_node_1, leaf_node_2])

    temp_g = networkx.DiGraph(network.edges)
    networkx.draw_networkx(temp_g)
    plt.show()

    cycle_node = pyn_nodes.FullyConnectedNode("FC_cycle", (8,), 2)
    network.generic_add_node(cycle_node, [leaf_node_1], [root_node_1])

    temp_g = networkx.DiGraph(network.edges)
    networkx.draw_networkx(temp_g)
    plt.show()


def test_sequential_network():

    network = pyn_networks.SequentialNetwork("Test Sequential Network", "X")

    print(network.is_empty())

    input_node = pyn_nodes.FullyConnectedNode("FC_1", (2,), 16)
    hidden_node1 = pyn_nodes.ReLUNode("ReLU_1", (16,))
    hidden_node2 = pyn_nodes.FullyConnectedNode("FC_2", (16,), 32)
    hidden_node3 = pyn_nodes.ReLUNode("ReLU_2", (32,))
    output_node = pyn_nodes.FullyConnectedNode("FC_3", (32,), 8)

    network.add_node(input_node)
    network.add_node(hidden_node1)
    network.add_node(hidden_node2)
    network.add_node(hidden_node3)
    network.add_node(output_node)

    temp_g = networkx.DiGraph(network.edges)
    networkx.draw_networkx(temp_g)
    plt.show()

    print(network.is_empty())
    print(network.get_last_node())
    print(network.get_first_node())
    print(network.get_next_node(network.get_first_node()))
    print(network.get_next_node(network.get_last_node()))


def test_acyclic_network():

    network = pyn_networks.AcyclicNetwork("Test Neural Network", ['X', 'Z'])

    root_node_1 = pyn_nodes.FullyConnectedNode("FC_1", (2,), 16)
    root_node_2 = pyn_nodes.FullyConnectedNode("FC_2", (3,), 32)

    child_node_1 = pyn_nodes.ReLUNode("ReLU_1", (16,))
    child_node_2 = pyn_nodes.ReLUNode("ReLU_2", (32,))

    multi_input_node = pyn_nodes.ConcatNode("Concat_1", [(16,), (32,)])

    leaf_node_1 = pyn_nodes.FullyConnectedNode("FC_3", (48,), 8)
    leaf_node_2 = pyn_nodes.FullyConnectedNode("FC_4", (48,), 4)

    network.add_node(root_node_1)
    network.add_node(root_node_2)

    network.add_node(child_node_1, [root_node_1])
    network.add_node(child_node_2, [root_node_2])

    network.add_node(leaf_node_1)

    network.add_node(multi_input_node, [child_node_1, child_node_2], [leaf_node_1])

    network.add_node(leaf_node_2, [multi_input_node])

    temp_g = networkx.DiGraph(network.edges)
    networkx.draw_networkx(temp_g)
    plt.show()

    print(network.get_roots())
    print(network.get_leaves())

    print(network.get_children(multi_input_node))
    print(network.get_parents(multi_input_node))

    print(network.get_children(leaf_node_1))
    print(network.get_parents(root_node_1))

    print(network.has_parents(multi_input_node))
    print(network.has_children(multi_input_node))
    print(network.has_parents(root_node_2))
    print(network.has_children(leaf_node_2))

    cycle_node = pyn_nodes.FullyConnectedNode("FC_cycle", (8,), 2)
    network.add_node(cycle_node, [leaf_node_1], [root_node_1])


if __name__ == "__main__":
    test_neural_network()





