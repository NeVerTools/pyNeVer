import pynever.networks as network
import pynever.nodes as nodes
import pynever.strategies.conversion as conversion


def fully_connected_node_test(converter: conversion.ConversionStrategy, has_bias: bool):
    print("Keras: fully connected node test")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.FullyConnectedNode("FullyConnected_1", (3, 4, 5), 5, has_bias=has_bias))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.FullyConnectedNode)
    assert isinstance(end_node, nodes.FullyConnectedNode)
    assert start_node.in_dim == end_node.in_dim
    assert start_node.out_dim == end_node.out_dim
    assert start_node.in_features == end_node.in_features
    assert start_node.out_features == end_node.out_features
    if not has_bias:
        assert start_node.bias is None and end_node.bias is None
    assert start_node.has_bias == end_node.has_bias


def conv_node_test(converter: conversion.ConversionStrategy, has_bias: bool):
    print("Keras: convolutional node test")

    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.ConvNode("Conv_1", (3, 32, 32), 3, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 1, has_bias))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.ConvNode)
    assert isinstance(end_node, nodes.ConvNode)
    assert start_node.in_dim == end_node.in_dim
    assert start_node.out_dim == end_node.out_dim
    assert start_node.in_channels == end_node.in_channels
    assert start_node.out_channels == end_node.out_channels
    assert start_node.kernel_size == end_node.kernel_size
    assert start_node.stride == end_node.stride
    assert start_node.padding == end_node.padding
    assert start_node.dilation == end_node.dilation
    assert start_node.groups == end_node.groups
    if not has_bias:
        assert start_node.bias is None and end_node.bias is None
    assert start_node.has_bias == end_node.has_bias


conv = conversion.TensorflowConverter()

fully_connected_node_test(conv, True)
fully_connected_node_test(conv, False)
conv_node_test(conv, True)
conv_node_test(conv, False)
