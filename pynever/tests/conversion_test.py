import math

import pynever.networks as network
import pynever.nodes as nodes
import pynever.strategies.conversion as conversion

float_tolerance = 1e-5


def relu_node_test(converter: conversion.ConversionStrategy):
    print("RELU NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.ReLUNode("ReLU_1", (3, 3, 3)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()
    assert isinstance(start_node, nodes.ReLUNode)
    assert isinstance(end_node, nodes.ReLUNode)
    assert start_node.in_dim == end_node.in_dim
    assert start_node.out_dim == end_node.out_dim
    assert start_node.identifier == end_node.identifier


def sigmoid_node_test(converter: conversion.ConversionStrategy):
    print("SIGMOID NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.SigmoidNode("Sigmoid_1", (3, 3, 3)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()
    assert isinstance(start_node, nodes.SigmoidNode)
    assert isinstance(end_node, nodes.SigmoidNode)
    assert start_node.in_dim == end_node.in_dim
    assert start_node.out_dim == end_node.out_dim
    assert start_node.identifier == end_node.identifier


def fully_connected_node_test(converter: conversion.ConversionStrategy, has_bias: bool):
    print("FULLY CONNECTED NODE TEST")
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
    assert (start_node.weight == end_node.weight).all()
    if start_node.bias is not None:
        assert (start_node.bias == end_node.bias).all()
    else:
        assert start_node.bias is None and end_node.bias is None
    assert start_node.has_bias == end_node.has_bias
    assert start_node.identifier == end_node.identifier


def batchnorm_node_test(converter: conversion.ConversionStrategy):
    print("BATCHNORM NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.BatchNormNode("Batchnorm_1", (4, 5, 6, 3)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.BatchNormNode)
    assert isinstance(end_node, nodes.BatchNormNode)
    assert start_node.in_dim == end_node.in_dim
    assert start_node.out_dim == end_node.out_dim
    assert start_node.num_features == end_node.num_features
    assert (start_node.weight == end_node.weight).all()
    assert (start_node.bias == end_node.bias).all()
    assert (start_node.running_mean == end_node.running_mean).all()
    assert (start_node.running_var == end_node.running_var).all()
    assert math.isclose(start_node.eps, end_node.eps, abs_tol=float_tolerance)
    assert start_node.track_running_stats == end_node.track_running_stats
    assert start_node.affine == end_node.affine
    assert math.isclose(start_node.momentum, end_node.momentum, abs_tol=float_tolerance)
    assert start_node.identifier == end_node.identifier


def conv_node_test(converter: conversion.ConversionStrategy, has_bias: bool):
    print("CONV NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    # dilatation = (0,0) doesn't work somehow
    start_network.add_node(nodes.ConvNode("Conv_1", (3, 32, 32), 3, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 1,
                                          has_bias))
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
    assert (start_node.weight == end_node.weight).all()
    if start_node.bias is not None:
        assert (start_node.bias == end_node.bias).all()
    else:
        assert start_node.bias is None and end_node.bias is None
    assert start_node.has_bias == end_node.has_bias
    assert start_node.identifier == end_node.identifier


def averagepool_node_test(converter: conversion.ConversionStrategy):
    print("AVERAGEPOOL NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.AveragePoolNode("AveragePool_1", (3, 32, 32), (3, 3), (1, 1), (1, 1, 1, 1)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.AveragePoolNode)
    assert isinstance(end_node, nodes.AveragePoolNode)
    assert start_node.in_dim == end_node.in_dim
    assert start_node.out_dim == end_node.out_dim
    assert start_node.kernel_size == end_node.kernel_size
    assert start_node.stride == end_node.stride
    assert start_node.padding == end_node.padding
    assert start_node.ceil_mode == end_node.ceil_mode
    assert start_node.count_include_pad == end_node.count_include_pad
    assert start_node.identifier == end_node.identifier


def maxpool_node_test(converter: conversion.ConversionStrategy):
    print("MAXPOOL NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.MaxPoolNode("Maxpool_1", (3, 32, 32), (3, 3), (1, 1), (1, 1, 1, 1), (0, 0)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.MaxPoolNode)
    assert isinstance(end_node, nodes.MaxPoolNode)
    assert start_node.in_dim == end_node.in_dim
    assert start_node.out_dim == end_node.out_dim
    assert start_node.kernel_size == end_node.kernel_size
    assert start_node.stride == end_node.stride
    assert start_node.padding == end_node.padding
    assert start_node.ceil_mode == end_node.ceil_mode
    assert start_node.return_indices == end_node.return_indices
    assert start_node.identifier == end_node.identifier


def lrn_node_test(converter: conversion.ConversionStrategy):
    print("LRN NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.LRNNode("LRN_1", (3, 32, 32), 3))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.LRNNode)
    assert isinstance(end_node, nodes.LRNNode)
    assert math.isclose(start_node.alpha, end_node.alpha, abs_tol=float_tolerance)
    assert math.isclose(start_node.beta, end_node.beta, abs_tol=float_tolerance)
    assert math.isclose(start_node.k, end_node.k, abs_tol=float_tolerance)
    assert start_node.size == end_node.size
    assert start_node.identifier == end_node.identifier


def softmax_node_test(converter: conversion.ConversionStrategy):
    print("SOFTMAX NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.SoftMaxNode("Softmax_1", (3, 4, 5, 6)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.SoftMaxNode)
    assert isinstance(end_node, nodes.SoftMaxNode)
    assert start_node.axis == end_node.axis
    assert start_node.identifier == end_node.identifier


def unsqueeze_node_test(converter: conversion.ConversionStrategy):
    print("UNSQUEEZE NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.UnsqueezeNode("Unsqueeze_1", (3, 4, 5, 6), (0, 3)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.UnsqueezeNode)
    assert isinstance(end_node, nodes.UnsqueezeNode)
    assert start_node.axes == end_node.axes
    assert start_node.identifier == end_node.identifier


def reshape_node_test(converter: conversion.ConversionStrategy):
    print("RESHAPE NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.ReshapeNode("Reshape_1", (3, 4, 5, 6), (3, 0, -1), False))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.ReshapeNode)
    assert isinstance(end_node, nodes.ReshapeNode)
    assert start_node.shape == end_node.shape
    assert start_node.identifier == end_node.identifier


def flatten_node_test(converter: conversion.ConversionStrategy):
    print("FLATTEN NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.FlattenNode("Flatten_1", (3, 4, 5, 6)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.FlattenNode)
    assert isinstance(end_node, nodes.FlattenNode)
    assert start_node.axis == end_node.axis
    assert start_node.identifier == end_node.identifier


def dropout_node_test(converter: conversion.ConversionStrategy):
    print("DROPOUT NODE TEST")
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.DropoutNode("Dropout_1", (3, 4, 5, 6)))
    alt_network = converter.from_neural_network(start_network)
    end_network = converter.to_neural_network(alt_network)
    assert isinstance(end_network, network.SequentialNetwork)
    start_node = start_network.get_first_node()
    end_node = end_network.get_first_node()

    assert isinstance(start_node, nodes.DropoutNode)
    assert isinstance(end_node, nodes.DropoutNode)
    assert start_node.p == end_node.p
    assert start_node.identifier == end_node.identifier


converters = [conversion.ONNXConverter(), conversion.PyTorchConverter()]

for conv in converters:
    print(f"Test for {conv.__class__.__name__}")
    relu_node_test(conv)
    sigmoid_node_test(conv)
    fully_connected_node_test(conv, True)
    fully_connected_node_test(conv, False)
    batchnorm_node_test(conv)
    conv_node_test(conv, True)
    conv_node_test(conv, False)
    averagepool_node_test(conv)
    maxpool_node_test(conv)
    lrn_node_test(conv)
    softmax_node_test(conv)
    unsqueeze_node_test(conv)
    reshape_node_test(conv)
    flatten_node_test(conv)
    dropout_node_test(conv)
