import numpy as np
import tensorflow as tf

import pynever.networks as network
import pynever.nodes as nodes
import pynever.strategies.conversion as conversion

float_tolerance = 1e-5


def relu_node_test(converter: conversion.ConversionStrategy):
    print("RELU NODE TEST")
    tensor = np.random.rand(3, 3, 3)
    print(tensor)

    start_network = network.SequentialNetwork("NET_TEST", "tensor")
    start_network.add_node(nodes.ReLUNode("ReLU_1", (3, 3, 3)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    layer = tf.keras.layers.ReLU()
    output = layer(tensor)
    print("** TENSORFLOW TEST ** \n")
    print(output)


def sigmoid_node_test(converter: conversion.ConversionStrategy):
    print("SIGMOID NODE TEST")
    tensor = np.random.rand(3, 3, 3)
    print(tensor)

    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.SigmoidNode("Sigmoid_1", (3, 3, 3)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    layer = tf.keras.activations.sigmoid(tensor)
    print("** TENSORFLOW TEST ** \n")
    print(layer)


def fully_connected_node_test(converter: conversion.ConversionStrategy, has_bias: bool):
    # TODO: check kernel
    print("FULLY CONNECTED NODE TEST")
    tensor = np.random.rand(3, 4, 5)
    # print(tensor)

    # weight = np.array([[0.21652315, -0.04425954, 0.12699856, -0.09156641, -0.14583177],
    #                    [-0.13583256, -0.08782442, -0.28941352, -0.34822181, -0.32571547],
    #                    [-0.3720666, 0.03280868, 0.40765258, 0.20728782, 0.14109302],
    #                    [-0.15650549, -0.3554072, 0.13704456, -0.09792434, 0.15857265],
    #                    [-0.34614734, -0.25804695, 0.18050925, 0.38407865, 0.0462972]])
    weight = np.ones((5, 5))
    # print(weight)

    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.FullyConnectedNode("FullyConnected_1", (3, 4, 5), 5, weight=weight, has_bias=has_bias))
    alt_network = converter.from_neural_network(start_network)

    # kernel = alt_network.tensorflow_network.layers[0].kernel
    # if has_bias:
    #     bias = alt_network.tensorflow_network.layers[0].bias
    # else:
    #     bias = np.zeros((5,))

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    # layer = tf.keras.layers.Dense(5, activation=None, use_bias=has_bias, kernel_constraint=MyConstraint(weight))
    layer = tf.keras.layers.Dense(5, activation=None, use_bias=has_bias, kernel_initializer='ones',
                                  bias_initializer='zeros')
    output = layer(tensor)
    print("** TENSORFLOW TEST ** \n")
    print(output)


def batchnorm_node_test(converter: conversion.ConversionStrategy):
    print("BATCHNORM NODE TEST")
    tensor = np.random.rand(4, 5, 6, 3, 1)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.BatchNormNode("Batchnorm_1", (4, 5, 6, 3)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    output = tf.keras.layers.BatchNormalization(4, 0.1, 1e-05, True, True)(tensor)
    print("** TENSORFLOW TEST ** \n")
    print(output)


def conv_node_test(converter: conversion.ConversionStrategy, has_bias: bool):
    # TODO: check kernel
    print("CONV NODE TEST")
    weight = np.ones((3, 3, 3, 3))
    tensor = np.random.rand(3, 32, 32, 3)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.ConvNode("Conv_1", (3, 32, 32), 3, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 1,
                                          has_bias, weight=weight))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    # If we set kernel_initializer='ones' in the Conv2d class we get the same output
    output = tf.keras.layers.Conv2D(3, (3, 3), (1, 1), "valid", "channels_last", (1, 1), 1, activation=None,
                                    use_bias=has_bias, kernel_initializer='ones')(tensor)
    print(output)


def averagepool_node_test(converter: conversion.ConversionStrategy):
    print("AVERAGEPOOL NODE TEST")
    tensor = np.random.rand(3, 32, 32, 3)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.AveragePoolNode("AveragePool_1", (3, 32, 32), (3, 3), (1, 1), (1, 1, 1, 1)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    output = tf.keras.layers.AveragePooling2D(3, (1, 1), "valid", "channels_last")(tensor)
    print(output)


def maxpool_node_test(converter: conversion.ConversionStrategy):
    print("MAXPOOL NODE TEST")
    tensor = np.random.rand(3, 32, 32, 3)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.MaxPoolNode("Maxpool_1", (3, 32, 32), (3, 3), (1, 1), (1, 1, 1, 1), (0, 0)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    output = tf.keras.layers.MaxPooling2D(3, (1, 1), "valid", "channels_last")(tensor)
    print(output)


def lrn_node_test(converter: conversion.ConversionStrategy):
    print("LRN NODE TEST")
    tensor = np.random.rand(3, 32, 32, 3)
    # float64 not accepted
    tensor = np.float32(tensor)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.LRNNode("LRN_1", (3, 32, 32), 3))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    output = tf.nn.local_response_normalization(tensor, depth_radius=3, bias=1, alpha=0.0001, beta=0.75, name=None)
    print(output)


def softmax_node_test(converter: conversion.ConversionStrategy):
    print("SOFTMAX NODE TEST")
    tensor = np.random.rand(1, 2, 3)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.SoftMaxNode("Softmax_1", (3, 4, 5, 6)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    output = tf.keras.layers.Softmax(-1)(tensor)
    print(output)


def unsqueeze_node_test(converter: conversion.ConversionStrategy):
    print("UNSQUEEZE NODE TEST")
    tensor = np.random.rand(1, 2, 3)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.UnsqueezeNode("Unsqueeze_1", (3, 4, 5, 6), (0,)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    output = tf.expand_dims(tensor, 0)
    print(output)


def reshape_node_test(converter: conversion.ConversionStrategy):
    print("RESHAPE NODE TEST")
    tensor = np.random.rand(2, 2, 2)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.ReshapeNode("Reshape_1", (2, 2, 2), (2, 4), False))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    # batch dimension to consider
    output = tf.reshape(tensor, [1, 2, 4])
    print(output)


def flatten_node_test(converter: conversion.ConversionStrategy):
    print("FLATTEN NODE TEST")
    tensor = np.random.rand(3, 4, 5, 6, 6)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.FlattenNode("Flatten_1", (3, 4, 5, 6)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    output = tf.keras.layers.Flatten('channels_last')(tensor)
    print(output)


def dropout_node_test(converter: conversion.ConversionStrategy):
    print("DROPOUT NODE TEST")
    tensor = np.random.rand(3, 1, 2, 4)
    print(tensor)
    start_network = network.SequentialNetwork("NET_TEST", "X")
    start_network.add_node(nodes.DropoutNode("Dropout_1", (3, 4, 5, 6)))
    alt_network = converter.from_neural_network(start_network)

    output = alt_network.tensorflow_network(tensor)
    print("** CONVERSION TEST ** \n")
    print(output)
    print("\n")

    print("** TENSORFLOW TEST ** \n")
    output = tf.keras.layers.Dropout(0.5)(tensor)
    print(output)


converters = [conversion.TensorflowConverter()]
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
