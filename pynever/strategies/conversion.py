import abc
import copy

import numpy as np
import onnx
import onnx.numpy_helper
import torch

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.pytorch_layers as pyt_l


class AlternativeRepresentation(abc.ABC):
    """
    An abstract class used to represent an alternative representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation
    up_to_date : bool, optional
        flag which indicates if the alternative representation is up to date with respect
        to the internal representation of the network (optional: True).

    """

    def __init__(self, identifier: str, up_to_date: bool = True):
        self.identifier = identifier
        self.up_to_date = up_to_date


class ONNXNetwork(AlternativeRepresentation):
    """
    An class used to represent a ONNX representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation
    onnx_network : onnx.ModelProto
        Real ONNX network.
    up_to_date : bool
        flag which indicates if the alternative representation is up to date with respect
        to the internal representation of the network (optional: True).

    """

    def __init__(self, identifier: str, onnx_network: onnx.ModelProto, up_to_date: bool = True):
        super().__init__(identifier, up_to_date)
        self.onnx_network = copy.deepcopy(onnx_network)


class PyTorchNetwork(AlternativeRepresentation):
    """
    An class used to represent a PyTorch representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation
    pytorch_network : torch.nn.Module
        Real PyTorch network.
    up_to_date : bool
        flag which indicates if the alternative representation is up to date with respect
        to the internal representation of the network (optional: True).

    """

    def __init__(self, identifier: str, pytorch_network: torch.nn.Module, up_to_date: bool = True):
        super().__init__(identifier, up_to_date)
        self.pytorch_network = copy.deepcopy(pytorch_network)


class TensorflowNetwork(AlternativeRepresentation):
    """
    An class used to represent a Tensorflow representation for a neural network.

    Attributes
    ----------
    identifier : str
        identifier for the alternative representation
    up_to_date : bool
        flag which indicates if the alternative representation is up to date with respect
        to the internal representation of the network (optional: True).

    """

    def __init__(self, identifier: str, up_to_date: bool = True):
        super().__init__(identifier, up_to_date)


class ConversionStrategy(abc.ABC):
    """
    An abstract class used to represent a Conversion Strategy.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to an alternative representation determined in the concrete children.
    to_neural_network(AlternativeRepresentation)
        Convert the alternative representation of interest to our internal representation of a Neural Network.

    """

    @abc.abstractmethod
    def from_neural_network(self, network: networks.NeuralNetwork) -> AlternativeRepresentation:
        """
        Convert the neural network of interest to an alternative representation determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        AlternativeRepresentation
            The alternative representation resulting from the conversion of the original network.
        """
        pass

    @abc.abstractmethod
    def to_neural_network(self, alt_rep: AlternativeRepresentation) -> networks.NeuralNetwork:
        """
        Convert the alternative representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : AlternativeRepresentation
            The Alternative Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of Alternative Representation.
        """
        pass


class ONNXConverter(ConversionStrategy):
    """
    A class used to represent the conversion strategy for ONNX models.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to a ONNXNetwork model.
    to_neural_network(ONNXNetwork)
        Convert the ONNXNetwork of interest to our internal representation of a Neural Network.

    """

    @staticmethod
    def __add_onnx_relu(current_input: str, current_output: str, onnx_nodes: list):

        onnx_node = onnx.helper.make_node(
            'Relu',
            inputs=[current_input],
            outputs=[current_output],
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_sigmoid(current_input: str, current_output: str, onnx_nodes: list):

        onnx_node = onnx.helper.make_node(
            'Sigmoid',
            inputs=[current_input],
            outputs=[current_output],
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_linear(current_node: nodes.FullyConnectedNode, current_input: str, current_output: str,
                          onnx_nodes: list, input_info: list, initializers: list):

        input_weight = current_node.identifier + "_weight"

        weight_value_info = onnx.helper.make_tensor_value_info(input_weight, onnx.TensorProto.DOUBLE,
                                                               [current_node.out_features,
                                                                current_node.in_features])

        weight_tensor = onnx.numpy_helper.from_array(current_node.weight.T, input_weight)

        if current_node.has_bias:
            input_bias = current_node.identifier + "_bias"
            bias_value_info = onnx.helper.make_tensor_value_info(input_bias, onnx.TensorProto.DOUBLE,
                                                                 [current_node.out_features])
            bias_tensor = onnx.numpy_helper.from_array(current_node.bias, input_bias)

            onnx_node = onnx.helper.make_node(
                'Gemm',
                inputs=[current_input, input_weight, input_bias],
                outputs=[current_output],
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=0
            )

            input_info.append(bias_value_info)
            initializers.append(bias_tensor)

        else:
            onnx_node = onnx.helper.make_node(
                'Gemm',
                inputs=[current_input, input_weight],
                outputs=[current_output],
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=0
            )

        input_info.append(weight_value_info)
        initializers.append(weight_tensor)
        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_batchnorm(current_node: nodes.BatchNormNode, current_input: str, current_output: str,
                             onnx_nodes: list, input_info: list, initializers: list):

        input_scale = current_node.identifier + "_scale"
        input_bias = current_node.identifier + "_bias"
        input_mean = current_node.identifier + "_mean"
        input_var = current_node.identifier + "_var"

        scale_value_info = onnx.helper.make_tensor_value_info(input_scale, onnx.TensorProto.DOUBLE,
                                                              [current_node.num_features])
        bias_value_info = onnx.helper.make_tensor_value_info(input_bias, onnx.TensorProto.DOUBLE,
                                                             [current_node.num_features])
        mean_value_info = onnx.helper.make_tensor_value_info(input_mean, onnx.TensorProto.DOUBLE,
                                                             [current_node.num_features])
        var_value_info = onnx.helper.make_tensor_value_info(input_var, onnx.TensorProto.DOUBLE,
                                                            [current_node.num_features])

        scale_tensor = onnx.numpy_helper.from_array(current_node.weight, input_scale)
        bias_tensor = onnx.numpy_helper.from_array(current_node.bias, input_bias)
        mean_tensor = onnx.numpy_helper.from_array(current_node.running_mean, input_mean)
        var_tensor = onnx.numpy_helper.from_array(current_node.running_var, input_var)

        onnx_node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=[current_input, input_scale, input_bias, input_mean, input_var],
            outputs=[current_output],
            epsilon=current_node.eps,
            momentum=current_node.momentum,
            training_mode=int(current_node.track_running_stats)
        )

        input_info.append(scale_value_info)
        input_info.append(bias_value_info)
        input_info.append(mean_value_info)
        input_info.append(var_value_info)

        initializers.append(scale_tensor)
        initializers.append(bias_tensor)
        initializers.append(mean_tensor)
        initializers.append(var_tensor)

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_conv(current_node: nodes.ConvNode, current_input: str, current_output: str,
                        onnx_nodes: list, input_info: list, initializers: list):

        weight_size = list(current_node.weight.shape)
        input_weight = current_node.identifier + "_weight"

        weight_value_info = onnx.helper.make_tensor_value_info(input_weight, onnx.TensorProto.DOUBLE,
                                                               weight_size)

        weight_tensor = onnx.numpy_helper.from_array(current_node.weight, input_weight)

        if current_node.has_bias:

            input_bias = current_node.identifier + "_bias"
            bias_size = list(current_node.bias.shape)

            bias_value_info = onnx.helper.make_tensor_value_info(input_bias, onnx.TensorProto.DOUBLE,
                                                                 bias_size)
            bias_tensor = onnx.numpy_helper.from_array(current_node.bias, input_bias)

            onnx_node = onnx.helper.make_node(
                'Conv',
                inputs=[current_input, input_weight, input_bias],
                outputs=[current_output],
                kernel_shape=list(current_node.kernel_size),
                strides=list(current_node.stride),
                dilations=list(current_node.dilation),
                groups=current_node.groups,
                pads=list(current_node.padding)
            )

            input_info.append(bias_value_info)
            initializers.append(bias_tensor)

        else:
            onnx_node = onnx.helper.make_node(
                'Conv',
                inputs=[current_input, input_weight],
                outputs=[current_output],
                kernel_shape=list(current_node.kernel_size),
                strides=list(current_node.stride),
                dilations=list(current_node.dilation),
                groups=current_node.groups,
                pads=list(current_node.padding)
            )

        input_info.append(weight_value_info)
        initializers.append(weight_tensor)

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_averagepool(current_node: nodes.AveragePoolNode, current_input: str, current_output: str,
                               onnx_nodes: list):

        onnx_node = onnx.helper.make_node(
            'AveragePool',
            inputs=[current_input],
            outputs=[current_output],
            ceil_mode=int(current_node.ceil_mode),
            count_include_pad=int(current_node.count_include_pad),
            kernel_shape=list(current_node.kernel_size),
            strides=list(current_node.stride),
            pads=list(current_node.padding)
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_maxpool(current_node: nodes.MaxPoolNode, current_input: str, current_output: str,
                           onnx_nodes: list):

        # N.B. we do not support the attribute storage_order of ONNX
        # ONNX does not support the return_indices parameters

        onnx_node = onnx.helper.make_node(
            'MaxPool',
            inputs=[current_input],
            outputs=[current_output],
            ceil_mode=int(current_node.ceil_mode),
            dilations=current_node.dilation,
            kernel_shape=list(current_node.kernel_size),
            strides=list(current_node.stride),
            pads=list(current_node.padding)
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_lrn(current_node: nodes.LRNNode, current_input: str, current_output: str,
                       onnx_nodes: list):

        onnx_node = onnx.helper.make_node(
            'LRN',
            inputs=[current_input],
            outputs=[current_output],
            alpha=current_node.alpha,
            beta=current_node.beta,
            bias=current_node.k,
            size=current_node.size
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_softmax(current_node: nodes.SoftMaxNode, current_input: str, current_output: str,
                           onnx_nodes: list):

        # Since our representation do not consider the batch dimension we need to scale the axis by 1
        # when we pass to the onnx representation.
        temp_axis = current_node.axis + 1
        onnx_node = onnx.helper.make_node(
            'Softmax',
            inputs=[current_input],
            outputs=[current_output],
            axis=temp_axis
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_unsqueeze(current_node: nodes.UnsqueezeNode, current_input: str, current_output: str,
                             onnx_nodes: list, input_info: list, initializers: list):

        axes_size = [len(current_node.axes)]
        input_axes = current_node.identifier + "_axes"

        axes_value_info = onnx.helper.make_tensor_value_info(input_axes, onnx.TensorProto.INT64,
                                                             axes_size)

        # Since our representation do not consider the batch dimension we need to scale all the axes
        # by 1 when we pass to the onnx representation.
        temp_axes = [e + 1 for e in current_node.axes]
        axes_tensor = onnx.numpy_helper.from_array(np.array(temp_axes), input_axes)

        onnx_node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=[current_input, input_axes],
            outputs=[current_output]
        )

        input_info.append(axes_value_info)
        initializers.append(axes_tensor)

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_reshape(current_node: nodes.ReshapeNode, current_input: str, current_output: str,
                           onnx_nodes: list, input_info: list, initializers: list):

        # Need to add the batch dimension to the shape
        temp_shape = [1]
        for e in current_node.shape:
            temp_shape.append(e)
        shape_size = [len(current_node.shape)]
        input_shape = current_node.identifier + "_shape"

        shape_value_info = onnx.helper.make_tensor_value_info(input_shape, onnx.TensorProto.INT64,
                                                              shape_size)

        shape_tensor = onnx.numpy_helper.from_array(np.array(temp_shape), input_shape)

        onnx_node = onnx.helper.make_node(
            'Reshape',
            inputs=[current_input, input_shape],
            outputs=[current_output],
            allow_zero=int(current_node.allow_zero)
        )

        input_info.append(shape_value_info)
        initializers.append(shape_tensor)

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_flatten(current_node: nodes.FlattenNode, current_input: str, current_output: str,
                           onnx_nodes: list):

        # Since our representation do not consider the batch dimension we need to scale the axis by 1
        # when we pass to the onnx representation.
        temp_axis = current_node.axis + 1

        onnx_node = onnx.helper.make_node(
            'Flatten',
            inputs=[current_input],
            outputs=[current_output],
            axis=temp_axis
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_dropout(current_node: nodes.DropoutNode, current_input: str, current_output: str,
                           onnx_nodes: list, input_info: list, initializers: list):

        # N.B. we do not support the seed attribute and the training_mode input.

        ratio_size = [1]
        input_ratio = current_node.identifier + "_ratio"

        ratio_value_info = onnx.helper.make_tensor_value_info(input_ratio, onnx.TensorProto.DOUBLE,
                                                              ratio_size)

        ratio_tensor = onnx.numpy_helper.from_array(np.array([current_node.p]), input_ratio)

        onnx_node = onnx.helper.make_node(
            'Dropout',
            inputs=[current_input, input_ratio],
            outputs=[current_output]
        )

        input_info.append(ratio_value_info)
        initializers.append(ratio_tensor)

        onnx_nodes.append(onnx_node)

    def from_neural_network(self, network: networks.NeuralNetwork) -> ONNXNetwork:
        """
        Convert the neural network of interest to a ONNX representation.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        ONNXNetwork
            The ONNX representation resulting from the conversion of the original network.

        """

        alt_net = None
        for alt_rep in network.alt_rep_cache:
            if isinstance(alt_rep, ONNXNetwork) and alt_rep.up_to_date:
                alt_net = alt_rep

        if alt_net is None:

            if not network.up_to_date:

                for alt_rep in network.alt_rep_cache:

                    if alt_rep.up_to_date:

                        if isinstance(alt_rep, PyTorchNetwork):
                            pytorch_cv = PyTorchConverter()
                            network = pytorch_cv.to_neural_network(alt_rep)

                        else:
                            raise NotImplementedError
                        break

            if isinstance(network, networks.SequentialNetwork):

                current_node = None
                previous_output = network.input_id
                input_info = []
                output_info = []
                initializers = []
                onnx_nodes = []

                while network.get_next_node(current_node) is not None:

                    current_node = network.get_next_node(current_node)
                    current_input = previous_output
                    current_output = current_node.identifier

                    input_dim = [1]
                    for e in current_node.in_dim:
                        input_dim.append(e)

                    output_dim = [1]
                    for e in current_node.out_dim:
                        output_dim.append(e)

                    input_value_info = onnx.helper.make_tensor_value_info(current_input, onnx.TensorProto.DOUBLE,
                                                                          input_dim)
                    output_value_info = onnx.helper.make_tensor_value_info(current_output, onnx.TensorProto.DOUBLE,
                                                                           output_dim)

                    input_info.append(input_value_info)
                    output_info.append(output_value_info)

                    if isinstance(current_node, nodes.ReLUNode):
                        self.__add_onnx_relu(current_input, current_output, onnx_nodes)

                    elif isinstance(current_node, nodes.SigmoidNode):
                        self.__add_onnx_sigmoid(current_input, current_output, onnx_nodes)

                    elif isinstance(current_node, nodes.FullyConnectedNode):
                        self.__add_onnx_linear(current_node, current_input, current_output, onnx_nodes, input_info,
                                               initializers)

                    elif isinstance(current_node, nodes.BatchNormNode):
                        self.__add_onnx_batchnorm(current_node, current_input, current_output, onnx_nodes, input_info,
                                                  initializers)

                    elif isinstance(current_node, nodes.ConvNode):
                        self.__add_onnx_conv(current_node, current_input, current_output, onnx_nodes, input_info,
                                             initializers)

                    elif isinstance(current_node, nodes.AveragePoolNode):
                        self.__add_onnx_averagepool(current_node, current_input, current_output, onnx_nodes)

                    elif isinstance(current_node, nodes.MaxPoolNode):
                        self.__add_onnx_maxpool(current_node, current_input, current_output, onnx_nodes)

                    elif isinstance(current_node, nodes.LRNNode):
                        self.__add_onnx_lrn(current_node, current_input, current_output, onnx_nodes)

                    elif isinstance(current_node, nodes.SoftMaxNode):
                        self.__add_onnx_softmax(current_node, current_input, current_output, onnx_nodes)

                    elif isinstance(current_node, nodes.UnsqueezeNode):
                        self.__add_onnx_unsqueeze(current_node, current_input, current_output, onnx_nodes, input_info,
                                                  initializers)

                    elif isinstance(current_node, nodes.ReshapeNode):
                        self.__add_onnx_reshape(current_node, current_input, current_output, onnx_nodes, input_info,
                                                initializers)

                    elif isinstance(current_node, nodes.FlattenNode):
                        self.__add_onnx_flatten(current_node, current_input, current_output, onnx_nodes)

                    elif isinstance(current_node, nodes.DropoutNode):
                        self.__add_onnx_dropout(current_node, current_input, current_output, onnx_nodes, input_info,
                                                initializers)

                    else:
                        raise NotImplementedError

                    previous_output = current_output

                onnx_graph = onnx.helper.make_graph(
                    nodes=onnx_nodes,
                    name=network.identifier,
                    inputs=[input_info[0]],
                    outputs=[output_info[-1]],
                    initializer=initializers,
                    value_info=input_info
                )

                onnx_network = onnx.helper.make_model(graph=onnx_graph)
                alt_net = ONNXNetwork(network.identifier, onnx_network)

            else:
                raise NotImplementedError

        return alt_net

    def to_neural_network(self, alt_rep: ONNXNetwork) -> networks.NeuralNetwork:
        """
        Convert the ONNX representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : ONNXNetwork
            The ONNX Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of ONNX Representation.

        """

        identifier = alt_rep.identifier
        network = networks.SequentialNetwork(identifier, alt_rep.onnx_network.graph.input[0].name)

        parameters = {}
        for initializer in alt_rep.onnx_network.graph.initializer:
            parameters[initializer.name] = onnx.numpy_helper.to_array(initializer)

        shape_info = {}
        for i in alt_rep.onnx_network.graph.input:
            shape = []
            for dim in i.type.tensor_type.shape.dim:
                shape.append(dim.dim_value)
            shape_info[i.name] = shape

        node_index = 1
        in_dim = tuple(shape_info[alt_rep.onnx_network.graph.input[0].name][1:])

        for node in alt_rep.onnx_network.graph.node:

            if node.op_type == "Relu":

                # We assume that the real input of the node is always the first element of node.input
                # and the first element of the shape is the batch placeholder

                network.add_node(nodes.ReLUNode(node.output[0], in_dim))

            elif node.op_type == "Sigmoid":

                network.add_node(nodes.SigmoidNode(node.output[0], in_dim))

            elif node.op_type == "Gemm":
                # We assume that the weight tensor is always the second element of node.input and the bias tensor
                # is always the third.
                # N.B: We do not support the attributes transA and transB,
                # therefore we need to transpose the weight vector.

                if (node.attribute[2].name == 'transA' or node.attribute[2].name == 'transB') and \
                        node.attribute[2].i == 0:
                    weight = parameters[node.input[1]].T
                else:
                    weight = parameters[node.input[1]]
                if len(node.input) <= 2:
                    has_bias = False
                    bias = None
                else:
                    has_bias = True
                    bias = parameters[node.input[2]]
                in_features = weight.shape[1]
                out_features = weight.shape[0]
                network.add_node(nodes.FullyConnectedNode(node.output[0], in_dim,
                                                          out_features, weight, bias, has_bias))
            elif node.op_type == "BatchNormalization":
                # We assume that the real input is always the first element of node.input, the weight tensor
                # is always the second, the bias tensor is always the third, the running_mean always the fourth
                # and the running_var always the fifth.

                weight = parameters[node.input[1]]
                bias = parameters[node.input[2]]
                running_mean = parameters[node.input[3]]
                running_var = parameters[node.input[4]]
                # We assume that eps is always the first attribute and momentum is always the second.
                eps = node.attribute[0].f
                momentum = node.attribute[1].f
                network.add_node(nodes.BatchNormNode(node.output[0], in_dim, weight,
                                                     bias, running_mean, running_var, eps, momentum))

            elif node.op_type == "Conv":
                # We assume that the real input is always the first element of node.input, the weight tensor
                # is always the second and the bias tensor is always the third.

                weight = parameters[node.input[1]]
                if len(node.input) <= 2:
                    has_bias = False
                    bias = None
                else:
                    has_bias = True
                    bias = parameters[node.input[2]]

                out_channels = weight.shape[0]
                # We assume that the attribute are in the following order: dilation, groups, kernel_shape, pads and
                # strides
                dilation = tuple(node.attribute[0].ints)
                groups = node.attribute[1].i
                kernel_size = tuple(node.attribute[2].ints)
                padding = tuple(node.attribute[3].ints)
                stride = tuple(node.attribute[4].ints)

                network.add_node(nodes.ConvNode(node.output[0], in_dim, out_channels, kernel_size,
                                                stride, padding, dilation, groups, has_bias, bias, weight))

            elif node.op_type == "AveragePool":

                # We assume that the attribute are in the following order: ceil_mode, count_include_pad,
                # kernel_shape, pads and strides
                ceil_mode = bool(node.attribute[0].i)
                count_include_pad = bool(node.attribute[1].i)
                kernel_size = tuple(node.attribute[2].ints)
                padding = tuple(node.attribute[3].ints)
                stride = tuple(node.attribute[4].ints)

                network.add_node(nodes.AveragePoolNode(node.output[0], in_dim, kernel_size, stride,
                                                       padding, ceil_mode, count_include_pad))

            elif node.op_type == "MaxPool":

                # We assume that the attribute are in the following order: ceil_mode, dilation,
                # kernel_shape, pads and strides
                ceil_mode = bool(node.attribute[0].i)
                dilation = tuple(node.attribute[1].ints)
                kernel_size = tuple(node.attribute[2].ints)
                padding = tuple(node.attribute[3].ints)
                stride = tuple(node.attribute[4].ints)

                network.add_node(nodes.MaxPoolNode(node.output[0], in_dim, kernel_size, stride, padding,
                                                   dilation, ceil_mode))

            elif node.op_type == "LRN":

                alpha = node.attribute[0].f
                beta = node.attribute[1].f
                k = node.attribute[2].f
                size = node.attribute[3].i

                network.add_node(nodes.LRNNode(node.output[0], in_dim, size, alpha, beta, k))

            elif node.op_type == "Softmax":

                # Since the ONNX representation consider the batch dimension we need to scale the axis by 1
                # when we pass to our representation.
                axis = node.attribute[0].i - 1
                network.add_node(nodes.SoftMaxNode(node.output[0], in_dim, axis))

            elif node.op_type == "Unsqueeze":

                temp_axes = tuple(parameters[node.input[1]])
                # Since our representation do not consider the batch dimension we need to scale all the axes
                # by 1 when we pass to the onnx representation.
                axes = tuple([e - 1 for e in temp_axes])
                network.add_node(nodes.UnsqueezeNode(node.output[0], in_dim, axes))

            elif node.op_type == "Reshape":

                shape = tuple(parameters[node.input[1]])
                # We need to eliminate the first dimension corresponding to the batch dimension
                shape = shape[1:]
                allow_zero = node.attribute[0].i
                network.add_node(nodes.ReshapeNode(node.output[0], in_dim, shape, allow_zero))

            elif node.op_type == "Flatten":

                # We need to scale the axis value since our representation does not have the batch dimension
                axis = node.attribute[0].i - 1
                network.add_node(nodes.FlattenNode(node.output[0], in_dim, axis))

            elif node.op_type == "Dropout":

                ratio = parameters[node.input[1]][0]
                network.add_node(nodes.DropoutNode(node.output[0], in_dim, ratio))

            else:
                raise NotImplementedError

            node_index += 1
            in_dim = network.get_last_node().out_dim

        return network


class PyTorchConverter(ConversionStrategy):
    """
    A class used to represent the conversion strategy for PyTorch models.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to a PyTorchNetwork model.
    to_neural_network(PyTorchNetwork)
        Convert the PyTorchNetwork of interest to our internal representation of a Neural Network.

    """

    def from_neural_network(self, network: networks.NeuralNetwork) -> PyTorchNetwork:
        """
        Convert the neural network of interest to a PyTorch representation.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        PyTorchNetwork
            The PyTorch representation resulting from the conversion of the original network.

        """

        alt_net = None
        pytorch_network = None
        for alt_rep in network.alt_rep_cache:
            if isinstance(alt_rep, PyTorchNetwork) and alt_rep.up_to_date:
                alt_net = alt_rep

        if alt_net is None:

            if not network.up_to_date:

                for alt_rep in network.alt_rep_cache:

                    if alt_rep.up_to_date:

                        if isinstance(alt_rep, ONNXNetwork):
                            onnx_cv = ONNXConverter()
                            network = onnx_cv.to_neural_network(alt_rep)

                        else:
                            raise NotImplementedError
                        break

            if isinstance(network, networks.SequentialNetwork):
                pytorch_layers = []
                for layer in network.nodes.values():

                    new_layer = None
                    if isinstance(layer, nodes.ReLUNode):
                        new_layer = pyt_l.ReLU(layer.identifier, layer.in_dim, layer.out_dim)

                    elif isinstance(layer, nodes.SigmoidNode):
                        new_layer = pyt_l.Sigmoid(layer.identifier, layer.in_dim, layer.out_dim)

                    elif isinstance(layer, nodes.FullyConnectedNode):

                        if layer.bias is not None:
                            has_bias = True
                        else:
                            has_bias = False

                        new_layer = pyt_l.Linear(layer.identifier, layer.in_dim, layer.out_dim,
                                                 in_features=layer.in_features, out_features=layer.out_features,
                                                 bias=has_bias)

                        weight = torch.from_numpy(layer.weight)
                        new_layer.weight.data = weight

                        if has_bias:
                            bias = torch.from_numpy(layer.bias)
                            new_layer.bias.data = bias

                    elif isinstance(layer, nodes.BatchNormNode):

                        if len(layer.in_dim) == 1 or len(layer.in_dim) == 2:

                            new_layer = pyt_l.BatchNorm1d(layer.identifier, layer.in_dim, layer.out_dim,
                                                          num_features=layer.num_features,
                                                          eps=layer.eps, momentum=layer.momentum,
                                                          affine=layer.affine,
                                                          track_running_stats=layer.track_running_stats)
                        elif len(layer.in_dim) == 3:

                            new_layer = pyt_l.BatchNorm2d(layer.identifier, layer.in_dim, layer.out_dim,
                                                          num_features=layer.num_features,
                                                          eps=layer.eps, momentum=layer.momentum,
                                                          affine=layer.affine,
                                                          track_running_stats=layer.track_running_stats)

                        elif len(layer.in_dim) == 4:

                            new_layer = pyt_l.BatchNorm3d(layer.identifier, layer.in_dim, layer.out_dim,
                                                          num_features=layer.num_features,
                                                          eps=layer.eps, momentum=layer.momentum,
                                                          affine=layer.affine,
                                                          track_running_stats=layer.track_running_stats)

                        else:
                            raise Exception("Pytorch does not support batchnorm layer for input with more than"
                                            "4 or less than 1 dimension excluding the batch dimension")

                        new_layer.weight.data = torch.from_numpy(layer.weight)
                        new_layer.bias.data = torch.from_numpy(layer.bias)
                        new_layer.running_mean.data = torch.from_numpy(layer.running_mean)
                        new_layer.running_var.data = torch.from_numpy(layer.running_var)

                    elif isinstance(layer, nodes.ConvNode):

                        # Pytorch support only symmetric padding, thereore we assume that the padding given is
                        # symmetric. Padding mode is not supported in our representation therefore we let it be
                        # set to the default value.
                        padding = layer.padding[:int(len(layer.padding) / 2)]

                        if len(layer.in_dim) == 2:

                            new_layer = pyt_l.Conv1d(layer.identifier, layer.in_dim, layer.out_dim,
                                                     layer.in_channels, layer.out_channels, layer.kernel_size,
                                                     layer.stride, padding, layer.dilation, layer.groups,
                                                     layer.has_bias)

                        elif len(layer.in_dim) == 3:

                            new_layer = pyt_l.Conv2d(layer.identifier, layer.in_dim, layer.out_dim,
                                                     layer.in_channels, layer.out_channels, layer.kernel_size,
                                                     layer.stride, padding, layer.dilation, layer.groups,
                                                     layer.has_bias)

                        elif len(layer.in_dim) == 4:

                            new_layer = pyt_l.Conv3d(layer.identifier, layer.in_dim, layer.out_dim,
                                                     layer.in_channels, layer.out_channels, layer.kernel_size,
                                                     layer.stride, padding, layer.dilation, layer.groups,
                                                     layer.has_bias)

                        else:
                            raise Exception("Pytorch does not support Conv layer for input with more than"
                                            "4 or less than 2 dimension excluding the batch dimension")

                        new_layer.weight.data = torch.from_numpy(layer.weight)
                        if layer.has_bias:
                            new_layer.bias.data = torch.from_numpy(layer.bias)

                    elif isinstance(layer, nodes.AveragePoolNode):

                        # Pytorch support only symmetric padding, thereore we assume that the padding given is
                        # symmetric.
                        padding = layer.padding[:int(len(layer.padding) / 2)]

                        if len(layer.in_dim) == 2:

                            new_layer = pyt_l.AvgPool1d(layer.identifier, layer.in_dim, layer.out_dim,
                                                        layer.kernel_size, layer.stride, padding,
                                                        layer.ceil_mode, layer.count_include_pad)

                        elif len(layer.in_dim) == 3:

                            new_layer = pyt_l.AvgPool2d(layer.identifier, layer.in_dim, layer.out_dim,
                                                        layer.kernel_size, layer.stride, padding,
                                                        layer.ceil_mode, layer.count_include_pad)

                        elif len(layer.in_dim) == 4:

                            new_layer = pyt_l.AvgPool3d(layer.identifier, layer.in_dim, layer.out_dim,
                                                        layer.kernel_size, layer.stride, padding,
                                                        layer.ceil_mode, layer.count_include_pad)

                        else:
                            raise Exception("Pytorch does not support Conv layer for input with more than"
                                            "4 or less than 2 dimension excluding the batch dimension")

                    elif isinstance(layer, nodes.MaxPoolNode):

                        # Pytorch support only symmetric padding, thereore we assume that the padding given is
                        # symmetric.
                        padding = layer.padding[:int(len(layer.padding) / 2)]

                        if len(layer.in_dim) == 2:

                            new_layer = pyt_l.MaxPool1d(layer.identifier, layer.in_dim, layer.out_dim,
                                                        layer.kernel_size, layer.stride, padding,
                                                        layer.dilation, layer.return_indices, layer.ceil_mode)

                        elif len(layer.in_dim) == 3:

                            new_layer = pyt_l.MaxPool2d(layer.identifier, layer.in_dim, layer.out_dim,
                                                        layer.kernel_size, layer.stride, padding,
                                                        layer.dilation, layer.return_indices, layer.ceil_mode)

                        elif len(layer.in_dim) == 4:

                            new_layer = pyt_l.MaxPool3d(layer.identifier, layer.in_dim, layer.out_dim,
                                                        layer.kernel_size, layer.stride, padding,
                                                        layer.dilation, layer.return_indices, layer.ceil_mode)

                        else:
                            raise Exception("Pytorch does not support Conv layer for input with more than"
                                            "4 or less than 2 dimension excluding the batch dimension")

                    elif isinstance(layer, nodes.LRNNode):

                        new_layer = pyt_l.LocalResponseNorm(layer.identifier, layer.in_dim, layer.out_dim,
                                                            layer.size, layer.alpha, layer.beta, layer.k)

                    elif isinstance(layer, nodes.SoftMaxNode):

                        # We need to scale the axis by one since our representation does not support the batch dimension
                        new_layer = pyt_l.Softmax(layer.identifier, layer.in_dim, layer.out_dim, layer.axis + 1)

                    elif isinstance(layer, nodes.UnsqueezeNode):

                        # Our representation does not consider batch dimension, therefore we need to scale
                        # the axes values.
                        axes = tuple([e + 1 for e in layer.axes])
                        new_layer = pyt_l.Unsqueeze(layer.identifier, layer.in_dim, layer.out_dim, axes)

                    elif isinstance(layer, nodes.ReshapeNode):

                        # Pytorch does not support the allow_zero attribute and the corresponding reshape with 0
                        # dimensions.
                        if layer.allow_zero:
                            raise Exception("allow_zero not supported by pytorch")

                        # Our representation does not consider batch dimension, therefore we need to add it to
                        # the shape.
                        shape = [1]
                        for e in layer.shape:
                            shape.append(e)
                        shape = tuple(shape)

                        new_layer = pyt_l.Reshape(layer.identifier, layer.in_dim, layer.out_dim, shape)

                    elif isinstance(layer, nodes.FlattenNode):

                        # We need to scale the axis by one since our representation does not support the batch dimension
                        new_layer = pyt_l.Flatten(layer.identifier, layer.in_dim, layer.out_dim, layer.axis + 1)

                    elif isinstance(layer, nodes.DropoutNode):

                        new_layer = pyt_l.Dropout(layer.identifier, layer.in_dim, layer.out_dim, layer.p)

                    else:
                        raise NotImplementedError

                    if new_layer is not None:
                        pytorch_layers.append(new_layer)

                pytorch_network = pyt_l.Sequential(network.identifier, network.input_id, pytorch_layers)

            if alt_net is None and pytorch_network is None:
                print("WARNING: network to convert is not valid, the alternative representation is None")

            identifier = network.identifier
            alt_net = PyTorchNetwork(identifier=identifier, pytorch_network=pytorch_network)

        return alt_net

    def to_neural_network(self, alt_rep: PyTorchNetwork) -> networks.NeuralNetwork:
        """
        Convert the PyTorch representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : PyTorchNetwork
            The PyTorch Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of PyTorch Representation.

        """

        identifier = alt_rep.identifier
        network = networks.SequentialNetwork(identifier, alt_rep.pytorch_network.input_id)

        node_index = 0
        alt_rep.pytorch_network.cpu()
        for m in alt_rep.pytorch_network.modules():

            new_node = None

            if isinstance(m, pyt_l.ReLU):
                new_node = nodes.ReLUNode(m.identifier, m.in_dim)

            elif isinstance(m, pyt_l.Sigmoid):
                new_node = nodes.SigmoidNode(m.identifier, m.in_dim)

            elif isinstance(m, pyt_l.Linear):
                out_features = m.out_features
                weight = m.weight.detach().numpy()
                bias = None
                has_bias = False
                if m.bias is not None:
                    bias = m.bias.detach().numpy()
                    has_bias = True
                new_node = nodes.FullyConnectedNode(m.identifier, m.in_dim, out_features, weight, bias, has_bias)

            elif isinstance(m, pyt_l.BatchNorm1d) or isinstance(m, pyt_l.BatchNorm2d) or \
                    isinstance(m, pyt_l.BatchNorm3d):

                eps = m.eps
                momentum = m.momentum
                track_running_stats = m.track_running_stats
                affine = m.affine

                weight = m.weight.detach().numpy()
                bias = m.bias.detach().numpy()
                running_mean = m.running_mean.numpy()
                running_var = m.running_var.numpy()

                new_node = nodes.BatchNormNode(m.identifier, m.in_dim, weight,
                                               bias, running_mean, running_var, eps, momentum, affine,
                                               track_running_stats)

            elif isinstance(m, pyt_l.Conv1d) or isinstance(m, pyt_l.Conv2d) or isinstance(m, pyt_l.Conv3d):

                out_channels = m.out_channels
                kernel_size = m.kernel_size
                stride = m.stride
                temp_padding = list(m.padding)
                for e in m.padding:
                    temp_padding.append(e)
                padding = tuple(temp_padding)
                dilation = m.dilation
                groups = m.groups
                weight = m.weight.detach().numpy()
                if m.bias is None:
                    has_bias = False
                    bias = None
                else:
                    has_bias = True
                    bias = m.bias.detach().numpy()

                new_node = nodes.ConvNode(m.identifier, m.in_dim, out_channels, kernel_size,
                                          stride, padding, dilation, groups, has_bias, bias, weight)

            elif isinstance(m, pyt_l.AvgPool1d) or isinstance(m, pyt_l.AvgPool2d) or \
                    isinstance(m, pyt_l.AvgPool3d):

                stride = m.stride
                temp_padding = list(m.padding)
                for e in m.padding:
                    temp_padding.append(e)
                padding = tuple(temp_padding)
                kernel_size = m.kernel_size
                ceil_mode = m.ceil_mode
                count_include_pad = m.count_include_pad

                new_node = nodes.AveragePoolNode(m.identifier, m.in_dim, kernel_size, stride, padding,
                                                 ceil_mode, count_include_pad)

            elif isinstance(m, pyt_l.MaxPool1d) or isinstance(m, pyt_l.MaxPool2d) or \
                    isinstance(m, pyt_l.MaxPool3d):

                stride = m.stride
                temp_padding = list(m.padding)
                for e in m.padding:
                    temp_padding.append(e)
                padding = tuple(temp_padding)
                kernel_size = m.kernel_size
                ceil_mode = m.ceil_mode
                dilation = m.dilation
                return_indices = m.return_indices

                new_node = nodes.MaxPoolNode(m.identifier, m.in_dim, kernel_size, stride, padding, dilation,
                                             ceil_mode, return_indices)

            elif isinstance(m, pyt_l.LocalResponseNorm):

                new_node = nodes.LRNNode(m.identifier, m.in_dim, m.size, m.alpha, m.beta, m.k)

            elif isinstance(m, pyt_l.Softmax):

                new_node = nodes.SoftMaxNode(m.identifier, m.in_dim, m.dim - 1)

            elif isinstance(m, pyt_l.Unsqueeze):

                axes = tuple([e - 1 for e in m.axes])
                new_node = nodes.UnsqueezeNode(m.identifier, m.in_dim, axes)

            elif isinstance(m, pyt_l.Reshape):

                shape = m.shape[1:]
                new_node = nodes.ReshapeNode(m.identifier, m.in_dim, shape)

            elif isinstance(m, pyt_l.Flatten):

                new_node = nodes.FlattenNode(m.identifier, m.in_dim, m.axis - 1)

            elif isinstance(m, pyt_l.Dropout):

                new_node = nodes.DropoutNode(m.identifier, m.in_dim, m.p)

            elif isinstance(m, pyt_l.Sequential):
                pass

            else:
                raise NotImplementedError

            if new_node is not None:
                node_index += 1
                network.add_node(new_node)

        return network


class TensorflowConverter(ConversionStrategy):
    """
    A class used to represent the conversion strategy for Tensorflow models.

    Methods
    ----------
    from_neural_network(NeuralNetwork)
        Convert the neural network of interest to a TensorflowNetwork model.
    to_neural_network(ONNXNetwork)
        Convert the TensorflowNetwork of interest to our internal representation of a Neural Network.

    """

    def from_neural_network(self, network: networks.NeuralNetwork) -> TensorflowNetwork:
        """
        Convert the neural network of interest to a Tensorflow representation.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to convert.

        Returns
        ----------
        TensorflowNetwork
            The Tensorflow representation resulting from the conversion of the original network.

        """

    def to_neural_network(self, alt_rep: TensorflowNetwork) -> networks.NeuralNetwork:
        """
        Convert the Tensorflow representation of interest to the internal one.

        Parameters
        ----------
        alt_rep : TensorflowNetwork
            The Tensorflow Representation to convert.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the conversion of Tensorflow Representation.

        """
