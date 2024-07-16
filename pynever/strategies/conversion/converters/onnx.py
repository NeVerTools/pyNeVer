import numpy as np
import onnx
import onnx.numpy_helper

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.tensors as tensors
from pynever.strategies.conversion.representation import ConversionStrategy, ONNXNetwork


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
    def __add_onnx_elu(current_node: nodes.ELUNode, current_input: str, current_output: str,
                       onnx_nodes: list):

        onnx_node = onnx.helper.make_node(
            'Elu',
            inputs=[current_input],
            outputs=[current_output],
            alpha=current_node.alpha
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_leakyrelu(current_node: nodes.LeakyReLUNode, current_input: str, current_output: str,
                             onnx_nodes: list):

        onnx_node = onnx.helper.make_node(
            'LeakyRelu',
            inputs=[current_input],
            outputs=[current_output],
            alpha=current_node.negative_slope
        )

        onnx_nodes.append(onnx_node)

    @staticmethod
    def __add_onnx_celu(current_node: nodes.CELUNode, current_input: str, current_output: str,
                        onnx_nodes: list):

        onnx_node = onnx.helper.make_node(
            'Celu',
            inputs=[current_input],
            outputs=[current_output],
            alpha=current_node.alpha
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
    def __add_onnx_tanh(current_input: str, current_output: str, onnx_nodes: list):

        onnx_node = onnx.helper.make_node(
            'Tanh',
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

    @staticmethod
    def __add_onnx_transpose(current_node: nodes.TransposeNode, current_input: str, current_output: str,
                             onnx_nodes: list):

        # Since our representation do not consider the batch dimension we need to scale the perm by 1
        # and add the 0 dimension when we pass to the onnx representation.
        perm = [0]
        for p in current_node.perm:
            perm.append(p + 1)

        onnx_node = onnx.helper.make_node(
            'Transpose',
            inputs=[current_input],
            outputs=[current_output],
            perm=perm
        )

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

        if not isinstance(network, networks.SequentialNetwork):
            raise NotImplementedError

        current_node = network.get_first_node()
        previous_output = network.get_input_id()
        input_info = []
        output_info = []
        initializers = []
        onnx_nodes = []

        while current_node is not None:

            current_input = previous_output
            current_output = current_node.identifier

            input_dim = [1]
            if isinstance(current_node.get_input_dim(), list):
                raise Exception('Sequential networks cannot have multiple input layers')

            for e in current_node.get_input_dim():
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

            elif isinstance(current_node, nodes.ELUNode):
                self.__add_onnx_elu(current_node, current_input, current_output, onnx_nodes)

            elif isinstance(current_node, nodes.LeakyReLUNode):
                self.__add_onnx_leakyrelu(current_node, current_input, current_output, onnx_nodes)

            elif isinstance(current_node, nodes.CELUNode):
                self.__add_onnx_celu(current_node, current_input, current_output, onnx_nodes)

            elif isinstance(current_node, nodes.SigmoidNode):
                self.__add_onnx_sigmoid(current_input, current_output, onnx_nodes)

            elif isinstance(current_node, nodes.TanhNode):
                self.__add_onnx_tanh(current_input, current_output, onnx_nodes)

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

            elif isinstance(current_node, nodes.TransposeNode):
                self.__add_onnx_transpose(current_node, current_input, current_output, onnx_nodes)

            else:
                raise NotImplementedError

            previous_output = current_output
            current_node = network.get_next_node(current_node)

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
        if in_dim == ():
            in_dim = tuple(shape_info[alt_rep.onnx_network.graph.input[0].name])

        temp_fc = None
        matmul_found = False

        for node in alt_rep.onnx_network.graph.node:

            """ This part of code handles the case when there is a MatMul + Add for a linear transformation """
            if matmul_found:
                if node.op_type == "Add":

                    # We assume that the bias is always the second element of node.input

                    bias = tensors.array(parameters[node.input[1]])
                    network.append_node(nodes.FullyConnectedNode(temp_fc.identifier, temp_fc.get_input_dim(),
                                                                 temp_fc.out_features, temp_fc.weight, bias, True))
                    matmul_found = False
                    temp_fc = None
                    node_index += 1
                    in_dim = network.get_last_node().out_dim
                    continue

                else:
                    network.append_node(nodes.FullyConnectedNode(temp_fc.identifier, temp_fc.get_input_dim(),
                                                                 temp_fc.out_features, temp_fc.weight, None, False))
                    matmul_found = False
                    temp_fc = None
                    node_index += 1
                    in_dim = network.get_last_node().out_dim

            if node.op_type == "MatMul":

                # If the weight is the second parameter we need to transpose it

                if node.input[0] in parameters.keys():
                    weight = parameters[node.input[0]]
                else:
                    weight = parameters[node.input[1]].T

                weight = tensors.array(weight)

                out_features = weight.shape[0]
                temp_fc = nodes.FullyConnectedNode(node.output[0], in_dim, out_features, weight, None, False)
                matmul_found = True

                continue

            elif node.op_type == 'Sub':
                """ This node is not relevant for verification and should not appear at all in benchmarks """
                continue

            elif node.op_type == 'Div':
                """ This node is not relevant for verification and should not appear at all in benchmarks """
                continue

            elif node.op_type == "Relu":

                # We assume that the real input of the node is always the first element of node.input
                # and the first element of the shape is the batch placeholder

                network.append_node(nodes.ReLUNode(node.output[0], in_dim))

            elif node.op_type == "Elu":

                alpha = 1.0

                for att in node.attribute:
                    if att.name == 'alpha':
                        alpha = att.f

                network.append_node(nodes.ELUNode(node.output[0], in_dim, alpha))

            elif node.op_type == "LeakyRelu":

                negative_slope = 1.0

                for att in node.attribute:
                    if att.name == 'alpha':
                        negative_slope = att.f

                network.append_node(nodes.LeakyReLUNode(node.output[0], in_dim, negative_slope))

            elif node.op_type == "Celu":

                alpha = 1.0

                for att in node.attribute:
                    if att.name == 'alpha':
                        alpha = att.f

                network.append_node(nodes.CELUNode(node.output[0], in_dim, alpha))

            elif node.op_type == "Sigmoid":

                network.append_node(nodes.SigmoidNode(node.output[0], in_dim))

            elif node.op_type == "Tanh":

                network.append_node(nodes.TanhNode(node.output[0], in_dim))

            elif node.op_type == "Gemm":
                # We assume that the weight tensor is always the second element of node.input and the bias tensor
                # is always the third.
                # N.B: We do not support the attributes transA and transB,
                # therefore we need to transpose the weight vector.
                # TODO: Can we support transA and transB in some way?

                weight = parameters[node.input[1]]
                for att in node.attribute:
                    if (att.name == 'transA' or att.name == 'transB') and att.i == 0:
                        weight = parameters[node.input[1]].T

                if len(node.input) <= 2:
                    has_bias = False
                    bias = None
                else:
                    has_bias = True
                    bias = parameters[node.input[2]]

                out_features = weight.shape[0]
                network.append_node(nodes.FullyConnectedNode(node.output[0], in_dim,
                                                             out_features, weight, bias, has_bias))
            elif node.op_type == "BatchNormalization":
                # We assume that the real input is always the first element of node.input, the weight tensor
                # is always the second, the bias tensor is always the third, the running_mean always the fourth
                # and the running_var always the fifth.

                weight = tensors.array(parameters[node.input[1]])
                bias = tensors.array(parameters[node.input[2]])
                running_mean = tensors.array(parameters[node.input[3]])
                running_var = tensors.array(parameters[node.input[4]])

                eps = 1e-05
                momentum = 0.9
                for att in node.attribute:

                    if att.name == 'epsilon':
                        eps = att.f
                    elif att.name == 'momentum':
                        momentum = att.f

                network.append_node(nodes.BatchNormNode(node.output[0], in_dim, weight,
                                                        bias, running_mean, running_var, eps, momentum))

            elif node.op_type == "Conv":
                # We assume that the real input is always the first element of node.input, the weight tensor
                # is always the second and the bias tensor is always the third.

                weight = tensors.array(parameters[node.input[1]])
                if len(node.input) <= 2:
                    has_bias = False
                    bias = None
                else:
                    has_bias = True
                    bias = parameters[node.input[2]]

                out_channels = weight.shape[0]

                # TODO: at present we do not support auto_pad and implicit kernel_shape.
                groups = 1
                # We need to exclude the first axis (channels) from the following quantities.
                dilation = tuple(np.ones(len(in_dim[1:]), dtype=np.intc))
                padding = tuple(np.zeros(2 * len(in_dim[1:]), dtype=np.intc))
                stride = tuple(np.ones(len(in_dim[1:]), dtype=np.intc))

                for att in node.attribute:
                    if att.name == 'dilations':
                        dilation = tuple(att.ints)
                    elif att.name == 'groups':
                        groups = att.i
                    elif att.name == 'kernel_shape':
                        kernel_shape = tuple(att.ints)
                    elif att.name == 'pads':
                        padding = tuple(att.ints)
                    elif att.name == 'strides':
                        stride = tuple(att.ints)

                network.append_node(nodes.ConvNode(node.output[0], in_dim, out_channels, kernel_shape,
                                                   stride, padding, dilation, groups, has_bias, bias, weight))

            elif node.op_type == "AveragePool":

                # TODO: at present we do not support auto_pad.

                ceil_mode = False
                count_include_pad = False
                padding = tuple(np.zeros(2 * len(in_dim[1:]), dtype=np.intc))
                stride = tuple(np.ones(len(in_dim[1:]), dtype=np.intc))

                for att in node.attribute:
                    if att.name == 'ceil_mode':
                        ceil_mode = bool(att.i)
                    elif att.name == 'count_include_pad':
                        count_include_pad = bool(att.i)
                    elif att.name == 'kernel_shape':
                        kernel_shape = tuple(att.ints)
                    elif att.name == 'pads':
                        padding = tuple(att.ints)
                    elif att.name == 'strides':
                        stride = tuple(att.ints)

                network.append_node(nodes.AveragePoolNode(node.output[0], in_dim, kernel_shape, stride,
                                                          padding, ceil_mode, count_include_pad))

            elif node.op_type == "MaxPool":

                ceil_mode = False
                dilation = tuple(np.ones(len(in_dim[1:]), dtype=np.intc))
                padding = tuple(np.zeros(2 * len(in_dim[1:]), dtype=np.intc))
                stride = tuple(np.ones(len(in_dim[1:]), dtype=np.intc))

                for att in node.attribute:
                    if att.name == 'ceil_mode':
                        ceil_mode = bool(att.i)
                    elif att.name == 'dilations':
                        dilation = tuple(att.ints)
                    elif att.name == 'kernel_shape':
                        kernel_shape = tuple(att.ints)
                    elif att.name == 'pads':
                        padding = tuple(att.ints)
                    elif att.name == 'strides':
                        stride = tuple(att.ints)

                network.append_node(nodes.MaxPoolNode(node.output[0], in_dim, kernel_shape, stride, padding,
                                                      dilation, ceil_mode))

            elif node.op_type == "LRN":

                alpha = 0.0001
                beta = 0.75
                k = 1.0

                for att in node.attribute:
                    if att.name == 'alpha':
                        alpha = att.f
                    elif att.name == 'beta':
                        beta = att.f
                    elif att.name == 'bias':
                        k = att.f
                    elif att.name == 'size':
                        size = att.i

                network.append_node(nodes.LRNNode(node.output[0], in_dim, size, alpha, beta, k))

            elif node.op_type == "Softmax":

                # Since the ONNX representation consider the batch dimension we need to scale the axis by 1
                # when we pass to our representation.
                axis = -1
                for att in node.attribute:
                    if att.name == 'axis':
                        axis = att.i - 1

                network.append_node(nodes.SoftMaxNode(node.output[0], in_dim, axis))

            elif node.op_type == "Unsqueeze":

                temp_axes = tuple(parameters[node.input[1]])
                # Since our representation do not consider the batch dimension we need to scale all the axes
                # by 1 when we pass to the onnx representation.
                axes = tuple([e - 1 for e in temp_axes])
                network.append_node(nodes.UnsqueezeNode(node.output[0], in_dim, axes))

            elif node.op_type == "Reshape":

                shape = tuple(parameters[node.input[1]])
                # We need to eliminate the first dimension corresponding to the batch dimension
                shape = shape[1:]
                allow_zero = 0
                for att in node.attribute:
                    if att.name == 'allowzero':
                        allow_zero = att.i

                network.append_node(nodes.ReshapeNode(node.output[0], in_dim, shape, allow_zero))

            elif node.op_type == "Flatten":

                # We need to scale the axis value since our representation does not have the batch dimension
                axis = 0
                for att in node.attribute:
                    if att.name == 'axis':
                        axis = att.i - 1
                network.append_node(nodes.FlattenNode(node.output[0], in_dim, axis))

            elif node.op_type == "Dropout":

                ratio = tensors.array(parameters[node.input[1]][0])
                network.append_node(nodes.DropoutNode(node.output[0], in_dim, ratio))

            elif node.op_type == "Transpose":

                perm = None
                for att in node.attribute:
                    if att.name == 'perm':
                        # Must manage batch dimension
                        if len(att.ints) > len(in_dim):
                            perm = [att.ints[k] - 1 for k in range(1, len(att.ints))]
                        else:
                            perm = att.ints

                network.append_node(nodes.TransposeNode(node.output[0], in_dim, perm))

            else:
                raise NotImplementedError

            node_index += 1
            in_dim = network.get_last_node().out_dim

        return network
