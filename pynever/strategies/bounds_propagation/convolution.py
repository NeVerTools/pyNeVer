import numpy as np

from pynever.nodes import ConvNode
from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions


class ConvLinearization:
    def __init__(self):
        self.i = None
        self.j = None
        self.k = None

    def compute_output_equation(self, conv_node: ConvNode, inputs: SymbolicLinearBounds):
        filter_n = conv_node.weight.shape[0]
        weights_col = conv_node.weight.reshape(filter_n, -1)
        weights_plus = np.maximum(weights_col, np.zeros(weights_col.shape))
        weights_minus = np.minimum(weights_col, np.zeros(weights_col.shape))

        # First compute the output matrices

        # Assume that the dimensions of the inputs equation matrices are (M, N)
        #
        # Transpose so as to have columns first to be able to treat every column as an inputs vector,
        # and then proceed similarly to get_dot_product
        input_lower_matrix = inputs.get_lower().get_matrix()
        input_lower_matrix = input_lower_matrix.transpose()
        input_upper_matrix = inputs.get_upper().get_matrix()
        input_upper_matrix = input_upper_matrix.transpose()

        # Do all the reshuffling once instead of each time in the loop
        # input_shape = (conv_node.get_input_dim()[1], conv_node.get_input_dim()[2], conv_node.get_input_dim()[0])
        # input_lower_matrix = input_lower_matrix.reshape((input_lower_matrix.shape[0], 1) + input_shape)
        # input_lower_matrix = input_lower_matrix.transpose(0, 1, 4, 2, 3)
        # input_upper_matrix = input_upper_matrix.reshape((input_upper_matrix.shape[0], 1) + input_shape)
        # input_upper_matrix = input_upper_matrix.transpose(0, 1, 4, 2, 3)

        input_shape = (conv_node.get_input_dim()[0], conv_node.get_input_dim()[1], conv_node.get_input_dim()[2])
        input_lower_matrix = input_lower_matrix.reshape((input_lower_matrix.shape[0], 1) + input_shape)
        input_upper_matrix = input_upper_matrix.reshape((input_upper_matrix.shape[0], 1) + input_shape)

        self.initialise_im2col_indices(conv_node)

        # Handle each column separately
        output_lower_matrix = []
        output_upper_matrix = []
        for column_i in range(input_lower_matrix.shape[0]):
            in_lower_vector = input_lower_matrix[column_i]
            in_upper_vector = input_upper_matrix[column_i]

            input_lower_vector_col = self._get_input_col_opt(conv_node, in_lower_vector)
            input_upper_vector_col = self._get_input_col_opt(conv_node, in_upper_vector)

            output_lower_vector = weights_plus.dot(input_lower_vector_col) + \
                                  weights_minus.dot(input_upper_vector_col)
            output_upper_vector = weights_plus.dot(input_upper_vector_col) + \
                                  weights_minus.dot(input_lower_vector_col)

            # Collect the vectors
            output_lower_matrix.append(output_lower_vector)
            output_upper_matrix.append(output_upper_vector)

        # The vectors in the matrices are of the shape (filters_n, h_out * w_out * 1)
        # Rearrange them so as to have channels after rows and cols,
        # finally transpose the matrices as they consist of columns
        output_lower_matrix = np.array(output_lower_matrix)
        output_lower_matrix = output_lower_matrix.reshape(output_lower_matrix.shape[0],
                                                          conv_node.out_dim[0],
                                                          conv_node.out_dim[1],
                                                          conv_node.out_dim[2])
        # output_lower_matrix = output_lower_matrix.transpose(0, 2, 3, 1)
        output_lower_matrix = output_lower_matrix.reshape(output_lower_matrix.shape[0], -1)
        output_lower_matrix = output_lower_matrix.transpose()

        output_upper_matrix = np.array(output_upper_matrix)
        output_upper_matrix = output_upper_matrix.reshape(output_upper_matrix.shape[0],
                                                          conv_node.out_dim[0],
                                                          conv_node.out_dim[1],
                                                          conv_node.out_dim[2])
        # output_upper_matrix = output_upper_matrix.transpose(0, 2, 3, 1)
        output_upper_matrix = output_upper_matrix.reshape(output_upper_matrix.shape[0], -1)
        output_upper_matrix = output_upper_matrix.transpose()

        # Second, compute the offsets
        input_lower_offset = inputs.get_lower().get_offset()
        input_lower_offset = input_lower_offset.reshape(1, -1)
        input_upper_offset = inputs.get_upper().get_offset()
        input_upper_offset = input_upper_offset.reshape(1, -1)

        input_lower_offset_col = ConvLinearization._get_input_col(conv_node, input_lower_offset)
        input_upper_offset_col = ConvLinearization._get_input_col(conv_node, input_upper_offset)

        bias = conv_node.bias if conv_node.bias is not None else np.zeros((weights_plus.shape[0], 1))

        # Second compute the output offsets
        output_lower_offset = weights_plus.dot(input_lower_offset_col) + weights_minus.dot(input_upper_offset_col) + \
                              bias.reshape(-1, 1)
        output_upper_offset = weights_plus.dot(input_upper_offset_col) + weights_minus.dot(input_lower_offset_col) + \
                              bias.reshape(-1, 1)

        # Similarly, rearrange the offsets to have channels after rows and cols
        output_lower_offset = output_lower_offset.reshape(conv_node.out_dim[1], conv_node.out_dim[2],
                                                          conv_node.out_dim[0], 1)
        # output_lower_offset = output_lower_offset.transpose(3, 1, 2, 0)
        output_lower_offset = output_lower_offset.reshape(-1)

        output_upper_offset = output_upper_offset.reshape(conv_node.out_dim[1], conv_node.out_dim[2],
                                                          conv_node.out_dim[0], 1)
        # output_upper_offset = output_upper_offset.transpose(3, 1, 2, 0)
        output_upper_offset = output_upper_offset.reshape(-1)

        return SymbolicLinearBounds(LinearFunctions(output_lower_matrix, output_lower_offset),
                                    LinearFunctions(output_upper_matrix, output_upper_offset))

    def initialise_im2col_indices(self, conv_node: ConvNode):
        input_shape = (1, conv_node.get_input_dim()[0], conv_node.get_input_dim()[1], conv_node.get_input_dim()[2])
        n_filters, d_filter, h_filter, w_filter = conv_node.weight.shape

        self.k, self.i, self.j = ConvLinearization.get_im2col_indices(
            input_shape, h_filter, w_filter, conv_node.padding[0], conv_node.stride[0])

    @staticmethod
    def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
        """
        Some black magic I found on the Internet.
        """
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        # assert (H + 2 * padding - field_height) % stride == 0
        # assert (W + 2 * padding - field_height) % stride == 0
        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return k.astype(int), i.astype(int), j.astype(int)

    @staticmethod
    def _get_input_col(conv_node: ConvNode, inputs):
        """
        This method prepares inputs for a fast pass of convolution over it.

        Input is assumed to be 2-dimensional so that inputs[i] is a flattened
        'image' being inputs to the convolutional layer.
        In some cases it is an array of gurobi variables,
        in others it is a matrix of symbolic bound equations.

        As usual all inputs are assumed to be ordered (rows, cols, channels).
        """

        # rearrange the inputs so as to have channels before rows and cols
        inputs = inputs.reshape((inputs.shape[0],) + (
            conv_node.get_input_dim()[1], conv_node.get_input_dim()[2], conv_node.get_input_dim()[0]))
        inputs = inputs.transpose(0, 3, 1, 2)

        n_filters, d_filter, h_filter, w_filter = conv_node.weight.shape

        input_col = ConvLinearization.im2col_indices(inputs, h_filter, w_filter,
                                                     padding=conv_node.padding[0],
                                                     stride=conv_node.stride[0])
        return input_col

    @staticmethod
    def im2col_indices(x, field_height, field_width, padding=1, stride=1):
        """
        An implementation of im2col based on some fancy indexing

        Some black magic I found on the Internet.
        """
        # Zero-pad the inputs
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = ConvLinearization.get_im2col_indices(x.shape, field_height, field_width, padding, stride)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def _get_input_col_opt(self, conv_node: ConvNode, inputs):
        """
                This is a slight optimisation of the method above.
                Works only for inputs where the first dimension is 1
                """
        # The inputs are already rearranged
        # inputs = inputs.reshape((inputs.shape[0],) + self.input_shape)
        # inputs = inputs.transpose(0, 3, 1, 2)

        n_filters, d_filter, h_filter, w_filter = conv_node.weight.shape

        input_col = self.im2col_indices_opt(inputs, h_filter, w_filter,
                                            padding=conv_node.padding[0],
                                            stride=conv_node.stride[0])
        return input_col

    def im2col_indices_opt(self, x, field_height, field_width, padding=1, stride=1):
        """
        An implementation of im2col based on some fancy indexing

        Some black magic I found on the Internet.
        """
        # Zero-pad the inputs
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        cols = x_padded[:, self.k, self.i, self.j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols
