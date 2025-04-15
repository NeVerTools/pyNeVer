import numpy as np
import torch

from pynever.nodes import MaxPoolNode
from pynever.strategies.abstraction.bounds_propagation.bounds import SymbolicLinearBounds, HyperRectangleBounds
from pynever.strategies.abstraction.linearfunctions import LinearFunctions

#NON COMPLETED

class MaxPoolLinearization:
    def __init__(self):
        pass

    def compute_output_equation(self, max_pool_node: MaxPoolNode, input_dim: tuple,
                                sym_inputs: SymbolicLinearBounds, numeric_inputs: HyperRectangleBounds):
        # Ensure the input dimension is 3D: (out_channels, out_height, out_width)
        if len(input_dim) != 3:
            raise ValueError("MaxPoolLinearization is only supported for 3-dimensional inputs")

        # Check MaxPool constraints: no padding, dilation, or ceil_mode
        if max_pool_node.dilation not in [(1, 1), None]:
            raise ValueError("Only dilation of (1, 1) or None is supported")
        if max_pool_node.ceil_mode:
            raise ValueError("MaxPool ceil_mode is not supported")

        # Check that the flattened vectors in input are compatible with the input_dim
        if input_dim[0] * input_dim[1] * input_dim[2] != numeric_inputs.size:
            raise ValueError("The product of the input_dim dimensions id different from the numeric_inputs size")

        # Check that the flattened vectors in input are compatible with the input_dim
        if input_dim[0] * input_dim[1] * input_dim[2] != sym_inputs.size:
            raise ValueError("The product of the input_dim dimensions id different from the sym_input size")

        def calculate_output_size(input_size, kernel_size, stride, padding, dilation=1):
            """Calculate output size of the max pooling operation."""
            return int(np.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

        # Calculate output height and width
        output_channel = input_dim[0]

        if type(max_pool_node.padding) == int:
            padding_h = max_pool_node.padding
            padding_w = max_pool_node.padding
            padding = max_pool_node.padding

        elif type(max_pool_node.padding) == tuple and len(max_pool_node.padding) == 2:
            padding_h = max_pool_node.padding[0]
            padding_w = max_pool_node.padding[1]
            padding = (padding_h, padding_w)

        elif type(max_pool_node.padding) == tuple and len(max_pool_node.padding) == 4:
            if max_pool_node.padding[2] != max_pool_node.padding[3] or max_pool_node.padding[1] != max_pool_node.padding[0]:
                raise ValueError("Only symmetrical padding is supported. Left must be equal to right padding as well as top and bottom padding")
            padding_h = max_pool_node.padding[2] + max_pool_node.padding[3]
            padding_w = max_pool_node.padding[0] + max_pool_node.padding[1]
            padding = (padding_h, padding_w)



        else:
            raise ValueError("MaxPool padding format is not supported. It can be only an integer or a tuple of integers of 2 or 4 values.")


        output_height = calculate_output_size(input_dim[1], max_pool_node.kernel_size[0],
                                              max_pool_node.stride[0], padding_h)
        output_width = calculate_output_size(input_dim[2], max_pool_node.kernel_size[1],
                                             max_pool_node.stride[1], padding_w)

        # Dimension of elements of the output flattened
        output_flattened = output_height * output_width * output_channel
        c_output_flattened = output_height * output_width

        # Number of variables to express the equations
        num_variables = sym_inputs.get_lower().matrix.shape[1]

        # Reshaping input symb and numeric lower and upper bounds
        lower_matrix = sym_inputs.lower.matrix.reshape(output_channel, -1, num_variables)
        lower_offset = sym_inputs.lower.offset.reshape(output_channel, -1)

        upper_matrix = sym_inputs.upper.matrix.reshape(output_channel, -1, num_variables)
        upper_offset = sym_inputs.upper.offset.reshape(output_channel, -1)

        numeric_lower = numeric_inputs.get_lower().reshape(output_channel, -1)
        numeric_upper = numeric_inputs.get_upper().reshape(output_channel, -1)

        # Generating output symb and numeric lower and upper bounds
        output_lower_matrix = np.zeros(shape = (output_channel, c_output_flattened, num_variables), dtype=np.float32)
        output_lower_offset = np.zeros(shape = (output_channel, c_output_flattened), dtype=np.float32)

        output_upper_matrix = np.zeros(shape = (output_channel, c_output_flattened, num_variables), dtype=np.float32)
        output_upper_offset = np.zeros(shape = (output_channel, c_output_flattened), dtype=np.float32)

        output_numeric_lower_bounds = np.zeros(shape = (output_channel, c_output_flattened), dtype=np.float32)
        output_numeric_upper_bounds = np.zeros(shape = (output_channel, c_output_flattened), dtype=np.float32)

        indexes = torch.arange(input_dim[1] * input_dim[2]).view(1, input_dim[1], input_dim[2])
        indexes = indexes.to(dtype=torch.float)
        Unfold = torch.nn.Unfold(kernel_size=max_pool_node.kernel_size, padding=padding, stride=max_pool_node.stride)
        patches = Unfold(indexes)
        patches = patches.numpy()
        patches = patches.T

        for c in range(output_channel):
            c_lower_matrix = lower_matrix[c, :]
            c_upper_matrix = upper_matrix[c, :]

            c_lower_offset = lower_offset[c, :]
            c_upper_offset = upper_offset[c, :]

            c_numeric_lower = numeric_lower[c, :]
            c_numeric_upper = numeric_upper[c, :]

            for index, patch in enumerate(patches):
                patch = patch.astype(np.int32)
                extracted_min = c_numeric_lower[patch]
                min_index = patch[np.argmax(extracted_min)]

                # Get maximum lower values, to handle rare cases where there are multiple max values
                min_value = np.max(extracted_min)
                min_values = np.where(extracted_min == min_value)
                min_indexes = patch[min_values]

                extracted_max = c_numeric_upper[patch]
                max_value = np.max(extracted_max)
                max_values = np.where(extracted_max == max_value)
                max_indexes = patch[max_values]
                intersection = np.intersect1d(min_indexes, max_indexes)


                if len(intersection) >= 0:
                    output_upper_matrix[c, index] = c_upper_matrix[intersection[0], :]
                    output_upper_offset[c, index] = c_upper_offset[intersection[0]]
                    output_numeric_upper_bounds[c, index] = c_numeric_upper[intersection[0]]

                    output_lower_matrix[c, index] = c_lower_matrix[intersection[0], :]
                    output_lower_offset[c, index] = c_lower_offset[intersection[0]]
                    output_numeric_lower_bounds[c, index] = c_numeric_lower[intersection[0]]
                else:
                    output_upper_offset[c, index] = c_upper_offset[max_indexes[0], :]
                    output_numeric_upper_bounds[c, index] = c_numeric_upper[max_indexes[0], :]
                    output_lower_matrix[c, index] = c_lower_matrix[min_indexes[0], :]
                    output_lower_offset[c, index] = c_lower_offset[min_indexes[0]]
                    output_numeric_lower_bounds[c, index] = c_numeric_lower[min_indexes[0]]



        # Check that the flattened output is equal to the output_channel * output_height * output_width
        output_lower_matrix = output_lower_matrix.reshape(-1, c_output_flattened, num_variables)
        output_lower_offset = output_lower_offset.reshape(-1)

        output_upper_matrix = output_upper_matrix.reshape(-1, c_output_flattened, num_variables)
        output_upper_offset = output_upper_offset.reshape(-1)

        output_numeric_lower_bounds = output_numeric_lower_bounds.reshape(-1)
        output_numeric_upper_bounds = output_numeric_upper_bounds.reshape(-1)

        output_hyperect = HyperRectangleBounds(output_numeric_lower_bounds, output_numeric_upper_bounds)
        lower = LinearFunctions(output_lower_matrix, output_lower_offset)
        upper = LinearFunctions(output_upper_matrix, output_upper_offset)
        output_symb = SymbolicLinearBounds(lower, upper)

        return output_symb, output_hyperect


