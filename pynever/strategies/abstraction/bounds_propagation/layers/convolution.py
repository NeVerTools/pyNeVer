import numpy as np
import torch

from pynever.nodes import ConvNode
from pynever.strategies.abstraction.bounds_propagation.bounds import SymbolicLinearBounds
from pynever.strategies.abstraction.linearfunctions import LinearFunctions


class LinearizeConv:

    def compute_output_equations(self, conv_node: ConvNode, inputs: SymbolicLinearBounds):

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        DATA_TYPE = torch.float32

        sym_lower_bounds = inputs.get_lower()
        sym_upper_bounds = inputs.get_upper()

        # Conversion torch: this conversion in only temporary
        if isinstance(sym_lower_bounds.matrix, np.ndarray):
            sym_lower_bounds.matrix = sym_lower_bounds.matrix
            sym_upper_bounds.matrix = sym_upper_bounds.matrix
            sym_lower_bounds.offset = sym_lower_bounds.offset
            sym_upper_bounds.offset = sym_upper_bounds.offset
        elif isinstance(sym_lower_bounds.matrix, torch.Tensor):
            sym_lower_bounds.matrix = sym_lower_bounds.matrix
            sym_upper_bounds.matrix = sym_upper_bounds.matrix
            sym_lower_bounds.offset = sym_lower_bounds.offset
            sym_upper_bounds.offset = sym_upper_bounds.offset

        weights = conv_node.weight

        if conv_node.has_bias:
            bias_weights = conv_node.bias

        # Extract kernel dimensions
        if isinstance(conv_node.kernel_size, tuple) and len(conv_node.kernel_size) == 2:
            kernel_height, kernel_width = conv_node.kernel_size
            kernel_size = conv_node.kernel_size
        elif isinstance(conv_node.kernel_size, int):
            kernel_height = kernel_width = conv_node.kernel_size
            kernel_size = (conv_node.kernel_size, conv_node.kernel_size)
        else:
            raise ValueError("Kernel size must be an int or a tuple of two integers.")

        # Determine padding values
        if isinstance(conv_node.padding, int):
            pad_tuple = (conv_node.padding, conv_node.padding, conv_node.padding, conv_node.padding)
        elif isinstance(conv_node.padding, tuple) and len(conv_node.padding) == 2:
            pad_tuple = (conv_node.padding[1], conv_node.padding[1], conv_node.padding[0], conv_node.padding[0])
        elif isinstance(conv_node.padding, tuple) and len(conv_node.padding) == 4:
            if conv_node.padding[0] != conv_node.padding[1] or conv_node.padding[2] != conv_node.padding[3]:
                raise ValueError(
                    "Only symmetrical padding is supported. Top must equal bottom and left must equal right.")
            pad_tuple = conv_node.padding
        elif conv_node.padding == 0 or conv_node.padding is None:
            pad_tuple = (0, 0, 0, 0)
        else:
            raise ValueError("Padding must be an int or a tuple of appropriate dimensions.")

        # Extract input shape information: channels, height, width
        input_channels = conv_node.in_channels

        # Flatten filter weights for sparse matrix operations
        num_filters = weights.shape[0]
        filter_weights = weights.to(DATA_TYPE).to(device)

        if filter_weights.ndim == 3:
            filter_weights = filter_weights.reshape(num_filters, -1)
        elif filter_weights.ndim == 4:
            filter_weights = filter_weights.reshape(num_filters, input_channels, -1)

        if conv_node.has_bias:
            filter_biases = bias_weights.to(DATA_TYPE).to(device)

        input_shape = conv_node.in_dims[0]
        input_flattened_size = input_shape[1] * input_shape[2]

        # Calculate output dimensions of the convolution
        pad_top, pad_bottom, pad_left, pad_right = pad_tuple
        output_height = int(((input_shape[1] - kernel_height + pad_top + pad_bottom) / conv_node.stride[0]) + 1)
        output_width = int(((input_shape[2] - kernel_width + pad_left + pad_right) / conv_node.stride[1]) + 1)
        output_flattened_size = output_height * output_width

        assert output_height == conv_node.out_dim[1] and output_width == conv_node.out_dim[
            2], "The predicted output dim is different from the real one"

        # Create an index matrix for image patches
        index_matrix = torch.arange(0, input_flattened_size * input_channels, dtype=DATA_TYPE, device=device).reshape(
            input_channels, input_shape[1],
            input_shape[2])

        list_index_matrixes = []
        # Apply padding to input tensors. When there is padding the indexing system must be modified
        if conv_node.padding is not None:
            for index in range(input_channels):
                # The matrix used to generate is padded with -1 padding if necessary
                idx_matrix = torch.nn.functional.pad(index_matrix[index], pad=pad_tuple, mode='constant', value=-1)
                list_index_matrixes.append(idx_matrix)
        else:
            list_index_matrixes.append(index_matrix)

        # Unfold the input indices to get patch indices
        list_patches_indices = list()
        for index in range(len(list_index_matrixes)):
            patch_indices = torch.nn.functional.unfold(list_index_matrixes[index].unsqueeze(0), kernel_size=kernel_size,
                                                       stride=conv_node.stride).transpose(0,
                                                                                          1).to(torch.int32)
            list_patches_indices.append(patch_indices)

        num_patches = list_patches_indices[0].shape[0]

        assert output_flattened_size == num_patches, "Mismatch between the calculated output_size of the image and the real one"

        # Ensure the number of patches matches the expected output size
        assert num_patches == output_flattened_size, f"Mismatch in patch count: {num_patches} != {output_flattened_size}."

        # This conversion is temporary
        filter_temp_matrix_pos_results = []
        filter_temp_matrix_neg_results = []
        filter_temp_offset_pos_results = []
        filter_temp_offset_neg_results = []

        # This FOR loop handles the generations of the filters
        for filter_idx in range(num_filters):
            patch_temp_matrix_pos_results = []
            patch_temp_matrix_neg_results = []
            patch_temp_offset_pos_results = []
            patch_temp_offset_neg_results = []

            # This FOR loop cycle generates the symbolic formulas for the elements of a single filter
            for patch_idx in range(num_patches):

                padding_re_indexing_list = list()
                patch_list = list()
                for patch_indices in list_patches_indices:
                    patch = patch_indices[patch_idx]
                    filter_pad = patch != -1
                    patch = patch[filter_pad]
                    padding_re_indexing_list.append(filter_pad)
                    patch_list.append(patch)

                padding_re_indexing = patch_list[0] != -1

                temp_pos_results = []
                temp_neg_results = []

                temp_offset_pos_results = []
                temp_offset_neg_results = []

                for in_ch_idx in range(input_channels):

                    if filter_weights.ndim == 2:
                        filter = filter_weights[filter_idx].reshape(-1)
                    elif filter_weights.ndim == 3:
                        filter = filter_weights[filter_idx, in_ch_idx].reshape(-1)
                    else:
                        assert False, "Case not implemented."

                    pos_filter = torch.max(filter, torch.tensor(0.).to(DATA_TYPE).to(device))
                    neg_filter = torch.min(filter, torch.tensor(0.).to(DATA_TYPE).to(device))
                    pos_filter = pos_filter[padding_re_indexing_list[in_ch_idx]]
                    neg_filter = neg_filter[padding_re_indexing_list[in_ch_idx]]

                    # Phantom dimension for broadcasting
                    pos_filter = pos_filter[:, None]
                    neg_filter = neg_filter[:, None]

                    # patch indexes mast bu update
                    pos_matrix = sym_upper_bounds.matrix[patch_list[in_ch_idx]]
                    neg_matrix = sym_lower_bounds.matrix[patch_list[in_ch_idx]]
                    pos_offset = sym_upper_bounds.offset[patch_list[in_ch_idx]]
                    neg_offset = sym_lower_bounds.offset[patch_list[in_ch_idx]]

                    i_ch_lower_symb_matrix_bounds = neg_matrix * pos_filter + pos_matrix * neg_filter
                    i_ch_upper_symb_matrix_bounds = pos_matrix * pos_filter + neg_matrix * neg_filter
                    i_ch_lower_symb_bias_bounds = neg_offset * pos_filter.squeeze() + pos_offset * neg_filter.squeeze()
                    i_ch_upper_symb_bias_bounds = pos_offset * pos_filter.squeeze() + neg_offset * neg_filter.squeeze()

                    temp_neg_results.append(i_ch_lower_symb_matrix_bounds)
                    temp_pos_results.append(i_ch_upper_symb_matrix_bounds)
                    temp_offset_neg_results.append(i_ch_lower_symb_bias_bounds)
                    temp_offset_pos_results.append(i_ch_upper_symb_bias_bounds)

                pos_matrix_stack = torch.cat(temp_neg_results, dim=0)
                neg_matrix_stack = torch.cat(temp_pos_results, dim=0)
                pos_offset_stack = torch.cat(temp_offset_neg_results, dim=0)
                neg_offset_stack = torch.cat(temp_offset_pos_results, dim=0)

                pos_matrix_sum = torch.sum(pos_matrix_stack, dim=0)
                neg_matrix_sum = torch.sum(neg_matrix_stack, dim=0)
                pos_offset_sum = torch.sum(pos_offset_stack, dim=0)
                neg_offset_sum = torch.sum(neg_offset_stack, dim=0)

                if conv_node.has_bias:
                    bias = filter_biases[filter_idx]
                    pos_offset_sum += bias
                    neg_offset_sum += bias

                patch_temp_matrix_pos_results.append(pos_matrix_sum)
                patch_temp_matrix_neg_results.append(neg_matrix_sum)
                patch_temp_offset_pos_results.append(pos_offset_sum)
                patch_temp_offset_neg_results.append(neg_offset_sum)

            i_pos_filter_matrix = torch.stack(patch_temp_matrix_pos_results, dim=0)
            i_neg_filter_matrix = torch.stack(patch_temp_matrix_neg_results, dim=0)

            i_pos_filter_offset = torch.stack(patch_temp_offset_pos_results, dim=0)
            i_neg_filter_offset = torch.stack(patch_temp_offset_neg_results, dim=0)

            filter_temp_matrix_pos_results.append(i_pos_filter_matrix)
            filter_temp_matrix_neg_results.append(i_neg_filter_matrix)

            filter_temp_offset_pos_results.append(i_pos_filter_offset)
            filter_temp_offset_neg_results.append(i_neg_filter_offset)

        pos_matrix = torch.cat(filter_temp_matrix_pos_results, dim=0)
        neg_matrix = torch.cat(filter_temp_matrix_neg_results, dim=0)

        pos_offset = torch.cat(filter_temp_offset_pos_results, dim=0)
        neg_offset = torch.cat(filter_temp_offset_neg_results, dim=0)

        if device.type == "cuda":
            pos_matrix = pos_matrix.cpu()
            neg_matrix = neg_matrix.cpu()
            pos_offset = pos_offset.cpu()
            neg_offset = neg_offset.cpu()

        # # Converting to numpy ndarray
        # pos_matrix = pos_matrix.numpy()
        # neg_matrix = neg_matrix.numpy()
        # pos_offset = pos_offset.numpy()
        # neg_offset = neg_offset.numpy()

        upper = LinearFunctions(pos_matrix, pos_offset)
        lower = LinearFunctions(neg_matrix, neg_offset)

        return SymbolicLinearBounds(lower, upper)
