import os
import pstats
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cProfile

seed_value = 42
torch.manual_seed(seed_value)
DATA_TYPE = torch.float32


def generate_heatmaps_and_save(matrix, output_dir="heatmaps"):
    """
    Generates heatmaps for each filter in a 3D matrix and saves them to a specified directory.

    Args:
        matrix (numpy.ndarray): A 3D matrix containing negative, positive, and zero values (num_filters x H x W).
        output_dir (str): Path to the directory where the heatmaps will be saved.

    Output:
        Heatmaps will be saved as PNG files in the specified directory.
    """
    num_filters, height, width = matrix.shape
    if height != width:
        return 0  # Early exit if the matrix is not square.

    # Check if the matrix is 3D
    if len(matrix.shape) == 3:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a custom colormap (blue for negative, white for zero, red for positive)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_colormap", ["blue", "white", "red"]
        )

        # Generate and save the heatmaps
        for i in range(num_filters):
            plt.figure(figsize=(6, 6))
            plt.imshow(matrix[i], cmap=cmap, aspect='equal', interpolation='nearest')
            plt.colorbar(label="Value")
            plt.title(f"Filter Heatmap {i + 1} (0=white, positives=red, negatives=blue)")

            # Save the image
            file_path = os.path.join(output_dir, f"heatmap_filter_{i + 1}.png")
            plt.savefig(file_path, dpi=300)
            plt.close()  # Close the figure to free memory

        print(f"Heatmaps saved in the directory: {output_dir}")


def generate_array_int32(initial_array, k, n, device):
    """
    Generates a new array by concatenating elements of the initial array with multiples of k.

    Args:
        initial_array (torch.Tensor): Initial 1D array (torch.int32).
        k (int): Base value to add to the elements.
        n (int): Maximum number of multiples of k to consider.

    Returns:
        torch.Tensor: New array generated (torch.int32).
    """
    # Ensure the initial array is of type int32
    initial_array = initial_array.to(torch.int32)
    result = initial_array.clone()  # Copy the initial array
    for i in range(1, n + 1):
        result = torch.cat([result, initial_array + i * k])  # Concatenate
    return result.to(device)


def compare_tensors(tensor1, tensor2, atol=1e-3, rtol=1e-3):
    """
    Compares two PyTorch tensors to check if they are equal or similar within a tolerance.

    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
        atol (float): Absolute tolerance for similarity.
        rtol (float): Relative tolerance for similarity.

    Returns:
        dict: Comparison results with keys:
              - "exact_match": True if tensors are exactly equal.
              - "close_match": True if tensors are similar within the tolerance.
              - "max_difference": Maximum absolute difference between tensors.
    """
    # Check for exact equality
    exact_match = torch.equal(tensor1, tensor2)

    # Check for approximate similarity
    close_match = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)

    # Calculate the maximum difference
    max_difference = torch.max(torch.abs(tensor1 - tensor2)).item()

    # Compute the difference tensor
    diff_tensor = tensor1 - tensor2

    sum_tensor1 = torch.sum(tensor1)
    sum_tensor2 = torch.sum(tensor2)


    assert tensor1.shape == tensor2.shape  # Ensure tensors have the same shape

    return {
        "exact_match": exact_match,
        "close_match": close_match,
        "max_difference": max_difference,
        "sum_over_tensor1": sum_tensor1,
        "sum_over_tensor2": sum_tensor2,
    }, diff_tensor


def propagate_conv_bp_sparse(kernel_size, padding, stride, lb, ub, device, filter_weights, filter_biases=None,
                             differentiable=True):

    # Ensure that the lower bounds (lb) and upper bounds (ub) have the same dimensions
    assert lb.shape == ub.shape, "The dimensions of 'lb' and 'ub' must match."
    batch_size = lb.shape[0]

    # Ensure that the inputs (lb and ub) have 4 dimensions: batch_size, channels, height, width
    assert lb.dim() == 4, "'lb' must have shape (batch_size, channels, height, width)."
    assert ub.dim() == 4, "'ub' must have shape (batch_size, channels, height, width)."

    # Extract kernel dimensions for the convolution
    if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        kernel_height, kernel_width = kernel_size
    elif isinstance(kernel_size, int):
        kernel_height = kernel_width = kernel_size
        kernel_size = (kernel_size, kernel_size)
    else:
        raise ValueError("Kernel size must be an int or a tuple of two integers.")

    # Determine padding configuration
    if isinstance(padding, int):
        pad_tuple = (padding, padding, padding, padding)  # Same padding for all sides
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_tuple = (padding[1], padding[1], padding[0], padding[0])  # Symmetric padding for height and width
    elif isinstance(padding, tuple) and len(padding) == 4:
        # Ensure padding is symmetric (top == bottom and left == right)
        if padding[0] != padding[1] or padding[2] != padding[3]:
            raise ValueError("Only symmetrical padding is supported. Top must equal bottom and left must equal right.")
        pad_tuple = padding
    elif padding == 0 or padding is None:
        pad_tuple = (0, 0, 0, 0)  # No padding
    else:
        raise ValueError("Padding must be an int or a tuple of appropriate dimensions.")

    # Determine stride values
    if isinstance(stride, int):
        stride_tuple = (stride, stride)
    elif isinstance(stride, tuple) and len(stride) == 2:
        stride_tuple = stride
    else:
        raise ValueError("Stride must be an int or a tuple of appropriate dimensions.")

    # Extract the number of input channels from the lower bound (lb)
    input_channels = lb.shape[1]

    # Move the lower and upper bounds tensors to the specified device and data type
    lb = lb.to(DATA_TYPE).to(device)
    ub = ub.to(DATA_TYPE).to(device)

    # Flatten the filter weights for sparse matrix operations
    num_filters = filter_weights.shape[0]
    filter_weights = filter_weights.reshape(num_filters, -1).to(DATA_TYPE).to(device)

    # Handle filter biases if provided
    if filter_biases is not None:
        filter_biases = filter_biases.to(DATA_TYPE).to(device)

    # Input shape and flattened size calculations
    input_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
    input_flattened_size = lb.shape[2] * lb.shape[3]

    # Calculate output dimensions after the convolution operation
    pad_top, pad_bottom, pad_left, pad_right = pad_tuple
    output_height = int(((input_shape[1] - kernel_height + pad_top + pad_bottom) / stride_tuple[0]) + 1)
    output_width = int(((input_shape[2] - kernel_width + pad_left + pad_right) / stride_tuple[1]) + 1)
    output_shape = (output_height, output_width)
    output_flattened_size = output_height * output_width

    # Apply padding to the input tensors (if padding is not None)
    if padding is not None:
        lb = torch.nn.functional.pad(lb, pad=pad_tuple, mode='constant', value=0)
        ub = torch.nn.functional.pad(ub, pad=pad_tuple, mode='constant', value=0)
        input_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
        input_flattened_size = lb.shape[2] * lb.shape[3]

    # Flatten the lower and upper bounds for sparse matrix operations
    lb_flattened = lb.reshape(batch_size, -1).to(DATA_TYPE).to(device)
    ub_flattened = ub.reshape(batch_size, -1).to(DATA_TYPE).to(device)

    # Create an index matrix for image patches
    index_matrix = torch.arange(0, input_flattened_size, dtype=DATA_TYPE, device=device).reshape(1, input_shape[1],
                                                                                                 input_shape[2])

    # Use unfold to get patch indices
    patch_indices = torch.nn.functional.unfold(index_matrix, kernel_size=kernel_size, stride=stride_tuple).transpose(0, 1).to(
        torch.int32)
    num_patches = patch_indices.shape[0]

    assert num_patches == (output_height * output_width), "Wrong number of patches. Please check."

    # Ensure the number of patches matches the expected output size
    assert num_patches == output_flattened_size, f"Mismatch in patch count: {num_patches} != {output_flattened_size}."

    # Initialize a bias matrix for all filters (if biases are provided)
    if filter_biases is not None:
        bias_matrix = torch.zeros(num_filters, num_patches, dtype=DATA_TYPE, device=device)

    indices_list = []  # List to store indices for sparse matrix
    temp_indices = []

    # Loop through each filter to create sparse matrix components
    for filter_idx in range(num_filters):
        for patch_idx in range(num_patches):
            # Generate the indices for the current patch
            indices = generate_array_int32(patch_indices[patch_idx, :], input_flattened_size, input_channels - 1,
                                           device=device)

            # Store the indices for the sparse matrix
            temp_indices.append(
                torch.stack((
                    torch.full(indices.shape, filter_idx * num_patches + patch_idx, dtype=torch.long, device=device),
                    indices
                ), dim=1)
            )

        # Apply biases to the bias matrix if provided
        if filter_biases is not None:
            bias_matrix[filter_idx, :] = filter_biases[filter_idx]

    indices_list.append(torch.cat(temp_indices, dim=0))

    # Positive and negative values of filters array
    pos_values_list = torch.maximum(filter_weights, torch.tensor(0.0, device=filter_weights.device))
    neg_values_list = torch.minimum(filter_weights, torch.tensor(0.0, device=filter_weights.device))

    # Replica i valori per il numero di patch
    pos_sparse_values = pos_values_list.repeat_interleave(num_patches, 0).reshape(-1)
    neg_sparse_values = neg_values_list.repeat_interleave(num_patches, 0).reshape(-1)

    # Concatenate the indices and values for the sparse matrix
    sparse_indices = torch.cat(indices_list, dim=0)

    # Create sparse tensors for positive and negative filter values
    pos_filter_tensor = torch.sparse_coo_tensor(sparse_indices.T, pos_sparse_values,
                                                size=(num_filters * num_patches, input_channels * input_flattened_size),
                                                device=device)
    neg_filter_tensor = torch.sparse_coo_tensor(sparse_indices.T, neg_sparse_values,
                                                size=(num_filters * num_patches, input_channels * input_flattened_size),
                                                device=device)

    # Convert the sparse tensors to CSR format if not differentiable
    if not differentiable:
        pos_filter_tensor = pos_filter_tensor.to_sparse_csr()
        neg_filter_tensor = neg_filter_tensor.to_sparse_csr()


    batched_lb = lb_flattened.view(batch_size,  input_channels * input_flattened_size)
    batched_ub = ub_flattened.view(batch_size,  input_channels * input_flattened_size)


    # Perform the matrix multiplication (sparse operation) for the current batch
    output_lb = torch.sparse.mm(batched_lb, pos_filter_tensor.T) + torch.sparse.mm(batched_ub, neg_filter_tensor.T)
    output_ub = torch.sparse.mm(batched_ub, pos_filter_tensor.T) + torch.sparse.mm(batched_lb, neg_filter_tensor.T)


    # Making the matrix dense and reshaping
    output_lb_tensor =output_lb.to_dense().view(batch_size, num_filters, output_height,
                                                                     output_width)
    output_ub_tensor = output_ub.to_dense().view(batch_size, num_filters, output_height,
                                          output_width)

    # Add biases if applicable
    if filter_biases is not None:
        bias_matrix = bias_matrix.view(1, num_filters, output_height, output_width)
        output_lb_tensor += bias_matrix
        output_ub_tensor += bias_matrix

    # Ensure that lower bounds are always less than or equal to upper bounds
    assert torch.all(output_lb_tensor <= output_ub_tensor), "Lower bounds must always be lower than upper bounds."

    # Return the flattened lower and upper bounds tensors
    return output_lb_tensor.reshape(-1), output_ub_tensor.reshape(-1), sparse_indices



def partial_conv_sparse(kernel_size, padding, stride, lb, ub, device, filter_weights,  filter_biases=None, sparse_indices=None, differentiable=True):
    # Flatten the lower and upper bounds for sparse matrix operations
    batch_size = lb.shape[0]
    input_channels = lb.shape[1]
    input_flattened_size = lb.shape[2] * lb.shape[3]
    num_filters = filter_weights.shape[0]

    # Ensure that the lower bounds (lb) and upper bounds (ub) have the same dimensions
    assert lb.shape == ub.shape, "The dimensions of 'lb' and 'ub' must match."

    # Ensure that the inputs (lb and ub) have 4 dimensions: batch_size, channels, height, width
    assert lb.dim() == 4, "'lb' must have shape (batch_size, channels, height, width)."
    assert ub.dim() == 4, "'ub' must have shape (batch_size, channels, height, width)."

    # Extract kernel dimensions for the convolution
    if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        kernel_height, kernel_width = kernel_size
    elif isinstance(kernel_size, int):
        kernel_height = kernel_width = kernel_size
        kernel_size = (kernel_size, kernel_size)
    else:
        raise ValueError("Kernel size must be an int or a tuple of two integers.")

    # Determine padding configuration
    if isinstance(padding, int):
        pad_tuple = (padding, padding, padding, padding)  # Same padding for all sides
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_tuple = (padding[1], padding[1], padding[0], padding[0])  # Symmetric padding for height and width
    elif isinstance(padding, tuple) and len(padding) == 4:
        # Ensure padding is symmetric (top == bottom and left == right)
        if padding[0] != padding[1] or padding[2] != padding[3]:
            raise ValueError("Only symmetrical padding is supported. Top must equal bottom and left must equal right.")
        pad_tuple = padding
    elif padding == 0 or padding is None:
        pad_tuple = (0, 0, 0, 0)  # No padding
    else:
        raise ValueError("Padding must be an int or a tuple of appropriate dimensions.")

    # Determine stride values
    if isinstance(stride, int):
        stride_tuple = (stride, stride)
    elif isinstance(stride, tuple) and len(stride) == 2:
        stride_tuple = stride
    else:
        raise ValueError("Stride must be an int or a tuple of appropriate dimensions.")

    pad_top, pad_bottom, pad_left, pad_right = pad_tuple
    input_shape = (lb.shape[1], lb.shape[2], lb.shape[3])

    output_height = int(((input_shape[1] - kernel_height + pad_top + pad_bottom) / stride_tuple[0]) + 1)
    output_width = int(((input_shape[2] - kernel_width + pad_left + pad_right) / stride_tuple[1]) + 1)

    lb_flattened = lb.reshape(batch_size, -1).to(DATA_TYPE).to(device)
    ub_flattened = ub.reshape(batch_size, -1).to(DATA_TYPE).to(device)

    # Positive and negative values of filters array
    pos_values_list = torch.maximum(filter_weights, torch.tensor(0.0, device=filter_weights.device))
    neg_values_list = torch.minimum(filter_weights, torch.tensor(0.0, device=filter_weights.device))

    # Replica i valori per il numero di patch
    num_patches = output_height * output_width
    pos_sparse_values = pos_values_list.repeat_interleave(num_patches, 0).reshape(-1)
    neg_sparse_values = neg_values_list.repeat_interleave(num_patches, 0).reshape(-1)

    # Initialize a bias matrix for all filters (if biases are provided)
    if filter_biases is not None:
        bias_matrix = torch.zeros(num_filters, num_patches, dtype=DATA_TYPE, device=device)

        for filter_idx in range(num_filters):
            # Apply biases to the bias matrix if provided
                bias_matrix[filter_idx, :] = filter_biases[filter_idx]

    # Create sparse tensors for positive and negative filter values
    pos_filter_tensor = torch.sparse_coo_tensor(sparse_indices.T, pos_sparse_values,
                                                size=(num_filters * num_patches, input_channels * input_flattened_size),
                                                device=device)
    neg_filter_tensor = torch.sparse_coo_tensor(sparse_indices.T, neg_sparse_values,
                                                size=(num_filters * num_patches, input_channels * input_flattened_size),
                                                device=device)

    # Convert the sparse tensors to CSR format if not differentiable
    if not differentiable:
        pos_filter_tensor = pos_filter_tensor.to_sparse_csr()
        neg_filter_tensor = neg_filter_tensor.to_sparse_csr()


    batched_lb = lb_flattened.view(batch_size,  input_channels * input_flattened_size)
    batched_ub = ub_flattened.view(batch_size,  input_channels * input_flattened_size)


    # Perform the matrix multiplication (sparse operation) for the current batch
    output_lb = torch.sparse.mm(batched_lb, pos_filter_tensor.T) + torch.sparse.mm(batched_ub, neg_filter_tensor.T)
    output_ub = torch.sparse.mm(batched_ub, pos_filter_tensor.T) + torch.sparse.mm(batched_lb, neg_filter_tensor.T)


    # Making the matrix dense and reshaping
    output_lb_tensor =output_lb.to_dense().view(batch_size, num_filters, output_height,
                                                                     output_width)
    output_ub_tensor = output_ub.to_dense().view(batch_size, num_filters, output_height,
                                          output_width)

    # Add biases if applicable
    if filter_biases is not None:
        bias_matrix = bias_matrix.view(1, num_filters, output_height, output_width)
        output_lb_tensor += bias_matrix
        output_ub_tensor += bias_matrix

    # Ensure that lower bounds are always less than or equal to upper bounds
    assert torch.all(output_lb_tensor <= output_ub_tensor), "Lower bounds must always be lower than upper bounds."

    # Return the flattened lower and upper bounds tensors
    return output_lb_tensor.reshape(-1), output_ub_tensor.reshape(-1)



def propagate_conv_bp(kernel_size, padding, stride, lb, ub, device, filter_weights, filter_biases=None,
                      differentiable=True):
    """
    Perform a backward propagation through a convolutional layer by simulating
    the convolution behavior using fully connected operations. The function
    handles lower and upper bounds of input tensors for interval-based analysis.

    Arguments:
        kernel_size: Size of the convolutional kernel (int or tuple of ints).
        padding: Padding configuration for the convolution (int, tuple, or None).
        stride: Stride size for the convolution (int or tuple of ints).
        lb: Tensor containing the lower bounds of the input.
        ub: Tensor containing the upper bounds of the input.
        device: Device on which the tensors and computations will reside.
        filter_weights: Convolutional filter weights.
        filter_biases: (Optional) Biases for the filters.
        differentiable: Whether to use differentiable computations (default: True).

    Returns:
        Flattened tensors of lower and upper bounds after convolution.
    """

    # Ensure lb and ub have matching dimensions
    assert lb.shape == ub.shape, "The dimensions of 'lb' and 'ub' must match."
    batch_size = lb.shape[0]

    # Validate the dimensionality of lb and ub
    assert lb.dim() == 4, "'lb' must have shape (batch_size, channels, height, width)."
    assert ub.dim() == 4, "'ub' must have shape (batch_size, channels, height, width)."

    # Parse kernel dimensions (supports both int and tuple inputs)
    if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        kernel_height, kernel_width = kernel_size
    elif isinstance(kernel_size, int):
        kernel_height = kernel_width = kernel_size
        kernel_size = (kernel_size, kernel_size)
    else:
        raise ValueError("Kernel size must be an int or a tuple of two integers.")

    # Configure padding (supports various formats)
    if isinstance(padding, int):
        pad_tuple = (padding, padding, padding, padding)  # Uniform padding
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_tuple = (padding[1], padding[1], padding[0], padding[0])  # Symmetric padding
    elif isinstance(padding, tuple) and len(padding) == 4:
        # Validate symmetric padding (top == bottom and left == right)
        if padding[0] != padding[1] or padding[2] != padding[3]:
            raise ValueError("Only symmetrical padding is supported. Top must equal bottom and left must equal right.")
        pad_tuple = padding
    elif padding == 0 or padding is None:
        pad_tuple = (0, 0, 0, 0)  # No padding
    else:
        raise ValueError("Padding must be an int or a tuple of appropriate dimensions.")

    # Determine stride (supports int or tuple inputs)
    if isinstance(stride, int):
        stride_tuple = (stride, stride)
    elif isinstance(stride, tuple) and len(stride) == 2:
        stride_tuple = stride
    else:
        raise ValueError("Stride must be an int or a tuple of appropriate dimensions.")

    # Input channels and data preparation
    input_channels = lb.shape[1]

    # Move tensors to the specified device and set data type
    lb = lb.to(DATA_TYPE).to(device)
    ub = ub.to(DATA_TYPE).to(device)

    # Flatten filter weights for compatibility with matrix operations
    num_filters = filter_weights.shape[0]
    filter_weights = filter_weights.reshape(num_filters, -1).to(DATA_TYPE).to(device)

    # Handle optional filter biases
    if filter_biases is not None:
        filter_biases = filter_biases.to(DATA_TYPE).to(device)

    # Compute input shape and output dimensions after convolution
    pad_top, pad_bottom, pad_left, pad_right = pad_tuple
    output_height = int(((lb.shape[2] - kernel_height + pad_top + pad_bottom) / stride_tuple[0]) + 1)
    output_width = int(((lb.shape[3] - kernel_width + pad_left + pad_right) / stride_tuple[1]) + 1)
    output_shape = (output_height, output_width)
    output_flattened_size = output_height * output_width

    # Apply padding to input tensors if necessary
    if padding is not None:
        lb = torch.nn.functional.pad(lb, pad=pad_tuple, mode='constant', value=0)
        ub = torch.nn.functional.pad(ub, pad=pad_tuple, mode='constant', value=0)

    # Flatten inputs for compatibility with sparse operations
    lb_flattened = lb.reshape(batch_size, -1).to(DATA_TYPE).to(device)
    ub_flattened = ub.reshape(batch_size, -1).to(DATA_TYPE).to(device)

    # Generate patch indices for sparse convolution simulation
    index_matrix = torch.arange(0, lb.shape[2] * lb.shape[3], dtype=DATA_TYPE, device=device).reshape(
        1, lb.shape[2], lb.shape[3])
    patch_indices = torch.nn.functional.unfold(index_matrix, kernel_size=kernel_size, stride=stride_tuple).transpose(0,
                                                                                                                     1).to(
        torch.int32)
    num_patches = patch_indices.shape[0]

    # Separate positive and negative contributions for weights
    pos_values_list = []
    neg_values_list = []
    for filter_idx in range(num_filters):
        for patch_idx in range(num_patches):
            pos_values_list.append(torch.maximum(filter_weights[filter_idx], torch.tensor(0.0)))
            neg_values_list.append(torch.minimum(filter_weights[filter_idx], torch.tensor(0.0)))

    # Ensure number of patches matches the expected output size
    assert num_patches == output_flattened_size, f"Mismatch in patch count: {num_patches} != {output_flattened_size}."

    # Instantiate matrices to simulate convolution as fully connected operations
    convolution_expanded_matrix_pos = torch.zeros(num_filters, num_patches, input_channels * lb.shape[2] * lb.shape[3],
                                                  dtype=DATA_TYPE, device=device)
    convolution_expanded_matrix_neg = torch.zeros_like(convolution_expanded_matrix_pos)
    bias_expanded_matrix = torch.zeros(num_filters, num_patches, dtype=DATA_TYPE, device=device)

    # Build sparse convolution matrices
    for f_idx in range(num_filters):
        for i in range(num_patches):
            temp_pos = torch.zeros(input_channels * lb.shape[2] * lb.shape[3], dtype=DATA_TYPE, device=device)
            temp_neg = torch.zeros_like(temp_pos)
            indices = generate_array_int32(patch_indices[i, :], lb.shape[2] * lb.shape[3], input_channels - 1,
                                           device=device)
            temp_pos[indices] = pos_values_list[f_idx * num_patches + i]
            temp_neg[indices] = neg_values_list[f_idx * num_patches + i]
            convolution_expanded_matrix_pos[f_idx, i, :] = temp_pos
            convolution_expanded_matrix_neg[f_idx, i, :] = temp_neg
            if filter_biases is not None:
                bias_expanded_matrix[f_idx, i] = filter_biases[f_idx]

    # Iterate through batches to compute output bounds
    ub_output_batch = []
    lb_output_batch = []
    for b_idx in range(batch_size):
        lb_single = lb_flattened[b_idx, :]
        ub_single = ub_flattened[b_idx, :]
        output_tensor_ub = torch.matmul(convolution_expanded_matrix_neg, lb_single.T) + \
                           torch.matmul(convolution_expanded_matrix_pos, ub_single.T)
        output_tensor_lb = torch.matmul(convolution_expanded_matrix_neg, ub_single.T) + \
                           torch.matmul(convolution_expanded_matrix_pos, lb_single.T)
        ub_output_batch.append(output_tensor_ub.view(1, num_filters, output_height, output_width))
        lb_output_batch.append(output_tensor_lb.view(1, num_filters, output_height, output_width))

    # Concatenate outputs for final results
    output_tensor_batch_lb = torch.cat(lb_output_batch, dim=0)
    output_tensor_batch_ub = torch.cat(ub_output_batch, dim=0)

    # Add biases if applicable
    if filter_biases is not None:
        bias_expanded_matrix = bias_expanded_matrix.view(1, num_filters, output_height, output_width)
        output_tensor_batch_lb += bias_expanded_matrix
        output_tensor_batch_ub += bias_expanded_matrix

    # Validate bounds consistency
    assert torch.all(output_tensor_batch_lb <= output_tensor_batch_ub), "Lower bounds must always be <= upper bounds."

    return output_tensor_batch_lb.view(-1), output_tensor_batch_ub.view(-1)


def main():
    # Definizione di un modello semplice per test
    class SimpleConvNet(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super(SimpleConvNet, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=True)

            # Imposta manualmente i pesi dei filtri
            #filters_weights = torch.arange(0, kernel_size*kernel_size*out_channels*in_channels, dtype=DATA_TYPE, device=device)*10
            filters_weights = torch.randn(kernel_size * kernel_size * out_channels * in_channels, dtype=DATA_TYPE,
                                          device=device)
            # filters_weights = torch.tensor(
            #     [
            #         [[[1, 1], [1, 1]]],  # Primo filtro
            #         [[[2, 2], [2, 2]]]  # Secondo filtro
            #     ],
            #     dtype=DATA_TYPE
            # )
            self.conv.weight.data = filters_weights.reshape(out_channels, in_channels, kernel_size, kernel_size)

            self.conv.bias.data = torch.zeros(out_channels, dtype=DATA_TYPE)

        def forward(self, x):
            return self.conv(x)

    # Parametri di esempio
    kernel_size = 3
    padding = (0, 0, 0, 0)
    stride = 1

    #  il problema é qui nel filtro
    filters_number = 1
    batch_size = 2
    in_channels = 3
    img_size_w = 28
    img_size_h = 28
    noise = 0.0000000001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"

    # Modello
    if len(padding) == 4:
        padding = (padding[0], padding[2])
    model = SimpleConvNet(in_channels, filters_number, kernel_size, stride=stride, padding=padding).to(device)

    # Input di esempio
    inputs = torch.randn(batch_size, in_channels, img_size_w, img_size_h, dtype=DATA_TYPE, device=device)
    #inputs = torch.ones(img_size_w * img_size_h * in_channels * batch_size, dtype=DATA_TYPE, device=device)
    #inputs = inputs.reshape(batch_size, in_channels, img_size_w, img_size_h)

    #inputs =  inputs.reshape(batch_size, in_channels, img_size_w, img_size_h)
    lb = inputs - noise
    ub = inputs + noise

    # # Misura il tempo di esecuzione per `propagate_conv`
    # start_time = time.time()
    #
    # profiler = cProfile.Profile()
    # profiler.enable()
    # with torch.no_grad():
    #     results_bp = propagate_conv(model, kernel_size, padding, stride, inputs, device=device)
    # propagate_conv_time = time.time() - start_time
    #
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.strip_dirs()
    # stats.sort_stats('time')
    # stats.print_stats()
    #
    # print(f"{propagate_conv_time=}")

    # Misura il tempo di esecuzione per il modello convoluzionale nativo
    start_time = time.time()
    with torch.no_grad():
        results_conv = model(inputs).view(-1)
    model_time = time.time() - start_time
    print(f"{model_time=}")

    # Misura il tempo di esecuzione per `propagate_conv_bp`
    parameters = list(model.parameters())
    start_time = time.time()

    profiler = cProfile.Profile()
    profiler.enable()

    lb, ub, _ = propagate_conv_bp_sparse(kernel_size, padding, stride, lb, ub, device=device, filter_weights=parameters[0], differentiable=True)
    propagate_conv_bp_time = time.time() - start_time
    print(f"{propagate_conv_bp_time=}")

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.print_stats()

    # Comparazione dei risultati
    # comparation_dict, diff_tensor = compare_tensors(results_conv, results_bp)
    #    generate_heatmaps_and_save(diff_tensor.squeeze(0).cpu())

    comparation_dict_sparse, diff_tensor_sparse = compare_tensors(results_conv, lb)
    #    generate_heatmaps_and_save(diff_tensor_sparse.squeeze(0).cpu())

    #print(comparation_dict)
    print(comparation_dict_sparse)
    #print(f"{results_bp.shape=}")
    print(f"{results_conv.shape=}")

    print(f"{lb=}")
    print(f"{results_conv=}")

if __name__ == "__main__":
    main()


