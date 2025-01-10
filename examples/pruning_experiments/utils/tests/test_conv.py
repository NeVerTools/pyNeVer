from itertools import product
import torch
import torch.nn.functional as F
import pytest
from torch import nn
from torch.nn import init

from examples.pruning_experiments.utils.convolution_bp import propagate_conv_bp_sparse, propagate_conv_bp

def create_model(input_channels, filters_number, kernel_size, stride, padding, device):
    """
    Creates a model with a convolutional layer, a flattening layer, and a fully connected layer.
    The convolutional filters are randomly initialized.

    Args:
        input_channels (int): Number of input channels.
        filters_number (int): Number of filters in the convolutional layer.
        kernel_size (int): Kernel size (filters).
        stride (int): Stride for the convolutional layer.
        padding (int): Padding for the convolutional layer.
        device (torch.device): Device on which to create the model.

    Returns:
        torch.nn.Module: The created PyTorch model.
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(
                input_channels, filters_number, kernel_size, stride=stride, padding=padding, bias=True
            )
            self.flatten = nn.Flatten()
            self._initialize_weights()

        def _initialize_weights(self):
            # Initialize the weights of the convolutional layer
            init.normal_(self.conv.weight, mean=0.0, std=0.02)  # Normal distribution

        def forward(self, x):
            x = self.conv(x)  # Pass through the convolutional layer
            x = self.flatten(x)  # Flatten all dimensions except the batch dimension
            return x

    # Instantiate the model and move it to the specified device
    model = SimpleModel().to(device)
    return model

def generate_optimized_test_params():
    """
    Generate a set of test parameters for the convolutional layer tests.

    Returns:
        list: A list of parameter combinations.
    """
    kernel_sizes = [1, 2, 3]
    paddings = [0, 1, 2, 3]
    strides = [1, 2, 3]
    filter_numbers = [8, 12, 16]
    batch_sizes = [8, 16, 32]
    input_channels = [1, 3, 6]
    height_dims = [6, 10, 16]
    width_dims = [6, 10, 16]

    # Generate all combinations of the parameters
    test_params = list(product(
        kernel_sizes,
        paddings,
        strides,
        filter_numbers,
        batch_sizes,
        input_channels,
        height_dims,
        width_dims
    ))

    return test_params

@pytest.mark.parametrize(
    "kernel_size, padding, stride, filters_number, batch_size, input_channels, height_dim, width_dim",
    generate_optimized_test_params()
)
def test_propagate_conv(kernel_size, padding, stride, filters_number, batch_size, input_channels, height_dim, width_dim):
    """
    Test the custom convolutional backpropagation function against PyTorch's built-in functionality.

    Args:
        kernel_size (int): Kernel size.
        padding (int): Padding size.
        stride (int): Stride size.
        filters_number (int): Number of filters.
        batch_size (int): Batch size.
        input_channels (int): Number of input channels.
        height_dim (int): Height of the input.
        width_dim (int): Width of the input.
    """
    noise = 1e-9

    # Set the device for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate random inputs
    inputs = torch.randn(batch_size, input_channels, width_dim, height_dim, dtype=torch.float32, device=device)

    lb = inputs - noise  # Lower bound
    ub = inputs + noise  # Upper bound

    # Create the model
    model = create_model(input_channels, filters_number, kernel_size, stride, padding, device=device)

    # Call the custom function
    with torch.no_grad():
        output_custom_sparse, _ = propagate_conv_bp_sparse(kernel_size, padding, stride, lb, ub, device=device,
                                                    filter_weights=model.conv.weight, filter_biases=model.conv.bias)
        output_custom, _ = propagate_conv_bp(kernel_size, padding, stride, lb, ub, device=device,
                                                           filter_weights=model.conv.weight,
                                                           filter_biases=model.conv.bias)

    # Get PyTorch output
    with torch.no_grad():
        output_torch = model(inputs)

    # Verify that the results are similar
    # Assume a tolerance for the difference between the results
    assert torch.allclose(output_custom, output_torch.view(-1), atol=1e-5), \
        f"Test failed for kernel_size={kernel_size}, padding={padding}, stride={stride}, filters_number={filters_number}, " \
        f"batch_size={batch_size}, input_channels={input_channels}, height_dim={height_dim}, width_dim={width_dim} .\n" \
        f"Custom output: {output_custom}\nTorch output: {output_torch}"

    assert torch.allclose(output_custom_sparse, output_torch.view(-1), atol=1e-5), \
        f"Test failed for kernel_size={kernel_size}, padding={padding}, stride={stride}, filters_number={filters_number}, " \
        f"batch_size={batch_size}, input_channels={input_channels}, height_dim={height_dim}, width_dim={width_dim} .\n" \
        f"Custom output: {output_custom}\nTorch output: {output_torch}"

if __name__ == "__main__":
    # Example test invocation with specific parameters
    test_propagate_conv(2, 0, 1, 8, 8, 3, 6, 6)
