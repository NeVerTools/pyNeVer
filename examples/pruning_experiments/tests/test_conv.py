from itertools import product
import torch
import torch.nn.functional as F
import pytest

from examples.pruning_experiments.testinh import propagate_conv


def generate_optimized_test_params():
    kernel_sizes = [1, 2, 3]
    paddings = [1, 2, 3]
    strides = [1, 2, 3]
    filter_numbers = [8, 16, 28]
    batch_sizes = [8, 16, 32]
    input_channels = [1, 3, 6]
    height_dims = [16, 28, 32]
    width_dims = [16, 28, 32]

    # Generate all combinations
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
    # Parametri di test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Crea input aleatori
    inputs = torch.randn(batch_size, input_channels, width_dim, height_dim, dtype=torch.float32, device=device)

    model = torch.nn.Conv2d(input_channels, filters_number, kernel_size, stride=stride, padding=padding, bias=False).to(device)


    # Chiamata della funzione
    with torch.no_grad():
        output_custom = propagate_conv(model, kernel_size, padding, stride, inputs, device=device)

    with torch.no_grad():
        output_torch = model(inputs)



    # Verifica che i risultati siano simili
    # Assumiamo una tolleranza di errore per la differenza tra i risultati
    assert torch.allclose(output_custom, output_torch, atol=1e-5), \
        f"Test failed for kernel_size={kernel_size}, padding={padding}, stride={stride}, filters_number={filters_number}, " \
        f"batch_size={batch_size}, input_channels={input_channels}, height_dim={height_dim}, width_dim={width_dim} .\n" \
        f"Custom output: {output_custom}\nTorch output: {output_torch}"

