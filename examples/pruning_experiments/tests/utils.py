import torch
import torch.nn.functional as F
import pytest

from examples.pruning_experiments.testinh import propagate_conv

# Funzione di test per confrontare i risultati
@pytest.mark.parametrize(
    "kernel_size, padding, stride, filters_number",
    [(3, 1, 1, 8), (5, 2, 2, 16), (3, 0, 1, 32)]
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("input_channels", [3, 6])
@pytest.mark.parametrize("height_width", [(28, 28), (32, 32)])
def test_propagate_conv(kernel_size, padding, stride, filters_number, batch_size, input_channels, height_width):
    # Parametri di test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Crea input aleatori
    inputs = torch.randn(batch_size, input_channels, *height_width, dtype=torch.float32, device=device)


    # Crea un modello fittizio con il numero di filtri e kernel indicati
    model = torch.nn.Conv2d(input_channels, filters_number, kernel_size, stride=stride, padding=padding).to(device)

    # Chiamata alla tua funzione
    output_custom = propagate_conv(model, kernel_size, padding, stride, inputs, device)

    # Chiamata alla funzione conv2d di PyTorch per ottenere i risultati attesi
    output_torch = F.conv2d(inputs, model.weight, model.bias, stride=stride, padding=padding)

    # Verifica che i risultati siano simili
    # Assumiamo una tolleranza di errore per la differenza tra i risultati
    assert torch.allclose(output_custom, output_torch, atol=1e-5), \
        f"Test failed for kernel_size={kernel_size}, padding={padding}, stride={stride}, filters_number={filters_number}, " \
        f"batch_size={batch_size}, input_channels={input_channels}, height_width={height_width}.\n" \
        f"Custom output: {output_custom}\nTorch output: {output_torch}"

