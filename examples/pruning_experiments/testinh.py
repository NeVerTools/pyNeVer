import torch


def compare_tensors(tensor1, tensor2, atol=1e-8, rtol=1e-5):
    """
    Confronta due tensori PyTorch per verificare se sono uguali o simili entro una tolleranza.

    Args:
        tensor1 (torch.Tensor): Primo tensore.
        tensor2 (torch.Tensor): Secondo tensore.
        atol (float): Tolleranza assoluta per la somiglianza.
        rtol (float): Tolleranza relativa per la somiglianza.

    Returns:
        dict: Risultati del confronto con chiavi:
              - "exact_match": True se i tensori sono esattamente uguali.
              - "close_match": True se i tensori sono simili entro la tolleranza.
              - "max_difference": Valore massimo della differenza assoluta tra i tensori.
    """
    # Controlla uguaglianza esatta
    exact_match = torch.equal(tensor1, tensor2)

    # Controlla somiglianza approssimativa
    close_match = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)

    # Calcola la differenza massima
    max_difference = torch.max(torch.abs(tensor1 - tensor2)).item()

    # Calcola il vettore differenza
    new_tensor = tensor1 - tensor2

    # Salva il vettore differenza su file
    with open("tensor_difference.txt", "w") as f:
        f.write(str(new_tensor.cpu().detach().numpy()))

    assert tensor1.shape == tensor2.shape

    return {
        "exact_match": exact_match,
        "close_match": close_match,
        "max_difference": max_difference,
    }

def calculate_rs_loss_regularizer(model, kernel_size, padding, stride, filters_number, inputs, lb, ub, normalized):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checking that the lb and ub dims are equal
    assert lb.shape == ub.shape, "The dims of the ub and lb inputs do not match."
    batch_dim = lb.shape[0]

    # Checking that the input has 4 dims
    assert lb.dim() == 4, "Input must be shaped as n_batch, n_channels, n_height, n_width."

    # Getting the information of the input images like number of input channels, height, width
    n_input_channels = inputs.shape[1]

    lb = lb.to(torch.float32).to(device)
    ub = ub.to(torch.float32).to(device)

    # lb_flattened has shape (batch_dim, n_input_channels, -1)
    lb_flatted = lb.reshape(batch_dim, n_input_channels, -1).to(torch.float32).to(device)
    ub_flatted = ub.reshape(batch_dim, n_input_channels, -1).to(torch.float32).to(device)

    # Getting the filters weights and biases
    params = list(model.parameters())
    assert n_input_channels == n_input_channels, "The number of input channels does not match with the filters channels."
    filter_weights = params[0].reshape(filters_number, n_input_channels, -1).to(torch.float32).to(device)
    #filter_biases = params[1].to(torch.float32).to(device)

    kernel_param_size = kernel_size * kernel_size

    image_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
    image_flattened_dim = lb.shape[2] * lb.shape[3]

    assert image_shape[1] == image_shape[2], "The image must be squared"

    # Getting output convolution shape, excluding the number of channels in output
    output_conv_dim = int(((image_shape[1] - kernel_size + 2 * padding) / stride) + 1)
    output_conv_shape = (output_conv_dim, output_conv_dim)
    output_conv_shape_flatten = int(output_conv_dim * output_conv_dim)

    # Setting a matrix to handle the indexes
    matrix_index = torch.arange(0, image_flattened_dim, dtype=torch.float32, device=device).reshape(1, image_shape[1],
                                                                                                    image_shape[2])
    # No padding supported till now //TODO
    assert padding == 0, "No padding supported"

    # Using Unfold torch function to handles the indexes correctly
    patch_matrix = torch.nn.functional.unfold(matrix_index, kernel_size=kernel_size, stride=stride)
    patch_matrix = patch_matrix.transpose(0, 1).to(torch.int32)

    n_patches = patch_matrix.shape[0]

    # Instantiating the matrix that will simulate the conv behaviour through a fc operation
    convolution_expanded_matrix = torch.zeros(n_patches, n_input_channels, image_flattened_dim, filters_number, device=device)

    # Ciclo sui filtri
    for f_idx in range(filters_number):
        filter = filter_weights[f_idx, :, :]
        for i in range(n_patches):
            temp = torch.zeros(n_input_channels, image_flattened_dim, dtype=torch.float32, device=device)
            indices = patch_matrix[i, :]
            # the n_input_channels dim must be done automatically
            temp[:, indices] = filter
            convolution_expanded_matrix[i, :, :, f_idx] = temp

    reshaped_matrix = convolution_expanded_matrix.permute(3, 0, 1, 2).unsqueeze(0)
    lb_flatted = lb_flatted.unsqueeze(1).unsqueeze(2)
    output_tensor = reshaped_matrix * lb_flatted
    output_tensor = output_tensor.sum(-1).sum(-1).squeeze(-1)
    output_tensor = output_tensor.reshape(batch_dim, filters_number, output_conv_dim, output_conv_dim)

    return output_tensor


def main():
    # Definizione di un modello semplice per test
    class SimpleConvNet(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(SimpleConvNet, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)

            # Imposta manualmente i pesi dei filtri
            filters_weights = torch.arange(0, kernel_size*kernel_size*out_channels*in_channels, dtype=torch.float32, device=device)*10
            self.conv.weight.data = filters_weights.reshape(out_channels, in_channels, kernel_size, kernel_size)

            #self.conv.bias.data = torch.zeros(out_channels, dtype=torch.float32)

        def forward(self, x):
            return self.conv(x)

    # Parametri di esempio
    kernel_size = 3
    padding = 0
    stride = 1
    filters_number = 2
    batch_size = 128
    in_channels = 1
    img_size = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modello
    model = SimpleConvNet(in_channels, filters_number, kernel_size).to(device)

    # Input di esempio
    inputs = torch.arange(0, img_size*img_size*batch_size, device=device, dtype=torch.float32).reshape(batch_size, in_channels, img_size, img_size)
    lb = inputs
    ub = inputs

    # Chiamata della funzione
    results_bp = calculate_rs_loss_regularizer(model, kernel_size, padding, stride, filters_number, inputs, lb, ub, normalized=False)

    with torch.no_grad():
        results_conv = model(inputs)

    comparation_dict = compare_tensors(results_conv, results_bp)
    print(comparation_dict)

    pass


if __name__ == "__main__":
    main()
