import os

import torch


import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F

seed_value = 42
torch.manual_seed(seed_value)

DATA_TYPE = torch.float64

def generate_heatmaps_and_save(matrix, output_dir= "heatmaps"):
    """
    Genera heatmap per ogni filtro in una matrice 3D e le salva in una cartella specificata.

    Args:
        matrix (numpy.ndarray): Matrice 3D con valori negativi, positivi e zero (num_filters x H x W).
        output_dir (str): Percorso della cartella dove salvare le immagini.

    Output:
        Le heatmap vengono salvate nella cartella specificata come file PNG.
    """
    # Controllo che la matrice sia 3D
    if len(matrix.shape) == 3:

        num_filters, height, width = matrix.shape

        # Controllo che la matrice sia quadrata
        if height != width:
            raise ValueError("Ogni filtro deve essere una matrice quadrata!")

        # Creazione della cartella se non esiste
        os.makedirs(output_dir, exist_ok=True)

        # Creazione della mappa colori personalizzata
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_colormap", ["blue", "white", "red"]
        )

        # Generazione e salvataggio delle heatmap
        for i in range(num_filters):
            plt.figure(figsize=(6, 6))
            plt.imshow(matrix[i], cmap=cmap, aspect='equal', interpolation='nearest')
            plt.colorbar(label="Valore")
            plt.title(f"Heatmap Filtro {i + 1} (0=bianco, positivi=rosso, negativi=blu)")

            # Salvataggio dell'immagine
            file_path = os.path.join(output_dir, f"heatmap_filter_{i + 1}.png")
            plt.savefig(file_path, dpi=300)
            plt.close()  # Chiude la figura per liberare memoria

        print(f"Heatmap salvate nella cartella: {output_dir}")



def generate_array_int32(initial_array, k, n):
    """
    Genera un nuovo array concatenando gli elementi dell'array iniziale
    con quelli derivati aggiungendo multipli di k, con tipo torch.int32.

    Args:
        initial_array (torch.Tensor): Array iniziale 1D (torch.int32).
        k (int): Valore base da aggiungere agli elementi.
        n (int): Numero massimo di multipli di k da considerare.

    Returns:
        torch.Tensor: Nuovo array generato (torch.int32).
    """
    # Assicurati che l'array iniziale sia di tipo int32
    initial_array = initial_array.to(torch.int32)
    result = initial_array.clone()  # Copia l'array iniziale
    for i in range(1, n + 1):
        result = torch.cat([result, initial_array + i * k])  # Concatenazione
    return result




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
    diff_tensor = tensor1 - tensor2

    sum_tensor1 = torch.sum(tensor1)
    sum_tensor2 = torch.sum(tensor2)

    # Salva il vettore differenza su file
    with open("tensor_difference.txt", "w") as f:
        f.write(str(diff_tensor.cpu().detach().numpy()))

    assert tensor1.shape == tensor2.shape



    return {
        "exact_match": exact_match,
        "close_match": close_match,
        "max_difference": max_difference,
        "sum_over_tensor1": sum_tensor1,
        "sum_over_tensor2": sum_tensor2,
    }, diff_tensor

import torch

def propagate_conv(model, kernel_size, padding, stride, inputs, device):
    # Checking that the lb and ub dims are equal
    ub = inputs
    assert inputs.shape == ub.shape, "The dims of the ub and lb inputs do not match."
    batch_dim = inputs.shape[0]

    # Checking that the input has 4 dims
    assert inputs.dim() == 4, "Input must be shaped as n_batch, n_channels, n_height, n_width."

    # Getting kernel info
    if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        k_w, k_h = kernel_size
        kernel_param_size = kernel_size
    elif isinstance(kernel_size, int):
        k_w = k_h = kernel_size
        kernel_param_size = (kernel_size, kernel_size)
    else:
        assert False, "Kernel size must be either int or a two-dimension tuple."

    if type(padding) == int and padding != 0:
        pad_tuple = (padding, padding, padding, padding)

    elif type(padding) == tuple and len(padding) == 2:
        pad_tuple = (padding[1], padding[1], padding[0], padding[0])

    elif type(padding) == tuple and len(padding) == 4:
        # Verifica che i valori siano simmetrici (opzionale se vuoi evitare un padding asimmetrico)
        if padding[0] != padding[1] or padding[2] != padding[3]:
            raise ValueError(
                "Only symmetrical padding is supported. Left must be equal to right padding as well as top and bottom padding"
            )
        pad_tuple = padding

    elif padding == 0 or padding is None:
        pad_tuple = (0, 0, 0, 0)

    else:
        assert False, "Padding must be either int or a two-dimension tuple."

    # Getting the information of the input images like number of input channels, height, width
    n_input_channels = inputs.shape[1]

    inputs = inputs.to(DATA_TYPE).to(device)
    ub = ub.to(DATA_TYPE).to(device)

    # Getting the filters weights and biases
    params = list(model.parameters())
    filters_number = params[0].shape[0]
    filter_weights = params[0].reshape(filters_number, -1).to(DATA_TYPE).to(device)

    image_shape = (inputs.shape[1], inputs.shape[2], inputs.shape[3])
    image_flattened_dim = inputs.shape[2] * inputs.shape[3]

    # Getting output convolution shape, excluding the number of channels in output
    padding_left, padding_right, padding_top, padding_bottom = pad_tuple
    output_conv_dim_h = int(((image_shape[1] - k_h + padding_top + padding_bottom) / stride) + 1)
    output_conv_dim_w = int(((image_shape[2] - k_w + padding_left + padding_right) / stride) + 1)
    output_conv_shape = (output_conv_dim_h, output_conv_dim_w)
    output_conv_shape_flatten = output_conv_dim_h * output_conv_dim_w

    # After calculating the correct output dimension, we apply padding to input
    if padding is not None:
        inputs = F.pad(inputs, pad=pad_tuple, mode='constant', value=0)
        ub = F.pad(ub, pad=pad_tuple, mode='constant', value=0)
        image_shape = (inputs.shape[1], inputs.shape[2], inputs.shape[3])
        image_flattened_dim = inputs.shape[2] * inputs.shape[3]

    # lb_flattened has shape (batch_dim, n_input_channels, -1)
    lb_flatted = inputs.reshape(batch_dim, -1).to(DATA_TYPE).to(device)
    ub_flatted = ub.reshape(batch_dim, -1).to(DATA_TYPE).to(device)

    # Setting a matrix to handle the indexes (1, height, width)
    matrix_index = torch.arange(0, image_flattened_dim, dtype=DATA_TYPE, device=device).reshape(1, image_shape[1],
                                                                                                image_shape[2])

    # Using Unfold torch function to handle the indexes correctly
    patch_matrix = torch.nn.functional.unfold(matrix_index, kernel_size=kernel_size,stride=stride)
    patch_matrix = patch_matrix.transpose(0, 1).to(torch.int32)


    # The number of patches must be equal to the output_conv_shape_flatten
    n_patches = patch_matrix.shape[0]
    assert n_patches == output_conv_shape_flatten, f"The number of patches = {n_patches} does not match with the expected output shape flattened {output_conv_shape_flatten}."

    # Instantiating the matrix that will simulate the conv behaviour through a fc operation
    convolution_expanded_matrix = torch.zeros(n_patches, filters_number, n_input_channels * image_flattened_dim, dtype=DATA_TYPE, device=device)

    # Ciclo sui filtri per ogni batch
    output_batch = []
    for b_idx in range(batch_dim):
        lb_single = lb_flatted[b_idx, :]
        # Ciclo sui filtri
        for f_idx in range(filters_number):
            filter = filter_weights[f_idx, :]
            for i in range(n_patches):
                temp = torch.zeros(n_input_channels * image_flattened_dim, dtype=DATA_TYPE, device=device)
                indices = generate_array_int32(patch_matrix[i, :], image_flattened_dim, n_input_channels-1)
                # the n_input_channels dim must be done automatically
                temp[indices] = filter
                convolution_expanded_matrix[i, f_idx, :] = temp

        reshaped_matrix = convolution_expanded_matrix.permute(1, 0, 2).unsqueeze(0)

        # Perform the fully connected-like operation for each batch item separately
        output_tensor = torch.matmul(reshaped_matrix, lb_single.T)

        output_tensor = output_tensor.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
        output_batch.append(output_tensor)

    # Stack the output tensors along the batch dimension
    output_tensor_batch = torch.cat(output_batch, dim=0)

    return output_tensor_batch



def main():
    # Definizione di un modello semplice per test
    class SimpleConvNet(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super(SimpleConvNet, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

            # Imposta manualmente i pesi dei filtri
            #filters_weights = torch.arange(0, kernel_size*kernel_size*out_channels*in_channels, dtype=DATA_TYPE, device=device)*10
            filters_weights = torch.randn(kernel_size*kernel_size*out_channels*in_channels, dtype=DATA_TYPE, device=device)
            #filters_weights = torch.tensor([[[[-0.5, 1.2], [3.1, -0.7]]],  [[[0.9, -2.3], [-1.1, 4.4]]], [[[-3.2, 0.8], [2.7, -1.5]]]], dtype=DATA_TYPE)

            self.conv.weight.data = filters_weights.reshape(out_channels, in_channels, kernel_size, kernel_size)

            #self.conv.bias.data = torch.zeros(out_channels, dtype=DATA_TYPE)

        def forward(self, x):
            return self.conv(x)

    # Parametri di esempio
    kernel_size = 3
    padding = (1,1,2,2)
    stride = 1
    filters_number = 2
    batch_size = 26
    in_channels = 6
    img_size_w = 28
    img_size_h = 26
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"

    # Modello
    if len(padding) == 4:
        padding = (padding[0], padding[2])
    model = SimpleConvNet(in_channels, filters_number, kernel_size, stride=stride, padding=padding).to(device)

    # Input di esempio
    inputs = (torch.arange(0, img_size_w*img_size_h*batch_size*in_channels, device=device, dtype=DATA_TYPE))
    inputs =  inputs.reshape(batch_size, in_channels, img_size_w, img_size_h)
    lb = inputs
    ub = inputs

    # Chiamata della funzione
    with torch.no_grad():
        results_bp = propagate_conv(model, kernel_size, padding, stride, inputs, device=device)

    with torch.no_grad():
        results_conv = model(inputs)

    comparation_dict, diff_tensor = compare_tensors(results_conv, results_bp)
    generate_heatmaps_and_save(diff_tensor.squeeze(0).cpu())
    print(comparation_dict)
    print(f"{results_bp.shape=}")
    print(f"{results_conv.shape=}")





if __name__ == "__main__":
    main()
