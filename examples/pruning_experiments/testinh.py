import os

import torch


import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def fc_convLike(model, kernel_size, padding, stride, filters_number, inputs, lb, ub, normalized, device):
    # Checking that the lb and ub dims are equal
    assert lb.shape == ub.shape, "The dims of the ub and lb inputs do not match."
    batch_dim = lb.shape[0]

    # Checking that the input has 4 dims
    assert lb.dim() == 4, "Input must be shaped as n_batch, n_channels, n_height, n_width."

    # Getting the information of the input images like number of input channels, height, width
    n_input_channels = inputs.shape[1]

    lb = lb.to(DATA_TYPE).to(device)
    ub = ub.to(DATA_TYPE).to(device)

    # lb_flattened has shape (batch_dim, n_input_channels, -1)
    lb_flatted = lb.reshape(batch_dim, -1).to(DATA_TYPE).to(device)
    ub_flatted = ub.reshape(batch_dim, -1).to(DATA_TYPE).to(device)

    # Getting the filters weights and biases
    params = list(model.parameters())
    assert n_input_channels == n_input_channels, "The number of input channels does not match with the filters channels."
    filter_weights = params[0].reshape(filters_number, -1).to(DATA_TYPE).to(device)

    kernel_param_size = kernel_size * kernel_size

    image_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
    image_flattened_dim = lb.shape[2] * lb.shape[3]

    assert image_shape[1] == image_shape[2], "The image must be squared"

    # Getting output convolution shape, excluding the number of channels in output
    output_conv_dim = int(((image_shape[1] - kernel_size + 2 * padding) / stride) + 1)
    output_conv_shape = (output_conv_dim, output_conv_dim)
    output_conv_shape_flatten = int(output_conv_dim * output_conv_dim)

    # Setting a matrix to handle the indexes (1, height, width)
    matrix_index = torch.arange(0, image_flattened_dim, dtype=DATA_TYPE, device=device).reshape(1, image_shape[1],
                                                                                                    image_shape[2])
    # No padding supported till now //TODO
    assert padding == 0, "No padding supported"

    # Using Unfold torch function to handle the indexes correctly
    patch_matrix = torch.nn.functional.unfold(matrix_index, kernel_size=kernel_size, stride=stride)
    patch_matrix = patch_matrix.transpose(0, 1).to(torch.int32)


    # The number of patches must be equal to the output_conv_shape_flatten
    n_patches = patch_matrix.shape[0]
    assert n_patches == output_conv_shape_flatten, "The number of patches does not match with the expected output shape."

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

        output_tensor = output_tensor.reshape(1, filters_number, output_conv_dim, output_conv_dim)
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
            filters_weights = torch.arange(0, kernel_size*kernel_size*out_channels*in_channels, dtype=DATA_TYPE, device=device)*10
            #filters_weights = torch.randn(kernel_size*kernel_size*out_channels*in_channels, dtype=DATA_TYPE, device=device)
            #filters_weights = torch.tensor([[[[-0.5, 1.2], [3.1, -0.7]]],  [[[0.9, -2.3], [-1.1, 4.4]]], [[[-3.2, 0.8], [2.7, -1.5]]]], dtype=DATA_TYPE)

            self.conv.weight.data = filters_weights.reshape(out_channels, in_channels, kernel_size, kernel_size)

            #self.conv.bias.data = torch.zeros(out_channels, dtype=DATA_TYPE)

        def forward(self, x):
            return self.conv(x)

    # Parametri di esempio
    kernel_size = 3
    padding = 0
    stride = 1
    filters_number = 6
    batch_size = 128
    in_channels = 6
    img_size = 28
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"

    # Modello
    model = SimpleConvNet(in_channels, filters_number, kernel_size, stride=stride, padding=padding).to(device)

    # Input di esempio
    inputs = (torch.arange(0, img_size*img_size*batch_size*in_channels, device=device, dtype=DATA_TYPE))
    inputs =  inputs.reshape(batch_size, in_channels, img_size, img_size)
    lb = inputs
    ub = inputs

    # Chiamata della funzione
    with torch.no_grad():
        results_bp = fc_convLike(model, kernel_size, padding, stride, filters_number, inputs, inputs, inputs, normalized=False, device = device)

    with torch.no_grad():
        results_conv = model(inputs)

    comparation_dict, diff_tensor = compare_tensors(results_conv, results_bp)
    generate_heatmaps_and_save(diff_tensor.squeeze(0).cpu())
    print(comparation_dict)
    print(f"{results_bp.shape=}")
    print(f"{results_conv.shape=}")


    pass




if __name__ == "__main__":
    main()
