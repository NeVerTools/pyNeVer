import os

import torch
import time

import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F

seed_value = 42
torch.manual_seed(seed_value)

DATA_TYPE = torch.float32

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




def compare_tensors(tensor1, tensor2, atol=1e-5, rtol=1e-5):
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
    filter_biases = params[1].to(DATA_TYPE).to(device)

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
    bias_expanded_matrix = torch.zeros(filters_number, n_patches, dtype=DATA_TYPE, device=device)

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
                if b_idx == 0:
                    bias_expanded_matrix[f_idx, i] = filter_biases[f_idx]


        reshaped_matrix = convolution_expanded_matrix.permute(1, 0, 2).unsqueeze(0)

        # Perform the fully connected-like operation for each batch item separately
        output_tensor = torch.matmul(reshaped_matrix, lb_single.T)

        output_tensor = output_tensor.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
        output_batch.append(output_tensor)

    # Stack the output tensors along the batch dimension
    output_tensor_batch = torch.cat(output_batch, dim=0)

    #Adding biases
    bias_expanded_matrix = bias_expanded_matrix.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
    output_tensor_batch = output_tensor_batch + bias_expanded_matrix

    return output_tensor_batch


def propagate_conv_bp(filer_weights, kernel_size, padding, stride, lb, ub, device, filter_biases=None):
    times = []

    # Checking that the lb and ub dims are equal
    assert lb.shape == ub.shape, "The dims of the ub and lb inputs do not match."
    batch_dim = lb.shape[0]

    # Checking that the input has 4 dims
    assert lb.dim() == 4, "lb must be shaped as n_batch, n_channels, n_height, n_width."
    assert ub.dim() == 4, "ub must be shaped as n_batch, n_channels, n_height, n_width."
    # Getting kernel info

    if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        k_w, k_h = kernel_size[0], kernel_size[1]
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
    n_input_channels = lb.shape[1]

    lb = lb.to(DATA_TYPE).to(device)
    ub = ub.to(DATA_TYPE).to(device)

    # Getting the filters weights and biases
    filters_number = filer_weights.shape[0]
    filter_weights = filer_weights.reshape(filters_number, -1).to(DATA_TYPE).to(device)

    if filter_biases is not None:
        filter_biases = filter_biases.to(DATA_TYPE).to(device)

    image_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
    image_flattened_dim = lb.shape[2] * lb.shape[3]

    # Getting output convolution shape, excluding the number of channels in output
    padding_left, padding_right, padding_top, padding_bottom = pad_tuple
    output_conv_dim_h = int(((image_shape[1] - k_h + padding_top + padding_bottom) / stride) + 1)
    output_conv_dim_w = int(((image_shape[2] - k_w + padding_left + padding_right) / stride) + 1)
    output_conv_shape = (output_conv_dim_h, output_conv_dim_w)
    output_conv_shape_flatten = output_conv_dim_h * output_conv_dim_w

    # After calculating the correct output dimension, we apply padding to input
    if padding is not None:
        lb = F.pad(lb, pad=pad_tuple, mode='constant', value=0)
        ub = F.pad(ub, pad=pad_tuple, mode='constant', value=0)
        image_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
        image_flattened_dim = lb.shape[2] * lb.shape[3]

    # lb_flattened has shape (batch_dim, n_input_channels, -1)
    lb_flatted = lb.reshape(batch_dim, -1).to(DATA_TYPE).to(device)
    ub_flatted = ub.reshape(batch_dim, -1).to(DATA_TYPE).to(device)

    # Setting a matrix to handle the indexes (1, height, width)
    matrix_index = torch.arange(0, image_flattened_dim, dtype=DATA_TYPE, device=device).reshape(1, image_shape[1],
                                                                                                image_shape[2])

    # Using Unfold torch function to handle the indexes correctly
    patch_matrix = torch.nn.functional.unfold(matrix_index, kernel_size=kernel_size, stride=stride)
    patch_matrix = patch_matrix.transpose(0, 1).to(torch.int32)

    # The number of patches must be equal to the output_conv_shape_flatten
    n_patches = patch_matrix.shape[0]
    assert n_patches == output_conv_shape_flatten, f"The number of patches = {n_patches} does not match with the expected output shape flattened {output_conv_shape_flatten}."



    # Instantiating the matrix that will simulate the conv behaviour through a fc operation
    covolution_sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.empty((3, 0), dtype=torch.long),
        values=torch.empty(0, dtype=torch.float32),
        size=(n_patches, filters_number, n_input_channels * image_flattened_dim),
        dtype=DATA_TYPE,
        device=device
    )

    bias_expanded_matrix = torch.zeros(filters_number, n_patches, dtype=DATA_TYPE, device=device)

    # Ciclo sui filtri per ogni batch


    output_batch_lb = []
    output_batch_ub = []


    for b_idx in range(batch_dim):
        lb_single = lb_flatted[b_idx, :]
        ub_single = ub_flatted[b_idx, :]
        # Ciclo sui filtri
        for f_idx in range(filters_number):
            filter = filter_weights[f_idx, :]
            for i in range(n_patches):


                indices = generate_array_int32(patch_matrix[i, :], image_flattened_dim, n_input_channels - 1)
                indices_3d = torch.stack((torch.full(indices.shape, i, dtype=torch.long),
                                          torch.full(indices.shape, f_idx, dtype=torch.long),
                                          indices), dim=1)


                updating_matrix = torch.sparse_coo_tensor(
                    indices=indices_3d.T,
                    values=filter,
                    size=(n_patches, filters_number, n_input_channels * image_flattened_dim),
                    dtype=DATA_TYPE,
                    device=device
                )


                covolution_sparse_tensor = covolution_sparse_tensor + updating_matrix

                if b_idx == 0:
                    if filter_biases is not None:
                        bias_expanded_matrix[f_idx, i] = filter_biases[f_idx]

        expanded_filters_matrix = covolution_sparse_tensor.permute(1, 0, 2).unsqueeze(0)

        # Calculating the "positive" and "negative" filters matrix
        #F_max = torch.maximum(expanded_filters_matrix, torch.tensor(0.0))
        #F_min = torch.minimum(expanded_filters_matrix, torch.tensor(0.0))
        # F_max
        indices = expanded_filters_matrix._indices()
        values = expanded_filters_matrix._values()
        updated_values = torch.maximum(values, torch.tensor(0.0))

        # Getting F_max
        F_max = torch.sparse_coo_tensor(indices, updated_values, expanded_filters_matrix.size(), device=expanded_filters_matrix.device)
        updated_values = torch.minimum(values, torch.tensor(0.0))

        # Getting F_min
        F_min = torch.sparse_coo_tensor(indices, updated_values, expanded_filters_matrix.size(),
                                        device=expanded_filters_matrix.device)

        # Perform the fully connected-like operation for each batch item separately
        time_ = time.time()

        output_tensor_lb = torch.sparse.sum(F_max * lb_single.T, dim=-1) + torch.sparse.sum(F_min * ub_single.T, dim=-1)
        output_tensor_ub = torch.sparse.sum(F_max * ub_single.T, dim=-1) + torch.sparse.sum(F_min * lb_single.T, dim=-1)
        diff = time.time() - time_
        times.append(diff)

        #output_tensor_lb = output_tensor_lb.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
        #output_tensor_ub = output_tensor_ub.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
        output_batch_lb.append(output_tensor_lb)
        output_batch_ub.append(output_tensor_ub)

    # Stack the output tensors along the batch dimension


    output_tensor_batch_lb = torch.cat(output_batch_lb, dim=0)
    output_tensor_batch_ub = torch.cat(output_batch_ub, dim=0)
    times.append(diff)

    print(sum(times))

    # Adding biases
    # if filter_biases is not None:
    #     bias_expanded_matrix = bias_expanded_matrix.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
    #     output_tensor_batch_lb = output_tensor_batch_lb + bias_expanded_matrix
    #     output_tensor_batch_ub = output_tensor_batch_ub + bias_expanded_matrix
    #
    # assert torch.all(
    #     output_tensor_batch_lb <= output_tensor_batch_ub), "Lower bounds must always be lower than upper bounds."
    return output_tensor_batch_lb, output_tensor_batch_ub


def main():
    # Definizione di un modello semplice per test
    class SimpleConvNet(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super(SimpleConvNet, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)

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
    noise = 0.0000000001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"

    # Modello
    if len(padding) == 4:
        padding = (padding[0], padding[2])
    model = SimpleConvNet(in_channels, filters_number, kernel_size, stride=stride, padding=padding).to(device)

    # Input di esempio
    inputs = torch.randn(batch_size, in_channels, img_size_w, img_size_h, dtype=DATA_TYPE, device=device)

    inputs =  inputs.reshape(batch_size, in_channels, img_size_w, img_size_h)
    lb = inputs - noise
    ub = inputs + noise

    # Misura il tempo di esecuzione per `propagate_conv`
    start_time = time.time()
    with torch.no_grad():
        results_bp = propagate_conv(model, kernel_size, padding, stride, inputs, device=device)
    propagate_conv_time = time.time() - start_time
    print(f"{propagate_conv_time=}")

    # Misura il tempo di esecuzione per il modello convoluzionale nativo
    start_time = time.time()
    with torch.no_grad():
        results_conv = model(inputs)
    model_time = time.time() - start_time
    print(f"{model_time=}")


    # Misura il tempo di esecuzione per `propagate_conv_bp`
    parameters = list(model.parameters())
    start_time = time.time()
    with torch.no_grad():
        lb, ub = propagate_conv_bp(parameters[0], kernel_size, padding, stride, lb, ub, device=device,
                                   filter_biases=parameters[1])
    propagate_conv_bp_time = time.time() - start_time
    print(f"{propagate_conv_bp_time=}")


    # Comparazione dei risultati
    comparation_dict, diff_tensor = compare_tensors(results_conv, results_bp)
    generate_heatmaps_and_save(diff_tensor.squeeze(0).cpu())
    print(comparation_dict)
    print(f"{results_bp.shape=}")
    print(f"{results_conv.shape=}")





if __name__ == "__main__":
    main()
