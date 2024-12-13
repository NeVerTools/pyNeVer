import os
import pstats

import torch
import time

import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F
import cProfile

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
    num_filters, height, width = matrix.shape
    if height != width:
        return 0

    # Controllo che la matrice sia 3D
    if len(matrix.shape) == 3:


        # Controllo che la matrice sia quadrata


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

def propagate_conv(model, kernel_size, padding, stride, inputs, device, filter_biases_bool=True):
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

    if filter_biases_bool:
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
                    if filter_biases_bool:
                        bias_expanded_matrix[f_idx, i] = filter_biases[f_idx]


        reshaped_matrix = convolution_expanded_matrix.permute(1, 0, 2).unsqueeze(0)

        # Perform the fully connected-like operation for each batch item separately
        output_tensor = torch.matmul(reshaped_matrix, lb_single.T)

        output_tensor = output_tensor.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
        output_batch.append(output_tensor)

    # Stack the output tensors along the batch dimension
    output_tensor_batch = torch.cat(output_batch, dim=0)

    #Adding biases
    if filter_biases_bool:
        bias_expanded_matrix = bias_expanded_matrix.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
        output_tensor_batch = output_tensor_batch + bias_expanded_matrix

    return output_tensor_batch


def propagate_conv_bp(filter_weights, kernel_size, padding, stride, inputs, lb, ub, device, filter_biases=None):

    # Ensure the dimensions of lower bounds (lb) and upper bounds (ub) match
    assert lb.shape == ub.shape, "The dimensions of 'lb' and 'ub' must match."
    batch_size = lb.shape[0]

    # Ensure the inputs have 4 dimensions: batch_size, channels, height, width
    assert lb.dim() == 4, "'lb' must have shape (batch_size, channels, height, width)."
    assert ub.dim() == 4, "'ub' must have shape (batch_size, channels, height, width)."

    # Extract kernel dimensions
    if isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        kernel_height, kernel_width = kernel_size
    elif isinstance(kernel_size, int):
        kernel_height = kernel_width = kernel_size
        kernel_size = (kernel_size, kernel_size)
    else:
        raise ValueError("Kernel size must be an int or a tuple of two integers.")

    # Determine padding values
    if isinstance(padding, int):
        pad_tuple = (padding, padding, padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_tuple = (padding[1], padding[1], padding[0], padding[0])
    elif isinstance(padding, tuple) and len(padding) == 4:
        if padding[0] != padding[1] or padding[2] != padding[3]:
            raise ValueError("Only symmetrical padding is supported. Top must equal bottom and left must equal right.")
        pad_tuple = padding
    elif padding == 0 or padding is None:
        pad_tuple = (0, 0, 0, 0)
    else:
        raise ValueError("Padding must be an int or a tuple of appropriate dimensions.")

    # Extract input shape information: channels, height, width
    input_channels = lb.shape[1]

    # Move tensors to the specified device and data type
    lb = lb.to(DATA_TYPE).to(device)
    ub = ub.to(DATA_TYPE).to(device)

    # Flatten filter weights for sparse matrix operations
    num_filters = filter_weights.shape[0]
    filter_weights = filter_weights.reshape(num_filters, -1).to(DATA_TYPE).to(device)

    if filter_biases is not None:
        filter_biases = filter_biases.to(DATA_TYPE).to(device)

    input_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
    input_flattened_size = lb.shape[2] * lb.shape[3]

    # Calculate output dimensions of the convolution
    pad_top, pad_bottom, pad_left, pad_right = pad_tuple
    output_height = int(((input_shape[1] - kernel_height + pad_top + pad_bottom) / stride) + 1)
    output_width = int(((input_shape[2] - kernel_width + pad_left + pad_right) / stride) + 1)
    output_shape = (output_height, output_width)
    output_flattened_size = output_height * output_width

    # Apply padding to input tensors
    if padding is not None:
        lb = torch.nn.functional.pad(lb, pad=pad_tuple, mode='constant', value=0)
        ub = torch.nn.functional.pad(ub, pad=pad_tuple, mode='constant', value=0)
        inputs = torch.nn.functional.pad(inputs, pad=pad_tuple, mode='constant', value=0)
        input_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
        input_flattened_size = lb.shape[2] * lb.shape[3]

    # Flatten the tensors for sparse operations
    lb_flattened = lb.reshape(batch_size, -1).to(DATA_TYPE).to(device)
    ub_flattened = ub.reshape(batch_size, -1).to(DATA_TYPE).to(device)
    inputs_flattened = inputs.reshape(batch_size, -1).to(DATA_TYPE).to(device)

    # Create an index matrix for image patches
    index_matrix = torch.arange(0, input_flattened_size, dtype=DATA_TYPE, device=device).reshape(1, input_shape[1], input_shape[2])

    # Unfold the input indices to get patch indices
    patch_indices = torch.nn.functional.unfold(index_matrix, kernel_size=kernel_size, stride=stride).transpose(0, 1).to(torch.int32)
    num_patches = patch_indices.shape[0]

    # Ensure the number of patches matches the expected output size
    assert num_patches == output_flattened_size, f"Mismatch in patch count: {num_patches} != {output_flattened_size}."

    # Initialize bias matrix for all filters
    bias_matrix = torch.zeros(num_filters, num_patches, dtype=DATA_TYPE, device=device)


    indices_list = []  # To store sparse tensor indices

    # Loop over filters to create sparse matrix components
    for filter_idx in range(num_filters):
        filter_weights_current = filter_weights[filter_idx, :]

        temp_indices = []
        temp_values = []

        for patch_idx in range(num_patches):
            # Generate indices for the current patch
            indices = generate_array_int32(patch_indices[patch_idx, :], input_flattened_size, input_channels - 1)

            temp_indices.append(
                torch.stack((
                    torch.full(indices.shape, filter_idx, dtype=torch.long),
                    torch.full(indices.shape, patch_idx, dtype=torch.long),
                    indices
                ), dim=1)
            )
            temp_values.append(filter_weights_current)

        indices_list.append(torch.cat(temp_indices, dim=0))

        if filter_biases is not None:
            bias_matrix[filter_idx, :] = filter_biases[filter_idx]


    #Generating the values for the sparse matrix
    values_list = []
    for filter_idx in range(num_filters):
        for patch_idx in range(num_patches):
            values_list.append(filter_weights[filter_idx])

    # Concatenate all indices and values for the sparse matrix
    sparse_indices = torch.cat(indices_list, dim=0)
    sparse_values = torch.cat(values_list, dim=0)


    # Create sparse tensors for the filters
    filter_tensor = torch.sparse_coo_tensor(sparse_indices.T, sparse_values,
                                            size=(num_filters, num_patches, input_channels * input_flattened_size),
                                            device=device)

    # Initialize outputs
    outputs_lb, outputs_ub, outputs = [], [], []

    # Compute outputs for each batch
    for batch_idx in range(batch_size):
        lb_single = lb_flattened[batch_idx, :]
        ub_single = ub_flattened[batch_idx, :]
        input_single = inputs_flattened[batch_idx, :]

        # Calculate bounds and output
        output_lb = torch.sparse.sum(filter_tensor * lb_single.T, dim=-1)
        output_ub = torch.sparse.sum(filter_tensor * ub_single.T, dim=-1)
        output = torch.sparse.sum(filter_tensor * input_single.T, dim=-1)

        outputs_lb.append(output_lb)
        outputs_ub.append(output_ub)
        outputs.append(output)

    # Reshape outputs to match expected dimensions
    output_lb_tensor = torch.cat(outputs_lb, dim=0).to_dense().view(batch_size, num_filters, output_height, output_width)
    output_ub_tensor = torch.cat(outputs_ub, dim=0).to_dense().view(batch_size, num_filters, output_height, output_width)
    output_tensor = torch.cat(outputs, dim=0).to_dense().view(batch_size, num_filters, output_height, output_width)

    # Add biases if applicable
    if filter_biases is not None:
        bias_matrix = bias_matrix.view(1, num_filters, output_height, output_width)
        output_lb_tensor += bias_matrix
        output_ub_tensor += bias_matrix
        output_tensor += bias_matrix

    return output_lb_tensor, output_ub_tensor, output_tensor



def main():
    # Definizione di un modello semplice per test
    class SimpleConvNet(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super(SimpleConvNet, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)

            # Imposta manualmente i pesi dei filtri
            #filters_weights = torch.arange(0, kernel_size*kernel_size*out_channels*in_channels, dtype=DATA_TYPE, device=device)*10
            filters_weights = torch.randn(kernel_size*kernel_size*out_channels*in_channels, dtype=DATA_TYPE, device=device)
            #filters_weights = torch.tensor([[[[1, 1], [1, 1]]],  [[[2, 2], [2, 2]]]], dtype=DATA_TYPE)

            self.conv.weight.data = filters_weights.reshape(out_channels, in_channels, kernel_size, kernel_size)

            self.conv.bias.data = torch.zeros(out_channels, dtype=DATA_TYPE)

        def forward(self, x):
            return self.conv(x)

    # Parametri di esempio
    kernel_size = 2
    padding = (0,0,0,0)
    stride = 1

    #  il problema é qui nel filtro
    filters_number = 32
    batch_size = 128
    in_channels = 3
    img_size_w =28
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
    #inputs = torch.arange(img_size_w * img_size_h * in_channels * batch_size, dtype=DATA_TYPE, device=device)
    #inputs = inputs.reshape(batch_size, in_channels, img_size_w, img_size_h)

    inputs =  inputs.reshape(batch_size, in_channels, img_size_w, img_size_h)
    lb = inputs - noise
    ub = inputs + noise

    # Misura il tempo di esecuzione per `propagate_conv`
    start_time = time.time()

    profiler = cProfile.Profile()
    profiler.enable()
    with torch.no_grad():
        results_bp = propagate_conv(model, kernel_size, padding, stride, inputs, device=device)
    propagate_conv_time = time.time() - start_time

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.print_stats()

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



    profiler = cProfile.Profile()
    profiler.enable()

    with torch.no_grad():
        lb, ub, results_sparse = propagate_conv_bp(parameters[0], kernel_size, padding, stride, inputs,  lb, ub, device=device,
                                   filter_biases=parameters[1])
    propagate_conv_bp_time = time.time() - start_time
    print(f"{propagate_conv_bp_time=}")

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.print_stats()



    # Comparazione dei risultati
    comparation_dict, diff_tensor = compare_tensors(results_conv, results_bp)
#    generate_heatmaps_and_save(diff_tensor.squeeze(0).cpu())

    comparation_dict_sparse, diff_tensor_sparse = compare_tensors(results_conv, results_sparse)
#    generate_heatmaps_and_save(diff_tensor_sparse.squeeze(0).cpu())


    print(comparation_dict)
    print(comparation_dict_sparse)
    print(f"{results_bp.shape=}")
    print(f"{results_conv.shape=}")








if __name__ == "__main__":
    main()
