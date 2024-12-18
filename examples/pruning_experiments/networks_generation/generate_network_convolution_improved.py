import copy
import csv
import os
from logging import exception

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import time
from sympy import false
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.conversion as conv
import pynever.strategies.pruning as pruning
import pynever.strategies.training as training
import pynever.utilities as util

DATA_TYPE = torch.float32
DEBUG = False
INPUT_DIM = 28
OUTPUT_DIM = 28

kernel_size = 3
stride = 1
padding = 0
noise = 0.1

def load_yaml_config(yaml_file_path):
    """
    Carica la configurazione YAML da un file specificato.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Errore: Il file {yaml_file_path} non è stato trovato.")
        return None
    except yaml.YAMLError as exc:
        print(f"Errore nella lettura del file YAML: {exc}")
        return None

#######################################################################################################################

def generate_array_int32(initial_array, k, n, device):
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
    return result.to(device)

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
    patch_matrix = torch.nn.functional.unfold(matrix_index, kernel_size=kernel_size, stride=stride)
    patch_matrix = patch_matrix.transpose(0, 1).to(torch.int32)

    # The number of patches must be equal to the output_conv_shape_flatten
    n_patches = patch_matrix.shape[0]
    assert n_patches == output_conv_shape_flatten, f"The number of patches = {n_patches} does not match with the expected output shape flattened {output_conv_shape_flatten}."

    # Instantiating the matrix that will simulate the conv behaviour through a fc operation
    convolution_expanded_matrix = torch.zeros(n_patches, filters_number, n_input_channels * image_flattened_dim,
                                              dtype=DATA_TYPE, device=device)
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
                temp = torch.zeros(n_input_channels * image_flattened_dim, dtype=DATA_TYPE, device=device)
                indices = generate_array_int32(patch_matrix[i, :], image_flattened_dim, n_input_channels - 1, device=device)
                # the n_input_channels dim must be done automatically
                temp[indices] = filter
                convolution_expanded_matrix[i, f_idx, :] = temp
                if b_idx == 0:
                    bias_expanded_matrix[f_idx, i] = filter_biases[f_idx]

        expanded_filters_matrix = convolution_expanded_matrix.permute(1, 0, 2).unsqueeze(0)

        # Calculating the "positive" and "negative" filters matrix
        F_max = torch.maximum(expanded_filters_matrix, torch.tensor(0.0))
        F_min = torch.minimum(expanded_filters_matrix, torch.tensor(0.0))

        # Perform the fully connected-like operation for each batch item separately
        output_tensor_lb = torch.matmul(F_max, lb_single.T) + torch.matmul(F_min, ub_single.T)
        output_tensor_ub = torch.matmul(F_max, ub_single.T) + torch.matmul(F_min, lb_single.T)

        output_batch_lb = output_tensor_lb.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
        output_tensor_ub = output_tensor_ub.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
        output_batch_lb.append(output_batch_lb)
        output_batch_ub.append(output_tensor_ub)

    # Stack the output tensors along the batch dimension
    output_tensor_batch_lb = torch.cat(output_batch_lb, dim=0)
    output_tensor_batch_ub = torch.cat(output_batch_ub, dim=0)
    # Adding biases
    bias_expanded_matrix = bias_expanded_matrix.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
    output_tensor_batch_lb = output_tensor_batch_lb + bias_expanded_matrix
    output_tensor_batch_ub = output_tensor_batch_ub + bias_expanded_matrix

    assert output_tensor_batch_lb <= output_tensor_batch_ub, "Lower bounds must always be lower than upper bounds."

    return output_tensor_batch_lb, output_tensor_batch_ub

def propagate_conv_bpV2(filter_weights, kernel_size, padding, stride, lb, ub, device, filter_biases=None):

    _time = time.time()
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
        input_shape = (lb.shape[1], lb.shape[2], lb.shape[3])
        input_flattened_size = lb.shape[2] * lb.shape[3]

    # Flatten the tensors for sparse operations
    lb_flattened = lb.reshape(batch_size, -1).to(DATA_TYPE).to(device)
    ub_flattened = ub.reshape(batch_size, -1).to(DATA_TYPE).to(device)

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
    temp_indices = []

    # Loop over filters to create sparse matrix components
    for filter_idx in range(num_filters):
        filter_weights_current = filter_weights[filter_idx, :]

        for patch_idx in range(num_patches):
            # Generate indices for the current patch
            indices = generate_array_int32(patch_indices[patch_idx, :], input_flattened_size, input_channels - 1, device=device)

            temp_indices.append(
                torch.stack((
                    torch.full(indices.shape,filter_idx * num_patches + patch_idx, dtype=torch.long, device=device),
                    indices
                ), dim=1)
            )

        if filter_biases is not None:
            bias_matrix[filter_idx, :] = filter_biases[filter_idx]

    indices_list.append(torch.cat(temp_indices, dim=0))

    #Generating the values for the sparse matrix
    pos_values_list = []
    neg_values_list = []
    for filter_idx in range(num_filters):
        for patch_idx in range(num_patches):
            pos_values_list.append(torch.maximum(filter_weights[filter_idx], torch.tensor(0.0)))
            neg_values_list.append(torch.minimum(filter_weights[filter_idx], torch.tensor(0.0)))

    # Concatenate all indices and values for the sparse matrix
    sparse_indices = torch.cat(indices_list, dim=0)
    pos_sparse_values = torch.cat(pos_values_list, dim=0)
    neg_sparse_values = torch.cat(neg_values_list, dim=0)



    # Create sparse tensors for the filters
    pos_filter_tensor = torch.sparse_coo_tensor(sparse_indices.T, pos_sparse_values,
                                            size=(num_filters * num_patches, input_channels * input_flattened_size),
                                            device=device)
    neg_filter_tensor = torch.sparse_coo_tensor(sparse_indices.T, neg_sparse_values,
                                            size=(num_filters * num_patches, input_channels * input_flattened_size),
                                            device=device)

    # Initialize outputs
    outputs_lb, outputs_ub = [], []

    # Compute outputs for each batch
    for batch_idx in range(batch_size):
        lb_single = lb_flattened[batch_idx, :].unsqueeze(dim=0)
        ub_single = ub_flattened[batch_idx, :].unsqueeze(dim=0)

        # Perform the fully connected-like operation for each batch item separately
        output_lb = torch.sparse.mm(pos_filter_tensor, lb_single.T) + torch.sparse.mm(neg_filter_tensor, ub_single.T)
        output_ub = torch.sparse.mm(pos_filter_tensor, ub_single.T) + torch.sparse.mm(neg_filter_tensor, lb_single.T)

        outputs_lb.append(output_lb)
        outputs_ub.append(output_ub)

    # Reshape outputs to match expected dimensions
    output_lb_tensor = torch.cat(outputs_lb, dim=0).to_dense().view(batch_size, num_filters, output_height, output_width)
    output_ub_tensor = torch.cat(outputs_ub, dim=0).to_dense().view(batch_size, num_filters, output_height, output_width)

    # Add biases if applicable
    if filter_biases is not None:
        bias_matrix = bias_matrix.view(1, num_filters, output_height, output_width)
        output_lb_tensor += bias_matrix
        output_ub_tensor += bias_matrix

    assert torch.all(output_lb_tensor <= output_ub_tensor), "Lower bounds must always be lower than upper bounds."
    print(time.time() - _time)
    return output_lb_tensor, output_ub_tensor

"""RS Loss"""
def _l_relu_stable(lb, ub, norm_constant=1.0):
    loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))

    if loss < lb.shape[1] * -1 or loss > lb.shape[1]:
        raise Exception("Error in RS Loss, value exceeding the maximum")

    return loss



# Assume le forme Bxm, Bxm, mxn, n
def interval_arithmetic_fc(lb, ub, W, b):
    W = W.T
    W_max = torch.maximum(W, torch.tensor(0.0))
    W_min = torch.minimum(W, torch.tensor(0.0))

    new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
    new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b

    return new_lb, new_ub


def _compute_bounds_n_layers(lb, ub, W, b):
    # Base case
    if len(W.shape) == 2:
        naive_ia_bounds = interval_arithmetic_fc(lb, ub, W, b)
    else:
        raise NotImplementedError
    return naive_ia_bounds


########################################################################################################################
def train(model, device, train_loader, test_loader, optimizer_cls, optimizer_params, criterion_cls, num_epochs,
          filters_number, num_classes, rs_loss_regularizer=None, noise=0.1,
          scheduler_lr_cls=None, scheduler_lr_params=None, val_loader=None):
    # Initialize the optimizer
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    # Initialize the criterion
    criterion = criterion_cls()

    # Initialize the learning rate scheduler if provided
    scheduler = None
    if scheduler_lr_cls is not None:
        scheduler = scheduler_lr_cls(optimizer, **scheduler_lr_params)

    # Initialize metrics dictionary
    metrics = {
        'train_loss': None,
        'train_accuracy': None,
        'test_loss': None,
        'test_accuracy': None,
        'val_loss': None,
        'val_accuracy': None,
        'loss_1_train': None,  # Primary loss (train)
        'loss_2_train': None,  # RS loss or other additional regularization (train)
        'loss_1_test': None,   # Primary loss (test)
        'loss_2_test': None,   # RS loss or other additional regularization (test)
        'loss_1_val': None,    # Primary loss (validation)
        'loss_2_val': None,    # RS loss or other additional regularization (validation)
        'lambda': None         # Value of the RS regularizer, if applicable
    }

    def calculate_rs_loss_regularizer(model, kernel, padding, stride, filters_number, inputs, lbs, ubs, device, normalized=True):

        parameters = list(model.parameters())

        filter_weights = parameters[0]
        filter_biases = parameters[1]

        lb_1, ub_1 = propagate_conv_bpV2(filter_weights, kernel, padding, stride, lbs, ubs, device, filter_biases)

        lb_1 = lb_1.flatten(start_dim=1)
        ub_1 = ub_1.flatten(start_dim=1)

        #lb_1 = lb_1.mean(dim=0)
        #ub_1 = ub_1.mean(dim=0)

        lbh_1 = torch.relu(lb_1)
        ubh_1 = torch.relu(ub_1)

        # BATCH NORM
        b_weights = parameters[2]

        # Fc2
        W2 = parameters[3]
        b2 = parameters[4]

        lb_2, ub_2 = interval_arithmetic_fc(lbh_1, ubh_1, W2, b2)

        rsloss1 = _l_relu_stable(lb_1, ub_1)
        rsloss2 = _l_relu_stable(lb_2, ub_2)

        rs_loss = rsloss1 + rsloss2
        total_neurons_number = lb_1.shape[1] + lb_2.shape[1]
        if DEBUG:
            print(f"{total_neurons_number=}")

        if normalized:
            rs_loss = rs_loss / total_neurons_number

            rs_loss = (rs_loss + 1) / 2
            assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

        return rs_loss

    def propagate_conv_bp(filer_weights, kernel_size, padding, stride, lb, ub, device, filter_biases=None):
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
        bias_expanded_sparse = bias_expanded_matrix = torch.sparse_coo_tensor(indices=torch.empty((2, 0), dtype=torch.long), values=torch.empty(0, dtype=DATA_TYPE), size=(filters_number, n_patches), dtype=DATA_TYPE, device=device)

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
                    indices_2d = torch.stack((torch.full(indices.shape, i, dtype=torch.long),
                                              torch.full(indices.shape, f_idx, dtype=torch.long),
                                              indices), dim=1)

                    updating_matrix = torch.sparse_coo_tensor(
                        indices=indices_2d,
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
            F_max = torch.maximum(expanded_filters_matrix, torch.tensor(0.0))
            F_min = torch.minimum(expanded_filters_matrix, torch.tensor(0.0))

            # Perform the fully connected-like operation for each batch item separately
            output_tensor_lb = torch.matmul(F_max, lb_single.T) + torch.matmul(F_min, ub_single.T)
            output_tensor_ub = torch.matmul(F_max, ub_single.T) + torch.matmul(F_min, lb_single.T)

            output_tensor_lb = output_tensor_lb.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
            output_tensor_ub = output_tensor_ub.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
            output_batch_lb.append(output_tensor_lb)
            output_batch_ub.append(output_tensor_ub)

        # Stack the output tensors along the batch dimension
        output_tensor_batch_lb = torch.cat(output_batch_lb, dim=0)
        output_tensor_batch_ub = torch.cat(output_batch_ub, dim=0)
        # Adding biases
        if filter_biases is not None:
            bias_expanded_matrix = bias_expanded_matrix.view(1, filters_number, output_conv_dim_h, output_conv_dim_w)
            output_tensor_batch_lb = output_tensor_batch_lb + bias_expanded_matrix
            output_tensor_batch_ub = output_tensor_batch_ub + bias_expanded_matrix

        assert torch.all(
            output_tensor_batch_lb <= output_tensor_batch_ub), "Lower bounds must always be lower than upper bounds."
        return output_tensor_batch_lb, output_tensor_batch_ub

    def compute_rs_loss(inputs, model, kernel, padding, stride, filters_number, noise):
        """Helper function to compute RS loss."""
        ubs = inputs + noise
        lbs = inputs - noise
        return calculate_rs_loss_regularizer(model, kernel, padding, stride, filters_number, inputs, lbs, ubs, device, normalized=True)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_loss_1 = 0.0
        running_train_loss_2 = 0.0
        correct_train = 0
        total_train = 0

        # Training phase
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Compute primary loss
            if isinstance(criterion, nn.MSELoss):
                targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                loss = criterion(outputs, targets_hot_encoded)
            else:
                loss = criterion(outputs, targets)

            partial_loss_1 = loss.item()

            # Compute RS loss if applicable
            if rs_loss_regularizer is not None:
                rs_loss = compute_rs_loss(inputs, model, kernel=kernel_size, padding = padding, stride=stride,
                                          filters_number=filters_number, noise=noise)
                loss += rs_loss_regularizer * rs_loss
                partial_loss_2 = rs_loss.item()
            else:
                partial_loss_2 = 0.0

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            running_train_loss += loss.item()
            running_train_loss_1 += partial_loss_1
            running_train_loss_2 += partial_loss_2

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        # Compute average metrics for training
        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        loss_1_train = running_train_loss_1 / len(train_loader)
        loss_2_train = running_train_loss_2 / len(train_loader)

        # Update training metrics
        metrics.update({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'loss_1_train': loss_1_train,
            'loss_2_train': loss_2_train
        })

        # Testing phase
        model.eval()
        running_test_loss = 0.0
        running_test_loss_1 = 0.0
        running_test_loss_2 = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Compute primary loss
                if isinstance(criterion, nn.MSELoss):
                    targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                    loss = criterion(outputs, targets_hot_encoded)
                else:
                    loss = criterion(outputs, targets)

                partial_loss_1 = loss.item()

                # Compute RS loss if applicable
                if rs_loss_regularizer is not None:
                    rs_loss = compute_rs_loss(inputs, model, kernel=kernel_size, padding = padding, stride=stride,
                                          filters_number=filters_number, noise=noise)
                    loss += rs_loss_regularizer * rs_loss
                    partial_loss_2 = rs_loss.item()
                else:
                    partial_loss_2 = 0.0

                # Accumulate losses
                running_test_loss += loss.item()
                running_test_loss_1 += partial_loss_1
                running_test_loss_2 += partial_loss_2

                # Calculate testing accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        # Compute average metrics for testing
        test_loss = running_test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        loss_1_test = running_test_loss_1 / len(test_loader)
        loss_2_test = running_test_loss_2 / len(test_loader)

        # Update testing metrics
        metrics.update({
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'loss_1_test': loss_1_test,
            'loss_2_test': loss_2_test
        })

        # Validation phase (if applicable)
        if val_loader is not None:
            val_loss = 0.0
            running_val_loss_1 = 0.0
            running_val_loss_2 = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                    if isinstance(criterion, nn.MSELoss):
                        targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                        loss = criterion(outputs, targets_hot_encoded)
                    else:
                        loss = criterion(outputs, targets)

                    partial_loss_1 = loss.item()

                    if rs_loss_regularizer is not None:
                        rs_loss = compute_rs_loss(inputs, model, kernel=kernel_size, padding = padding, stride=stride,
                                          filters_number=filters_number, noise=noise)
                        loss += rs_loss_regularizer * rs_loss
                        partial_loss_2 = rs_loss.item()
                    else:
                        partial_loss_2 = 0.0

                    # Accumulate validation losses
                    val_loss += loss.item()
                    running_val_loss_1 += partial_loss_1
                    running_val_loss_2 += partial_loss_2

                    # Calculate validation accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()

            # Compute average metrics for validation
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            loss_1_val = running_val_loss_1 / len(val_loader)
            loss_2_val = running_val_loss_2 / len(val_loader)

            # Update validation metrics
            metrics.update({
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'loss_1_val': loss_1_val,
                'loss_2_val': loss_2_val
            })

        # Update RS regularizer value
        metrics['lambda'] = rs_loss_regularizer

        # Print metrics for the current epoch
        print(f"Epoch {epoch + 1}:")
        print(f"  Train -> Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Test  -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

        if val_loader is not None:
            print(f"  Validation -> Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    return metrics


class CustomNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, filters_number: int, kernel_size: int, stride: int,
                 padding: int, hidden_layer_dim: int):
        super(CustomNN, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(1, filters_number, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(filters_number)  # Batch normalization for conv layer
        self.flatten = nn.Flatten()

        # Calculate input size for fc1 dynamically
        dummy_input = torch.zeros(1, 1, input_dim, input_dim)
        conv_output = self.conv(dummy_input)
        conv_output = self.bn1(conv_output)  # Ensure BatchNorm compatibility
        conv_output_flatten = self.flatten(conv_output)
        fc1_in_features = conv_output_flatten.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(fc1_in_features, hidden_layer_dim)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        # Apply convolution, batch norm, and ReLU
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1_dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x



def save_metrics_to_csv(metrics, csv_file):
    headers = list(metrics.keys())
    data = list(metrics.values())

    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)  # Write headers if file is new
        writer.writerow(data)  # Write metrics data


def generate_no_batch_networks(data_dict, filters_number, rs_factor, hidden_layer_dim, save_bool=False):

    print(f"{filters_number=}")

    # Unpack data_dict
    optimizer_dict = data_dict['optimizer']
    scheduler_lr_dict = data_dict['scheduler_lr']

    # Create optimizer params dict
    opt_params_with_weight_decay = optimizer_dict.copy()
    optimizer_name = opt_params_with_weight_decay['type']
    del opt_params_with_weight_decay['type']

    opt_params = opt_params_with_weight_decay.copy()
    del opt_params['weight_decay']

    # Create scheduler_lr_dict params dict
    scheduler_lr_params = scheduler_lr_dict.copy()
    scheduler_lr_name = scheduler_lr_params['type']
    del scheduler_lr_params['type']

    # Dataset name
    dataset_name = data_dict['data']['dataset']

    # Dataset parameters
    train_dim = int(data_dict['data']['train_dim'])
    test_dim = int(data_dict['data']['test_dim'])

    # NN architectures
    input_dim = int(data_dict['data']['input_dim'])
    output_dim = int(data_dict['data']['output_dim'])

    # Training parameters
    #num_epochs = int(data_dict['training']['epochs'])

    if filters_number <= 90:
        num_epochs = 600
    else:
        num_epochs = 1200

    train_batch_size = int(data_dict['training']['train_batch_size'])
    test_batch_size = int(data_dict['training']['test_batch_size'])
    validation_batch_size = int(data_dict['training']['validation_batch_size'])
    validation_percentage = float(data_dict['training']['validation_percentage'])
    loss_name = data_dict['training']['loss_name']

    # Set the device (use GPU if available, otherwise fallback to CPU)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Dataset loading and transformation
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_shape = (28, 28)
    elif dataset_name == 'FMNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        input_shape = (28, 28)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_shape = (3, 32, 32)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create a subset
    train_subset = Subset(train_set, range(train_dim))  # Take the first train_dim elements from the training set
    test_subset = Subset(test_set, range(test_dim))  # Take the first test_dim elements from the test set

    if scheduler_lr_name == "ReduceLROnPlateau" and validation_percentage > 0.0:
        # Calculate the size of the validation set
        val_size = int(validation_percentage * len(train_subset))
        train_size = len(train_subset) - val_size

        # Split the training subset into training and validation sets
        train_dataset, val_dataset = random_split(train_subset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=True)
    else:
        val_loader = None
        train_dataset = train_subset

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

    # Initialize the model and move to the correct device

    model = CustomNN(input_shape[0], output_dim, filters_number=filters_number, kernel_size=3, stride=stride, padding=padding, hidden_layer_dim=hidden_layer_dim).to(device)

    # Define the optimizer
    if optimizer_name == 'Adam':
        optimizer_cls = optim.Adam
    elif optimizer_name == 'SGD':
        optimizer_cls = optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Define the loss function
    if loss_name == 'CrossEntropyLoss':
        criterion_cls = nn.CrossEntropyLoss
    elif loss_name == 'MSE':
        criterion_cls = nn.MSELoss
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    criterion_callable = criterion_cls()

    # Define the scheduler
    scheduler_lr_cls = None
    if scheduler_lr_name == "ReduceLROnPlateau":
        scheduler_lr_cls = optim.lr_scheduler.ReduceLROnPlateau
    else:
        scheduler_lr_params = None

    # Train the model without batch or L1 regularization
    print("Model Simple")
    model1 = copy.deepcopy(model).to(device)
    metrics1 = train(model1, device, train_loader, test_loader, optimizer_cls, opt_params, criterion_cls, num_epochs,
                     filters_number=filters_number, num_classes=output_dim, rs_loss_regularizer=rs_factor,
                     scheduler_lr_cls=scheduler_lr_cls, scheduler_lr_params=scheduler_lr_params, val_loader=val_loader)
    metrics1['filters_number'] = filters_number

    if save_bool:
        # Write results and save networks on file
        save_metrics_to_csv(metrics1, 'results\\accuracies_no_batch.csv')

        # Export the models to ONNX format
        # A dummy input (ensure it is on the same device as the model)
        dummy_input =  torch.rand(1, 1, 28, 28).to(device)

        torch.onnx.export(
            model1,
            dummy_input,
            f"results/no_batch/{filters_number}.onnx",
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
        )

        print(f"metrics1: {metrics1}")
        torch.save(model, f"results/no_batch/{filters_number}.pth")
        print(f"Model saved as results/no_batch/{filters_number}.pth")


    return metrics1, model1

