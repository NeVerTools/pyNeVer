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
PAPER_LIKE = False

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

    def calculate_rs_loss_regularizer(model, kernel, padding, stride, filters_number, lbs, ubs, device, sparce_indexes=None, normalized=True, paper_like = True):
        from examples.pruning_experiments.utils.convolution_bp import propagate_conv_bp_sparse, partial_conv_sparse
        batch_dim = lbs.shape[0]

        parameters = list(model.parameters())

        filter_weights = parameters[0]
        filter_biases = parameters[1]


        if sparce_indexes is None:
            lb_1, ub_1, sparce_indexes = propagate_conv_bp_sparse(kernel, padding, stride, lbs, ubs, device, filter_weights, filter_biases, differentiable=True)
        else:
            lb_1, ub_1 = partial_conv_sparse(kernel, padding, stride, lbs, ubs, device, filter_weights, filter_biases, sparse_indices=sparce_indexes, differentiable=True)

        #lb_1 = lb_1.flatten(start_dim=1)
        #ub_1 = ub_1.flatten(start_dim=1)

        lb_1 = lb_1.view(batch_dim, filters_number, -1).unsqueeze(0).unsqueeze(2)
        ub_1 = ub_1.view(batch_dim, filters_number, -1).unsqueeze(0).unsqueeze(2)


        # BATCH NORM
        gamma_quadro = parameters[2].detach().pow(2)
        gamma = parameters[2].view(1, -1, 1)#.detach()
        beta = parameters[3].view(1, -1, 1)#.detach()
        running_mean = model.bn1.running_mean.view(1, -1, 1)
        running_var = model.bn1.running_var.view(1, -1, 1)
        eps = model.bn1.eps

        # Calcolo di std dev per il batch norm
        std_dev = torch.sqrt(running_var + eps)

        lb_1_b_normalized = gamma * (lb_1 - running_mean) / std_dev + beta
        ub_1_b_normalized = gamma * (ub_1 - running_mean) / std_dev + beta

        # Calcolo dei limiti normali
        if paper_like:
            lb_1_loss = lb_1 / gamma_quadro.unsqueeze(0).unsqueeze(2)
            ub_1_loss = ub_1 / gamma_quadro.unsqueeze(0).unsqueeze(2)
        else:
            lb_1_loss = gamma * (lb_1 - running_mean) / std_dev + beta
            ub_1_loss = gamma * (ub_1 - running_mean) / std_dev + beta


        rsloss1 = _l_relu_stable(lb_1_loss.view(batch_dim, -1), ub_1_loss.view(batch_dim, -1))

        # Calcolo bounds come nel paper

        #lb_1= lb_1 / gamma_quadro.unsqueeze(0).unsqueeze(2)
        #ub_1 = ub_1 / gamma_quadro.unsqueeze(0).unsqueeze(2)

        # # Apply batch norm TODO
        # lb_1 = ((gamma * lb_1 - running_mean)/torch.sqrt(running_var + eps)) + beta
        # ub_1 = ((gamma * ub_1 - running_mean)/torch.sqrt(running_var + eps)) + beta


        lbh_1 = torch.relu(lb_1_b_normalized)
        ubh_1 = torch.relu(ub_1_b_normalized)

        # Fc2
        W2 = parameters[4]
        b2 = parameters[5]

        lb_2, ub_2 = interval_arithmetic_fc(lbh_1.view(batch_dim, -1), ubh_1.view(batch_dim, -1), W2, b2)

        rsloss2 = _l_relu_stable(lb_2.view(batch_dim, -1), ub_2.view(batch_dim, -1))

        rs_loss = rsloss1 + rsloss2
        total_neurons_number = lb_1.view(batch_dim, -1).shape[1]  + lb_2.view(batch_dim, -1).shape[1]
        if DEBUG:
            print(f"{total_neurons_number=}")

        if normalized:
            rs_loss = rs_loss / total_neurons_number

            rs_loss = (rs_loss + 1) / 2
            assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"



        return rs_loss, sparce_indexes



    def compute_rs_loss(inputs, model, kernel, padding, stride, filters_number, noise, sparse_indexes=None, paper_like=True):
        """Helper function to compute RS loss."""
        ubs = inputs + noise
        lbs = inputs - noise
        return calculate_rs_loss_regularizer(model, kernel, padding, stride, filters_number, lbs, ubs, device, sparce_indexes=sparse_indexes , normalized=True, paper_like=paper_like)

    sparse_indexes = None

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
            if rs_loss_regularizer is not None and rs_loss_regularizer != 0:
                rs_loss, sparse_indexes = compute_rs_loss(inputs, model, kernel=kernel_size, padding = padding, stride=stride,
                                          filters_number=filters_number, sparse_indexes=sparse_indexes, noise=noise, paper_like=PAPER_LIKE)
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
                if rs_loss_regularizer is not None and rs_loss_regularizer != 0.0:
                    if epoch == num_epochs - 1:
                        rs_loss, sparse_indexes = compute_rs_loss(inputs, model, kernel=kernel_size, padding=padding,
                                                                  stride=stride,
                                                                  filters_number=filters_number,
                                                                  sparse_indexes=sparse_indexes, noise=noise, paper_like=PAPER_LIKE)
                        loss += rs_loss_regularizer * rs_loss
                        partial_loss_2 = rs_loss.item()
                    else:
                        partial_loss_2 = 0.0
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

                    if rs_loss_regularizer is not None and rs_loss_regularizer != 0.0:
                        rs_loss, sparse_indexes = compute_rs_loss(inputs, model, kernel=kernel_size, padding=padding,
                                                                  stride=stride,
                                                                  filters_number=filters_number,
                                                                  sparse_indexes=sparse_indexes, noise=noise, paper_like=PAPER_LIKE)
                        assert False, "Never be here!"
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
        #print(f" Epoch train time: {train_time}")
        #print(f" Epoch test time: {test_time}")
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
        num_epochs = 3
    else:
        num_epochs = 3

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

