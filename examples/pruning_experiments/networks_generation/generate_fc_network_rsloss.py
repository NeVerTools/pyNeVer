import copy
import csv

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sympy import false
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.conversion as conv
import pynever.strategies.pruning as pruning
import pynever.strategies.training as training
import pynever.utilities as util

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
# Definiamo il custom regularizer
def calculate_rs_loss_regularizer(model, hidden_layer_dim, lb, ub, normalized):
    params = list(model.parameters())

    # Conv like a fully connected layer Fc1
    W1 = params[0]
    b1 = params[1]
    #print(f"lb0: {lb.shape}, ub0: {ub.shape}, W1: {W1.shape}, b1: {b1.shape}")
    lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)

    rs_loss = _l_relu_stable(lb_1, ub_1)


    if DEBUG:
        print(f"{hidden_layer_dim=}")

    if normalized:
        rs_loss = rs_loss / hidden_layer_dim

        rs_loss = (rs_loss + 1)/2
        assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"


    return rs_loss


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
def train(model, device, train_loader, test_loader, optimizer_cls, optimizer_params, criterion_cls, num_epochs, hidden_layer_dim,
          num_classes, rs_loss_regularizer=None, noise=0.1,
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

    def compute_rs_loss(inputs, model, hidden_layer_dim, noise):
        """Helper function to compute RS loss."""
        ubs = inputs + noise
        lbs = inputs - noise
        return calculate_rs_loss_regularizer(model, hidden_layer_dim, lbs, ubs, normalized=True)

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
                rs_loss = compute_rs_loss(inputs, model, hidden_layer_dim, noise)
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
                    rs_loss = compute_rs_loss(inputs, model, hidden_layer_dim, noise)
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
                        rs_loss = compute_rs_loss(inputs, model, hidden_layer_dim, noise)
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


import torch
import torch.nn as nn


class CustomNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_dim: int, output_dim: int):
        super(CustomNN, self).__init__()

        # Definizione del primo livello fully connected
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.relu = nn.ReLU()

        # Definizione del secondo livello fully connected (output layer)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        # Flatten del tensore di input se non è già piatto
        x = self.fc1(x)  # Primo livello fc
        x = self.relu(x)  # Attivazione ReLU
        x = self.fc2(x)  # Secondo livello fc (output)
        return x


def write_results_on_csv(file_path, dict_to_write):
    last_metrics = {
        'filters_number': dict_to_write['filters_number'],
        'train_loss': dict_to_write['train_loss'],
        'test_loss': dict_to_write['test_loss'],
        'train_accuracy': dict_to_write['train_accuracy'],
        'test_accuracy': dict_to_write['test_accuracy'],
        'loss_1': dict_to_write['loss_1'],
        'loss_2': dict_to_write['loss_2']
    }

    # Define the header
    header = ['filters_number', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'loss_1', 'loss_2']

    # Write the last metrics to the CSV file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)

        # Check if the file is empty to write header
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(last_metrics)

    print(f"Results appended to {file_path} successfully.")


def generate_no_batch_networks(data_dict, hidden_dim, old_weights, RS_FACTOR):

    print(f"{hidden_dim=}")

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
    num_epochs = int(data_dict['training']['epochs'])
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
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
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

    model = CustomNN(784, hidden_dim, 10)
    # Train the model without batch or L1 regularization
    print("Model Simple")
    model1 = copy.deepcopy(model).to(device)
    metrics1 = train(model1, device, train_loader, test_loader, optimizer_cls, opt_params, criterion_cls, num_epochs,
                     hidden_layer_dim=hidden_dim, num_classes=output_dim, rs_loss_regularizer=RS_FACTOR,
                     scheduler_lr_cls=scheduler_lr_cls, scheduler_lr_params=scheduler_lr_params, val_loader=val_loader)
    metrics1['h_dim'] = hidden_dim

    if False:
        # Write results and save networks on file
        write_results_on_csv('results\\accuracies_no_batch.csv', metrics1)

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

