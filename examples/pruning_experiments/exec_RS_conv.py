import copy
import csv
import time

import numpy as np
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

from pynever.bound_propagation_src.bounds_manager import BoundsManager
from pynever.bound_propagation_src.conversion.converters.onnx import ONNXConverter
from pynever.bound_propagation_src.conversion.representation import load_network_path, ONNXNetwork
from pynever.bound_propagation_src.neural_networks import networks, nodes
from pynever.bound_propagation_src.bounds import HyperRectangleBounds

INPUT_DIM = 28
OUTPUT_DIM = 28
RS_FACTOR = 0.001
RS_LOSS_BOOL = True
HIDDEN_LAYER_DIM = 100

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



def train(model, device, train_loader, test_loader, optimizer_cls, optimizer_params, criterion_cls, num_epochs,
          filters_number, num_classes, l1_lambda=None, rs_loss_bool=True, scheduler_lr_cls=None,
          scheduler_lr_params=None, val_loader=None):
    # Initialize the optimizer
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    # Initialize the criterion
    criterion = criterion_cls()

    # Initialize the learning rate scheduler if provided
    scheduler = None
    if scheduler_lr_cls is not None:
        scheduler = scheduler_lr_cls(optimizer, **scheduler_lr_params)

    # Placeholder for the metrics of the last epoch
    metrics = {
        'train_loss': None,
        'test_loss': None,
        'train_accuracy': None,
        'test_accuracy': None,
        'val_loss': None,
        'val_accuracy': None
    }

    for epoch in range(num_epochs):
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if isinstance(criterion, nn.MSELoss):
                targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                loss = criterion(outputs, targets_hot_encoded)
            else:
                loss = criterion(outputs, targets)

            # Add L1 regularization if l1_lambda is provided
            if l1_lambda is not None:
                l1_reg = 0
                for p in model.parameters():
                    l1_reg += p.abs().sum()

                loss += l1_lambda * l1_reg

            if rs_loss_bool:
                #chiamata a funzione di naive propagation
                ubs = inputs + noise
                lbs = inputs - noise

                rs_loss = rs_loss_regularizer(model, filters_number, inputs, lbs, ubs)
                loss += RS_FACTOR * rs_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)

                # Compute loss
                if isinstance(criterion, nn.MSELoss):
                    targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                    loss = criterion(outputs, targets_hot_encoded)
                else:
                    loss = criterion(outputs, targets)
                    running_test_loss += RS_FACTOR * loss.item()

                # Calculate testing accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        # Calculate average test loss and accuracy for the current epoch
        test_loss = running_test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test

        # Validation phase (if validation dataset is provided)
        if val_loader is not None:
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)

                    val_outputs = model(val_inputs)

                    if isinstance(criterion, nn.MSELoss):
                        val_targets_hot_encoded = F.one_hot(val_targets, num_classes=num_classes).float()
                        val_batch_loss = criterion(val_outputs, val_targets_hot_encoded)
                    else:
                        val_batch_loss = criterion(val_outputs, val_targets)
                    val_loss += val_batch_loss.item()

                    _, predicted_val = torch.max(val_outputs.data, 1)
                    total_val += val_targets.size(0)
                    correct_val += (predicted_val == val_targets).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100 * correct_val / total_val

            # Update metrics for validation
            metrics['val_loss'] = val_loss
            metrics['val_accuracy'] = val_accuracy

            # Step the scheduler if necessary
            if scheduler is not None:
                scheduler.step(val_loss)

        # Step the scheduler if it's not ReduceLROnPlateau
        elif scheduler is not None and scheduler_lr_cls != torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step()

        # Store metrics for the last epoch
        metrics['train_loss'] = train_loss
        metrics['train_accuracy'] = train_accuracy
        metrics['test_loss'] = test_loss
        metrics['test_accuracy'] = test_accuracy

        # Print metrics for the current epoch
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        if val_loader is not None:
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    return metrics


class FcLikeConvLayer(nn.Linear):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Numero di parametri in un singolo kernel
        self.kernel_param_size = in_channels * kernel_size * kernel_size

        super(FcLikeConvLayer, self).__init__(self.kernel_param_size, self.out_channels, bias=True)

    def forward(self, x):
        # Applica padding all'input
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        #print(x.shape)
        # Usa Unfold per estrarre le patch, dimensione di uscita: (batch, kernel_param_size, num_patches)
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)

        # Trasponi patches per avere ogni patch come input separato (batch_size * num_patches, kernel_param_size)
        patches = patches.transpose(1, 2).reshape(-1, self.kernel_param_size)

        # Applica il livello fully connected a ciascuna patch
        #out = self.fc(patches)
        out = super(FcLikeConvLayer, self).forward(patches)

        # Reshape per ottenere l'output con la struttura di una convoluzione
        batch_size, _, height, width = x.size(0), x.size(1), x.size(2), x.size(3)
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        out = out.view(batch_size, out_height, out_width, self.out_channels).permute(0, 3, 1, 2)

        # It has an output like a conv2d network
        return out

class CustomNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, filters_number, kernel_size, stride, padding):
        super(CustomNN, self).__init__()
        self.fcLikeConv = FcLikeConvLayer(1, filters_number, kernel_size=kernel_size, stride=stride, padding=padding)
        self.flatten = nn.Flatten()

        # Calculate input size for fc1 dynamically
        dummy_input = torch.zeros(1, 1, input_dim, input_dim)
        conv_output = self.fcLikeConv(dummy_input)
        conv_output_flatten = self.flatten(conv_output)
        fc1_in_features = conv_output_flatten.numel()

        # Define fully connected layers
        self.fc1 = nn.Linear(fc1_in_features, HIDDEN_LAYER_DIM)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(HIDDEN_LAYER_DIM, output_dim)

    def forward(self, x):
        # Apply FcLikeConv and ReLU
        x = self.fcLikeConv(x)
        x = F.relu(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1_dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def write_results_on_csv(file_path, dict_to_write):
    last_metrics = {
        'filters_number': dict_to_write['filters_number'],
        'train_loss': dict_to_write['train_loss'],
        'test_loss': dict_to_write['test_loss'],
        'train_accuracy': dict_to_write['train_accuracy'],
        'test_accuracy': dict_to_write['test_accuracy'],
    }

    # Define the header
    header = ['filters_number', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']

    # Write the last metrics to the CSV file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)

        # Check if the file is empty to write header
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(last_metrics)

    #print(f"Results appended to {file_path} successfully.")



#######################################################################################################################
# Definiamo il custom regularizer
def rs_loss_regularizer(model, filters_number, inputs, lb, ub, normalized = True):
    params = list(model.parameters())
    kernel_param_size = kernel_size*kernel_size*1

    patches_lb = F.unfold(lb, kernel_size=3, stride=1)
    lb = patches_lb.transpose(1, 2).reshape(-1, kernel_param_size)

    # Transform the inputs to make them like a conv
    patches_ub = F.unfold(ub, kernel_size=3, stride=1)
    ub = patches_ub.transpose(1, 2).reshape(-1, kernel_param_size)

    # Conv like a fully connected layer Fc1
    W1 = params[0]
    b1 = params[1]
    #print(f"lb0: {lb.shape}, ub0: {ub.shape}, W1: {W1.shape}, b1: {b1.shape}")
    lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)

    def reshape(bounds):
        # Reshape per ottenere l'output con la struttura di una convoluzione
        batch_size, _, height, width =inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1
        bounds = bounds.view(batch_size, out_height, out_width, filters_number).permute(0, 3, 1, 2)
        return bounds

    lb_1 = reshape(lb_1).flatten(start_dim=1)
    ub_1 = reshape(ub_1).flatten(start_dim=1)

    lbh_1 = torch.relu(lb_1)
    ubh_1 = torch.relu(ub_1)

    # Fc2
    W2 = params[2]
    b2 = params[3]
    lb_2, ub_2 = interval_arithmetic_fc(lbh_1, ubh_1, W2, b2)
    #print(f"lb1: {lb_1.shape}, ub1: {ub_1.shape}, W2: {W2.shape}, b2: {b2.shape}")


    rsloss1 = _l_relu_stable(lb_1, ub_1)
    rsloss2 = _l_relu_stable(lb_2, ub_2)



    rs_loss = rsloss1 + rsloss2
    total_neurons_number = lb_1.shape[1] + lb_2.shape[1]

    if normalized:
        rs_loss = rs_loss / total_neurons_number

        rs_loss = (rs_loss + 1)/2
        assert rs_loss >= 0 and rs_loss <= 1, "RS LOSS not in 0, 1 range"


    return rs_loss


"""RS Loss"""
def _l_relu_stable(lb, ub, normalized=False, norm_constant=1.0):
    loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))

    if loss < lb.shape[1] * -1 or loss > lb.shape[1]:
        raise Exception("Error in RS Loss, value exceeding the maximum")
    # The Rs loss should drop from 0 to 1
    if normalized:
        loss = loss / lb.shape[1]
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

def generate_no_batch_networks(data_dict, mul_factor):
    filters_number = mul_factor * 8

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
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'FMNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
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
        train_dataset=train_subset

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

    # Initialize the model and move to the correct device
    model = CustomNN(INPUT_DIM, OUTPUT_DIM, filters_number, kernel_size, stride, padding).to(device)

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
    #print("Model Simple")
    model1 = copy.deepcopy(model).to(device)
    metrics1 = train(model1, device, train_loader, test_loader, optimizer_cls, opt_params, criterion_cls, num_epochs,
                     filters_number=filters_number, num_classes=output_dim, l1_lambda=None, rs_loss_bool=RS_LOSS_BOOL,
                     scheduler_lr_cls=scheduler_lr_cls, scheduler_lr_params=scheduler_lr_params, val_loader=val_loader)
    metrics1['filters_number'] = filters_number//8

    # Write results and save networks on file
    write_results_on_csv('accuracies_no_batch.csv', metrics1)

    # Export the models to ONNX format
    # A dummy input (ensure it is on the same device as the model)
    dummy_input = torch.rand(1,1, 28, 28).to(device)

    torch.onnx.export(
        model1,
        dummy_input,
        f"{filters_number//8}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )



def generate_vectors(input_sample: np.ndarray, noise_eps) -> HyperRectangleBounds:
    num_lower_input = input_sample - noise_eps
    num_upper_input = input_sample + noise_eps
    return HyperRectangleBounds(num_lower_input, num_upper_input)

def calculate_numeric_bounds(input_sample, noise_eps, network):
    bp = BoundsManager()
    start_time = time.time()
    input_hyperect = generate_vectors(input_sample, noise_eps)
    results_dict = bp.compute_bounds(input_hyperect, network)
    num_bounds = results_dict["numeric_post"]
    end_time = time.time()
    execution_time = end_time - start_time

    return num_bounds

def generate_csv():
    # List of CSV files to update
    csv_files = [
        "accuracies_neuron_pruning.csv",
        "accuracies_no_batch.csv",
        "accuracies_no_batch_dropout.csv",
        "accuracies_no_batch_leaky.csv",
        "accuracies_no_batch_sparse.csv",
        "accuracies_no_batch_weight_decay.csv",
        "accuracies_weight_pruning.csv",
        "accuracies_with_batch.csv",
        "accuracies_with_batch_sparse.csv",
        "accuracies_with_batch_weight_decay.csv"
    ]

    # Header to write
    header = ['h_dim', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']

    # Iterate over each file and update the header
    for file_name in csv_files:
        try:
            file_path = f'results/{file_name}'
            # Open the file in write mode to overwrite existing content
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(header)
            print(f"Updated header for file: {file_name}")
        except Exception as e:
            print(f"Failed to update file {file_name}: {e}")


if __name__ == '__main__':
    yaml_file = 'configs/config_MNIST_filter_growth.yaml'
    config = load_yaml_config(yaml_file)

    mul_factors = [1,2,3,4,5,6]
    mul_factors = [1]


    old_weights = None

    for mul_factor in mul_factors:
        print(f"Number of filters {mul_factor} * 8: {mul_factor * 8}")
        #old_weights = generate_no_batch_networks(config, mul_factor, old_weights = old_weights)
        generate_no_batch_networks(config, mul_factor)

