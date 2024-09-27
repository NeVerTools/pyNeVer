import copy
import csv
import os.path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.conversion as conv
import pynever.strategies.pruning as pruning
import pynever.strategies.training as training
import pynever.utilities as util


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
          num_classes, l1_lambda=None, scheduler_lr_cls=None, scheduler_lr_params=None, val_loader=None):
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
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            outputs = model(inputs)

            if isinstance(criterion, nn.MSELoss):
                targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                loss = criterion(outputs, targets_hot_encoded)
            else:
                loss = criterion(outputs, targets)

            # Add L1 regularization if l1_lambda is provided
            if l1_lambda is not None:
                l1_reg = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_reg

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
                inputs = inputs.view(inputs.size(0), -1)

                outputs = model(inputs)

                # Compute loss
                if isinstance(criterion, nn.MSELoss):
                    targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                    loss = criterion(outputs, targets_hot_encoded)
                else:
                    loss = criterion(outputs, targets)
                    running_test_loss += loss.item()

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
                    val_inputs = val_inputs.view(val_inputs.size(0), -1).to(device)
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


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hdim, output_dim):
        super(SimpleNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, output_dim)
        )

    def forward(self, x):
        return self.sequential(x)



def write_results_on_csv(file_path, dict_to_write):
    last_metrics = {
        'h_dim': dict_to_write['h_dim'],
        'train_loss': dict_to_write['train_loss'],
        'test_loss': dict_to_write['test_loss'],
        'train_accuracy': dict_to_write['train_accuracy'],
        'test_accuracy': dict_to_write['test_accuracy'],
    }

    # Define the header
    header = ['h_dim', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']

    # Write the last metrics to the CSV file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)

        # Check if the file is empty to write header
        file_empty = file.tell() == 0
        if file_empty:
            writer.writeheader()

        writer.writerow(last_metrics)

    print(f"Results appended to {file_path} successfully.")


def generate_no_batch_networks(data_dict, hdim):
    # Unpack data_dict
    optimizer_dict = data_dict['optimizer']
    scheduler_lr_dict = data_dict['scheduler_lr']

    # Create optimizer params dict
    opt_params_with_weight_decay = optimizer_dict.copy()
    optimizer_name = opt_params_with_weight_decay['type']
    del opt_params_with_weight_decay['type']

    opt_params_without_weight_decay = opt_params_with_weight_decay.copy()
    del opt_params_without_weight_decay['weight_decay']

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

    # Other parameters for setting up the different pruning strategies
    l1_decay = float(data_dict['l1_sparse']['l1_decay'])
    wp_strength = float(data_dict['weight_pruning']['wp_strength'])
    np_strength = float(data_dict['neuron_pruning']['np_strength'])
    batch_norm_decay = float(data_dict['neuron_pruning']['batch_norm_decay'])
    dropout_rate = float(data_dict['dropout']['dropout_rate'])
    leaky_slope = float(data_dict['leaky']['leaky_slope'])

    # Set the device (use GPU if available, otherwise fallback to CPU)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    cuda = True if device_str == 'cuda' else False

    # Dataset loading and transformation
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'FMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
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

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

    # Initialize the model and move to the correct device
    model = SimpleNN(input_dim, hdim, output_dim).to(device)

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

    # Train the model without batch or L1 regularization
    generate_bool = data_dict['training']['further_training']

    print("Model Simple with ReduceLROnPlateau")
    if generate_bool:
        model1 = copy.deepcopy(model).to(device)
        optimizer_learning_rate =opt_params_without_weight_decay.copy()
        optimizer_learning_rate["lr"] = 0.1
        metrics1 = train(model1, device, train_loader, test_loader, optimizer_cls, optimizer_learning_rate,
                         criterion_cls, num_epochs,
                         output_dim, None, scheduler_lr_cls, scheduler_lr_params, val_loader)
        metrics1['h_dim'] = hdim

    print("Model Simple without ReduceLROnPlateau")
    model2 = copy.deepcopy(model).to(device)
    metrics2 = train(model2, device, train_loader, test_loader, optimizer_cls, opt_params_without_weight_decay,
                     criterion_cls, num_epochs,
                     output_dim, None, None, None, val_loader)
    metrics2['h_dim'] = hdim

    # Write results and save networks on file
    write_results_on_csv('results\\accuracies_no_batch_with_reduce_lr.csv', metrics2)

    if generate_bool:
        write_results_on_csv('results\\accuracies_no_batch.csv', metrics1)


    # Export the models to ONNX format
    # A dummy input (ensure it is on the same device as the model)
    dummy_input = torch.randn(1, input_dim).to(device)

    lr = opt_params_without_weight_decay["lr"]

    if generate_bool:
        torch.onnx.export(
            model1,
            dummy_input,
            f"results/no_batch/{hdim}.onnx",
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
        )

    torch.onnx.export(
        model2,
        dummy_input,
        f"results/no_batch/{hdim}_lr_{lr}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )
