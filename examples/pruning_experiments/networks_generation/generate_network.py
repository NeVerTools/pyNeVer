import copy
import csv

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


class DropNN(nn.Module):
    def __init__(self, input_dim, hdim, output_dim, dropout_rate):
        super(DropNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hdim, output_dim)
        )

    def forward(self, x):
        return self.sequential(x)


class LeakyNN(nn.Module):
    def __init__(self, input_dim, hdim, output_dim, leaky_slope):
        super(LeakyNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(hdim, output_dim)
        )

    def forward(self, x):
        return self.sequential(x)


def create_batched_NN(input_dim, hdim, output_dim):
    i = 1
    net = networks.SequentialNetwork("Net", "X")
    net.add_node(nodes.FullyConnectedNode(f'Linear_{i}', (input_dim,), hdim))
    i = i + 1
    net.add_node(nodes.BatchNormNode(f'BatchNorm_{i}', (hdim,)))
    i = i + 1
    net.add_node(nodes.ReLUNode(f'ReLU_{i}', (hdim,)))
    i = i + 1

    net.add_node(nodes.FullyConnectedNode(f'Linear_{i}', (hdim,), output_dim))
    return net


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
    print("Model Simple")
    model1 = copy.deepcopy(model).to(device)
    metrics1 = train(model1, device, train_loader, test_loader, optimizer_cls, opt_params_without_weight_decay,
                     criterion_cls, num_epochs,
                     output_dim, None, scheduler_lr_cls, scheduler_lr_params, val_loader)
    metrics1['h_dim'] = hdim

    # Train the model with weight decay
    print("Model Simple with Weight Decay")
    model2 = copy.deepcopy(model).to(device)
    metrics2 = train(model2, device, train_loader, test_loader, optimizer_cls, opt_params_with_weight_decay,
                     criterion_cls, num_epochs,
                     output_dim, None, scheduler_lr_cls, scheduler_lr_params, val_loader)
    metrics2['h_dim'] = hdim

    # Train the model with L1 regularization
    print("Model Simple with L1 Regularization")
    model3 = copy.deepcopy(model).to(device)
    metrics3 = train(model3, device, train_loader, test_loader, optimizer_cls, opt_params_without_weight_decay,
                     criterion_cls, num_epochs, output_dim,
                     l1_decay, scheduler_lr_cls, scheduler_lr_params, val_loader)
    metrics3['h_dim'] = hdim

    print("Model Simple with Dropout")
    # Train the model with Drop regularization
    model4 = DropNN(input_dim, hdim, output_dim, dropout_rate).to(device)
    metrics4 = train(model4, device, train_loader, test_loader, optimizer_cls, opt_params_without_weight_decay,
                     criterion_cls, num_epochs, output_dim,
                     None, scheduler_lr_cls, scheduler_lr_params, val_loader)
    metrics4['h_dim'] = hdim

    # Train the model with LeakyRelu
    print("Model Simple with LeakyRelu")
    model5 = LeakyNN(input_dim, hdim, output_dim, leaky_slope).to(device)
    metrics5 = train(model5, device, train_loader, test_loader, optimizer_cls, opt_params_without_weight_decay,
                     criterion_cls, num_epochs, output_dim,
                     None, scheduler_lr_cls, scheduler_lr_params, val_loader)
    metrics5['h_dim'] = hdim

    # Trainer with Weight Pruning
    trainer_wp = training.PytorchTraining(
        optimizer_con=optimizer_cls,
        opt_params=opt_params_without_weight_decay,
        loss_function=criterion_callable,
        n_epochs=num_epochs,
        validation_percentage=validation_percentage,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        r_split=True,
        scheduler_con=scheduler_lr_cls,
        sch_params=scheduler_lr_params,
        precision_metric=training.PytorchMetrics.inaccuracy,
        network_transform=pruning.WPTransform(l1_decay, True, cuda),
        device=device_str
    )

    # Trainer with Neuron Pruning
    trainer_ns = training.PytorchTraining(
        optimizer_con=optimizer_cls,
        opt_params=opt_params_without_weight_decay,
        loss_function=criterion_callable,
        n_epochs=num_epochs,
        validation_percentage=validation_percentage,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        r_split=True,
        scheduler_con=scheduler_lr_cls,
        sch_params=scheduler_lr_params,
        precision_metric=training.PytorchMetrics.inaccuracy,
        network_transform=pruning.NSTransform(batch_norm_decay, True, cuda),
        device=device_str
    )

    # Baseline Trainer
    trainer_baseline = training.PytorchTraining(
        optimizer_con=optimizer_cls,
        opt_params=opt_params_without_weight_decay,
        loss_function=criterion_callable,
        n_epochs=num_epochs,
        validation_percentage=validation_percentage,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        r_split=True,
        scheduler_con=scheduler_lr_cls,
        sch_params=scheduler_lr_params,
        precision_metric=training.PytorchMetrics.inaccuracy,
        device=device_str
    )

    # Baseline Trainer
    trainer_baseline_weight_decay = training.PytorchTraining(
        optimizer_con=optimizer_cls,
        opt_params=opt_params_with_weight_decay,
        loss_function=criterion_callable,
        n_epochs=num_epochs,
        validation_percentage=validation_percentage,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        r_split=True,
        scheduler_con=scheduler_lr_cls,
        sch_params=scheduler_lr_params,
        precision_metric=training.PytorchMetrics.inaccuracy,
        device=device_str
    )

    # Training and pruning of the networks of interest
    print("Model Batch")
    small_net = create_batched_NN(input_dim, hdim, output_dim)
    baseline_net = copy.deepcopy(small_net)
    baseline_net.identifier = "Baseline"
    baseline_net = trainer_baseline.train(baseline_net, train_subset.dataset)

    # Training and pruning of the networks of interest
    print("Model Batch with Weight Decay")
    baseline_net_weight_decay = copy.deepcopy(small_net)
    baseline_net_weight_decay.identifier = "Baseline_Weight_Decay"
    baseline_net_weight_decay = trainer_baseline_weight_decay.train(baseline_net_weight_decay, train_subset.dataset)

    print("Model Batch with Ns Pruning")
    sparse_net = copy.deepcopy(small_net)
    sparse_net.identifier = "Sparse"
    trainer_ns.network_transform.fine_tuning = False
    sparse_net = trainer_ns.train(sparse_net, train_subset.dataset)
    trainer_ns.network_transform.fine_tuning = True

    wp_pruner = pruning.WeightPruning(wp_strength, trainer_wp, pre_training=True)
    ns_pruner = pruning.NetworkSlimming(np_strength, trainer_ns, pre_training=False)

    wp_pruned_net = copy.deepcopy(small_net)
    wp_pruned_net.identifier = "WP_PRUNED"
    wp_pruned_net = wp_pruner.prune(wp_pruned_net, train_subset.dataset)

    ns_pruned_net = copy.deepcopy(sparse_net)
    ns_pruned_net.identifier = "NS_PRUNED"
    ns_pruned_net = ns_pruner.prune(ns_pruned_net, train_subset.dataset)

    tester = training.PytorchTesting(training.PytorchMetrics.inaccuracy, {}, test_batch_size, device_str)

    baseline_loss, baseline_accuracy = tester.test(baseline_net, test_subset.dataset)
    baseline_net_weight_decay_loss, baseline_net_weight_decay_accuracy = (
        tester.test(baseline_net_weight_decay, test_subset.dataset))
    sparse_loss, sparse_accuracy = tester.test(sparse_net, test_subset.dataset)
    ns_loss, ns_accuracy = tester.test(ns_pruned_net, test_subset.dataset)
    wp_loss, wp_accuracy = tester.test(wp_pruned_net, test_subset.dataset)

    # Batch norm fusion for the networks of interest (needed for verification and abstraction).
    com_baseline_net = util.combine_batchnorm1d_net(baseline_net)
    com_baseline_net_weight_decay = util.combine_batchnorm1d_net(baseline_net_weight_decay)
    com_sparse_net = util.combine_batchnorm1d_net(sparse_net)
    com_wp_pruned_net = util.combine_batchnorm1d_net(wp_pruned_net)
    com_ns_pruned_net = util.combine_batchnorm1d_net(ns_pruned_net)

    # Calcolo delle metriche per i vari modelli
    train_com_baseline_loss, train_com_baseline_accuracy = tester.test(com_baseline_net, train_subset.dataset)
    train_com_baseline_weight_loss, train_com_baseline_weight_accuracy = tester.test(com_baseline_net_weight_decay,
                                                                                     train_subset.dataset)
    train_com_sparse_loss, train_com_sparse_accuracy = tester.test(com_sparse_net, train_subset.dataset)
    train_com_ns_loss, train_com_ns_accuracy = tester.test(com_ns_pruned_net, train_subset.dataset)
    train_com_wp_loss, train_com_wp_accuracy = tester.test(com_wp_pruned_net, train_subset.dataset)

    # Valutazione sui test set
    test_com_baseline_loss, test_com_baseline_accuracy = tester.test(com_baseline_net, test_subset.dataset)
    test_com_baseline_weight_loss, test_com_baseline_weight_accuracy = tester.test(com_baseline_net_weight_decay,
                                                                                   test_subset.dataset)
    test_com_sparse_loss, test_com_sparse_accuracy = tester.test(com_sparse_net, test_subset.dataset)
    test_com_ns_loss, test_com_ns_accuracy = tester.test(com_ns_pruned_net, test_subset.dataset)
    test_com_wp_loss, test_com_wp_accuracy = tester.test(com_wp_pruned_net, test_subset.dataset)

    # Registrazione delle metriche per il modello di base
    baseline_metrics = {
        'h_dim': hdim,  # Define the value of hdim based on your configuration
        'train_loss': train_com_baseline_loss,
        'test_loss': test_com_baseline_loss,
        'train_accuracy': train_com_baseline_accuracy,
        'test_accuracy': test_com_baseline_accuracy
    }

    baseline_weight_decay_metrics = {
        'h_dim': hdim,  # Define the value of hdim based on your configuration
        'train_loss': train_com_baseline_weight_loss,
        'test_loss': train_com_baseline_weight_loss,
        'train_accuracy': train_com_baseline_weight_accuracy,
        'test_accuracy': train_com_baseline_weight_accuracy
    }

    # Metrics for the sparse model
    sparse_metrics = {
        'h_dim': hdim,  # Insert hdim here
        'train_loss': train_com_sparse_loss,
        'test_loss': test_com_sparse_loss,
        'train_accuracy': train_com_sparse_accuracy,
        'test_accuracy': test_com_sparse_accuracy
    }

    # Metrics for the pruned model with normal pruning (NS)
    ns_metrics = {
        'h_dim': hdim,  # Insert hdim here
        'train_loss': train_com_ns_loss,
        'test_loss': test_com_ns_loss,
        'train_accuracy': train_com_ns_accuracy,
        'test_accuracy': test_com_ns_accuracy
    }

    # Metrics for the pruned model with weight pruning (WP)
    wp_metrics = {
        'h_dim': hdim,  # Insert hdim here
        'train_loss': train_com_wp_loss,
        'test_loss': test_com_wp_loss,
        'train_accuracy': train_com_wp_accuracy,
        'test_accuracy': test_com_wp_accuracy
    }

    onnx_baseline = conv.ONNXConverter().from_neural_network(com_baseline_net)
    onnx_baseline_weight_decay = conv.ONNXConverter().from_neural_network(com_baseline_net_weight_decay)
    onnx_sparse = conv.ONNXConverter().from_neural_network(com_sparse_net)
    onnx_wp = conv.ONNXConverter().from_neural_network(com_wp_pruned_net)
    onnx_ns = conv.ONNXConverter().from_neural_network(com_ns_pruned_net)

    # Write results and save networks on file
    write_results_on_csv('results\\accuracies_no_batch.csv', metrics1)
    write_results_on_csv('results\\accuracies_no_batch_weight_decay.csv', metrics2)
    write_results_on_csv('results\\accuracies_no_batch_sparse.csv', metrics3)
    write_results_on_csv('results\\accuracies_no_batch_dropout.csv', metrics4)
    write_results_on_csv('results\\accuracies_no_batch_leaky.csv', metrics5)
    write_results_on_csv('results\\accuracies_with_batch.csv', baseline_metrics)
    write_results_on_csv('results\\accuracies_with_batch_weight_decay.csv', baseline_weight_decay_metrics)
    write_results_on_csv('results\\accuracies_with_batch_sparse.csv', sparse_metrics)
    write_results_on_csv('results\\accuracies_neuron_pruning.csv', ns_metrics)
    write_results_on_csv('results\\accuracies_weight_pruning.csv', wp_metrics)

    # Export the models to ONNX format
    # A dummy input (ensure it is on the same device as the model)
    dummy_input = torch.randn(1, input_dim).to(device)

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
        f"results/no_batch_weight_decay/{hdim}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )

    torch.onnx.export(
        model3,
        dummy_input,
        f"results/no_batch_sparse/{hdim}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )

    torch.onnx.export(
        model4,
        dummy_input,
        f"results/no_batch_dropout/{hdim}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )

    torch.onnx.export(
        model5,
        dummy_input,
        f"results/no_batch_leaky/{hdim}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )

    # Saving each ONNX model to respective files based on conditions
    onnx.save(onnx_baseline.onnx_network,
              "results/batch" + f"/baseline_{hdim}.onnx")
    onnx.save(onnx_baseline_weight_decay.onnx_network,
              "results/batch_weight_decay" + f"/baseline_weight_{hdim}.onnx")
    onnx.save(onnx_sparse.onnx_network, "results/batch_sparse" + f"/sparse_{hdim}.onnx")
    onnx.save(onnx_wp.onnx_network, "results/weight_pruning" + f"/wp_pruned_{hdim}.onnx")
    onnx.save(onnx_ns.onnx_network, "results/neuron_pruning" + f"/ns_pruned_{hdim}.onnx")
