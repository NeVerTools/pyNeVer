import copy
import csv

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.conversion as conv
import pynever.strategies.pruning as pruning
import pynever.strategies.training as training
import pynever.utilities as util



def train(model, device, train_loader, test_loader, optimizer_cls, optimizer_params, criterion_cls, num_epochs,
          num_classes,
          l1_lambda=None,
          scheduler_lr_cls=None, scheduler_lr_params=None, val_loader=None):
    # Initialize lists to store metrics for each epoch
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    # Initialize criterion
    criterion = criterion_cls()

    # Initialize scheduler_lr_params if provided
    scheduler = None
    if scheduler_lr_cls is not None:
        scheduler = scheduler_lr_cls(optimizer, **scheduler_lr_params)

    metrics = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
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
                targets = F.one_hot(targets, num_classes=num_classes).float()
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

                if isinstance(criterion, nn.MSELoss):
                    targets = F.one_hot(targets, num_classes=num_classes).float()

                # Compute loss
                loss = criterion(outputs, targets)
                running_test_loss += loss.item()

                # Calculate testing accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        # Calculate average test loss and accuracy for the current epoch
        test_loss = running_test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test

        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['test_loss'].append(test_loss)
        metrics['test_accuracy'].append(test_accuracy)

        # Validation phase (if validation dataset is provided)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.view(val_inputs.size(0), -1).to(device)
                    val_targets = val_targets.to(device)

                    val_outputs = model(val_inputs)

                    if isinstance(criterion, nn.MSELoss):
                        val_targets = F.one_hot(val_targets, num_classes=num_classes).float()

                    val_batch_loss = criterion(val_outputs, val_targets)
                    val_loss += val_batch_loss.item()

                    _, predicted_val = torch.max(val_outputs.data, 1)
                    total_val += val_targets.size(0)
                    correct_val += (predicted_val == val_targets).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100 * correct_val / total_val

            # Store validation metrics
            metrics['val_loss'] = val_loss
            metrics['val_accuracy'] = val_accuracy

            # Step the scheduler if necessary
            if scheduler is not None:
                scheduler.step(val_loss)

        # Step the scheduler if it's not ReduceLROnPlateau
        elif scheduler is not None and scheduler_lr_cls != torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step()

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
        'train_loss': dict_to_write['train_loss'][-1],
        'test_loss': dict_to_write['test_loss'][-1],
        'train_accuracy': dict_to_write['train_accuracy'][-1],
        'test_accuracy': dict_to_write['test_accuracy'][-1],
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


def generate_no_batch_networks(data_dict):
    # Unpack data_dict
    optimizer_name = data_dict['optimizer']
    hdim = int(data_dict['hdim'])
    learning_rate = float(data_dict['learning_rate'])
    train_batch_size = int(data_dict['train_batch_size'])
    test_batch_size = int(data_dict['test_batch_size'])
    validation_batch_size = int(data_dict['validation_batch_size'])
    validation_percentage = float(data_dict['validation_percentage'])
    scheduler_lr = data_dict['scheduler_lr']
    scheduler_patience = int(data_dict['scheduler_patience'])
    loss_type = data_dict['loss_type']
    dataset_name = data_dict['dataset']
    train_dim = int(data_dict['train_dim'])
    test_dim = int(data_dict['test_dim'])
    input_dim = int(data_dict['input_dim'])
    output_dim = int(data_dict['output_dim'])
    num_epochs = int(data_dict['num_epochs'])
    l1_decay = float(data_dict['l1_decay'])
    wp_strength = float(data_dict['wp_strength'])
    np_strength = float(data_dict['np_strength'])
    batch_norm_decay = float(data_dict['batch_norm_decay'])

    # Set the device (use GPU if available, otherwise fallback to CPU)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    cuda = True if device_str == 'cuda' else False

    # Dataset loading and transformation
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(x.size(0), -1))])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'FMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(x.size(0), -1))])
        train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(x.size(0), -1))])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create a subset
    train_subset = Subset(train_set, range(train_dim))  # Take the first train_dim elements from the training set
    test_subset = Subset(test_set, range(test_dim))  # Take the first test_dim elements from the test set

    if scheduler_lr == "ReduceLROnPlateau" and validation_percentage > 0.0:
        # Calculate the size of the validation set
        val_size = int(validation_percentage * len(train_subset))
        train_size = len(train_subset) - val_size

        # Split the training subset into training and validation sets
        train_dataset, val_dataset = random_split(train_subset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=True)
    else:
        val_loader = None

    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
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
    if loss_type == 'CrossEntropyLoss':
        criterion_cls = nn.CrossEntropyLoss
    elif loss_type == 'MSE':
        criterion_cls = nn.MSELoss
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")

    # Define the scheduler
    scheduler_lr_cls = None
    if scheduler_lr == "ReduceLROnPlateau":
        scheduler_lr_cls = optim.lr_scheduler.ReduceLROnPlateau

    # Train the model without batch or L1 regularization
    model1 = copy.deepcopy(model).to(device)
    metrics1 = train(model1, device, train_loader, test_loader, optimizer_cls, criterion_cls, num_epochs, output_dim,
                     None, scheduler_lr_cls, val_loader)
    metrics1['h_dim'] = hdim

    # Train the model with L1 regularization
    model2 = copy.deepcopy(model).to(device)
    metrics2 = train(model2, device, train_loader, test_loader, optimizer_cls, criterion_cls, num_epochs, output_dim,
                     l1_decay, scheduler_lr_cls, val_loader)
    metrics2['h_dim'] = hdim

    # Train the NNs with Batch using pynever
    opt_params_pr = {"lr": learning_rate}
    scheduler_params = {"patience": scheduler_patience}
    opt_params = {"lr": learning_rate}

    # Trainer with Weight Pruning
    trainer_wp = training.PytorchTraining(
        optimizer_con=optimizer_cls,
        opt_params=opt_params_pr,
        loss_function=criterion,
        n_epochs=num_epochs,
        validation_percentage=validation_percentage,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        r_split=True,
        scheduler_con=criterion,
        sch_params=scheduler_params,
        precision_metric=training.PytorchMetrics.inaccuracy,
        network_transform=pruning.WPTransform(l1_decay, True, cuda),
        device=device_str
    )

    # Trainer with Neuron Pruning
    trainer_ns = training.PytorchTraining(
        optimizer_con=optimizer_cls,
        opt_params=opt_params_pr,
        loss_function=criterion,
        n_epochs=num_epochs,
        validation_percentage=validation_percentage,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        r_split=True,
        scheduler_con=scheduler_lr_cls,
        sch_params=scheduler_params,
        precision_metric=training.PytorchMetrics.inaccuracy,
        network_transform=pruning.NSTransform(batch_norm_decay, True, cuda),
        device=device_str
    )

    # Baseline Trainer
    trainer_baseline = training.PytorchTraining(
        optimizer_con=optimizer_cls,
        opt_params=opt_params,
        loss_function=criterion,
        n_epochs=num_epochs,
        validation_percentage=validation_percentage,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        r_split=True,
        scheduler_con=scheduler_lr_cls,
        sch_params=scheduler_params,
        precision_metric=training.PytorchMetrics.inaccuracy,
        device=device_str
    )

    # Training and pruning of the networks of interest
    small_net = create_batched_NN(input_dim, hdim, output_dim)
    baseline_net = copy.deepcopy(small_net)
    baseline_net.identifier = "Baseline"
    baseline_net = trainer_baseline.train(baseline_net, train_subset.dataset)

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
    sparse_loss, sparse_accuracy = tester.test(sparse_net, test_subset.dataset)
    ns_loss, ns_accuracy = tester.test(ns_pruned_net, test_subset.dataset)
    wp_loss, wp_accuracy = tester.test(wp_pruned_net, test_subset.dataset)

    # Batch norm fusion for the networks of interest (needed for verification and abstraction).
    com_baseline_net = util.combine_batchnorm1d_net(baseline_net)
    com_sparse_net = util.combine_batchnorm1d_net(sparse_net)
    com_wp_pruned_net = util.combine_batchnorm1d_net(wp_pruned_net)
    com_ns_pruned_net = util.combine_batchnorm1d_net(ns_pruned_net)

    # Calcolo delle metriche per i vari modelli
    train_com_baseline_loss, train_com_baseline_accuracy = tester.test(com_baseline_net, test_subset.dataset)
    train_com_sparse_loss, train_com_sparse_accuracy = tester.test(com_sparse_net, test_subset.dataset)
    train_com_ns_loss, train_com_ns_accuracy = tester.test(com_ns_pruned_net, test_subset.dataset)
    train_com_wp_loss, train_com_wp_accuracy = tester.test(com_wp_pruned_net, test_subset.dataset)

    # Valutazione sui test set
    test_com_baseline_loss, test_com_baseline_accuracy = tester.test(com_baseline_net, test_subset.dataset)
    test_com_sparse_loss, test_com_sparse_accuracy = tester.test(com_sparse_net, test_subset.dataset)
    test_com_ns_loss, test_com_ns_accuracy = tester.test(com_ns_pruned_net, test_subset.dataset)
    test_com_wp_loss, test_com_wp_accuracy = tester.test(com_wp_pruned_net, test_subset.dataset)

    # Registrazione delle metriche per il modello di base
    baseline_metrics = {
        'h_dim': hdim,  # Dovresti definire il valore di hdim in base alla tua configurazione
        'train_loss': [train_com_baseline_loss],
        'test_loss': [test_com_baseline_loss],
        'train_accuracy': [train_com_baseline_accuracy],
        'test_accuracy': [test_com_baseline_accuracy]
    }

    # Registrazione delle metriche per il modello sparso
    sparse_metrics = {
        'h_dim': hdim,  # Inserisci qui hdim
        'train_loss': [train_com_sparse_loss],
        'test_loss': [test_com_sparse_loss],
        'train_accuracy': [train_com_sparse_accuracy],
        'test_accuracy': [test_com_sparse_accuracy]
    }

    # Registrazione delle metriche per il modello pruned con normal pruning (NS)
    ns_metrics = {
        'h_dim': hdim,  # Inserisci qui hdim
        'train_loss': [train_com_ns_loss],
        'test_loss': [test_com_ns_loss],
        'train_accuracy': [train_com_ns_accuracy],
        'test_accuracy': [test_com_ns_accuracy]
    }

    # Registrazione delle metriche per il modello pruned con weight pruning (WP)
    wp_metrics = {
        'h_dim': hdim,  # Inserisci qui hdim
        'train_loss': [train_com_wp_loss],
        'test_loss': [test_com_wp_loss],
        'train_accuracy': [train_com_wp_accuracy],
        'test_accuracy': [test_com_wp_accuracy]
    }

    onnx_baseline = conv.ONNXConverter().from_neural_network(com_baseline_net)
    onnx_sparse = conv.ONNXConverter().from_neural_network(com_sparse_net)
    onnx_wp = conv.ONNXConverter().from_neural_network(com_wp_pruned_net)
    onnx_ns = conv.ONNXConverter().from_neural_network(com_ns_pruned_net)

    # Write results and save networks on file
    write_results_on_csv('results\\accuracies_no_batch.csv', metrics1)
    write_results_on_csv('results\\accuracies_no_batch_sparse.csv', metrics2)
    write_results_on_csv('results\\accuracies_with_batch.csv', baseline_metrics)
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
        f"results/no_batch_sparse/{hdim}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )

    # Saving each ONNX model to respective files based on conditions
    onnx.save(onnx_baseline.onnx_network,
              "results/batch" + f"/baseline_{hdim}.onnx")
    onnx.save(onnx_sparse.onnx_network, "results/batch_sparse" + f"/sparse_{hdim}.onnx")
    onnx.save(onnx_wp.onnx_network, "results/weight_pruning" + f"/wp_pruned_{hdim}.onnx")
    onnx.save(onnx_ns.onnx_network, "results/neuron_pruning" + f"/ns_pruned_{hdim}.onnx")
