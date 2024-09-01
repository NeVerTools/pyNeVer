import os
import sys

# Add the path to the pynever module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pynever')))
import onnx
import pynever.networks as networks
import pynever.nodes as nodes
import pynever.utilities as util
import pynever.strategies.training as training
import pynever.strategies.pruning as pruning
import pynever.strategies.conversion as conv
import torchvision.transforms as tr
import torchvision
import torch.optim as opt
import torch.nn as nn
import torch
import copy

epochs = 1
train_batch_size = 128
validation_batch_size = 64
test_batch_size = 64
learning_rate = 0.1
batch_norm_decay = 0.001
weight_sparsity_rate = 0.7  # Prune 70% of the weights
neuron_sparsity_rate = 0.5  # Prune 50% of the neurons
validation_percentage = 0.3
scheduler_patience = 5
l1_decay = 0.0001
weight_decay = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"
cuda = True if device == "cuda" else False

# Dataset

INPUT_DIM = 784  # after flattening
OUTPUT_DIM = 10
DATASET_DIR = 'dataset'
checkpoints_root = None

# Data loading and transformations
transform = tr.Compose([
    tr.ToTensor(),
    tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image
])

train_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=False, download=True, transform=transform)


def train_and_prune_mnist_network(weight_sparsity_rate, neuron_sparsity_rate, architecture, folder_path):
    print(architecture)
    print(f"{weight_sparsity_rate=}, {neuron_sparsity_rate=}")
    label = '_'.join(map(str, architecture))
    first_layer = architecture.pop(0)
    i = 1

    # Building of the modified network
    small_net = networks.SequentialNetwork("SmallNetwork", "X")
    small_net.add_node(nodes.FullyConnectedNode(f'Linear_{i}', (INPUT_DIM,), first_layer))
    i = i + 1
    small_net.add_node(nodes.BatchNormNode(f'BatchNorm_{i}', (first_layer,)))
    i = i + 1
    small_net.add_node(nodes.ReLUNode(f'ReLU_{i}', (first_layer,)))
    i = i + 1
    last_layer = first_layer

    for value in architecture:
        small_net.add_node(nodes.FullyConnectedNode(f'Linear_{i}', (last_layer,), value))
        i = i + 1
        small_net.add_node(nodes.BatchNormNode(f'BatchNorm_{i}', (value,)))
        i = i + 1
        small_net.add_node(nodes.ReLUNode(f'ReLU_{i}', (value,)))
        i = i + 1
        last_layer = value

    small_net.add_node(nodes.FullyConnectedNode(f'Linear_{i}', (last_layer,), OUTPUT_DIM))

    # Initialization of the training and pruning parameters
    opt_params_pr = {"lr": learning_rate}
    scheduler_params = {"patience": scheduler_patience}
    opt_params = {"lr": learning_rate, "weight_decay": weight_decay}

    checkpoint_root = folder_path

    trainer_wp = training.PytorchTraining(opt.Adam, opt_params_pr, nn.CrossEntropyLoss(), epochs, validation_percentage,
                                          train_batch_size, validation_batch_size, True,
                                          opt.lr_scheduler.ReduceLROnPlateau,
                                          scheduler_params, training.PytorchMetrics.inaccuracy,
                                          pruning.WPTransform(l1_decay, True, cuda), device,
                                          checkpoints_root=checkpoint_root)

    trainer_ns = training.PytorchTraining(opt.Adam, opt_params_pr, nn.CrossEntropyLoss(), epochs, validation_percentage,
                                          train_batch_size, validation_batch_size, True,
                                          opt.lr_scheduler.ReduceLROnPlateau,
                                          scheduler_params, training.PytorchMetrics.inaccuracy,
                                          pruning.NSTransform(batch_norm_decay, True, cuda), device,
                                          checkpoints_root=checkpoint_root)

    trainer_baseline = training.PytorchTraining(opt.Adam, opt_params, nn.CrossEntropyLoss(), epochs,
                                                validation_percentage,
                                                train_batch_size, validation_batch_size,
                                                True,
                                                opt.lr_scheduler.ReduceLROnPlateau,
                                                scheduler_params, training.PytorchMetrics.inaccuracy, device=device,
                                                checkpoints_root=checkpoint_root)

    # Training and pruning of the networks of interest
    baseline_net = copy.deepcopy(small_net)
    baseline_net.identifier = "Baseline"
    baseline_net = trainer_baseline.train(baseline_net, train_dataset)

    sparse_net = copy.deepcopy(small_net)
    sparse_net.identifier = "Sparse"
    trainer_ns.network_transform.fine_tuning = False
    sparse_net = trainer_ns.train(sparse_net, train_dataset)
    trainer_ns.network_transform.fine_tuning = True

    wp_pruner = pruning.WeightPruning(weight_sparsity_rate, trainer_wp, pre_training=True)
    ns_pruner = pruning.NetworkSlimming(neuron_sparsity_rate, trainer_ns, pre_training=False)

    wp_pruned_net = copy.deepcopy(small_net)
    wp_pruned_net.identifier = "WP_PRUNED"
    wp_pruned_net = wp_pruner.prune(wp_pruned_net, train_dataset)

    ns_pruned_net = copy.deepcopy(sparse_net)
    ns_pruned_net.identifier = "NS_PRUNED"
    ns_pruned_net = ns_pruner.prune(ns_pruned_net, train_dataset)

    tester = training.PytorchTesting(training.PytorchMetrics.inaccuracy, {}, test_batch_size, device)

    baseline_loss, baseline_accuracy = tester.test(baseline_net, test_dataset)
    sparse_loss, sparse_accuracy = tester.test(sparse_net, test_dataset)
    ns_loss, ns_accuracy = tester.test(ns_pruned_net, test_dataset)
    wp_loss, wp_accuracy = tester.test(wp_pruned_net, test_dataset)

    # Batch norm fusion for the networks of interest (needed for verification and abstraction).
    com_baseline_net = util.combine_batchnorm1d_net(baseline_net)
    com_sparse_net = util.combine_batchnorm1d_net(sparse_net)
    com_wp_pruned_net = util.combine_batchnorm1d_net(wp_pruned_net)
    com_ns_pruned_net = util.combine_batchnorm1d_net(ns_pruned_net)

    com_baseline_loss, com_baseline_accuracy = tester.test(com_baseline_net, test_dataset)
    com_sparse_loss, com_sparse_accuracy = tester.test(com_sparse_net, test_dataset)
    com_ns_loss, com_ns_accuracy = tester.test(com_ns_pruned_net, test_dataset)
    com_wp_loss, com_wp_accuracy = tester.test(com_wp_pruned_net, test_dataset)

    onnx_baseline = conv.ONNXConverter().from_neural_network(com_baseline_net)
    onnx_sparse = conv.ONNXConverter().from_neural_network(com_sparse_net)
    onnx_wp = conv.ONNXConverter().from_neural_network(com_wp_pruned_net)
    onnx_ns = conv.ONNXConverter().from_neural_network(com_ns_pruned_net)

    # Saving each ONNX model to respective files based on conditions

    onnx.save(onnx_baseline.onnx_network,
              folder_path + f"/baseline_{neuron_sparsity_rate}_{weight_sparsity_rate}.onnx")
    onnx.save(onnx_sparse.onnx_network, folder_path + f"/sparse_{neuron_sparsity_rate}_{weight_sparsity_rate}.onnx")
    onnx.save(onnx_wp.onnx_network, folder_path + f"/wp_pruned_{neuron_sparsity_rate}_{weight_sparsity_rate}.onnx")
    onnx.save(onnx_ns.onnx_network, folder_path + f"/ns_pruned_{neuron_sparsity_rate}_{weight_sparsity_rate}.onnx")

    return {
        "architecture": label,
        "ns_rate": neuron_sparsity_rate,
        "wp_rate": weight_sparsity_rate,
        "baseline_accuracy": baseline_accuracy,
        "sparse_accuracy": sparse_accuracy,
        "ns_accuracy": ns_accuracy,
        "wp_accuracy": wp_accuracy,
    }
