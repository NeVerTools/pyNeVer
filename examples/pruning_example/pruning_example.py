import copy
import logging

import onnx
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.transforms as tr

import pynever.datasets as dt
import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.training as training
import pynever.utilities as util
from pynever.strategies import pruning
from pynever.strategies.conversion.converters.onnx import ONNXConverter

# Logger Setup
logger = logging.getLogger("pynever")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Building of the network of interest
small_net = networks.SequentialNetwork("SmallNetwork", "X")
small_net.append_node(nodes.FullyConnectedNode('Linear_1', (784,), 64))
small_net.append_node(nodes.BatchNormNode('BatchNorm_2', (64,)))
small_net.append_node(nodes.ReLUNode('ReLU_3', (64,)))
small_net.append_node(nodes.FullyConnectedNode('Linear_4', (64,), 32))
small_net.append_node(nodes.BatchNormNode('BatchNorm_5', (32,)))
small_net.append_node(nodes.ReLUNode('ReLU_6', (32,)))
small_net.append_node(nodes.FullyConnectedNode('Linear_7', (32,), 10))

onnx_net = ONNXConverter().from_neural_network(small_net)
onnx.save(onnx_net.onnx_network, "FMNIST_Example.onnx")

# Loading of the dataset of interest
transform = tr.Compose([tr.ToTensor(), tr.Normalize(1, 0.5), tr.Lambda(lambda x: torch.flatten(x))])
train_dataset = dt.TorchFMNIST("data/", True, transform)
test_dataset = dt.TorchFMNIST("data/", False, transform)

# Initialization of the training and pruning parameters
cuda = False  # If possible the experiment should be run with cuda, otherwise it will take quite some time.
epochs = 2
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
checkpoint_root = "checkpoints/"

# Creation of the trainers needed for baseline training and fine tuned pruning.
opt_params_pr = {"lr": learning_rate}
scheduler_params = {"patience": scheduler_patience}
opt_params = {"lr": learning_rate, "weight_decay": weight_decay}

trainer_wp = training.PytorchTraining(opt.Adam, opt_params_pr, nn.CrossEntropyLoss(), epochs,
                                      validation_percentage, train_batch_size, validation_batch_size,
                                      scheduler_con=opt.lr_scheduler.ReduceLROnPlateau, sch_params=scheduler_params,
                                      precision_metric=training.PytorchMetrics.inaccuracy,
                                      network_transform=pruning.WPTransform(l1_decay, True, cuda),
                                      device='cuda', checkpoints_root=checkpoint_root)

trainer_ns = training.PytorchTraining(opt.Adam, opt_params_pr, nn.CrossEntropyLoss(), epochs,
                                      validation_percentage, train_batch_size, validation_batch_size,
                                      scheduler_con=opt.lr_scheduler.ReduceLROnPlateau, sch_params=scheduler_params,
                                      precision_metric=training.PytorchMetrics.inaccuracy,
                                      network_transform=pruning.NSTransform(batch_norm_decay, True, cuda),
                                      device='cuda', checkpoints_root=checkpoint_root)

trainer_baseline = training.PytorchTraining(opt.Adam, opt_params, nn.CrossEntropyLoss(), epochs,
                                            validation_percentage, train_batch_size, validation_batch_size,
                                            scheduler_con=opt.lr_scheduler.ReduceLROnPlateau,
                                            sch_params=scheduler_params,
                                            precision_metric=training.PytorchMetrics.inaccuracy, device='cuda',
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

tester = training.PytorchTesting(training.PytorchMetrics.inaccuracy, {}, test_batch_size, 'cuda')

baseline_accuracy = tester.test(baseline_net, test_dataset)
sparse_accuracy = tester.test(sparse_net, test_dataset)
ns_accuracy = tester.test(ns_pruned_net, test_dataset)
wp_accuracy = tester.test(wp_pruned_net, test_dataset)

# Batch norm fusion for the networks of interest (needed for verification and abstraction).
if (isinstance(baseline_net, networks.SequentialNetwork) and
        isinstance(sparse_net, networks.SequentialNetwork) and
        isinstance(wp_pruned_net, networks.SequentialNetwork) and
        isinstance(ns_pruned_net, networks.SequentialNetwork)):
    com_baseline_net = util.combine_batchnorm1d_net(baseline_net)
    com_sparse_net = util.combine_batchnorm1d_net(sparse_net)
    com_wp_pruned_net = util.combine_batchnorm1d_net(wp_pruned_net)
    com_ns_pruned_net = util.combine_batchnorm1d_net(ns_pruned_net)

    com_baseline_accuracy = tester.test(com_baseline_net, test_dataset)
    com_sparse_accuracy = tester.test(com_sparse_net, test_dataset)
    com_ns_accuracy = tester.test(com_ns_pruned_net, test_dataset)
    com_wp_accuracy = tester.test(com_wp_pruned_net, test_dataset)

    logger.info("ACCURACIES (% of samples correctly classified):\n")
    logger.info(f"Baseline: {baseline_accuracy}, Sparse: {sparse_accuracy}, NS: {ns_accuracy}, WP: {wp_accuracy}")
    logger.info(f"COMBINED BATCHNORM NETWORKS")
    logger.info(f"Baseline: {com_baseline_accuracy}, Sparse: {com_sparse_accuracy}, NS: {com_ns_accuracy}, "
                f"WP: {com_wp_accuracy}")
