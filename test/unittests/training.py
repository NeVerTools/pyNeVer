import logging

import torch
import torchvision.transforms as tr
from torch import nn
from torch.optim import Adam, lr_scheduler

from pynever.datasets import GenericFileDataset, TorchMNIST
from pynever.strategies.conversion.converters.onnx import ONNXConverter
from pynever.strategies.conversion.representation import ONNXNetwork
from pynever.strategies.training import PytorchTraining, PytorchMetrics
from strategies.training import PytorchTesting

# Logger Setup
logger = logging.getLogger("pynever")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

train_batch_size = 128
validation_batch_size = 64
test_batch_size = 64
learning_rate = 0.01
validation_percentage = 0.3
scheduler_patience = 5
weight_decay = 0.001
checkpoint_root = ""
scheduler_params = {"patience": scheduler_patience}
opt_params = {"lr": learning_rate, "weight_decay": weight_decay}

data_train = GenericFileDataset('D:/Documents/MNIST/MNIST_training.csv', 784)
# data_train = TorchMNIST('mnist', train=True, download=True,
#                         transform=tr.Compose([tr.ToTensor(),
#                                               tr.Normalize(1, 0.5),
#                                               tr.Lambda(lambda x: torch.flatten(x))
#                                               ])
#                         )
data_test = GenericFileDataset('D:/Documents/MNIST/MNIST_test.csv', 784)

net = ONNXConverter().to_neural_network(ONNXNetwork('../../../mnist_fc.onnx'))

train_strategy = PytorchTraining(Adam, opt_params, nn.CrossEntropyLoss(), 50,
                                 validation_percentage, train_batch_size, validation_batch_size,
                                 scheduler_con=lr_scheduler.ReduceLROnPlateau, sch_params=scheduler_params,
                                 precision_metric=PytorchMetrics.inaccuracy,
                                 device='cpu', checkpoints_root=checkpoint_root)
test_strategy = PytorchTesting(PytorchMetrics.inaccuracy, {}, test_batch_size, 'cpu')

trained = train_strategy.train(net, data_train)
test_strategy.test(trained, data_test)
