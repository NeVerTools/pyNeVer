"""
This module contains the classes to train neural networks. It follows the Strategy design pattern providing the abstract
interfaces ``TrainingStrategy`` and ``TestingStrategy``.
At the moment, we provide only a single training and testing strategy using PyTorch.
"""
import abc
import logging
import math
import os
import shutil
from collections.abc import Callable

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as schedulers
import torch.utils.data as tdt

import pynever.datasets as datasets
import pynever.networks as networks
from pynever.strategies.conversion.converters.pytorch import PyTorchConverter
from pynever.strategies.conversion.representation import PyTorchNetwork

logger_name = "pynever.strategies.training"


class TrainingStrategy(abc.ABC):
    """
    An abstract class used to represent a Training Strategy.
    """

    @abc.abstractmethod
    def train(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Train the neural network of interest using a testing strategy determined in the concrete children.

        Parameters
        ----------
        network: NeuralNetwork
            The neural network to train.
        dataset: Dataset
            The dataset to use to train the neural network.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the training of the original network using the training strategy and the
            dataset.
        """
        raise NotImplementedError


class TestingStrategy(abc.ABC):
    """
    An abstract class used to represent a Testing Strategy.
    """

    @abc.abstractmethod
    def test(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> float:
        """
        Test the neural network of interest using a testing strategy determined in the concrete children.

        Parameters
        ----------
        network: NeuralNetwork
            The neural network to test.
        dataset: Dataset
            The dataset to use to test the neural network.

        Returns
        ----------
        float
            A measure of the correctness of the networks dependent on the concrete children

        """
        raise NotImplementedError


class PytorchTraining(TrainingStrategy):
    """
    Class used to represent the training strategy based on the Pytorch learning framework.
    It supports different optimization algorithms, schedulers, loss functions and others based on
    the attributes provided at instantiation time.

    Attributes
    ----------
    optimizer_con: type
        Reference to the class constructor for the Optimizer of choice for the training strategy.
    opt_params: dict
        Dictionary of the parameters to pass to the constructor of the optimizer excluding the first which is always
        assumed to be the parameters to optimize
    loss_function: Callable
        Loss function for the training strategy. We assume it to take as parameters two pytorch Tensor
        corresponding to the output of the network and the target. Other parameters should be given as attributes of
        the callable object.
    n_epochs: int
        Number of epochs for the training procedure.
    validation_percentage: float
        Percentage of the dataset to use as the validation set
    train_batch_size: int
        Dimension for the train batch size for the training procedure
    validation_batch_size: int
        Dimension for the validation batch size for the training procedure
    scheduler_con: type, Optional
        Reference to the class constructor for the Scheduler for the learning rate of choice for the training strategy
        (default: None)
    sch_params: dict, Optional
        Dictionary of the parameters to pass to the constructor of the scheduler excluding the first which is always
        assumed to be the optimizer whose learning rate must be updated. (default: None)
    precision_metric: Callable, Optional
        Function for measuring the precision of the neural network. It is used to choose the best model and to control
        the Plateau Scheduler and the early stopping. We assume it to take as parameters two pytorch Tensor
        corresponding to the output of the network and the target.It should produce a float value and such value should
        decrease for increasing correctness of the network (as the traditional loss value).
        Optional supplementary parameters should be given as attributes of the object. (default: None)
    network_transform: Callable, Optional
        We provide the possibility to define a function which will be applied to the network after
        the computation of backward and before the optimizer step. In practice, we use it for the manipulation
        needed to the pruning oriented training. It should take a pytorch module (i.e., the neural network) as
        input and optional supplementary parameters () should be given as attributes of the object. (default: None)
    train_patience: int, Optional
        The number of epochs in which the loss may not decrease before the
        training procedure is interrupted with early stopping (default: None).
    checkpoints_root: str, Optional
        Where to store the checkpoints of the training strategy. (default: '')
    verbose_rate: int, Optional
        After how many batch the strategy prints information about how the training is going.
    """

    def __init__(self, optimizer_con: type, opt_params: dict, loss_function: Callable, n_epochs: int,
                 validation_percentage: float, train_batch_size: int, validation_batch_size: int, r_split: bool = True,
                 scheduler_con: type = None, sch_params: dict = None, precision_metric: Callable = None,
                 network_transform: Callable = None, device: str = 'cpu', train_patience: int = None,
                 checkpoints_root: str = '', verbose_rate: int = None):

        TrainingStrategy.__init__(self)

        self.optimizer_con = optimizer_con
        self.opt_params = opt_params
        self.scheduler_con = scheduler_con
        self.sch_params = sch_params
        self.loss_function = loss_function

        if precision_metric is None:
            precision_metric = loss_function
        self.precision_metric = precision_metric

        self.n_epochs = n_epochs
        self.validation_percentage = validation_percentage
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.r_split = r_split
        self.network_transform = network_transform

        if device not in ['cpu', 'cuda']:
            raise Exception
        self.device = torch.device(device)

        if train_patience is None:
            train_patience = n_epochs + 1

        self.train_patience = train_patience
        self.verbose_rate = verbose_rate
        self.checkpoints_root = checkpoints_root

        self.logger = logging.getLogger(logger_name)

        # Sanitize checkpoints root as a path
        if len(self.checkpoints_root) > 0 and self.checkpoints_root[-1] != '/':
            if self.checkpoints_root[-1] != '\\':
                self.checkpoints_root = self.checkpoints_root + '/'

    def train(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        pytorch_converter = PyTorchConverter()
        py_net = self.pytorch_training(pytorch_converter.from_neural_network(network), dataset)

        return pytorch_converter.to_neural_network(py_net)

    def pytorch_training(self, net: PyTorchNetwork, dataset: datasets.Dataset) -> PyTorchNetwork:
        """
        Training procedure using PyTorch.

        Parameters
        ----------
        net: PyTorchNetwork
            The PyTorchNetwork to train.
        dataset: Dataset
            The dataset to use for the training of the :class:`~pynever.strategies.conversion.representation.PyTorchNetwork`

        Returns
        ----------
        PyTorchNetwork
            The trained :class:`~pynever.strategies.conversion.representation.PyTorchNetwork`.
        """

        # We set the model to the opportune device.
        net.pytorch_network.float()
        net.pytorch_network.to(self.device)

        # We build the optimizer and the scheduler
        optimizer = self.optimizer_con(net.pytorch_network.parameters(), **self.opt_params)
        scheduler = self.scheduler_con(optimizer, **self.sch_params) if self.scheduler_con else None

        # We split the dataset in training set and validation set.
        validation_len = int(len(dataset) * self.validation_percentage)
        training_len = len(dataset) - validation_len
        if self.r_split:
            training_set, validation_set = tdt.random_split(dataset, (training_len, validation_len))
        else:
            training_set = tdt.Subset(dataset, range(training_len))
            validation_set = tdt.Subset(dataset, range(training_len, len(dataset)))

        # We instantiate the data loaders
        train_loader = tdt.DataLoader(training_set, self.train_batch_size, shuffle=True)
        validation_loader = tdt.DataLoader(validation_set, self.validation_batch_size, shuffle=True)

        if self.verbose_rate is None:
            self.verbose_rate = int(len(train_loader) / 4)

        # If a checkpoint exists, we load it
        checkpoints_path = self.checkpoints_root + net.identifier + '.pth.tar'
        best_model_path = self.checkpoints_root + net.identifier + '_best.pth.tar'

        if os.path.isfile(checkpoints_path):
            self.logger.info(f"Loading Checkpoint: '{checkpoints_path}'")
            checkpoint = torch.load(checkpoints_path)
            start_epoch = checkpoint['epoch']
            net.pytorch_network.load_state_dict(checkpoint['network_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss_score = checkpoint['best_loss_score']
            epochs_without_decrease = checkpoint['epochs_without_decrease']
            self.logger.info(f"Loaded Checkpoint: '{checkpoints_path}'")
            self.logger.info(f"Epoch: {start_epoch}, Best Loss Score: {best_loss_score:.6f}")

        else:
            # Otherwise we initialize the values
            self.logger.info(f"No Checkpoint was found at '{checkpoints_path}'")
            # best_loss_score is set to a high number so that the first epoch will replace it
            best_loss_score = 999999
            epochs_without_decrease = 0
            start_epoch = 0

        # history_score is used to keep track of the training and validation loss evolution
        history_score = torch.zeros((self.n_epochs - start_epoch + 1, 2))
        train_accuracy = 0

        # We begin the real and proper training of the network. In the outer cycle we consider the epochs
        # and for each epoch until termination we consider all the batches
        for epoch in range(start_epoch, self.n_epochs):

            train_size = 0
            correct = 0

            if epochs_without_decrease > self.train_patience:
                break

            # We set all the values of the network to double.
            # net.pytorch_network.double()
            net.pytorch_network.float()
            # We set the network to train mode.
            net.pytorch_network.train()
            avg_loss = 0

            # For each batch we compute one learning step
            for batch_idx, (data, target) in enumerate(train_loader):

                target = target.squeeze()  # Preserves the correct shape
                data, target = data.to(self.device), target.to(self.device)
                output = net.pytorch_network(data)

                if isinstance(self.loss_function, torch.nn.MSELoss):
                    loss = self.loss_function(output, F.one_hot(target, num_classes=output.shape[1]).float())
                else:
                    loss = self.loss_function(output, target)

                avg_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, pred = torch.max(output.data, 1)
                correct += (pred == target.squeeze()).sum().item()
                train_size += self.train_batch_size

                if batch_idx % self.verbose_rate == 0:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(training_set),
                               100. * batch_idx / math.floor(len(training_set) / self.train_batch_size),
                        loss.data.item()))

            avg_loss = avg_loss / batch_idx
            history_score[epoch - start_epoch][0] = avg_loss
            train_accuracy = 100 * correct / train_size

            # EPOCH TEST
            net.pytorch_network.eval()
            validation_loss = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(validation_loader):

                    target = target.squeeze()  # Preserves the correct shape
                    data, target = data.to(self.device), target.to(self.device)
                    output = net.pytorch_network(data)

                    loss = self.precision_metric(output, target)
                    validation_loss += loss.item()

            validation_loss = validation_loss / batch_idx
            self.logger.info('\nValidation Set Metric Value: {:.4f}\n'.format(validation_loss))

            if validation_loss < best_loss_score:
                epochs_without_decrease = 0
                best_loss_score = validation_loss
            else:
                epochs_without_decrease += 1

            if scheduler is not None:
                if isinstance(scheduler, schedulers.ReduceLROnPlateau):
                    scheduler.step(validation_loss)
                elif scheduler is not None:
                    scheduler.step()

            # CHECKPOINT
            history_score[epoch - start_epoch][1] = validation_loss
            is_best = validation_loss < best_loss_score

            state = {
                'epoch': epoch + 1,
                'network_state_dict': net.pytorch_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss_score': best_loss_score,
                'epochs_without_decrease': epochs_without_decrease
            }

            torch.save(state, checkpoints_path)
            if is_best:
                shutil.copyfile(checkpoints_path, best_model_path)

        if os.path.isfile(best_model_path):
            best_checkpoint = torch.load(best_model_path)
            net.pytorch_network.load_state_dict(best_checkpoint['network_state_dict'])

        self.logger.info(f"Best Training Loss Score: {best_loss_score:.6f}")
        self.logger.info(f"Training Accuracy: {train_accuracy:.4f}")

        return net


class PytorchTesting(TestingStrategy):
    """
    Class used to represent the testing strategy based on the Pytorch learning framework.
    It supports different metrics measure for the correctness of the neural network.

    Attributes
    ----------
    metric: Callable
        Function for measuring the precision/correctness of the neural network.
    metric_params: dict
        Supplementary parameters for the metric other than the output and the target (which should always be the first
        two parameters of the metric). It is assumed that it produce a float value and such value
        decrease for increasing correctness of the network (as the traditional loss value).
    test_batch_size: int
        Dimension for the test batch size for the testing procedure
    save_results: bool, Optional
        Whether to save outputs, targets and losses as attributes.
    """

    def __init__(self, metric: Callable, metric_params: dict, test_batch_size: int, device: str = 'cpu',
                 save_results: bool = False):

        TestingStrategy.__init__(self)
        self.metric = metric
        self.metric_params = metric_params
        self.test_batch_size = test_batch_size
        if device not in ['cpu', 'cuda']:
            raise Exception
        self.device = torch.device(device)
        self.save_results = save_results
        self.logger = logging.getLogger(logger_name)

        if save_results:
            self.outputs = []
            self.targets = []
            self.losses = []
        else:
            self.outputs = None
            self.targets = None
            self.losses = None

    def test(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> float:

        pytorch_converter = PyTorchConverter()
        measure = self.pytorch_testing(pytorch_converter.from_neural_network(network), dataset)

        return measure

    def pytorch_testing(self, net: PyTorchNetwork, dataset: datasets.Dataset) -> float:

        net.pytorch_network.float()
        net.pytorch_network.to(self.device)

        # We instantiate the data loader
        test_loader = tdt.DataLoader(dataset, self.test_batch_size)

        net.pytorch_network.eval()
        test_loss = 0
        test_size = 0

        with torch.no_grad():

            correct = 0
            for batch_idx, (data, target) in enumerate(test_loader):

                data, target = data.to(self.device), target.to(self.device)
                output = net.pytorch_network(data)

                if isinstance(self.metric, torch.nn.MSELoss):
                    loss = self.metric(output, F.one_hot(target, num_classes=output.shape[1]).float())
                else:
                    loss = self.metric(output, target)

                if self.save_results:
                    self.outputs.append(output.cpu().detach().numpy())
                    self.targets.append(target.cpu().detach().numpy())
                    self.losses.append(loss.item())
                test_loss += loss.item()

                _, pred = torch.max(output.data, 1)
                correct += (pred == target.squeeze()).sum().item()
                test_size += self.test_batch_size

        # test_loss = test_loss / float(math.floor(len(dataset)) / self.test_batch_size)
        test_loss = test_loss / batch_idx
        test_accuracy = 100 * correct / len(dataset)

        self.logger.info(f"Best Test Loss Score: {test_loss:.6f}")
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        return test_loss


class PytorchMetrics:

    @staticmethod
    def inaccuracy(output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Function to compute the inaccuracy of a prediction. It assumes that the task is classification, the output is
        a Tensor of shape (n, d) where d is the number of possible classes. The target is a Tensor of shape (n, 1) of
        int whose elements correspond to the correct class for the n-th sample. The index of the output element with
        the greater value (considering the n-th Tensor) correspond to the class predicted. We consider the inaccuracy
        metric instead than the accuracy because our metric functions must follow the rule: "lower value equals to
        better network" like the loss functions.

        Parameters
        ----------
        output: torch.Tensor
            Output predicted by the network. It should be a Tensor of shape (n, d)
        target: torch.Tensor
            Correct class for the prediction. It should be a Tensor of shape (n, 1)

        Returns
        -------
        float
            Number of correct prediction / number of sample analyzed
        """

        pred = output.data.max(1, keepdim=True)[1]
        acc = pred.eq(target.data.view_as(pred)).cpu().sum() / len(target)
        return 1 - acc
