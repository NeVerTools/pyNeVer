import abc
import pynever.networks as networks
import pynever.datasets as datasets
import pynever.strategies.conversion as cv
import pynever.strategies.training as training
import torch
import math
import torch.nn as nn
import pynever.pytorch_layers as ptl


class PruningStrategy(abc.ABC):
    """
    An abstract class used to represent a Pruning Strategy.

    Methods
    ----------
    prune(NeuralNetwork, Dataset)
        Prune the neural network of interest using a pruning strategy determined in the concrete children.

    """

    @abc.abstractmethod
    def prune(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Prune the neural network of interest using a pruning strategy determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to prune.
        dataset: Dataset
            The dataset to use for the pre-training and fine-tuning procedure.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the application of the pruning strategy to the original network.

        """
        pass


class WPTransform:
    """
    Callable to pass to the training strategy used in the pruning in order to optimize it for the pruning.

    Attributes
    ----------
    l1_decay : float
        Coefficient of the L1 norm regularizer used on the weights of the linear layers of the network
        to push the weights to near-to-zero values.
    fine_tuning : bool
        If True the weight with value zero are not updated by the optimizer step, otherwise the L1 regularizer
        is used.
    cuda : bool, Optional
        It should be the same of the training strategy receiving the callable as network transform
    """

    def __init__(self, l1_decay: float, fine_tuning: bool, cuda: bool = False):
        self.l1_decay = l1_decay
        self.fine_tuning = fine_tuning
        self.cuda = cuda

    def __call__(self, net: cv.PyTorchNetwork):

        if self.fine_tuning:

            for m in net.pytorch_network.modules():

                if isinstance(m, nn.Linear):
                    weight_copy = m.weight.data.abs().clone()
                    if self.cuda:
                        mask = weight_copy.gt(0).float().cuda()
                    else:
                        mask = weight_copy.gt(0).float()
                    m.weight.grad.data.mul_(mask)

        else:

            for m in net.pytorch_network.modules():

                if isinstance(m, nn.Linear):
                    m.weight.grad.data.add_(self.l1_decay * torch.sign(m.weight.data))


class NSTransform:
    """
    Callable to pass to the training strategy used in the pruning in order to optimize it for the pruning.

    Attributes
    ----------
    batchnorm_decay : float
        Coefficient of the L1 norm regularizer used on the batchnorm layers of the network
        to push the weights to near-to-zero values.
    fine_tuning : bool
        If True the the L1 regularizer is not applied to the network.
    cuda : bool, Optional
        It should be the same of the training strategy receiving the callable as network transform
    """

    def __init__(self, batchnorm_decay: float, fine_tuning: bool, cuda: bool = False):
        self.batchnorm_decay = batchnorm_decay
        self.fine_tuning = fine_tuning
        self.cuda = cuda

    def __call__(self, net: cv.PyTorchNetwork):

        if not self.fine_tuning:
            for m in net.pytorch_network.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.grad.data.add_(self.batchnorm_decay * torch.sign(m.weight.data))


class WeightPruning(PruningStrategy):
    """
    A concrete class used to represent the weight pruning strategy.
    This kind of pruning select the least important weights of the neural network
    of interest and set them to 0. It assume vectorial input for the linear layers.
    We refer to https://arxiv.org/abs/1506.02626 for theoretical details on the strategy.

    Attributes
    ----------
    sparsity_rate : float
        It determines the percentage of neurons which will be removed. It must be a Real number between 0 and 1.
    training_strategy : PytorchTraining
        The training strategy to use for pre-training and/or fine-tuning. NB: Its network_transform parameter must
        be of the class WPTransform.
    pre_training : bool
        Flag to indicate if the network need to be pre-trained.

    Methods
    ----------
    prune(NeuralNetwork, Dataset)
        Prune the neural network of interest using the pruning strategy Weight Pruning.

    """

    def __init__(self, sparsity_rate: float, training_strategy: training.PytorchTraining = None,
                 pre_training: bool = False):

        self.sparsity_rate = sparsity_rate
        self.training_strategy = training_strategy
        if training_strategy is not None and not isinstance(training_strategy.network_transform, WPTransform):
            raise Exception("The network_transform attribute of the training_strategy should be of"
                            " the class WPTransform")
        self.pre_training = pre_training
        if training_strategy is None and pre_training:
            raise Exception("If pre_training is True then training_strategy must not be None")

    def prune(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Prune the neural network of interest using the pruning strategy Weight Pruning.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to prune.
        dataset : Dataset
            The dataset to use for the pre-training and fine-tuning procedure.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the application of the pruning strategy to the original network.

        """

        if self.training_strategy is not None and self.pre_training:

            fine_tuning = self.training_strategy.network_transform.fine_tuning
            self.training_strategy.fine_tuning = False
            network = self.training_strategy.train(network, dataset)
            self.training_strategy.network_transform.fine_tuning = fine_tuning

        pytorch_converter = cv.PyTorchConverter()
        py_net = pytorch_converter.from_neural_network(network)

        py_net = self.__pruning(py_net)

        network.alt_rep_cache.clear()
        network.alt_rep_cache.append(py_net)
        network.up_to_date = False

        if self.training_strategy is not None and self.training_strategy.fine_tuning:
            network = self.training_strategy.train(network, dataset)

        return network

    def __pruning(self, net: cv.PyTorchNetwork):
        """
        Procedure for the pruning of the weights of the PyTorchNetwork passed as an argument.

        Parameters
        ----------
        net : PyTorchNetwork
            The PyTorchNetwork to prune.

        Returns
        ----------
        PyTorchNetwork
            The pruned PyTorchNetwork.

        """

        # We transfer the internal pytorch model to the CPU for the pruning procedure.
        net.pytorch_network.cpu()

        # We compute the number of weights in the network
        num_weights = 0
        for m in net.pytorch_network.modules():

            if isinstance(m, ptl.Linear):
                num_weights += m.weight.numel()

        # We copy all the absolute values of the weights in a new tensor and we sort in ascending order
        weights = torch.zeros(num_weights)
        index = 0
        for m in net.pytorch_network.modules():

            if isinstance(m, ptl.Linear):
                size = m.weight.numel()
                weights[index:(index + size)] = m.weight.view(-1).abs().clone()
                index += size

        ordered_weights, ordered_indexes = torch.sort(weights)

        # We determine the number of weights we need to set to 0, given the sparsity rate.
        threshold_index = math.floor(num_weights * self.sparsity_rate)

        # We select the weight absolute value we will use as threshold value given the threshold index.
        threshold_value = ordered_weights[threshold_index]

        # We set all the weights of the different layers to 0 if they are less or equal than the threshold value
        # (in absolute value)

        for m in net.pytorch_network.modules():

            if isinstance(m, ptl.Linear):
                # The values of the mask are 0 when the corresponding weight is less then the threshold_value (in
                # absolute value), otherwise they are 1
                mask = m.weight.abs().gt(threshold_value).float()
                m.weight.data = torch.mul(m.weight, mask)

        return net


class NetworkSlimming(PruningStrategy):
    """
    A concrete class used to represent the network slimming pruning strategy.
    This kind of pruning select the least important neurons of the neural network
    of interest and eliminates them. It needs a batch normalization layer following each layer
    which should be pruned. We assume that the activation function is always applied after the batch
    normalization layer. It support only networks with linear and batchnorm layers with vectorial inputs
    We refer to https://arxiv.org/abs/1708.06519 for theoretical details on the strategy.

    Attributes
    ----------
    sparsity_rate : float
        It determines the percentage of neurons which will be removed. It must be a Real number between 0 and 1.
    training_strategy : PytorchTraining
        The training strategy to use for pre-training and/or fine-tuning. NB: Its network_transform parameter must
        be of the class NSTransform.
    pre_training : bool
        Flag to indicate if the network need to be pre-trained.

    Methods
    ----------
    prune(NeuralNetwork, Dataset)
        Prune the neural network of interest using the pruning strategy Network Slimming.

    """

    def __init__(self, sparsity_rate: float, training_strategy: training.PytorchTraining = None,
                 pre_training: bool = False):

        self.sparsity_rate = sparsity_rate
        self.training_strategy = training_strategy
        if training_strategy is not None and not isinstance(training_strategy.network_transform, NSTransform):
            raise Exception("The network_transform attribute of the training_strategy should be of"
                            " the class NSTransform")
        self.pre_training = pre_training
        if training_strategy is None and pre_training:
            raise Exception("If pre_training is True then training_strategy must not be None")

    def prune(self, network: networks.NeuralNetwork, dataset: datasets.Dataset) -> networks.NeuralNetwork:
        """
        Prune the neural network of interest using the pruning strategy Network Slimming.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to prune.
        dataset: Dataset
            The dataset to use for the pre-training and fine-tuning procedure.

        Returns
        ----------
        NeuralNetwork
            The Neural Network resulting from the application of the pruning strategy to the original network.

        """

        if self.training_strategy is not None and self.pre_training:
            fine_tuning = self.training_strategy.fine_tuning
            self.training_strategy.fine_tuning = False
            network = self.training_strategy.train(network, dataset)
            self.training_strategy.fine_tuning = fine_tuning

        pytorch_converter = cv.PyTorchConverter()
        py_net = pytorch_converter.from_neural_network(network)

        py_net = self.__pruning(py_net)

        network.alt_rep_cache.clear()
        network.alt_rep_cache.append(py_net)
        network.up_to_date = False

        if self.training_strategy is not None and self.training_strategy.network_transform.fine_tuning:
            network = self.training_strategy.train(network, dataset)

        return network

    def __pruning(self, net: cv.PyTorchNetwork):
        """
        Procedure for the pruning of the neurons of the PyTorchNetwork passed as an argument.

        Parameters
        ----------
        net : PyTorchNetwork
            The PyTorchNetwork to prune.

        Returns
        ----------
        PyTorchNetwork
            The PyTorchNetwork resulting from the application of the pure pruning procedure.

        """

        # We transfer the internal pytorch model to the CPU for the pruning procedure.
        net.pytorch_network.cpu()

        # We compute the total number of weights in the batch normalization layers (which, for fully connected networks,
        # is equal to the number of neurons in the corresponding fully-connected layer).
        num_bn_weights = 0
        for m in net.pytorch_network.modules():
            if isinstance(m, ptl.BatchNorm1d):
                num_bn_weights += m.weight.numel()

        # We copy all the absolute values of the batch norm weights in a new tensor and we sort in ascending order
        bn_weights = torch.zeros(num_bn_weights)
        bn_weights_index = 0
        for m in net.pytorch_network.modules():
            if isinstance(m, ptl.BatchNorm1d):
                size = m.weight.numel()
                bn_weights[bn_weights_index:(bn_weights_index + size)] = m.weight.abs().clone()
                bn_weights_index += size

        ordered_bn_weights, ordered_bn_indexes = torch.sort(bn_weights)

        # We determine the number of neurons we need to remove, given the sparsity rate.
        threshold_index = math.floor(num_bn_weights * self.sparsity_rate)

        # We select the batch norm weight absolute value we will use as threshold value given the threshold index.
        threshold_value = ordered_bn_weights[threshold_index]

        # We now need to create a new network with the correct number of neurons in the different layers.
        # To do so we assume that in the network after a linear layer there is always a batch norm layer.

        new_layers = []
        previous_layer_mask = None
        old_layers = [m for m in net.pytorch_network.modules()]
        orig_seq = old_layers[0]
        num_layers = len(old_layers)
        for i in range(num_layers):

            if (i == num_layers - 1) and isinstance(old_layers[i], ptl.Linear):

                # In this case we are considering the last layer of the network (which we assume to be a linear layer),
                # therefore the number of output of the new layer will be equal to the one of the old layer.

                previous_nonzero_indexes = previous_layer_mask.nonzero(as_tuple=True)[0]

                # If the old linear layer had bias then also the new linear layer has them.
                if old_layers[i].bias is None:
                    has_bias = False
                else:
                    has_bias = True

                # The number of input features for the new linear layer is equal to the number of non-zero elements in
                # the mask of the previous layer.
                num_in_features = int(previous_layer_mask.sum().item())

                # We create the new linear layer with the correct architecture.
                new_in_dim = list(old_layers[i].in_dim)
                new_in_dim[-1] = num_in_features
                new_in_dim = tuple(new_in_dim)
                new_linear_layer = ptl.Linear(old_layers[i].identifier, new_in_dim, old_layers[i].out_dim,
                                              num_in_features, old_layers[i].out_features, has_bias)

                # We copy the parameters corresponding to the still existing neurons.
                new_linear_layer.weight.data = old_layers[i].weight[:, previous_nonzero_indexes].clone()

                if has_bias:
                    new_linear_layer.bias.data = old_layers[i].bias.data

                # We save the new linear layer.
                new_layers.append(new_linear_layer)

            elif isinstance(old_layers[i], ptl.Linear) and isinstance(old_layers[i + 1], ptl.BatchNorm1d):

                # If the layer old_layers[i] is the first linear layer then the previous layer mask corrspond to the
                # complete input.
                if previous_layer_mask is None:
                    previous_layer_mask = torch.ones(old_layers[i].in_features)

                # We compute the mask corresponding to the batch normalization layer.
                layer_mask = old_layers[i + 1].weight.abs().gt(threshold_value).float()
                new_neuron_number = int(layer_mask.sum().item())

                # We compute the indexes of the non-zero weights for the current batch norm layer and the previous one.
                current_nonzero_indexes = layer_mask.nonzero(as_tuple=True)[0]
                previous_nonzero_indexes = previous_layer_mask.nonzero(as_tuple=True)[0]

                # We create the new batch norm layer with the new neuron number.

                new_in_dim = list(old_layers[i + 1].in_dim)
                new_in_dim[-1] = new_neuron_number
                new_in_dim = tuple(new_in_dim)
                new_bn_layer = ptl.BatchNorm1d(old_layers[i + 1].identifier, new_in_dim, new_in_dim, new_neuron_number,
                                               old_layers[i + 1].eps, old_layers[i + 1].momentum,
                                               old_layers[i + 1].affine, old_layers[i + 1].track_running_stats)

                # We copy the parameters corresponding to the still existing neurons from the old batch norm layer
                # to the new one. They are identified by the indexes in current_nonzero_indexes.

                new_bn_layer.weight.data = old_layers[i + 1].weight[current_nonzero_indexes].clone()
                new_bn_layer.bias.data = old_layers[i + 1].bias[current_nonzero_indexes].clone()
                new_bn_layer.running_mean = old_layers[i + 1].running_mean[current_nonzero_indexes].clone()
                new_bn_layer.running_var = old_layers[i + 1].running_var[current_nonzero_indexes].clone()

                # If the old linear layer had bias then also the new linear layer has them.
                if old_layers[i].bias is None:
                    has_bias = False
                else:
                    has_bias = True

                # The number of input features for the new linear layer is equal to the number of non-zero elements in
                # the mask of the previous layer.
                num_in_features = int(previous_layer_mask.sum().item())

                # We create the new linear layer with the correct architecture.
                new_in_dim = list(old_layers[i].in_dim)
                new_in_dim[-1] = num_in_features
                new_in_dim = tuple(new_in_dim)
                new_out_dim = list(old_layers[i].out_dim)
                new_out_dim[-1] = new_neuron_number
                new_out_dim = tuple(new_out_dim)
                new_linear_layer = ptl.Linear(old_layers[i].identifier, new_in_dim, new_out_dim,
                                              num_in_features, new_neuron_number, has_bias)

                # We copy the parameters corresponding to the still existing neurons.
                new_linear_layer.weight.data = old_layers[i].weight[current_nonzero_indexes, :].clone()
                new_linear_layer.weight.data = new_linear_layer.weight[:, previous_nonzero_indexes].clone()

                if has_bias:
                    new_linear_layer.bias.data = old_layers[i].bias[current_nonzero_indexes]

                # We save the new layers in the order in which they should be in our sequential model: first the linear
                # layer and then the batch norm layer.
                new_layers.append(new_linear_layer)
                new_layers.append(new_bn_layer)

                # We update the value of previous_layer_mask with the current mask.
                previous_layer_mask = layer_mask

            elif isinstance(old_layers[i], ptl.Linear) and not isinstance(old_layers[i + 1], ptl.BatchNorm1d):

                # If the layer old_layers[i] is the first linear layer then the previous layer mask correspond to the
                # complete input.
                if previous_layer_mask is None:
                    previous_layer_mask = torch.ones(old_layers[i].in_features)

                # If the linear layer is not followed by a batch normalization layer then it will not be neuron pruned,
                # therefore the layer_mask will be equals to the number of output features of the old layer.
                layer_mask = torch.ones(old_layers[i].out_features)

                # We compute the indexes of the non-zero weights for the current batch norm layer and the previous one.
                current_nonzero_indexes = layer_mask.nonzero(as_tuple=True)[0]
                previous_nonzero_indexes = previous_layer_mask.nonzero(as_tuple=True)[0]

                if old_layers[i].bias is None:
                    has_bias = False
                else:
                    has_bias = True

                # The number of input features for the new linear layer is equal to the number of non-zero elements in
                # the mask of the previous layer.
                num_in_features = previous_layer_mask.sum().item()

                # We create the new linear layer with the correct architecture.
                new_in_dim = list(old_layers[i].in_dim)
                new_in_dim[-1] = num_in_features
                new_in_dim = tuple(new_in_dim)
                new_linear_layer = ptl.Linear(old_layers[i].identifier, new_in_dim, old_layers[i].out_dim,
                                              num_in_features, old_layers[i].out_features, has_bias)

                # We copy the parameters corresponding to the still existing neurons.
                new_linear_layer.weight.data = old_layers[i].weight[current_nonzero_indexes, :].clone()
                new_linear_layer.weight.data = new_linear_layer.weight[:, previous_nonzero_indexes].clone()

                if has_bias:
                    new_linear_layer.bias.data = old_layers[i].bias[current_nonzero_indexes]

                # We save the new layers in the order in which they should be in our sequential model: first the linear
                # layer and then the batch norm layer.
                new_layers.append(new_linear_layer)

            elif isinstance(old_layers[i], ptl.ReLU):
                new_layers.append(ptl.ReLU(old_layers[i].identifier, old_layers[i].in_dim, old_layers[i].out_dim))

        pruned_network = ptl.Sequential(orig_seq.identifier, orig_seq.input_id, new_layers)
        net.pytorch_network = pruned_network
        net.identifier = net.identifier + '_pruned'
        return net
