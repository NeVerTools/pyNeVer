import copy

import pynever.strategies.conversion as cv
import pynever.nodes as nodes
import pynever.networks as networks
import torch
import torch.nn.functional as funct
from pynever.tensor import Tensor
import numpy as np
import pynever.pytorch_layers as ptl


def combine_batchnorm1d(linear: ptl.Linear, batchnorm: ptl.BatchNorm1d) -> ptl.Linear:
    """
    Utility function to combine a BatchNorm1D node with a Linear node in a corresponding Linear node.
    Parameters
    ----------
    linear : Linear
        Linear to combine.
    batchnorm : BatchNorm1D
        BatchNorm1D to combine.
    Return
    ----------
    Linear
        The Linear resulting from the fusion of the two input nodes.
    """

    l_weight = linear.weight
    l_bias = linear.bias
    bn_running_mean = batchnorm.running_mean
    bn_running_var = batchnorm.running_var
    bn_weight = batchnorm.weight
    bn_bias = batchnorm.bias
    bn_eps = batchnorm.eps

    fused_bias = torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps))
    fused_bias = torch.mul(fused_bias, torch.sub(l_bias, bn_running_mean))
    fused_bias = torch.add(fused_bias, bn_bias)

    fused_weight = torch.diag(torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps)))
    fused_weight = torch.matmul(fused_weight, l_weight)

    fused_linear = ptl.Linear(linear.identifier, linear.in_dim, linear.out_dim, linear.in_features, linear.out_features,
                              linear.bias)
    fused_linear.weight = fused_weight
    fused_linear.bias = fused_bias

    return fused_linear


def combine_batchnorm1d_net(network: networks.SequentialNetwork) -> networks.SequentialNetwork:
    """
    Utilities function to combine all the FullyConnectedNodes followed by BatchNorm1DNodes in corresponding
    FullyConnectedNodes.
    Parameters
    ----------
    network : SequentialNetwork
        Sequential Network of interest of which we want to combine the nodes.
    Return
    ----------
    SequentialNetwork
        Corresponding Sequential Network with the combined nodes.
    """

    py_net = cv.PyTorchConverter().from_neural_network(network)

    modules = [m for m in py_net.pytorch_network.modules()]
    modules = modules[1:]
    num_modules = len(modules)
    current_index = 0

    new_modules = []

    while current_index + 1 < num_modules:

        current_node = modules[current_index]
        next_node = modules[current_index + 1]

        if isinstance(current_node, ptl.Linear) and isinstance(next_node, ptl.BatchNorm1d):
            combined_node = combine_batchnorm1d(current_node, next_node)
            new_modules.append(combined_node)
            current_index = current_index + 1

        elif isinstance(current_node, ptl.Linear):
            new_modules.append(copy.deepcopy(current_node))

        elif isinstance(current_node, nodes.ReLUNode):
            new_modules.append(copy.deepcopy(current_node))

        else:
            raise Exception("Combine Batchnorm supports only ReLU, Linear and BatchNorm1D layers.")

        current_index = current_index + 1

    temp_pynet = ptl.Sequential(py_net.pytorch_network.identifier, py_net.pytorch_network.input_id, *new_modules)
    combined_pynet = cv.PyTorchNetwork(py_net.identifier, temp_pynet)
    combined_network = cv.PyTorchConverter().to_neural_network(combined_pynet)

    return combined_network


def generate_targeted_linf_robustness_query(data: Tensor, adv_target: int, bounds: tuple,
                                            num_classes: int, epsilon: float, filepath: str):
    """
    Function to generate a targeted Robustness SMTLIB query and to save it to a SMTLIB file.
    The robustness query is of the kind based on the infinity norm.
    It assumes that the data and target are from a classification task.

    Parameters
    ----------
    data : Tensor
        Input data of interest.
    adv_target : int
        Desired adversarial target for the input data.
    bounds : (int, int)
        Bounds for the input data (lower_bound, upper_bound).
    num_classes : int
        Number of possible classes.
    epsilon : float
        Perturbation with respect to the infinity norm.
    filepath : str
        Filepath for the resulting SMTLIB file.

    """
    with open(filepath, "w") as f:

        flattened_data = data.flatten()
        for i in range(len(flattened_data)):
            f.write(f"(declare-const X_{i} Real)\n")

        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        for i in range(len(flattened_data)):

            if flattened_data[i] - epsilon < bounds[0]:
                f.write(f"(assert (>= X_{i} {bounds[0]}))\n")
            else:
                f.write(f"(assert (>= X_{i} {flattened_data[i] - epsilon}))\n")

            if flattened_data[i] + epsilon > bounds[1]:
                f.write(f"(assert (<= X_{i} {bounds[1]}))\n")
            else:
                f.write(f"(assert (<= X_{i} {flattened_data[i] + epsilon}))\n")

        for i in range(num_classes):

            if i != adv_target:
                f.write(f"(assert (<= (- Y_{i} Y_{adv_target}) 0))\n")

def generate_untargeted_linf_robustness_query(data: Tensor, target: int, bounds: tuple,
                                              num_classes: int, epsilon: float, filepath: str):
    """
    Function to generate an untargeted Robustness SMTLIB query and to save it to a SMTLIB file.
    The robustness query is of the kind based on the infinity norm.
    It assumes that the data and target are from a classification task.

    Parameters
    ----------
    data : Tensor
        Input data of interest.
    adv_target : int
        Desired adversarial target for the input data.
    bounds : (int, int)
        Bounds for the input data (lower_bound, upper_bound).
    num_classes : int
        Number of possible classes.
    epsilon : float
        Perturbation with respect to the infinity norm.
    filepath : str
        Filepath for the resulting SMTLIB file.

    """
    with open(filepath, "w") as f:

        flattened_data = data.flatten()
        for i in range(len(flattened_data)):
            f.write(f"(declare-const X_{i} Real)\n")

        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        for i in range(len(flattened_data)):

            if flattened_data[i] - epsilon < bounds[0]:
                f.write(f"(assert (>= X_{i} {bounds[0]}))\n")
            else:
                f.write(f"(assert (>= X_{i} {flattened_data[i] - epsilon}))\n")

            if flattened_data[i] + epsilon > bounds[1]:
                f.write(f"(assert (<= X_{i} {bounds[1]}))\n")
            else:
                f.write(f"(assert (<= X_{i} {flattened_data[i] + epsilon}))\n")

        output_query = "(assert (or"
        for i in range(num_classes):

            if i != target:
                output_query += f" (<= (- Y_{target} Y_{i}) 0)"

        output_query += "))"
        f.write(output_query)


def parse_linf_robustness_smtlib(filepath: str) -> (bool, list, int):
    """
    Function to extract the parameters of a robustness query from the smtlib file.
    It assume the SMTLIB file is structured as following:

        ; definition of the variables of interest
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        ...
        (declare-const Y_1 Real)
        (declare-const Y_2 Real)
        ...
        ; definition of the constraints
        (assert (>= X_0 eps_0))
        (assert (<= X_0 eps_1))
        ...
        (assert (<= (- Y_0 Y_1) 0))
        ...

    Where the eps_i are Real numbers.

    Parameters
    ----------
    filepath : str
        Filepath to the SMTLIB file.

    Returns
    ----------
    (bool, list, int)
        Tuple of list: the first list contains the values eps_i for each variables as tuples (lower_bound, upper_bound),
        while the int correspond to the desired target for the related data.
    """
    targeted = True
    correct_target = -1
    lb = []
    ub = []
    with open(filepath, 'r') as f:

        for line in f:

            line = line.replace('(', '( ')
            line = line.replace(')', ' )')
            if line[0] == '(':
                aux = line.split()
                if aux[1] == 'assert':

                    if aux[4] == '(':
                        if aux[3] == 'or':
                            targeted = False
                            temp = aux[8].split("_")
                            correct_target = int(temp[1])
                        else:
                            targeted = True
                            temp = aux[7].split("_")
                            correct_target = int(temp[1])

                    else:

                        if aux[3] == ">=":
                            lb.append(float(aux[5]))
                        else:
                            ub.append(float(aux[5]))

    input_bounds = []
    for i in range(len(lb)):
        input_bounds.append((lb[i], ub[i]))

    return targeted, input_bounds, correct_target


def net_update(network: networks.NeuralNetwork) -> networks.NeuralNetwork:

    if not network.up_to_date:

        for alt_rep in network.alt_rep_cache:

            if alt_rep.up_to_date:
                if isinstance(alt_rep, cv.ONNXNetwork):
                    return cv.ONNXConverter().to_neural_network(alt_rep)
                elif isinstance(alt_rep, cv.PyTorchNetwork):
                    return cv.PyTorchConverter().to_neural_network(alt_rep)
                else:
                    raise NotImplementedError

    else:
        return network


def parse_acas_property(filepath: str) -> ((Tensor, Tensor), (Tensor, Tensor)):

    in_coeff = np.zeros((10, 5))
    in_bias = np.zeros((10, 1))
    out_coeff = []
    out_bias = []
    row_index = 0

    with open(filepath, 'r') as f:

        for line in f:

            if line[0] == "x":
                splitted_line = line.split(" ")
                var_index = int(splitted_line[0][1])
                if splitted_line[1] == ">=":
                    in_coeff[row_index, var_index] = -1
                    in_bias[row_index] = -float(splitted_line[2])
                else:
                    in_coeff[row_index, var_index] = 1
                    in_bias[row_index] = float(splitted_line[2])

            else:

                splitted_line = line.split(" ")
                if len(splitted_line) == 3:
                    var_index = int(splitted_line[0][1])
                    temp = np.zeros(5)
                    if splitted_line[1] == ">=":
                        temp[var_index] = -1
                        out_coeff.append(temp)
                        out_bias.append(-float(splitted_line[2]))
                    else:
                        temp[var_index] = 1
                        out_coeff.append(temp)
                        out_bias.append(float(splitted_line[2]))
                else:
                    var_index_1 = int(splitted_line[0][2])
                    var_index_2 = int(splitted_line[1][2])
                    temp = np.zeros(5)
                    if splitted_line[2] == ">=":
                        temp[var_index_1] = -1
                        temp[var_index_2] = 1
                        out_coeff.append(temp)
                        out_bias.append(-float(splitted_line[3]))
                    else:
                        temp[var_index_1] = 1
                        temp[var_index_2] = -1
                        out_coeff.append(temp)
                        out_bias.append(float(splitted_line[3]))

            row_index = row_index + 1

        out_coeff = np.array(out_coeff)
        array_out_bias = np.zeros((len(out_bias), 1))

        for i in range(len(out_bias)):
            array_out_bias[i, 0] = out_bias[i]

        out_bias = array_out_bias

    return (in_coeff, in_bias), (out_coeff, out_bias)


def parse_nnet(filepath: str) -> (list, list, list, list, list, list):

    with open(filepath) as f:

        line = f.readline()
        cnt = 1
        while line[0:2] == "//":
            line = f.readline()
            cnt += 1
        # numLayers does't include the input layer!
        numLayers, inputSize, outputSize, maxLayersize = [int(x) for x in line.strip().split(",")[:-1]]
        line = f.readline()

        # input layer size, layer1size, layer2size...
        layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        means = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        ranges = [float(x) for x in line.strip().split(",")[:-1]]

        weights = []
        biases = []
        for layernum in range(numLayers):

            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum + 1]
            # weights
            weights.append([])
            biases.append([])
            # weights
            for i in range(currentLayerSize):
                line = f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                weights[layernum].append([])
                for j in range(previousLayerSize):
                    weights[layernum][i].append(aux[j])
            # biases
            for i in range(currentLayerSize):
                line = f.readline()
                x = float(line.strip().split(",")[0])
                biases[layernum].append(x)

        numLayers = numLayers
        layerSizes = layerSizes
        inputSize = inputSize
        outputSize = outputSize
        maxLayersize = maxLayersize
        inputMinimums = inputMinimums
        inputMaximums = inputMaximums
        inputMeans = means[:-1]
        inputRanges = ranges[:-1]
        outputMean = means[-1]
        outputRange = ranges[-1]
        weights = weights
        biases = biases

        new_weights = []
        new_biases = []
        for i in range(numLayers):
            weight = np.array(weights[i])
            bias = np.array(biases[i])

            new_weights.append(weight)
            new_biases.append(bias)

        return new_weights, new_biases, inputMeans, inputRanges, outputMean, outputRange


def input_search(net: networks.NeuralNetwork, ref_output: Tensor, start_input: Tensor, max_iter: int, rate: float,
                 threshold: float = 1e-5):

    py_net = cv.PyTorchConverter().from_neural_network(net).pytorch_network
    py_ref_output = torch.from_numpy(ref_output)
    py_start_input = torch.from_numpy(start_input)
    current_input = py_start_input
    current_input.requires_grad = True

    optim = torch.optim.SGD(params=[current_input], lr=rate)

    real_output = py_net(current_input)
    py_ref_output = torch.unsqueeze(py_ref_output, 0)
    real_output = torch.unsqueeze(real_output, 0)
    dist = funct.pairwise_distance(real_output, py_ref_output, p=2)
    iteration = 0

    while dist > threshold and iteration < max_iter:
        #current_input.requires_grad = True
        optim.zero_grad()
        dist.backward()
        optim.step()
        real_output = py_net(current_input)
        real_output = torch.unsqueeze(real_output, 0)
        dist = funct.pairwise_distance(py_ref_output, real_output, p=2)
        print(f"Loss: {dist}")
        print(f"Current Input: {current_input}")
        print(f"Current Output: {real_output}")
        print(f"Grad: {current_input.grad}")
        print(f"Ref Output: {ref_output}")
        iteration = iteration + 1

    correct = False
    if dist <= threshold:
        correct = True

    return correct, np.array(current_input), np.array(real_output)


def compute_saliency(net: networks.NeuralNetwork, ref_input: Tensor):

    class BackHook:

        def __init__(self, module: torch.nn.Module, backward=True):
            if backward:
                self.hook = module.register_backward_hook(self.hook_fn)
            else:
                self.hook = module.register_forward_hook(self.hook_fn)
            self.m_input = None
            self.m_output = None

        def hook_fn(self, module, m_input, m_output):
            self.m_input = m_input
            self.m_output = m_output

        def close(self):
            self.hook.remove()

    py_net = cv.PyTorchConverter().from_neural_network(net).pytorch_network

    # We register the hooks on the modules of the networks
    backward_hooks = [BackHook(layer) for layer in py_net.modules()]
    forward_hooks = [BackHook(layer, False) for layer in py_net.modules()]

    ref_input = torch.from_numpy(ref_input)
    ref_input.requires_grad = True
    out = py_net(ref_input)
    i = 0
    print("FORWARD HOOKS")
    for m in py_net.modules():
        hook = forward_hooks[i]
        print(m)
        print("INPUT")
        print(hook.m_input)
        print("OUTPUT")
        print(hook.m_output)
        i = i + 1
    for k in range(len(out)):
        print(f"Variable {k} of output")
        out = py_net(ref_input)
        out[k].backward(retain_graph=True)
        print("INPUT GRAD:" + f"{ref_input.grad}")

        i = 0
        for m in py_net.modules():
            hook = backward_hooks[i]
            print(m)
            print("INPUT")
            print(hook.m_input[0])
            print("OUTPUT")
            print(hook.m_output[0])
            i = i + 1

    print(out)
    ref_input[0] = ref_input[0] + 10
    out = py_net(ref_input)
    print(out)

