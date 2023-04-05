import abc
import copy
import logging
import operator
import time
from typing import List, Optional, Callable

import numpy as np
import torch

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.pytorch_layers as pyt_layers
import pynever.strategies.abstraction as abst
import pynever.strategies.conversion as conv
import pynever.strategies.smt_reading as reading
import pynever.utilities as utils
from pynever.tensor import Tensor

logger_name = "pynever.strategies.verification"


class Property(abc.ABC):
    """
    An abstract class used to represent a generic property for a NeuralNetwork.
    """


class LocalRobustnessProperty(Property):
    """
    A concrete class used to represent a local robustness property for a NeuralNetwork.
    Formally the property check if the counterexample (i.e., the adversarial example) exists, therefore
    when the verification strategy check such property it should return True if the adversarial example exist and
    false otherwise.

    Attributes
    ----------
    data : Tensor
        Original data used to determine the local robustness.
    target : int
        If targeted is True then it is the desired target for the adversarial, otherwise it is the correct target of
        data.
    targeted : bool
        Flag which is True if the robustness property is targeted, False otherwise.
    norm : str
        Norm type used to determine the local robustness. At present the only acceptable value is Linf.
    epsilon : float
        Magnitude of the acceptable perturbation.
    bounds: list
        List of (lower_bound, upper_bound) for the data.

    """

    def __init__(self, data: Tensor, target: int, targeted: bool, norm: str, epsilon: float, bounds: list):
        self.data = data
        self.target = target
        self.targeted = targeted
        if norm != "Linf":
            raise NotImplementedError
        self.norm = norm
        self.epsilon = epsilon
        self.bounds = bounds


class NeVerProperty(Property):
    """
    A concrete class used to represent a NeVer property for a NeuralNetwork. We assume that the hyperplane
    out_coef_mat * y <= out_bias_mat represent the unsafe region (i.e., the negation of the desired property).
    At present the input set must be defined as in_coef_mat * x <= in_bias_mat

    Attributes
    ----------
    in_coef_mat: Tensor
        Matrix of the coefficients for the input constraints.
    in_bias_mat: Tensor
        Matrix of the biases for the input constraints.
    out_coef_mat: List[Tensor]
        Matrixes of the coefficients for the output constraints.
    out_bias_mat: List[Tensor]
        Matrixes of the biases for the output constraints.

    """

    def __init__(self, in_coef_mat: Tensor = None, in_bias_mat: Tensor = None,
                 out_coef_mat: List[Tensor] = None, out_bias_mat: List[Tensor] = None):
        self.in_coef_mat = in_coef_mat
        self.in_bias_mat = in_bias_mat
        self.out_coef_mat = out_coef_mat
        self.out_bias_mat = out_bias_mat

    def from_smt_file(self, filepath: str = '', input_name: str = 'X', output_name: str = 'Y'):
        """
        This method builds the property by reading the corresponding SMT-LIB file

        Parameters
        ----------
        filepath : str
            Path to the SMT-LIB file
        input_name : str, Optional
            The name of the input vector
        output_name : str, Optional
            The name of the output vector

        """

        smt_parser = reading.SmtPropertyParser(filepath, input_name, output_name)
        self.in_coef_mat, self.in_bias_mat, self.out_coef_mat, self.out_bias_mat = smt_parser.parse_property()

    def to_smt_file(self, input_id: str = 'X', output_id: str = 'Y', filepath: str = ''):
        """
        This method builds the SMT-LIB representation of the NeVerProperty, expressing
        the variables and the matrices as constraints in the corresponding logic

        Parameters
        ----------
        input_id : str, Optional
            Identifier of the input node (default: 'X')
        output_id : str, Optional
            Identifier of the output node (default: 'X')
        filepath : str
            Path to the SMT-LIB file to create

        """

        with open(filepath, 'w+') as f:
            # Variables definition
            input_vars = [f"{input_id}_{i}" for i in range(self.in_coef_mat.shape[1])]
            if self.out_coef_mat:
                output_vars = [f"{output_id}_{i}" for i in range(self.out_coef_mat[0].shape[1])]
            else:
                output_vars = []

            f.write(';; --- INPUT VARIABLES ---\n')
            for v_name in input_vars:
                f.write(f"(declare-fun {v_name} () Real)\n")

            f.write('\n;; --- OUTPUT VARIABLES ---\n')
            for v_name in output_vars:
                f.write(f"(declare-fun {v_name} () Real)\n")

            # Constraints definition
            f.write('\n;; --- INPUT CONSTRAINTS ---\n')

            infix_in_constraints = self.__create_infix_constraints(input_vars, self.in_coef_mat, self.in_bias_mat)
            for c in infix_in_constraints:
                prefix_smt_row = reading.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                f.write(f"(assert {prefix_smt_row})\n")

            f.write('\n;; --- OUTPUT CONSTRAINTS ---\n')

            # Allow multiple output properties
            infix_output_properties = []
            for out_mat, out_bias in zip(self.out_coef_mat, self.out_bias_mat):
                infix_constraints = self.__create_infix_constraints(output_vars, out_mat, out_bias)
                infix_output_properties.append(infix_constraints)

            if len(infix_output_properties) == 1:
                for c in infix_output_properties[0]:
                    prefix_smt_row = reading.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                    f.write(f"(assert {prefix_smt_row})\n")
            else:
                s = '(assert (or '
                for p in infix_output_properties:
                    if len(p) == 1:
                        prefix_smt_row = reading.ExpressionTreeConverter().build_from_infix(p[0]).as_prefix()
                        s = s + '\n' + prefix_smt_row
                    else:
                        s = s + '(and '
                        for c in p:
                            prefix_smt_row = reading.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                            s = s + '\n' + prefix_smt_row
                        s = s + ')\n'
                s = s + '))'
                f.write(s)

    @staticmethod
    def __create_infix_constraints(variables: list, coef_mat: Tensor, bias_mat: Tensor) -> list:
        c_list = []

        for row in range(coef_mat.shape[0]):
            coef = coef_mat[row, :]
            bias = bias_mat[row][0]
            s = '('

            # Assign coefficients
            for k in range(len(coef)):
                c = coef[k]
                if c != 0:
                    s = s + f"({float(c)} * {variables[k]})"
                    if k < len(coef) - 1 and any(coef[k + 1:]):
                        s = s + ' + '

            # Add bias
            s = s + f") <= ({float(bias)})"
            c_list.append(s)

        return c_list


class VerificationStrategy(abc.ABC):
    """
    An abstract class used to represent a Verification Strategy.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument
        using a verification strategy determined in the concrete children.

    """

    @abc.abstractmethod
    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, Optional[Tensor]):
        """
        Verify that the neural network of interest satisfy the property given as argument
        using a verification strategy determined in the concrete children.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to train.
        prop : Dataset
            The property which the neural network must satisfy.

        Returns
        ----------
        bool
            True is the neural network satisfy the property, False otherwise.

        """
        pass


class NeverVerification(VerificationStrategy):
    """
    Class used to represent the Never verification strategy.

    Attributes
    ----------

    heuristic : str
        Heuristic used to decide the refinement level of the ReLU abstraction.
        At present can be only one of the following:
        - given_flags: the neuron to be refined are selected referring to the list given in params
        - best_n_neurons: for each star the best n neuron to refine are selected based on the loss of precision
          the abstraction would incur using the coarse over_approximation.
        - overapprox: no neuron refinement.
        - complete: full neuron refinement.
        - mixed: equal number of neuron refined in each ReLU Layer.

    params : List
        Parameters for the heuristic of interest.
        If the heuristic is given_flags then params is a list whose first element is the list of refinement flags.
        If the heuristic is best_n_neurons then params is a list whose first element is the number of neurons to refine.

    refinement_level : int
        Refinement level for the sigmoid abstraction.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument.
    """

    def __init__(self, heuristic: str = "best_n_neurons", params: List = None,
                 refinement_level: int = None):

        self.heuristic = heuristic
        self.params = params
        self.refinement_level = refinement_level
        self.logger = logging.getLogger(logger_name)
        self.counterexample_stars = None

    @staticmethod
    def __build_abst_network(network: networks.NeuralNetwork, heuristic: str, params: List) -> abst.AbsSeqNetwork:

        if not isinstance(network, networks.SequentialNetwork):
            raise Exception("Only sequential networks are supported at present")

        abst_network = abst.AbsSeqNetwork("Abstract Network")

        current_node = network.get_first_node()
        relu_count = 0
        while current_node is not None:

            if isinstance(current_node, nodes.FullyConnectedNode):
                abst_network.add_node(abst.AbsFullyConnectedNode("ABST_" + current_node.identifier, current_node))

            elif isinstance(current_node, nodes.ReLUNode):

                if heuristic == "overapprox":
                    temp_params = [0]
                    temp_heuristic = "best_n_neurons"
                elif heuristic == "complete":
                    temp_params = [current_node.in_dim[0]]
                    temp_heuristic = "best_n_neurons"
                elif heuristic == "mixed":
                    temp_params = params
                    temp_heuristic = "best_n_neurons"
                elif params is None and heuristic == "best_n_neurons":
                    temp_params = [0]
                    temp_heuristic = heuristic
                elif params is None and heuristic == "given_flags":
                    temp_params = [False for i in range(len(current_node.in_dim[0]))]
                    temp_heuristic = heuristic
                else:
                    temp_params = params[relu_count]
                    temp_heuristic = heuristic

                abst_network.add_node(abst.AbsReLUNode("ABST_" + current_node.identifier, current_node,
                                                       temp_heuristic, temp_params))

                relu_count += 1

            else:
                raise Exception(f"Node type: {current_node.__class__} not supported")

            current_node = network.get_next_node(current_node)

        return abst_network

    def __compute_output_starset(self, abst_network: abst.AbsSeqNetwork, prop: NeVerProperty) -> (abst.StarSet, List):

        input_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
        input_starset = abst.StarSet({input_star})
        current_node = abst_network.get_first_node()
        output_starset = input_starset
        n_areas = []
        while current_node is not None:
            time_start = time.perf_counter()
            output_starset = current_node.forward(output_starset)
            time_end = time.perf_counter()
            if isinstance(current_node, abst.AbsReLUNode):
                n_areas.append(current_node.n_areas)
            self.logger.info(f"Computing starset for layer {current_node.identifier}. Current starset has dimension "
                             f"{len(output_starset.stars)}. Time to compute: {time_end - time_start}s.")

            current_node = abst_network.get_next_node(current_node)

        return output_starset, n_areas

    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, Optional[Tensor]):

        self.counterexample_stars = None
        abst_network = self.__build_abst_network(network, self.heuristic, self.params)
        ver_start_time = time.perf_counter()
        if isinstance(prop, NeVerProperty):

            output_starset, n_areas = self.__compute_output_starset(abst_network, prop)

            out_coef_mat = prop.out_coef_mat
            out_bias_mat = prop.out_bias_mat

        else:
            raise Exception("Only NeVerProperty are supported at present")

        # Now we check the itersection of the output starset with the output halfspaces defined by the output
        # constraints of our property of interest. We recall that the property is satisfiable if there exist at least
        # one non-void intersection between the output starset and the halfspaces.
        unsafe_stars = []
        all_empty = []
        for i in range(len(out_coef_mat)):

            empty = True
            for star in output_starset.stars:
                out_coef = out_coef_mat[i]
                out_bias = out_bias_mat[i]
                temp_star = abst.intersect_with_halfspace(star, out_coef, out_bias)
                if not temp_star.check_if_empty():
                    empty = False
                    if self.heuristic == 'complete':
                        unsafe_stars.append(temp_star)

            all_empty.append(empty)

        is_satisfied = not all(all_empty)

        if len(unsafe_stars) > 0:
            self.counterexample_stars = self.__get_counterexample_stars(prop, unsafe_stars)
        ver_end_time = time.perf_counter()
        self.logger.info(f"The property is satisfiable: {is_satisfied}.")
        self.logger.info(f"Verification Time: {ver_end_time - ver_start_time}\n")

        return is_satisfied

    def get_output_starset(self, network: networks.NeuralNetwork, prop: Property):

        self.counterexample_stars = None
        abst_network = self.__build_abst_network(network, self.heuristic, self.params)

        computing_start_time = time.perf_counter()

        if isinstance(prop, NeVerProperty):
            output_starset, n_areas = self.__compute_output_starset(abst_network, prop)
        else:
            raise NotImplementedError

        computing_end_time = time.perf_counter()
        computing_time = computing_end_time - computing_start_time

        return output_starset, computing_time

    def __get_counterexample_stars(self, prop: Property, unsafe_stars: List[abst.Star]):

        if not isinstance(prop, NeVerProperty):
            raise NotImplementedError

        counterexample_stars = []
        for unsafe_star in unsafe_stars:
            temp_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
            temp_star.predicate_matrix = copy.deepcopy(unsafe_star.predicate_matrix)
            temp_star.predicate_bias = copy.deepcopy(unsafe_star.predicate_bias)
            counterexample_stars.append(temp_star)

        return counterexample_stars


class NeverVerificationRef(VerificationStrategy):
    """
    Class used to represent the Never verification strategy with counter example guided refinement.

    Attributes
    ----------
    max_neurons : int
        Maximum number of neuron to refine.

    input_search_params : Dict
        Parameters for the counter-example search.

    precision : float
        Acceptable threshold for the counter-example search.

    rel_ref : bool
        Flag for choosing between the RO and PS refinement methodologies.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument.
    """

    def __init__(self, max_neurons: int, input_search_params: dict, precision: float = 0.005, rel_ref: bool = True):

        self.max_neurons = max_neurons
        self.input_search_params = input_search_params
        self.precision = precision
        self.heuristic = "best_n_neurons"
        self.params = None
        self.logger = logging.getLogger(logger_name)
        self.rel_ref = rel_ref

    @staticmethod
    def __build_abst_network(network: networks.NeuralNetwork, heuristic: str, params: List) -> abst.AbsSeqNetwork:

        if not isinstance(network, networks.SequentialNetwork):
            raise Exception("Only sequential networks are supported at present")

        abst_network = abst.AbsSeqNetwork("Abstract Network")

        current_node = network.get_first_node()
        relu_count = 0
        while current_node is not None:

            if isinstance(current_node, nodes.FullyConnectedNode):
                abst_network.add_node(abst.AbsFullyConnectedNode("ABST_" + current_node.identifier, current_node))

            elif isinstance(current_node, nodes.ReLUNode):

                if params is None and heuristic == "best_n_neurons":
                    temp_params = [0]
                elif params is None and heuristic == "given_flags":
                    temp_params = [False for i in range(len(current_node.in_dim[0]))]
                else:
                    temp_params = params[relu_count]

                abst_network.add_node(abst.AbsReLUNode("ABST_" + current_node.identifier, current_node,
                                                       heuristic, temp_params))

                relu_count += 1

            else:
                raise Exception(f"Node type: {current_node.__class__} not supported")

            current_node = network.get_next_node(current_node)

        return abst_network

    def __compute_output_starset(self, abst_network: abst.AbsSeqNetwork, prop: NeVerProperty) -> (abst.StarSet, List):

        input_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
        input_starset = abst.StarSet({input_star})
        current_node = abst_network.get_first_node()
        output_starset = input_starset
        n_areas = []
        while current_node is not None:
            time_start = time.perf_counter()
            output_starset = current_node.forward(output_starset)
            time_end = time.perf_counter()
            if isinstance(current_node, abst.AbsReLUNode):
                n_areas.append(current_node.n_areas)
            self.logger.info(f"Computing starset for layer {current_node.identifier}. Current starset has dimension "
                             f"{len(output_starset.stars)}. Time to compute: {time_end - time_start}s.")

            current_node = abst_network.get_next_node(current_node)

        return output_starset, n_areas

    def __compute_ref_flags(self, network: networks.NeuralNetwork, input_sample: Tensor, n_areas: List) -> List:

        lrp_a = LRPAnalyzer()
        relevances = lrp_a.analyze(network, input_sample)
        relevances = relevances[1:len(relevances) - 1]
        ref_guides = []

        for i in range(len(relevances)):
            neuron_ref_value = abs(relevances[i] * n_areas[i])
            ref_guides.append(neuron_ref_value.tolist())

        ref_guides_array = np.array(ref_guides)
        temp_list = []
        for i in range(ref_guides_array.shape[0]):
            for j in range(ref_guides_array.shape[1]):
                temp_element = ((i, j), ref_guides_array[i, j])
                temp_list.append(temp_element)

        temp_list.sort(key=operator.itemgetter(1))
        temp_list.reverse()

        ref_flags = np.zeros(ref_guides_array.shape)
        ref_flags = ref_flags.astype(np.bool)
        i = 0
        while temp_list[i][1] != 0 and i < self.max_neurons:
            ref_flags[temp_list[i][0]] = True
            i += 1

        ref_flags = ref_flags.tolist()

        return ref_flags

    def __compute_rel_params(self, network: networks.NeuralNetwork, input_sample: Tensor) -> List:

        lrp_a = LRPAnalyzer()
        relevances = lrp_a.analyze(network, input_sample)
        relevances = relevances[1:len(relevances) - 1]
        for i in range(len(relevances)):
            relevances[i] = (self.max_neurons, abs(relevances[i]))

        return relevances

    def __ref_step(self, network: networks.NeuralNetwork, current_star: abst.Star, starting_point: Tensor,
                   input_star: abst.Star, n_areas: List, only_rel: bool, prop: NeVerProperty,
                   out_coef: Tensor, out_bias: Tensor) -> (bool, Optional[Tensor]):

        samples = current_star.get_samples(1)
        for sample in samples:
            self.logger.debug(f"Sample from current_star is correct: "
                              f"{current_star.check_point_inside(sample, 0)}")

        correct, current_input, current_output = utils.input_search_cloud(network, samples[0],
                                                                          starting_point,
                                                                          **self.input_search_params)

        point_in_unsafe = current_star.check_point_inside(current_output, self.precision)
        concrete_counter_found = input_star.check_point_inside(current_input, self.precision)
        self.logger.debug(f"Point in Unsafe Zone: {point_in_unsafe}")
        self.logger.debug(f"Concrete Counter-example: {concrete_counter_found}")

        if point_in_unsafe and concrete_counter_found:
            return False, current_input

        if only_rel:
            rel_params = self.__compute_rel_params(network, current_input)
            ref_abst_net = self.__build_abst_network(network, heuristic="best_n_neurons_rel", params=rel_params)
        else:
            ref_flags = self.__compute_ref_flags(network, current_input, n_areas)
            ref_abst_net = self.__build_abst_network(network, heuristic="given_flags", params=ref_flags)

        ref_output, trash = self.__compute_output_starset(ref_abst_net, prop)

        verified = True
        for ref_star in ref_output.stars:
            temp_ref_star = abst.intersect_with_halfspace(ref_star, out_coef, out_bias)
            if not temp_ref_star.check_if_empty():
                verified = False

        return verified, None

    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, Optional[Tensor]):

        abst_network = self.__build_abst_network(network, self.heuristic, self.params)

        ver_start_time = time.perf_counter()

        if isinstance(prop, NeVerProperty):

            output_starset, n_areas = self.__compute_output_starset(abst_network, prop)

            out_coef_mat = prop.out_coef_mat
            out_bias_mat = prop.out_bias_mat

        else:
            raise Exception("Only NeVerProperty are supported at present")

        input_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
        starting_point = input_star.get_samples(1)[0]

        or_verified = []
        for i in range(len(out_coef_mat)):

            verified = True
            for star in output_starset.stars:

                out_coef = out_coef_mat[i]
                out_bias = out_bias_mat[i]
                temp_star = abst.intersect_with_halfspace(star, out_coef, out_bias)
                if not temp_star.check_if_empty():

                    verified, counterexample = self.__ref_step(network, temp_star, starting_point, input_star,
                                                               n_areas, self.rel_ref, prop, out_coef, out_bias)

                    if counterexample is not None:
                        return verified, counterexample

            or_verified.append(verified)

        final_verified = any(or_verified)

        ver_end_time = time.perf_counter()
        self.logger.info(f"Verification Result: {final_verified}.")
        self.logger.info(f"Verification Time: {ver_end_time - ver_start_time}\n")

        return final_verified, None

    def get_output_starset(self, network: networks.NeuralNetwork, prop: Property):

        abst_network = self.__build_abst_network(network, self.heuristic, self.params)

        computing_start_time = time.perf_counter()

        if isinstance(prop, NeVerProperty):
            output_starset, n_areas = self.__compute_output_starset(abst_network, prop)
        else:
            raise NotImplementedError

        computing_end_time = time.perf_counter()
        computing_time = computing_end_time - computing_start_time

        return output_starset, computing_time


class LRPAnalyzer:
    """
    Class used to represent the Layer-wise Relevance Propagation Algorithm used in the counter-example
    guided refinement. At present only works for fully connected layers and relu activation functions.

    Attributes
    ----------

    lrp_rule: Callable, optional
        Rule used for the LRP propagation: it should be a function which, when applied to the fully connected layers,
        modify the weights as needed. It should work with our version of the Linear pytorch layers. (default: None)

    epsilon: float, optional
        term used to avoid division by zero errors and to guarantee the behaviour 0/0 = 0. (default: 1e-9)

    Methods
    ----------
    analyze(NeuralNetwork, Tensor)
        Analyze the Neural Network with respect to the input Tensor of interest. It returns a list of list containing
        the relevance value for the hidden layers ReLU neurons.

    """

    def __init__(self, lrp_rule: Callable = None, epsilon: float = 1e-9):

        self.lrp_rule = lrp_rule
        self.epsilon = epsilon

    def analyze(self, network: networks.NeuralNetwork, sample: Tensor) -> List:
        """
        Analyze the Neural Network with respect to the input Tensor of interest. It returns a list of list containing
        the relevance value for the hidden layers ReLU neurons.

        Parameters
        ----------
        network : NeuralNetwork
            The network to analyze to extract the relevances of the ReLU neurons.
        sample : Tensor
            The data sample which is used to compute the relevances.

        Returns
        -------
        List
            List containing the relevances for each neuron of each ReLU hidden layer.

        """

        pyt_net = conv.PyTorchConverter().from_neural_network(network).pytorch_network
        pyt_net.float()
        pyt_sample = torch.Tensor(sample).squeeze()

        pyt_net.eval()
        activations = []
        layers = []

        current_activations = pyt_sample
        activations.append(current_activations)

        last_module = 0
        for m in pyt_net.modules():
            last_module = last_module + 1

        m_index = 0
        for m in pyt_net.modules():

            if isinstance(m, pyt_layers.Linear):

                current_activations = m(current_activations)
                if self.lrp_rule is None:
                    layers.append(m)
                else:
                    layers.append(self.lrp_rule(m))

            elif isinstance(m, pyt_layers.ReLU) and m_index < last_module - 1:

                current_activations = m(current_activations)
                activations.append(current_activations)

            m_index += 1

        activations.reverse()
        layers.reverse()

        # We begin with setting the relevances to the values of the output
        relevances = [pyt_net(pyt_sample)]
        current_relevance = relevances[0]
        for i in range(len(activations)):
            current_relevance = self.__rel_prop_mat(activations[i], layers[i], current_relevance)
            relevances.append(current_relevance)

        for i in range(len(relevances)):
            relevances[i] = relevances[i].detach().numpy()

        relevances.reverse()
        return relevances

    def __rel_prop(self, activation: torch.Tensor, layer: pyt_layers.Linear, p_rel: torch.Tensor) -> torch.Tensor:

        activation = activation.clone().detach().requires_grad_(True)
        activation.retain_grad()
        z = self.epsilon + layer.forward(activation)
        s = p_rel / z
        (z * s.data).sum().backward()
        c = activation.grad
        return activation * c

    def __rel_prop_mat(self, activation: torch.Tensor, layer: pyt_layers.Linear, p_rel: torch.Tensor) -> torch.Tensor:

        activation = activation.clone().detach()
        z = self.epsilon + torch.matmul(activation, torch.transpose(layer.weight, 0, 1))
        s = p_rel / z
        c = torch.matmul(s, layer.weight)
        return activation * c
