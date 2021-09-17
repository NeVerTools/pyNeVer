import abc
import logging
import time
from typing import List, Optional

import pynever.networks as networks
import pynever.nodes as nodes
import pynever.strategies.abstraction as abst
import pynever.utilities as util
from pynever.tensor import Tensor
import pynever.strategies.processing as processing

logger_name = "pynever.strategies.verification"


class Property(abc.ABC):
    """
    An abstract class used to represent a generic property for a NeuralNetwork.
    """


class SMTLIBProperty(Property):
    """
    A concrete class used to represent a generic property for a NeuralNetwork expressed as a SMTLIB query.

    Attributes
    ----------
    smtlib_path : str
        Filepath for the SMTLIB file in which the property is defined.

    """

    def __init__(self, smtlib_path: str):
        self.smtlib_path = smtlib_path


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

    def __init__(self, in_coef_mat: Tensor, in_bias_mat: Tensor, out_coef_mat: List[Tensor],
                 out_bias_mat: List[Tensor]):
        self.in_coef_mat = in_coef_mat
        self.in_bias_mat = in_bias_mat
        self.out_coef_mat = out_coef_mat
        self.out_bias_mat = out_bias_mat

    def to_smt_file(self, filepath: str):

        infix_constraints = []

        

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

    refinement_heuristic : str
        Heuristic used to decide the refinement level of the ReLU abstraction.
        At present can be only one of the following:
        - given_flags: the neuron to be refined are selected referring to the list given in params
        - best_n_neurons: for each star the n best neuron to refine are selected based on the loss of precision
          the abstraction would incur using the coarse over_approximation.

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

    def __init__(self, heuristic: str = "best_n_neurons", params: List = [0],
                 refinement_level: int = None):

        self.heuristic = heuristic
        self.params = params
        self.refinement_level = refinement_level

    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, Optional[Tensor]):

        if not isinstance(network, networks.SequentialNetwork):
            raise Exception("Only sequential networks are supported at present")

        logger = logging.getLogger(logger_name)
        abst_networks = abst.AbsSeqNetwork("Abstract Network")

        current_node = network.get_first_node()
        while current_node is not None:

            if isinstance(current_node, nodes.FullyConnectedNode):
                abst_networks.add_node(abst.AbsFullyConnectedNode("ABST_" + current_node.identifier, current_node))

            elif isinstance(current_node, nodes.ReLUNode):
                abst_networks.add_node(abst.AbsReLUNode("ABST_" + current_node.identifier, current_node,
                                                        self.heuristic, self.params))

            elif isinstance(current_node, nodes.SigmoidNode):
                abst_networks.add_node(abst.AbsSigmoidNode("ABST_" + current_node.identifier, current_node,
                                                           self.refinement_level))

            else:
                raise Exception(f"Node type: {current_node.__class__} not supported")

            current_node = network.get_next_node(current_node)

        ver_start_time = time.perf_counter()
        if isinstance(prop, NeVerProperty):

            input_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
            input_starset = abst.StarSet({input_star})
            current_node = abst_networks.get_first_node()
            output_starset = input_starset
            while current_node is not None:
                time_start = time.perf_counter()
                output_starset = current_node.forward(output_starset)
                time_end = time.perf_counter()

                logger.info(f"Computing starset for layer {current_node.identifier}. Current starset has dimension "
                            f"{len(output_starset.stars)}. Time to compute: {time_end - time_start}s.")

                current_node = abst_networks.get_next_node(current_node)

            out_coef_mat = prop.out_coef_mat
            out_bias_mat = prop.out_bias_mat

        else:
            raise NotImplementedError

        or_verified = []
        for i in range(len(out_coef_mat)):

            verified = True
            for star in output_starset.stars:
                out_coef = out_coef_mat[i]
                out_bias = out_bias_mat[i]
                temp_star = abst.intersect_with_halfspace(star, out_coef, out_bias)
                if not temp_star.check_if_empty():
                    verified = False
                    # print(f"Star {k}: Unsafe")

            or_verified.append(verified)

        final_verified = any(or_verified)

        ver_end_time = time.perf_counter()
        logger.info(f"Verification Result: {final_verified}.")
        logger.info(f"Verification Time: {ver_end_time - ver_start_time}\n")

        return final_verified


class NeverVerificationRef(VerificationStrategy):
    """
    Class used to represent the Never verification strategy with counter example guided refinement.

    Attributes
    ----------
    log_filepath: str
        Filepath for saving the log files of the verification procedure.

    Methods
    ----------
    verify(NeuralNetwork, Property)
        Verify that the neural network of interest satisfy the property given as argument.
    """

    def __init__(self, log_filepath: str):

        self.log_filepath = log_filepath

    def verify(self, network: networks.NeuralNetwork, prop: Property) -> (bool, Optional[Tensor]):

        assert isinstance(network, networks.SequentialNetwork), "Only sequential networks are supported at present"
        abst_networks = abst.AbsSeqNetwork("Abstract Network")

        current_node = network.get_first_node()
        while current_node is not None:

            if isinstance(current_node, nodes.FullyConnectedNode):
                abst_networks.add_node(abst.AbsFullyConnectedNode("ABST_" + current_node.identifier, current_node))

            elif isinstance(current_node, nodes.ReLUNode):
                abst_networks.add_node(abst.AbsReLUNode("ABST_" + current_node.identifier, current_node,
                                                        refinement_flags=[False for m in
                                                                          range(current_node.num_features)]))

            # elif isinstance(current_node, nodes.SigmoidNode):
            #    abst_networks.add_node(abst.AbsSigmoidNode("ABST_" + current_node.identifier, current_node,
            #                                               self.refinement_level))

            else:
                raise NotImplementedError

            current_node = network.get_next_node(current_node)

        with open(self.log_filepath, "w") as log_file:
            areas_log = self.log_filepath.replace(".txt", "_areas.txt")
            with open(areas_log, "w") as area_log_file:

                if isinstance(prop, NeVerProperty):

                    input_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
                    input_starset = abst.StarSet({input_star})
                    current_node = abst_networks.get_first_node()
                    output_starset = input_starset
                    while current_node is not None:
                        time_start = time.perf_counter()
                        output_starset = current_node.forward(output_starset)
                        time_end = time.perf_counter()

                        print(f"Computing starset for layer {current_node.identifier}. Current starset has dimension "
                              f"{len(output_starset.stars)}. Time to compute: {time_end - time_start}s.")
                        log_file.write(f"Computing starset for layer {current_node.identifier}. "
                                       f"Current starset has dimension {len(output_starset.stars)}."
                                       f"Time to compute: {time_end - time_start}s.\n")

                        current_node = abst_networks.get_next_node(current_node)

                    out_coef_mat = prop.out_coef_mat
                    out_bias_mat = prop.out_bias_mat

                else:
                    raise NotImplementedError

                verified = True
                for i in range(len(out_coef_mat)):

                    for star in output_starset.stars:
                        out_coef = out_coef_mat[i]
                        out_bias = out_bias_mat[i]
                        temp_star = abst.intersect_with_halfspace(star, out_coef, out_bias)
                        if not temp_star.check_if_empty():

                            # In this case we trigger the search for the counter example and the following refinement
                            # It could be useful to compute more than 1 sample and to make the mean of the saliency
                            # values?
                            out_sample = temp_star.get_samples(1)[0]
                            input_star = abst.Star(prop.in_coef_mat, prop.in_bias_mat)
                            # Finding a good starting input can be critical for the search. How can be sure that we
                            # take a good starting input?
                            # Could we take a set of starting input and searching in parallel?
                            starting_input = input_star.get_samples(1)[0]
                            # The parameter to determine are max_iter, rate and threshold which correspond to the
                            # max number of gradient step, the learning rate, and the acceptable error respectively.
                            found, ref_in, ref_out = util.input_search(network, out_sample, starting_input,
                                                                       max_iter=1000, rate=0.1, threshold=1e-6)

                            # We need to check if the output is in the unsafe region and if the input is in the input
                            # boundaries defined by the properties. To do so we build two temporary star in order to
                            # leverage the check_alpha_inside method.

                            in_input_set = input_star.check_alpha_inside(ref_in)
                            temp_out_star = abst.Star(out_coef, out_bias)
                            in_unsafe_zone = temp_out_star.check_alpha_inside(ref_out)

                            # At this point we need to analyse the input found to understand what kind of counterexample
                            # we are working with.
                            # If found is false than the input point found is not close to the desired output. Should
                            # we still analize it? With what criterion?

                            if not in_input_set and in_unsafe_zone:
                                # It means that the input point corresponding to the output point of interest it is not
                                # in the input set of the property. This can be considered as a spurious counterexample.
                                pass

                            elif in_input_set and in_unsafe_zone:
                                # In this case the sample is a real counterexample and therefore the property is proved
                                # to be unsafe.
                                verified = False

                            elif in_input_set and not in_unsafe_zone:
                                # In this case the sample is compliant with the property, therefore there is nothing
                                # to do with it. It still means that the search didn't manage to find the correct point
                                # can we do something with this information?
                                pass

                            elif not in_input_set and not in_unsafe_zone:
                                # The point found is not in the input set of the property and the corresponding output
                                # is not in the unsafe zone. Again it is not clear what we can do with it.
                                pass

                            # Once we have checked the counterexample property we need to compute the saliency values.

                            verified = False
                            # print(f"Star {k}: Unsafe")

                log_file.write(f"Verification Result: {verified}.\n")

        return verified


def never2smt(prt: NeVerProperty, input_prefix: str, output_prefix: str, filepath: str):

    with open(filepath, "w") as f:

        input_matrix = prt.in_coef_mat
        input_biases = prt.in_bias_mat
        list_out_matrixes = prt.out_coef_mat
        list_out_biases = prt.out_bias_mat

        for i in range(len(input_matrix[0])):
            f.write(f"(declare-fun {input_prefix}_{i} () Real)\n")

        for i in range(len(list_out_matrixes[0][0])):
            f.write(f"(declare-fun {output_prefix}_{i} () Real)\n")

        for i in range(len(input_matrix)):

            vars_coef = input_matrix[i, :]
            bias = input_biases[i][0]
            s = "("
            for k in range(len(vars_coef)):
                var_coef = vars_coef[k]
                if var_coef != 0:
                    s = s + f"({float(var_coef)} * {input_prefix}_{k})"
                    if k < len(vars_coef) - 1 and any(vars_coef[k + 1:]):
                        s = s + " + "

            s = s + f") <= ({float(bias)})"
            smt_s = processing.ExpressionTreeConverter().build_from_infix(s).as_prefix()
            f.write(f"(assert {smt_s})" + "\n")

        s = ""
        for i in range(len(list_out_matrixes)):

            out_matrix = list_out_matrixes[i]
            out_biases = list_out_biases[i]
            s = s + "("
            for j in range(len(out_matrix)):
                vars_coef = out_matrix[j, :]
                bias = out_biases[j][0]
                s = s + "(("
                for k in range(len(vars_coef)):
                    var_coef = vars_coef[k]
                    if var_coef != 0:
                        s = s + f"({float(var_coef)} * {output_prefix}_{k})"
                        if k < len(vars_coef) - 1 and any(vars_coef[k + 1:]):
                            s = s + " + "

                s = s + f") <= ({float(bias)}))"
                if j < len(out_matrix) - 1:
                    s = s + " & "

            s = s + ")"
            if i < len(list_out_matrixes) - 1:
                s = s + " | "

        smt_s = processing.ExpressionTreeConverter().build_from_infix(s).as_prefix()
        smt_s = smt_s.replace("&", "and")
        smt_s = smt_s.replace("|", "or")
        f.write(f"(assert {smt_s})" + "\n")

def temp_never2smt(prt: NeVerProperty, input_prefix: str, output_prefix: str, filepath: str):

    with open(filepath, "w") as f:

        input_matrix = prt.in_coef_mat
        input_biases = prt.in_bias_mat
        out_matrix = prt.out_coef_mat[0]
        out_biases = prt.out_bias_mat[0]

        for i in range(len(input_matrix[0])):
            f.write(f"(declare-fun {input_prefix}_{i} () Real)\n")

        for i in range(len(out_matrix[0])):
            f.write(f"(declare-fun {output_prefix}_{i} () Real)\n")

        for i in range(len(input_matrix)):

            vars_coef = input_matrix[i, :]
            bias = input_biases[i][0]
            s = "("
            for k in range(len(vars_coef)):
                var_coef = vars_coef[k]
                if var_coef != 0:
                    s = s + f"({float(var_coef)} * {input_prefix}_{k})"
                    if k < len(vars_coef) - 1 and any(vars_coef[k + 1:]):
                        s = s + " + "

            s = s + f") <= ({float(bias)})"
            smt_s = processing.ExpressionTreeConverter().build_from_infix(s).as_prefix()
            f.write(f"(assert {smt_s})" + "\n")

        for i in range(len(out_matrix)):

            vars_coef = out_matrix[i, :]
            bias = out_biases[i][0]
            s = "("
            for k in range(len(vars_coef)):
                var_coef = vars_coef[k]
                if var_coef != 0:
                    s = s + f"({float(var_coef)} * {output_prefix}_{k})"
                    if k < len(vars_coef) - 1 and any(vars_coef[k + 1:]):
                        s = s + " + "

            s = s + f") <= ({float(bias)})"
            smt_s = processing.ExpressionTreeConverter().build_from_infix(s).as_prefix()
            f.write(f"(assert {smt_s})" + "\n")

        smt_s = processing.ExpressionTreeConverter().build_from_infix(s).as_prefix()
        smt_s = smt_s.replace("&", "and")
        smt_s = smt_s.replace("|", "or")
        f.write(f"(assert {smt_s})" + "\n")