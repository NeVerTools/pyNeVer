import abc
import copy
import time

import numpy as np

import pynever.networks as networks
import pynever.strategies.bounds_propagation.bounds_manager as bm
import pynever.strategies.verification.ssbp.intersection as ssbp_intersect
import pynever.strategies.verification.ssbp.propagation as ssbp_prop
import pynever.strategies.verification.ssbp.split as ssbp_split
from pynever.strategies.abstraction.networks import AbsSeqNetwork
from pynever.strategies.abstraction.star import StarSet, Star, ExtendedStar
from pynever.strategies.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions
from pynever.strategies.verification import LOGGER
from pynever.strategies.verification.parameters import SSLPVerificationParameters, SSBPVerificationParameters
from pynever.strategies.verification.properties import NeverProperty
from pynever.strategies.verification.ssbp.constants import BoundsBackend, PropagationStrategy, RefinementTarget, \
    RefinementStrategy
from pynever.tensors import Tensor


class VerificationStrategy(abc.ABC):
    """
    An abstract class used to represent a Verification Strategy.

    Methods
    ----------
    verify(NeuralNetwork, NeverProperty)
        Verify that the neural network of interest satisfy the property given as argument
        using a verification strategy determined in the concrete children.

    """

    @abc.abstractmethod
    def verify(self, network: networks.NeuralNetwork, prop: NeverProperty) -> tuple[bool, Tensor | None]:
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

        raise NotImplementedError


class SSLPVerification(VerificationStrategy):
    """
    Class used to represent the Never verification strategy.

    Attributes
    ----------

    Methods
    ----------
    verify(NeuralNetwork, NeverProperty)
        Verify that the neural network of interest satisfy the property given as argument.

    """

    def __init__(self, params: SSLPVerificationParameters):

        self.params = params
        self.logger = LOGGER

        self.counterexample_stars = None
        self.layers_bounds = {}

    def verify(self, network: networks.NeuralNetwork, prop: NeverProperty) -> tuple[bool, Tensor | None]:
        """
        Entry point for the verification algorithm for a network and a property

        Parameters
        ----------
        network : NeuralNetwork
            The network model in the internal representation
        prop : NeverProperty
            The property specification

        Returns
        ----------
        bool
            True if the network is safe, False otherwise

        """

        self.counterexample_stars = None

        if not isinstance(network, networks.SequentialNetwork):
            raise NotImplementedError('Only Sequential Networks are currently supported by NeverVerification')

        abst_network = AbsSeqNetwork(network, self.params)

        ver_start_time = time.perf_counter()

        # Compute symbolic bounds first. If the network architecture or the property
        # does not have a corresponding bound propagation method we skip the computation
        # TODO remove assert in bound propagation
        try:
            manager = bm.BoundsManager()
            self.layers_bounds = manager.compute_bounds_from_property(network, prop)

        except AssertionError:
            self.logger.warning(f"Warning: Bound propagation unsupported")
            self.layers_bounds = {}

        input_star = Star(prop.in_coef_mat, prop.in_bias_mat)
        input_starset = StarSet({input_star})

        output_starset = abst_network.forward(input_starset)
        if not isinstance(output_starset, StarSet):
            raise Exception

        # Now we check the intersection of the output starset with the output halfspaces defined by the output
        # constraints of our property of interest. We recall that the property is satisfiable if there exist at least
        # one non-void intersection between the output starset and the halfspaces and SAFE = NOT SAT.

        unsafe_stars = []
        all_empty = []
        for i in range(len(prop.out_coef_mat)):

            empty = True
            for star in output_starset.stars:
                out_coef = prop.out_coef_mat[i]
                out_bias = prop.out_bias_mat[i]
                temp_star = star.intersect_with_halfspace(out_coef, out_bias)
                if not temp_star.check_if_empty():
                    empty = False
                    if self.params.heuristic == 'complete':
                        unsafe_stars.append(temp_star)

            all_empty.append(empty)

        is_satisfied = not all(all_empty)

        counterexample: Tensor | None = None
        if len(unsafe_stars) > 0:
            self.counterexample_stars = SSLPVerification.get_counterexample_stars(prop, unsafe_stars)
            counterexample = self.counterexample_stars[0].get_samples(num_samples=1)[0]

        ver_end_time = time.perf_counter()

        self.logger.info(f"The property is satisfiable: {is_satisfied}.")
        self.logger.info(f"Verification Time: {ver_end_time - ver_start_time}\n")

        return not is_satisfied, counterexample

    @staticmethod
    def get_counterexample_stars(prop: NeverProperty, unsafe_stars: list[Star]):

        counterexample_stars = []
        for unsafe_star in unsafe_stars:
            temp_star = Star(prop.in_coef_mat, prop.in_bias_mat)
            temp_star.predicate_matrix = copy.deepcopy(unsafe_star.predicate_matrix)
            temp_star.predicate_bias = copy.deepcopy(unsafe_star.predicate_bias)
            counterexample_stars.append(temp_star)

        return counterexample_stars


class SSBPVerification(VerificationStrategy):
    """
    Class used to represent the search-based verification strategy. It employs
    star propagation with Symbolic Bounds Propagation and an abstraction-refinement
    loop for better readability, structure and functionality

    Attributes
    ----------
    parameters : dict
        The parameters to guide the search algorithm

    Methods
    ----------
    verify(SequentialNetwork, NeVerProperty)
        Verify that the neural network of interest satisfies the property given as argument.

    """

    def __init__(self, parameters: SSBPVerificationParameters):
        self.parameters = parameters
        self.logger = LOGGER

    def init_search(self, network: networks.SequentialNetwork, prop: NeverProperty) \
            -> tuple[ExtendedStar, HyperRectangleBounds, dict]:
        """
        Initialize the search algorithm and compute the
        starting values for the bounds, the star and the target

        """

        star0 = prop.to_input_star()
        star0 = ExtendedStar(LinearFunctions(star0.predicate_matrix, star0.predicate_bias),
                             LinearFunctions(star0.basis_matrix, star0.center))

        bounds = self.get_bounds(network, prop, self.parameters.bounds)
        star1 = ssbp_prop.propagate_and_init_star_before_relu_layer(star0, bounds, network, skip=False)

        return star1, bm.BoundsManager.get_input_bounds(prop), bounds

    def get_bounds(self, nn: networks.SequentialNetwork, prop: NeverProperty, strategy: BoundsBackend) -> dict:
        """
        This function gets the bounds of the neural network for the given property
        of interest. The bounds are computed based on a strategy that allows to
        plug and play different bound propagation algorithms

        Parameters
        ----------
        nn : SequentialNetwork
            The neural network of interest in the internal representation
        prop : NeVerProperty
            The property of interest
        strategy : str
            The strategy to use for computing the bounds [symbolic, ...]

        Returns
        ----------
        dict
            The dictionary of the bounds computed by the Bounds Manager

        """

        match strategy:
            case BoundsBackend.SYMBOLIC:
                return bm.BoundsManager().compute_bounds_from_property(nn, prop)

            case _:
                # TODO add more strategies
                raise NotImplementedError

    def compute_intersection(self, star: ExtendedStar, nn_bounds: dict, nn: networks.SequentialNetwork,
                             prop: NeverProperty) -> tuple[bool, list[float]]:
        """

        """

        match self.parameters.propagation:
            case PropagationStrategy.STAR_LP:
                raise NotImplementedError('To refactor')

            case PropagationStrategy.ADAPTIVE:
                return ssbp_intersect.intersect_adaptive(star, nn, nn_bounds, prop)

            case _:
                raise NotImplementedError('Intersection strategy not supported')

    def get_next_target(self, star: ExtendedStar, nn_bounds: dict, network: networks.SequentialNetwork) \
            -> tuple[RefinementTarget | None, ExtendedStar]:
        """

        """

        match self.parameters.heuristic:

            case RefinementStrategy.SEQUENTIAL:
                raise NotImplementedError('To refactor')

            case RefinementStrategy.LOWEST_APPROX:
                return ssbp_split.get_target_sequential(star, nn_bounds, network)

            case _:
                raise NotImplementedError('Only sequential refinement supported')

    def verify(self, network: networks.NeuralNetwork, prop: NeverProperty) -> tuple[bool, Tensor | None]:
        """
        Entry point for the abstraction-refinement search algorithm

        Parameters
        ----------
        network : NeuralNetwork
            The network model in the internal representation
        prop : Property
            The property specification

        Returns
        ----------
        bool, Optional[Tensor]
            True if the network is safe, False otherwise. If the result is False and the
            search is complete it also returns a counterexample

        """

        if isinstance(network, networks.SequentialNetwork):
            in_star, input_bounds, in_bounds = self.init_search(network, prop)
        else:
            raise NotImplementedError('Only SequentialNetwork objects are supported at present')

        self.logger.info(f"Inactive neurons: {in_bounds['stability_info'][bm.StabilityInfo.INACTIVE]}\n"
                         f"  Active neurons: {in_bounds['stability_info'][bm.StabilityInfo.ACTIVE]}\n\n")

        # Frontier is a stack of tuples (ExtendedStar, dict)
        frontier = [(in_star, in_bounds)]
        stop_flag = False

        # Start timer
        timer = 0
        start_time = time.perf_counter()

        while len(frontier) > 0 and not stop_flag:

            # self.logger.info(f"{datetime.datetime.now()} Start of the loop")
            current_star, nn_bounds = frontier.pop()

            intersects, candidate_cex = self.compute_intersection(current_star, nn_bounds, network, prop)
            # self.logger.info(f"{datetime.datetime.now()} Intersection computed")

            if intersects:
                # Check if the answer is a valid counter-example
                if ssbp_intersect.check_valid_counterexample(candidate_cex, network, prop):

                    # Found a counterexample. Can stop here
                    self.logger.info('Counterexample in branch {}.\n'
                                     'Execution time: {:.5f} s'.format(current_star.fixed_neurons, timer))
                    return False, Tensor(np.array(candidate_cex))

                else:
                    # We cannot conclude anything at this point. Should try to split.
                    # Pick the neuron to split according to the heuristic
                    target, current_star = self.get_next_target(current_star, nn_bounds, network)

                    if target is not None:
                        # Split the current branch according to the target
                        frontier.extend(
                            ssbp_split.split_star_opt(current_star, target, network, nn_bounds)
                        )
                        # self.logger.info(f"{datetime.datetime.now()} Split computed")
                    else:
                        # There is no more refinement to do, i.e., all neurons have been fixed.
                        # We can end up here because the bounds might not be aware that all neurons have been fixed.
                        # So there can be some overapproximation.
                        # We should detect and throw more exact intersection check
                        # TODO we stopped here
                        intersects, unsafe_stars = self.compute_intersection(current_star, nn_bounds, network, prop)
                        target, current_star = self.get_next_target(current_star, nn_bounds, network)

                        raise Exception("This point should not be reachable")

            else:
                # This branch is safe, no refinement needed
                self.logger.info(f"Branch {current_star.fixed_neurons} is safe")

            timer += (time.perf_counter() - start_time)
            if timer > self.parameters.timeout:
                stop_flag = True
            else:
                start_time = time.perf_counter()

        if stop_flag:
            self.logger.info(' ----- TIMEOUT -----\nExecution time: {:.5f} s'.format(timer))
            return False, None
        else:
            self.logger.info(' ----- SAFE -----\nExecution time: {:.5f} s'.format(timer))
            return True, None
