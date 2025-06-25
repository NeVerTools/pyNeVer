"""
This module contains the classes used to implement the different verification strategies. Following the strategy pattern,
the class ``VerificationStrategy`` is the abstract class providing a ``verify`` method that requires a neural network and
a property. The concrete classes ``SSLPVerification`` and ``SSBPVerification`` implement the verification strategies based
on starsets and symbolic bounds propagation, respectively.
"""

import abc
import copy
import datetime
import time

import torch

import pynever.networks as networks
import pynever.strategies.verification.ssbp.intersection as ssbp_intersect
import pynever.strategies.verification.ssbp.propagation as ssbp_prop
import pynever.strategies.verification.ssbp.split as ssbp_split
from pynever.networks import NeuralNetwork
from pynever.strategies.abstraction.bounds_propagation import ReLUStatus
from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.abstraction.bounds_propagation.manager import BoundsManager
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.strategies.abstraction.networks import AbsSeqNetwork
from pynever.strategies.abstraction.star import StarSet, Star, ExtendedStar
from pynever.strategies.verification import VERIFICATION_LOGGER
from pynever.strategies.verification.parameters import (
    VerificationParameters,
    SSLPVerificationParameters,
    SSBPVerificationParameters
)
from pynever.strategies.verification.properties import NeverProperty
from pynever.strategies.verification.ssbp.constants import (
    BoundsBackend,
    IntersectionStrategy,
    RefinementTarget,
    RefinementStrategy
)
from pynever.strategies.verification.statistics import VerboseBounds


class VerificationStrategy(abc.ABC):
    """
    An abstract class used to represent a Verification Strategy.

    Attributes
    ----------
    parameters: VerificationParameters
        Parameters to guide the verification algorithm
    logger: Logger
        Custom logger for the verification package
    """

    def __init__(self, parameters: VerificationParameters):
        self.parameters = parameters
        self.logger = VERIFICATION_LOGGER

    @abc.abstractmethod
    def verify(self, network: networks.NeuralNetwork, prop: NeverProperty) -> tuple[bool, torch.Tensor | None]:
        """
        Verify that the neural network of interest satisfies the property given as argument
        using a verification strategy determined in the concrete children.

        Parameters
        ----------
        network: NeuralNetwork
            The neural network to train.
        prop: NeverProperty
            The property which the neural network must satisfy.

        Returns
        ----------
        tuple[bool, torch.Tensor | None]
            True is the neural network satisfies the property, False otherwise.
            If False, also returns a counterexample
        """
        raise NotImplementedError


class SSLPVerification(VerificationStrategy):
    """
    Class used to represent the SSLP (Star Sets with Linear Programming) verification strategy.

    Attributes
    ----------
    counterexample_stars: list[Star]
        List of :class:`~pynever.strategies.abstraction.star.Star` objects containing a counterexample
    layers_bounds: dict
        Bounds obtained through bounds propagation to support verification
    """

    def __init__(self, params: SSLPVerificationParameters):
        super().__init__(params)

        self.counterexample_stars = None
        self.layers_bounds = {}

    def verify(self, network: networks.NeuralNetwork, prop: NeverProperty) -> tuple[bool, torch.Tensor | None]:
        """
        Entry point for the verification algorithm for a network and a property

        Parameters
        ----------
        network: NeuralNetwork
            The network model in the internal representation
        prop: NeverProperty
            The property specification

        Returns
        ----------
        tuple[bool, torch.Tensor | None]
            True is the neural network satisfies the property, False otherwise.
            If False, also returns a counterexample
        """
        self.counterexample_stars = None

        if not isinstance(network, networks.SequentialNetwork):
            raise NotImplementedError('Only Sequential Networks are currently supported by NeverVerification')

        abst_network = AbsSeqNetwork(network, self.parameters)

        ver_start_time = time.perf_counter()
        # Compute symbolic bounds first. If the network architecture or the property
        # does not have a corresponding bound propagation method we skip the computation
        try:
            manager = BoundsManager(network, prop)
            self.layers_bounds = manager.compute_bounds()

        except AssertionError:
            self.logger.warning(f"Warning: Bound propagation unsupported")
            self.layers_bounds = {}

        abst_network.set_bounds(self.layers_bounds.numeric_pre_bounds)

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
                    if self.parameters.heuristic == 'complete':
                        unsafe_stars.append(temp_star)

            all_empty.append(empty)

        sat = not all(all_empty)

        counterexample: torch.Tensor | None = None
        if len(unsafe_stars) > 0:
            self.counterexample_stars = SSLPVerification.get_counterexample_stars(prop, unsafe_stars)
            counterexample = self.counterexample_stars[0].get_samples(num_samples=1)[0]

        ver_end_time = time.perf_counter()

        self.logger.info(f"The property is satisfiable: {sat}")
        self.logger.info(f"Verification result:         {'Unsafe' if sat else 'Verified'}")
        self.logger.info(f"Verification Time:           {ver_end_time - ver_start_time:.4f} s\n")

        return not sat, counterexample

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
    Class used to represent the SSBP (Star Sets with Bounds Propagation) verification strategy.
    It uses star propagation with Symbolic Bounds Propagation and an abstraction-refinement
    loop for better readability, structure and functionality.

    Attributes
    ----------
    network: NeuralNetwork
        The neural network to verify
    prop: NeverProperty
        The property specification
    """

    def __init__(self, parameters: SSBPVerificationParameters):
        super().__init__(parameters)

        self.network = None
        self.prop = None

    def init_search(self, network: networks.NeuralNetwork, prop: NeverProperty) \
            -> tuple[ExtendedStar, HyperRectangleBounds, VerboseBounds]:
        """
        Initialize the search algorithm and compute the starting values for the bounds and the star.

        Parameters
        ----------
        network: NeuralNetwork
            The neural network in use
        prop: NeverProperty
            The property specification

        Returns
        -------
        tuple[ExtendedStar, HyperRectangleBounds, VerboseBounds]
            The starting values for the star and the bounds.
        """
        self.network = network
        self.prop = prop

        star0 = self.prop.to_star()
        star0 = ExtendedStar(LinearFunctions(star0.predicate_matrix, star0.predicate_bias),
                             LinearFunctions(star0.basis_matrix, star0.center))

        bounds = self.get_bounds()
        star1 = ssbp_prop.propagate_and_init_star_before_relu_layer(star0, bounds, self.network, skip=False)

        return star1, self.prop.to_numeric_bounds(), bounds

    def get_bounds(self) -> VerboseBounds:
        """
        This method gets the bounds of the neural network for the given property
        of interest. The bounds are computed based on a strategy that allows to
        plug and play different bound propagation algorithms

        Returns
        -------
        VerboseBounds
            The collection of the bounds computed by the Bounds Manager
        """
        match self.parameters.bounds:
            case BoundsBackend.SYMBOLIC:
                return BoundsManager(self.network, self.prop).compute_bounds()

            case _:
                # TODO add more strategies
                raise NotImplementedError

    def compute_intersection(self, star: ExtendedStar, nn_bounds: VerboseBounds) -> tuple[bool, torch.Tensor]:
        """
        This method computes the intersection between a star and the output property
        using the intersection algorithm specified by the parameters

        Parameters
        ----------
        star: ExtendedStar
            The :class:`~pynever.strategies.abstraction.star.ExtendedStar` object containing the star to intersect
        nn_bounds: VerboseBounds
            The bounds obtained through bounds propagation

        Returns
        -------
        tuple[bool, torch.Tensor]
            The result of the intersection. If True, a counterexample is returned too.
        """
        match self.parameters.intersection:
            case IntersectionStrategy.STAR_LP:
                return ssbp_intersect.intersect_star_lp(star, self.prop, self.network, nn_bounds)

            case IntersectionStrategy.ADAPTIVE:
                intersects, candidate = ssbp_intersect.intersect_adaptive(star, self.network, nn_bounds, self.prop)
                cex = None if len(candidate) == 0 else torch.Tensor(candidate)

                return intersects, cex

            case _:
                raise NotImplementedError('Intersection strategy not supported')

    def get_next_target(self, star: ExtendedStar, nn_bounds: VerboseBounds) \
            -> tuple[RefinementTarget | None, ExtendedStar]:
        """
        This method computes the next refinement target for the verification algorithm
        based on the strategy specified by the parameters.

        Parameters
        ----------
        star: ExtendedStar
            The :class:`~pynever.strategies.abstraction.star.ExtendedStar` object containing the star to refine
        nn_bounds: VerboseBounds
            The bounds obtained through bounds propagation.

        Returns
        -------
        tuple[RefinementTarget | None, ExtendedStar]
            The next refinement target and the extended star to refine. If no more refinement is needed, None is returned.
        """
        match self.parameters.heuristic:

            case RefinementStrategy.SEQUENTIAL:
                return ssbp_split.get_target_sequential(star, nn_bounds, self.network)

            case RefinementStrategy.LOWEST_APPROX:
                return ssbp_split.get_target_lowest_overapprox(star, nn_bounds)

            case RefinementStrategy.LOWEST_APPROX_CURRENT_LAYER:
                return ssbp_split.get_target_lowest_overapprox_current_layer(star, nn_bounds, self.network)

            case RefinementStrategy.INPUT_BOUNDS_CHANGE:
                return ssbp_split.get_target_most_input_change(star, nn_bounds, self.network, self.parameters)

            case _:
                raise NotImplementedError('Only sequential refinement supported')

    def verify(self, network: networks.NeuralNetwork, prop: NeverProperty) -> tuple[bool, torch.Tensor | None]:
        """
        Entry point for the abstraction-refinement search algorithm

        Parameters
        ----------
        network: NeuralNetwork
            The network model in the internal representation
        prop: NeverProperty
            The property specification

        Returns
        ----------
        tuple[bool, torch.Tensor | None]
            True if the network is safe, False otherwise. If the result is False and the
            search is complete it also returns a counterexample
        """
        # Start timer
        timer = 0
        start_time = time.perf_counter()

        in_star, input_num_bounds, input_symb_bounds = self.init_search(network, prop)

        n_unstable = input_symb_bounds.statistics.count_unstable()
        stable_ratio = (input_symb_bounds.statistics.stability_info['stable_count'] /
                        (input_symb_bounds.statistics.stability_info['stable_count'] + n_unstable))
        self.logger.info(f"Started {datetime.datetime.now()}\n"
                         f"Inactive neurons: {input_symb_bounds.statistics.stability_info[ReLUStatus.INACTIVE]}\n"
                         f"  Active neurons: {input_symb_bounds.statistics.stability_info[ReLUStatus.ACTIVE]}\n"
                         f"    Stable count: {input_symb_bounds.statistics.stability_info['stable_count']}\n"
                         f"    Stable ratio: {stable_ratio}\n"
                         f"\n")

        # Frontier is a stack of tuples (ExtendedStar, dict)
        frontier = [(in_star, input_symb_bounds)]
        stop_flag = False

        node_counter = 0

        while len(frontier) > 0 and not stop_flag:

            current_star, nn_bounds = frontier.pop()
            self.logger.info(f"Node {node_counter}. Frontier size {len(frontier) + 1}. "
                             f"Depth {len(current_star.fixed_neurons)}. "
                             f"Stable count {nn_bounds.statistics.stability_info['stable_count']}")

            intersects, candidate_cex = self.compute_intersection(current_star, nn_bounds)

            if intersects:
                # Check if the answer is a valid counter-example
                if ssbp_intersect.check_valid_counterexample(candidate_cex, self.network, prop):

                    # Found a counterexample. Can stop here
                    self.logger.info('Counterexample in branch {}.\n'
                                     'Explored nodes {}.\n'
                                     'Execution time: {:.5f} s'.format(current_star.fixed_neurons,
                                                                       node_counter, timer))
                    return False, candidate_cex

                else:
                    # We cannot conclude anything at this point. Should try to split.
                    # Pick the neuron to split according to the heuristic
                    target, current_star = self.get_next_target(current_star, nn_bounds)

                    if target is not None:
                        # Split the current branch according to the target
                        frontier.extend(
                            ssbp_split.split_star_opt(current_star, target, self.network, nn_bounds, self.parameters)
                        )

                    else:
                        # There is no more refinement to do, i.e., all neurons have been fixed.
                        # We can end up here because the bounds might not be aware that all neurons have been fixed.
                        # So there can be some over-approximation.
                        # We should detect and throw more exact intersection check
                        input_num_bounds = nn_bounds.numeric_pre_bounds[self.network.get_first_node().identifier]

                        self.logger.info(f"\tBranch {current_star.fixed_neurons} is inconsistent with bounds, "
                                         f"input {input_num_bounds.get_lower()} {input_num_bounds.get_upper()}")

                        raise Exception("This point should not be reachable")

            else:
                # This branch is safe, no refinement needed
                self.logger.info(f"\tBranch {current_star.fixed_neurons} is safe")

            timer += (time.perf_counter() - start_time)
            if timer > self.parameters.timeout:
                stop_flag = True
            else:
                start_time = time.perf_counter()

            node_counter += 1

        if stop_flag:
            self.logger.info(' ----- TIMEOUT -----\nExecution time: {:.5f} s'.format(timer))
            return False, None

        else:
            self.logger.info(' ----- SAFE -----\n'
                             'Explored nodes {}.\n'
                             'Execution time: {:.5f} s'.format(node_counter, timer))
            return True, None
