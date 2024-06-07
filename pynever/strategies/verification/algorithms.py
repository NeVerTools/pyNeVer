import abc
import copy
import logging
import time

import pynever.networks as networks
import pynever.strategies.bp.bounds_manager as bm
import pynever.strategies.verification.search as sf
from pynever.strategies.abstraction.networks import AbsSeqNetwork
from pynever.strategies.abstraction.star import StarSet, Star
from pynever.strategies.verification import LOGGER
from pynever.strategies.verification.parameters import NeverVerificationParameters, SearchVerificationParameters, \
    PropagationStrategy
from pynever.strategies.verification.properties import NeverProperty
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
    verify(NeuralNetwork, NeverProperty)
        Verify that the neural network of interest satisfy the property given as argument.

    """

    def __init__(self, params: NeverVerificationParameters):

        self.params = params
        self.logger = logging.getLogger(LOGGER)

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
            bound_manager = bm.BoundsManager()
            _, _, self.layers_bounds = bound_manager.compute_bounds_from_property(network, prop)
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
            self.counterexample_stars = NeverVerification.get_counterexample_stars(prop, unsafe_stars)
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


class SearchVerification(VerificationStrategy):
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

    def __init__(self, parameters: SearchVerificationParameters):
        self.parameters = parameters
        self.logger = logging.getLogger(LOGGER)

    def init_search(self, network: networks.SequentialNetwork, prop: NeverProperty):
        """
        Initialize the search algorithm and compute the starting values for
        the bounds, the star and the target

        """

        in_star = prop.to_input_star()
        in_star.ref_layer = 0

        return (in_star, bm.BoundsManager.get_input_bounds(prop),
                sf.get_bounds(network, prop, self.parameters.bounds), bm.net2list(network))

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
            in_star, input_bounds, nn_bounds, net_list = self.init_search(network, prop)
        else:
            raise NotImplementedError('Only SequentialNetwork objects are supported at present')

        # Frontier is a stack of tuples (Star, AbstractBounds)
        frontier = [(in_star, nn_bounds)]
        stop_flag = False

        # Start timer
        timer = 0
        start_time = time.perf_counter()

        # Flag to update bounds
        update = True

        while len(frontier) > 0 and not stop_flag:
            current_star, nn_bounds = frontier.pop()
            prev_layer = current_star.ref_layer

            if self.parameters.propagation == PropagationStrategy.STAR_LP:
                intersects, unsafe_stars = sf.intersect_star_lp(current_star, net_list, nn_bounds, prop)
            elif self.parameters.propagation == PropagationStrategy.BOUNDS:
                intersects, unsafe_stars = sf.intersect_symb_lp(input_bounds, nn_bounds, prop)
            else:
                raise NotImplementedError('Intersection strategy not supported')

            if intersects:
                # If new target is None there is no more refinement to do
                target, current_star = sf.get_next_target(self.parameters.heuristic, current_star, net_list)

                # Update bounds once per layer
                if current_star.ref_layer > prev_layer:
                    update = True
                else:
                    update = False

                if target is None:
                    # Nothing else to split, or
                    # Found a counterexample
                    cex = sf.get_counterexample(unsafe_stars, prop)
                    return False, cex

                else:
                    # We cannot conclude anything at this point.
                    # Split the current branch according to the target
                    frontier.extend(
                        sf.split_star(current_star, target, net_list, nn_bounds, update)
                    )

            else:
                """This branch is safe, no refinement needed"""

            timer += (time.perf_counter() - start_time)
            if timer > self.parameters.timeout:
                stop_flag = True
            else:
                start_time = time.perf_counter()

        if stop_flag:
            return False, None
        else:
            return True, None
