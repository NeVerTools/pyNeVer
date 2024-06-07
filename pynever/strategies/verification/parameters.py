import abc
import enum


class NeuronState(enum.Enum):
    """
    This class is used as an enumerator to represent the state of a neuron.
    The possible states are:

    - positive stable when the lower bound is greater than zero
    - negative stable when the upper bound is lower than zero
    - unstable when the upper bound is greater than zero and the lower bound lower than zero

    """

    POSITIVE_STABLE = 1
    NEGATIVE_STABLE = -1
    UNSTABLE = 0


class BoundsBackend(enum.Enum):
    """
    This class is used as an enumerator to represent different bounds propagation
    strategies.

    - symbolic bounds propagation from the Venus2 verifier
    - autoLiRPA propagation from the α,β-CROWN (alpha-beta-CROWN) verifier

    """

    SYMBOLIC = 1
    LIRPA = 2


class RefinementStrategy(enum.Enum):
    """
    This class is used as an enumerator to represent different refinement strategies.

    - sequential refinement refines each neuron in the order they appear in the network

    """

    SEQUENTIAL = 1


class PropagationStrategy(enum.Enum):
    """
    This class is used as an enumerator to represent different propagation strategies.

    - star with LP propagates the star abstraction
    - bounds propagates the symbolic bounds

    """

    STAR_LP = 1
    BOUNDS = 2


# TODO add documentation

class VerificationParameters(abc.ABC):
    raise NotImplementedError


class NeverVerificationParameters(VerificationParameters):
    def __init__(self, heuristic: str, neurons_to_refine: list | None = None,
                 approx_levels: int | list[int] | None = None):
        if heuristic not in ['overapprox', 'complete', 'mixed']:
            raise Exception(f'Selected heuristic {heuristic} is not valid')

        if heuristic == 'mixed' and neurons_to_refine is None:
            raise Exception('neurons_to_refine cannot be None if heuristic is mixed')

        # TODO add checks for approx_levels if needed

        self.heuristic = heuristic
        self.neurons_to_refine = neurons_to_refine
        self.approx_levels = approx_levels


class SearchVerificationParameters(VerificationParameters):
    def __init__(self,
                 heuristic: RefinementStrategy = RefinementStrategy.SEQUENTIAL,
                 bounds: BoundsBackend = BoundsBackend.SYMBOLIC,
                 propagation: PropagationStrategy = PropagationStrategy.STAR_LP,
                 timeout: int = 300):
        self.heuristic = heuristic
        self.bounds = bounds
        self.propagation = propagation
        self.timeout = timeout
