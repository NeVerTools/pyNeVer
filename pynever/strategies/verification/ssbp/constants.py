import enum


class RefinementTarget:
    """
    This class represents the refinement target for the verification.

    """

    def __init__(self, layer: int, neuron: int):
        self.layer_idx = layer
        self.neuron_idx = neuron

    def __repr__(self):
        return f'({self.layer_idx}, {self.neuron_idx})'

    def to_pair(self):
        return self.layer_idx, self.neuron_idx


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

    """

    SYMBOLIC = 1


class RefinementStrategy(enum.Enum):
    """
    This class is used as an enumerator to represent different refinement strategies.

    - sequential refinement refines each neuron in the order they appear in the network

    """

    SEQUENTIAL = 1
    LOWEST_APPROX = 2


class PropagationStrategy(enum.Enum):
    """
    This class is used as an enumerator to represent different propagation strategies.

    - star with LP propagates the star abstraction
    - bounds propagates the symbolic bounds

    """

    STAR_LP = 1
    ADAPTIVE = 2
