import enum


class RefinementTarget:
    """
    This class represents the refinement target for the verification.

    """

    def __init__(self, layer_id: str, neuron_n: int):
        self.layer_id = layer_id
        self.neuron_idx = neuron_n

    def __repr__(self):
        return f'({self.layer_id}, {self.neuron_idx})'

    def to_pair(self):
        return self.layer_id, self.neuron_idx


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
    - lowest approx selects the neuron that contributes with the lowest approximation
    - lowest approx - current layer selects the neuron that contributes with the lowest approximation in the layer
    - input bounds change selects the neuron that contributes most to change the input bounds when refined

    """

    SEQUENTIAL = 1
    LOWEST_APPROX = 2
    LOWEST_APPROX_CURRENT_LAYER = 3
    INPUT_BOUNDS_CHANGE = 4


class IntersectionStrategy(enum.Enum):
    """
    This class is used as an enumerator to represent different intersection strategies.

    - star with LP propagates the star abstraction and uses an LP to check for an intersection
    - adaptive uses different (precise or approximate) intersection checks based on the state of the search

    """

    STAR_LP = 1
    ADAPTIVE = 2
