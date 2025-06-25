"""
This module defines utility enumerators suppoorting the SSBP verification algorithm
"""

import enum


class RefinementTarget:
    """
    This class represents the refinement target for the verification.

    Attributes
    ----------
    layer_id: str
        Identifier of the target layer
    neuron_idx: int
        Index of the target neuron
    """

    def __init__(self, layer_id: str, neuron_idx: int):
        self.layer_id = layer_id
        self.neuron_idx = neuron_idx

    def __repr__(self):
        return f'({self.layer_id}, {self.neuron_idx})'

    def to_pair(self) -> tuple[str, int]:
        """Procedure to convert the object to a tuple

        Returns
        -------
        tuple[str, int]
            A tuple representation of the object
        """
        return self.layer_id, self.neuron_idx


class NeuronSplit(enum.Enum):
    """
    This class is used as an enumerator to distinguish the two outputs
    of a neuron split (positive or negative)
    """
    NEGATIVE = 0
    POSITIVE = 1


class BoundsDirection(enum.Enum):
    """
    This class is used as an enumerator to choose the direction of
    the bounds computation (forwards or backwards)
    """
    FORWARDS = 0
    BACKWARDS = 1


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
