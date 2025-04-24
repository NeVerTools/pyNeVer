import logging
from enum import Enum

BOUNDS_LOGGER = logging.getLogger("pynever.strategies.bounds_propagation")


class ReLUStatus(Enum):
    """This enumerator registers the status of a ReLU neuron

    ACTIVE means that the input is positive, i.e., ReLU acts as identity
    INACTIVE means that the input is negative, i.e., ReLU outputs zero
    UNSTABLE means that the input is both positive and negative

    """
    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2
