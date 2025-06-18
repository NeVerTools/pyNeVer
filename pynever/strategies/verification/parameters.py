"""
This module contains the classes used to define verification parameters for different strategies.
The class ``VerificationParameters`` is the abstract class from which derive the concrete classes
``SSLPVerificationParameters`` and ``SSBPVerificationParameters``.
"""

import abc

from pynever.strategies.verification.ssbp.constants import (
    RefinementStrategy,
    BoundsBackend,
    IntersectionStrategy,
    BoundsDirection
)


class VerificationParameters(abc.ABC):
    """
    This class is the abstract base for defining the verification parameters
    related to a specific verification algorithm
    """
    pass


class SSLPVerificationParameters(VerificationParameters):
    """
    This class defines the parameters for the SSLP verification algorithm.

    Attributes
    ----------
    heuristic: str
        The verification heuristic (complete, mixed or approximate)
    neurons_to_refine: int
        The number of neurons to refine in the mixed setting
    """

    def __init__(self, heuristic: str = 'complete', neurons_to_refine: list | None = None):
        if heuristic not in ['overapprox', 'complete', 'mixed']:
            raise Exception(f'Selected heuristic {heuristic} is not valid')

        if heuristic == 'mixed' and neurons_to_refine is None:
            raise Exception('neurons_to_refine cannot be None if heuristic is mixed')

        self.heuristic = heuristic
        self.neurons_to_refine = neurons_to_refine
        self.compute_areas = True if self.heuristic == 'mixed' else False


class SSBPVerificationParameters(VerificationParameters):
    """
    This class defines the parameters for the SSBP verification algorithm.

    Attributes
    ----------
    heuristic: RefinementStrategy
        The refinement heuristic to apply
    bounds: BoundsBackend
        The bounds backend structure to use
    bounds_direction: BoundsDirection
        The direction to compute the bounds
    intersection: IntersectionStrategy
        The intersection strategy to use
    timeout: int
        The timeout in seconds
    """

    def __init__(self,
                 heuristic: RefinementStrategy = RefinementStrategy.INPUT_BOUNDS_CHANGE,
                 bounds: BoundsBackend = BoundsBackend.SYMBOLIC,
                 bounds_direction: BoundsDirection = BoundsDirection.FORWARDS,
                 intersection: IntersectionStrategy = IntersectionStrategy.ADAPTIVE,
                 timeout: int = 60):
        self.heuristic = heuristic
        self.bounds = bounds
        self.bounds_direction = bounds_direction
        self.intersection = intersection
        self.timeout = timeout
