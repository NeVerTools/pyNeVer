import abc

from pynever.strategies.verification.ssbp.constants import (
    RefinementStrategy,
    BoundsBackend,
    IntersectionStrategy,
    BoundsDirection
)


# TODO add documentation

class VerificationParameters(abc.ABC):
    pass


class SSLPVerificationParameters(VerificationParameters):
    def __init__(self, heuristic: str = 'complete', neurons_to_refine: list | None = None):
        if heuristic not in ['overapprox', 'complete', 'mixed']:
            raise Exception(f'Selected heuristic {heuristic} is not valid')

        if heuristic == 'mixed' and neurons_to_refine is None:
            raise Exception('neurons_to_refine cannot be None if heuristic is mixed')

        self.heuristic = heuristic
        self.neurons_to_refine = neurons_to_refine
        self.compute_areas = True if self.heuristic == 'mixed' else False


class SSBPVerificationParameters(VerificationParameters):
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
