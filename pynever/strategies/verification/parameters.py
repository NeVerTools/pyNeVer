import abc

from pynever.strategies.verification.ssbp.constants import RefinementStrategy, BoundsBackend, PropagationStrategy


# TODO add documentation

class VerificationParameters(abc.ABC):
    pass


class SSLPVerificationParameters(VerificationParameters):
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


class SSBPVerificationParameters(VerificationParameters):
    def __init__(self,
                 heuristic: RefinementStrategy = RefinementStrategy.LOWEST_APPROX,
                 bounds: BoundsBackend = BoundsBackend.SYMBOLIC,
                 propagation: PropagationStrategy = PropagationStrategy.ADAPTIVE,
                 timeout: int = 600):
        self.heuristic = heuristic
        self.bounds = bounds
        self.propagation = propagation
        self.timeout = timeout
