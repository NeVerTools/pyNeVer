import abc

from pynever.strategies.verification.ssbp.constants import RefinementStrategy, BoundsBackend, IntersectionStrategy, \
    BoundsDirection


# TODO add documentation

class VerificationParameters(abc.ABC):
    pass


class SSLPVerificationParameters(VerificationParameters):
    """
    A class to hold the parameters for the SSLP verification.

    Attributes
    ----------
    heuristic : str
        Heuristic used to control the refinement level of the abstraction.
        It can be one of the following:
        - complete: all the neurons are processed with a precise abstraction
        - mixed: a given number of neurons is processed with a precise abstraction
        - overapprox: all the neurons are processed with a coarse abstraction

    neurons_to_refine : list[int], optional
        List of the neurons to process with a precise abstraction when using the
        mixed heuristic

    approx_levels : int | list[int], optional
        Refinement level for the s-shaped functions (a bigger level is more accurate).
        If it is a single int then that refinement level is applied to all the layers,
        otherwise it is a list containing the refinement levels for each layer

    """

    def __init__(self, heuristic: str = 'complete',
                 neurons_to_refine: list | None = None,
                 approx_levels: int | list[int] | None = None):
        if heuristic not in ['overapprox', 'complete', 'mixed']:
            raise Exception(f'Selected heuristic {heuristic} is not valid')

        if heuristic == 'mixed' and neurons_to_refine is None:
            raise Exception('neurons_to_refine cannot be None if heuristic is mixed')

        # TODO add checks for approx_levels if needed

        self.heuristic = heuristic
        self.neurons_to_refine = neurons_to_refine
        self.approx_levels = approx_levels
        self.compute_areas = True if self.heuristic == 'mixed' else False


class SSBPVerificationParameters(VerificationParameters):
    def __init__(self,
                 heuristic: RefinementStrategy = RefinementStrategy.INPUT_BOUNDS_CHANGE,
                 bounds: BoundsBackend = BoundsBackend.SYMBOLIC,
                 bounds_direction: BoundsDirection = BoundsDirection.BACKWARDS,
                 intersection: IntersectionStrategy = IntersectionStrategy.ADAPTIVE,
                 timeout: int = 1200):
        self.heuristic = heuristic
        self.bounds = bounds
        self.bounds_direction = bounds_direction
        self.intersection = intersection
        self.timeout = timeout
