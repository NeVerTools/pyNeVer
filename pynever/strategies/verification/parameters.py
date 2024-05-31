import abc


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
    def __init__(self, heuristic: str = 'sequential', bounds: str = 'symbolic',
                 intersection: str = 'bounds_lp', timeout: int = 300):
        # TODO better error messages
        if heuristic not in ['sequential']:
            raise Exception(f'Selected heuristic {heuristic} is not valid')
        if bounds not in ['symbolic']:
            raise Exception(f'Selected bounds {bounds} is not valid')
        if intersection not in ['bounds_lp']:
            raise Exception(f'Selected intersection {intersection} is not valid')

        self.heuristic = heuristic
        self.bounds = bounds
        self.intersection = intersection
        self.timeout = timeout
