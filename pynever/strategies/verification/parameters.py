import abc


# TODO add documentation

class VerificationParameters(abc.ABC):
    pass


class NeverVerificationParameters(VerificationParameters):
    def __init__(self, heuristic: str, neurons_to_refine: list | None, approx_levels: int | list[int] | None):
        if heuristic not in ['overapprox', 'complete', 'mixed']:
            raise Exception(f'Selected heuristic {heuristic} is not valid')

        self.heuristic = heuristic
        self.neurons_to_refine = neurons_to_refine
        self.approx_levels = approx_levels


class SearchVerificationParameters(VerificationParameters):
    def __init__(self, heuristic: str = 'sequential', bounds: str = 'symbolic',
                 intersection: str = 'bounds_lp', timeout: int = 300):
        self.heuristic = heuristic
        self.bounds = bounds
        self.intersection = intersection
        self.timeout = timeout
