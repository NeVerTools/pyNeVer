import abc


# TODO add documentation

class VerificationParameters(abc.ABC):
    pass


class NeverVerificationParameters(VerificationParameters):
    def __init__(self, heuristic: str, params: list | None, approx_levels: int | list[int] | None):
        self.heuristic = heuristic
        self.params = params
        self.approx_levels = approx_levels


class SearchVerificationParameters(VerificationParameters):
    def __init__(self, heuristic: str = 'sequential', bounds: str = 'symbolic',
                 intersection: str = 'bounds_lp', timeout: int = 300):
        self.heuristic = heuristic
        self.bounds = bounds
        self.intersection = intersection
        self.timeout = timeout
