import abc


# TODO add documentation
class ReLUParameters:
    def __init__(self, heuristic: str, params: list | None):
        self.heuristic = heuristic
        self.params = params


class SigmoidParameters:
    def __init__(self, approx_levels: int | list[int] | None):
        self.approx_levels = approx_levels


class VerificationParameters(abc.ABC):

    def __init__(self, relu_params: ReLUParameters | None = None, sigmoid_params: SigmoidParameters | None = None):
        self.relu_params = relu_params
        self.sigmoid_params = sigmoid_params
