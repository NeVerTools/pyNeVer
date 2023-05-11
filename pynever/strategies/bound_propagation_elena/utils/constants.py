from enum import Enum


class NeuralOutputSettings(Enum):
    Q_VALUES = 0
    ONE_HOT_ARGMAX = 1
    INTEGER_ARGMAX = 2


class ActivationFunctions(Enum):
    UNKNOWN = 0
    LINEAR = 1
    RELU = 2
    SIGMOID = 3


class RNNUnrollingMethods:
    INPUT_ON_START = 1
    INPUT_ON_DEMAND = 2

