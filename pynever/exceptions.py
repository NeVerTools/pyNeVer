from multipledispatch import dispatch


class InvalidDimensionError(Exception):
    def __init__(self, message):
        super().__init__(message)


class OutOfRangeError(Exception):
    @dispatch(str)
    def __init__(self, message):
        super().__init__(message)

    @dispatch((int, float), (int, float), (int, float))
    def __init__(self, parameter, min_value, max_value):
        super().__init__(
            f'Parameter {parameter.__name__} with value {parameter} is out of range: [{min_value}, {max_value}]')


class EmptyNetworkError(Exception):
    def __init__(self, message='The network is empty'):
        super(EmptyNetworkError, self).__init__(message)


class InvalidNodeError(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotInNetworkError(Exception):

    @dispatch(str)
    def __init__(self, message):
        super().__init__(message)

    @dispatch()
    def __init__(self, node):
        super().__init__(f'{node.identifier} is not a node of the Network')


class NonOptimalLPError(Exception):
    def __init__(self, message='The LP problem was not Optimal'):
        super().__init__(message)


class FixedConflictWithBounds(Exception):
    pass
