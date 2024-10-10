"""
This file contains the different representations of symbolic
and numeric bounds for the verification of neural networks

"""

import copy

PRECISION_GUARD = 10e-15


class AbstractBounds:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def get_lower(self):
        return self.lower

    def get_upper(self):
        return self.upper


class HyperRectangleBounds(AbstractBounds):

    def __init__(self, lower, upper):
        super(HyperRectangleBounds, self).__init__(lower, upper)

        self.size = len(lower)

    def __repr__(self):
        # return "Input Bounds: " + ', '.join(map(str, zip(self.lower, self.upper)))

        return ', '.join(["({:.5f}, {:.5f})".format(self.lower[i], self.upper[i])
                          for i in range(self.size)])

    def clone(self):
        return HyperRectangleBounds(copy.deepcopy(self.lower), copy.deepcopy(self.upper))

    def get_size(self):
        return self.size

    def get_upper_bounds(self):
        return self.upper

    def get_lower_bounds(self):
        return self.lower

    def get_dimension_bounds(self, dim):
        assert 0 <= dim < self.size
        return self.lower[dim], self.upper[dim]


class SymbolicLinearBounds(AbstractBounds):

    def __init__(self, lower, upper):
        super(SymbolicLinearBounds, self).__init__(lower, upper)

        self.size = lower.get_size()

    def get_size(self):
        return self.size

    def get_upper_bounds(self, input_bounds):
        return self.upper.compute_max_values(input_bounds)

    def get_lower_bounds(self, input_bounds):
        return self.lower.compute_min_values(input_bounds)

    def get_all_bounds(self, input_bounds):
        return self.lower.compute_min_values(input_bounds), \
            self.lower.compute_max_values(input_bounds), \
            self.upper.compute_min_values(input_bounds), \
            self.upper.compute_max_values(input_bounds)

    def to_hyper_rectangle_bounds(self, input_bounds):
        return HyperRectangleBounds(self.lower.compute_min_values(input_bounds),
                                    self.upper.compute_max_values(input_bounds))


class BoundsStats:
    """
    This class contains the statistics about stability and
    approximation obtained through Bounds Propagation

    """

    def __init__(self, stability_info: dict, overapprox_area: dict):
        self.stability_info = copy.deepcopy(stability_info)
        self.overapprox_area = copy.deepcopy(overapprox_area)


class VerboseBounds:
    """
    This class is a complete collection of different bounds and information
    about them, used in the bounds propagation loop

    Attributes
    ----------
    symbolic_bounds : dict[layer_id: str, bounds: SymbolicLinearBounds]
        The dictionary of the symbolic bounds for the network
    numeric_pre_bounds : dict[layer_id: str, bounds: HyperRectangleBounds]
        The dictionary of the numeric pre-activation bounds for the network
    numeric_post_bounds : dict[layer_id: str, bounds: HyperRectangleBounds]
        The dictionary of the numeric post-activation bounds for the network
    stable_count : int
        The (cumulative) number of stable neurons in the network
    statistics : BoundsStats
        The data structure of stability and approximation statistics

    """

    def __init__(self, symbolic: dict = None,
                 numeric_pre: dict = None,
                 numeric_post: dict = None,
                 stable_count: int = 0):
        self.symbolic_bounds = copy.deepcopy(symbolic) if symbolic is not None else dict()
        self.numeric_pre_bounds = copy.deepcopy(numeric_pre) if numeric_pre is not None else dict()
        self.numeric_post_bounds = copy.deepcopy(numeric_post) if numeric_post is not None else dict()
        self.stable_count = stable_count
        self.statistics = None
