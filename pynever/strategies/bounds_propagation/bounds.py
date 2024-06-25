import copy


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
