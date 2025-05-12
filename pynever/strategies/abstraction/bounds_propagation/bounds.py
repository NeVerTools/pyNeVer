"""
Module bounds.py

This file contains the different representations of symbolic and
numeric bounds for the verification of neural networks
"""
import abc
import copy
from abc import abstractmethod

import torch

from pynever.exceptions import InvalidDimensionError
from pynever.strategies.abstraction.linearfunctions import LinearFunctions


class AbstractBounds(abc.ABC):
    """
    Abstract class that defines the abstraction of lower and upper bounds for a neural network layer

    Attributes
    ----------
    lower: Any
        The lower bounds
    upper: Any
        The upper bounds
    size: int
        The number of dimensions of the lower and upper bounds
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.size = self.get_size()

    def __repr__(self):
        return ', '.join(["({}, {})".format(self.lower[i], self.upper[i]) for i in range(self.size)])

    @abstractmethod
    def get_lower(self):
        raise NotImplementedError

    @abstractmethod
    def get_upper(self):
        raise NotImplementedError

    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError


class HyperRectangleBounds(AbstractBounds):
    """
    Class that defines the hyper-rectangle bounds for a neural network layer, i.e.,
    bounding the variables with individual lower and upper bounds.

    Methods
    -------
    get_dimension_bounds(int)
        Procedure to get the bounds for a specific dimension
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        super(HyperRectangleBounds, self).__init__(lower, upper)

    def __repr__(self):
        return ', '.join(["(lb[{}]: {:.5f}, ub[{}]: {:.5f})".format(i, self.lower[i], i, self.upper[i])
                          for i in range(self.size)])

    def get_lower(self) -> torch.Tensor:
        return self.lower

    def get_upper(self) -> torch.Tensor:
        return self.upper

    def get_size(self) -> int:
        return len(self.lower)

    def clone(self):
        return HyperRectangleBounds(copy.deepcopy(self.lower), copy.deepcopy(self.upper))

    def get_dimension_bounds(self, dim: int) -> tuple[float, float]:
        """Procedure to get the bounds for a specific dimension"""
        if 0 <= dim < self.size:
            return self.lower[dim].item(), self.upper[dim].item()
        else:
            raise InvalidDimensionError("Dimension {} is out of range for size {}".format(dim, self.size))


class SymbolicLinearBounds(AbstractBounds):
    """
    Class that defines the symbolic linear bounds for a neural network layer, i.e.,
    the linear equations for the lower and upper bounds.

    Methods
    -------
    get_upper_bounds(HyperRectangleBounds) -> torch.Tensor
        Procedure to compute the numeric upper bounds
    get_lower_bounds(HyperRectangleBounds) -> torch.Tensor
        Procedure to compute the numeric lower bounds
    get_all_bounds(HyperRectangleBounds) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Procedure to compute all bounds
    to_hyper_rectangle_bounds(HyperRectangleBounds) -> HyperRectangleBounds
        Procedure to compute the hyper-rectangle bounds
    """

    def __init__(self, lower: LinearFunctions, upper: LinearFunctions):
        super(SymbolicLinearBounds, self).__init__(lower, upper)

    def get_lower(self) -> LinearFunctions:
        return self.lower

    def get_upper(self) -> LinearFunctions:
        return self.upper

    def get_size(self) -> int:
        return self.lower.get_size()

    def get_upper_bounds(self, input_bounds: HyperRectangleBounds) -> torch.Tensor:
        """Procedure to compute the numeric upper bounds
        Parameters
        ----------
        input_bounds: HyperRectangleBounds
            The initial bounds
        """
        return self.upper.compute_max_values(input_bounds)

    def get_lower_bounds(self, input_bounds: HyperRectangleBounds) -> torch.Tensor:
        """Procedure to compute the numeric lower bounds
        Parameters
        ----------
        input_bounds: HyperRectangleBounds
            The initial bounds
        """
        return self.lower.compute_min_values(input_bounds)

    def get_all_bounds(self, input_bounds: HyperRectangleBounds) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Procedure to compute all bounds
        Parameters
        ----------
        input_bounds: HyperRectangleBounds
            The initial bounds
        """
        return self.lower.compute_min_values(input_bounds), \
            self.lower.compute_max_values(input_bounds), \
            self.upper.compute_min_values(input_bounds), \
            self.upper.compute_max_values(input_bounds)

    def to_hyper_rectangle_bounds(self, input_bounds: HyperRectangleBounds) -> HyperRectangleBounds:
        """Procedure to compute the hyper-rectangle bounds
        Parameters
        ----------
        input_bounds: HyperRectangleBounds
            The initial bounds
        """
        return HyperRectangleBounds(self.lower.compute_min_values(input_bounds),
                                    self.upper.compute_max_values(input_bounds))
