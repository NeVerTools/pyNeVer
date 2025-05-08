"""Module logistic.py

This module contains the classes for the linearization of s-like functions
such as sigmoid and tanh. Following the method used by the Venus2 verifier.

"""

import copy

import torch

from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds, SymbolicLinearBounds


def atleast1d(x: torch.Tensor):
    if x.dim() == 0:
        return x.reshape(1)
    else:
        return x


class LinearizeSLikeActivation:
    """
    This class factorizes common initializations and procedures to compute
    a linearization of s-like functions such as sigmoid and tanh.
    Specialized, activation-specific components are implemented in the
    corresponding child classes

    Attributes
    ----------
    input_bounds: HyperRectangleBounds
        The input numeric bounds for this layer
    num_iterations: int
        The number of iterations for the search of the optimal midpoint
    """

    def __init__(self, input_hyper_rect: HyperRectangleBounds, num_iterations: int = 2):
        self.input_bounds = input_hyper_rect
        self.num_iterations = num_iterations

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the activation function for the input x
        """
        raise NotImplementedError

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the first order derivative for the input x
        """
        raise NotImplementedError

    def compute_split_point(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Compute the optimal split point for the linearization
        """
        raise NotImplementedError

    def _update_xi(self, xi: torch.Tensor, x_bound: torch.Tensor, upper: bool) -> torch.Tensor:
        """
        Calculates the new xi for the iterative tangent method as described in the
        paper 'Efficient Neural Network Verification via Adaptive Refinement and
        Adversarial Search'.

        Parameters
        ----------
        xi: torch.Tensor
            The last tangent point calculated.
        x_bound: torch.Tensor
            The lower/upper input bound for calculating upper/lower relaxation
            respectively.
        upper: bool
            If True computes the upper tangent, otherwise the lower.

        Returns
        ----------
            The new xi
        """
        raise NotImplementedError

    def get_intercepting_lines(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> torch.Tensor:
        """
        This method computes the lines intercepting the activation function in all the
        lower and upper bounds and builds a Nx2 tensor with the a and b coefficients
        of the lines equation y = ax + b

        Parameters
        ----------
        lower_bounds: torch.Tensor
            The concrete lower bounds
        upper_bounds: torch.Tensor
            The concrete upper bounds

        Returns
        -------
            The tensor of the line coefficients
        """
        xl = atleast1d(lower_bounds)
        xu = atleast1d(upper_bounds)

        a = (self.activation(xu) - self.activation(xl)) / xu - xl
        b = self.activation(xu) - a * xu

        return torch.cat((a.unsqueeze(1), b.unsqueeze(1)), dim=1)

    def get_tangent_lines(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor,
                          tg_x: torch.Tensor = None) -> torch.Tensor:
        """
        This method computes the lines tangent to the activation function in all the
        lower and upper bounds and builds a Nx2 tensor with the a and b coefficients
        of the lines equation y = ax + b

        Parameters
        ----------
        lower_bounds: torch.Tensor
            The concrete lower bounds
        upper_bounds: torch.Tensor
            The concrete upper bounds
        tg_x: torch.Tensor, optional
            The tangent point. If None, it is computed by the function

        Returns
        -------
            The tensor of the line coefficients
        """
        xl = atleast1d(lower_bounds)
        xu = atleast1d(upper_bounds)

        if tg_x is None:
            tg_x = (xu + xl) / 2

        a = self.derivative(tg_x)
        b = self.activation(tg_x) - a * tg_x

        return torch.cat((a.unsqueeze(1), b.unsqueeze(1)), dim=1)

    def get_iterative_tangent_lines(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor,
                                    upper: bool) -> torch.Tensor:
        """
        This method computes the lines tangent to the activation function in all the
        lower and upper bounds, using the iterative search of the tangent point
        and builds a Nx2 tensor with the a and b coefficients of the lines equation y = ax + b

        Parameters
        ----------
        lower_bounds: torch.Tensor
            The concrete lower bounds
        upper_bounds: torch.Tensor
            The concrete upper bounds
        upper: bool
            Flag for the upper or lower bound

        Returns
        -------
            The tensor of the line coefficients
        """
        if upper:
            x_bound = lower_bounds
            xi = upper_bounds
        else:
            x_bound = upper_bounds
            xi = lower_bounds

        for i in range(self.num_iterations):
            xi = self._update_xi(xi, x_bound, upper)

        return self.get_tangent_lines(lower_bounds, upper_bounds, xi)

    def compute_output_numeric_bounds(self, cur_numeric_bounds: HyperRectangleBounds) -> HyperRectangleBounds:
        """
        Computes the concrete output bounds for this layer given the input concrete bounds

        Parameters
        ----------
        cur_numeric_bounds: HyperRectangleBounds
            The input numeric bounds

        Returns
        -------
            The output numeric bounds
        """
        return HyperRectangleBounds(
            self.activation(cur_numeric_bounds.get_lower()),
            self.activation(cur_numeric_bounds.get_upper())
        )

    @staticmethod
    def compute_output_linear_bounds(input_eq: SymbolicLinearBounds, lower_relax: torch.Tensor,
                                     upper_relax: torch.Tensor) -> SymbolicLinearBounds:
        """
        Propagates the given symbolic equations through the linear relaxations.

        Parameters
        ----------
        input_eq: SymbolicLinearBounds
            The input bounds to this layer
        lower_relax: torch.Tensor
            The lower bound relaxations
        upper_relax: torch.Tensor
            The upper bound relaxations

        Returns
        -------
            The symbolic bounds after the layer
        """
        out_symbolic = copy.deepcopy(input_eq)

        # relaxations:
        # A 2xNx2 tensor where the first dimension indicates the lower and upper
        # relaxation, the second dimension contains the neurons in the current
        # node and the last dimension contains the parameters
        # [a, b] in l(const_terms) = ax + b.

        out_symbolic.get_lower().matrix *= lower_relax[:, 0:1]
        out_symbolic.get_upper().matrix *= upper_relax[:, 0:1]

        out_symbolic.get_lower().offset += lower_relax[:, 1]
        out_symbolic.get_upper().offset += upper_relax[:, 1]

        return out_symbolic

    def compute_linear_relaxation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method computes the linear relaxation of the s-like activation function
        and returns the lower and upper linearization

        Returns
        -------
            The two relaxation tensors, lower and upper
        """
        return self.__single_linear_relaxation(upper=False), self.__single_linear_relaxation(upper=True)

    def __single_linear_relaxation(self, upper: bool) -> torch.Tensor:
        """
        This method computes the actual linear relaxation for both the lower
        and upper bound

        Parameters
        ----------
        upper: bool
            Flag to signal the upper or lower relaxation

        Returns
        -------
            The relaxation as a 2xN torch.Tensor
        """
        lower_bounds = self.input_bounds.get_lower()
        upper_bounds = self.input_bounds.get_upper()

        # Initialize the relaxation matrix and the unstable bounds
        relaxation = self.__init_relaxation()

        unstable_idx = torch.nonzero(lower_bounds != upper_bounds).squeeze()
        unstable_idx = atleast1d(unstable_idx)
        solved = torch.zeros_like(unstable_idx)

        unstable_lbs = lower_bounds[unstable_idx]
        unstable_ubs = upper_bounds[unstable_idx]

        if upper:
            activation = self.activation(unstable_lbs).squeeze()
            derivative = self.derivative(unstable_ubs).squeeze()

        else:
            activation = self.activation(unstable_ubs).squeeze()
            derivative = self.derivative(unstable_lbs).squeeze()

        # Try 1: the line that intercepts both endpoints
        lines = self.get_intercepting_lines(unstable_lbs, unstable_ubs)
        valid = torch.nonzero(lines[:, 0] <= derivative)

        # Save the valid lines
        relaxation[unstable_idx[valid]] = lines[valid]
        solved[valid] = 1

        if not all(solved):
            # Try 2: the optimal tangent line
            lines = self.get_tangent_lines(unstable_lbs[solved != 1], unstable_ubs[solved != 1])

            if upper:
                valid = torch.nonzero(lines[:, 0] * unstable_lbs + lines[:, 1] >= activation)
            else:
                valid = torch.nonzero(lines[:, 0] * unstable_ubs + lines[:, 1] <= activation)

            relaxation[unstable_idx[valid]] = lines[valid]
            solved[valid] = 1

            if not all(solved):
                # Try 3: iterative method
                lines = self.get_iterative_tangent_lines(unstable_lbs[solved != 1],
                                                         unstable_ubs[solved != 1], upper)
                relaxation[unstable_idx[solved != 1]] = lines  # this method is always valid

        return relaxation

    def __init_relaxation(self) -> torch.Tensor:
        """
        Relaxation that identifies whether there are some bounds with lb = ub.
        In that case, uses the linear relaxation y = ax + b where a = 0 and
        b = activation(lb) = activation(ub), otherwise sets the relaxation to zero.

        Returns
        -------
            The relaxation tensor initialized
        """
        # See https://github.com/vas-group-imperial/VeriNet/blob/main/verinet/sip_torch/operations/abstract_operation.py
        relaxation = torch.zeros((self.input_bounds.size, 2))
        equal_bounds_idx = torch.nonzero(self.input_bounds.get_lower() == self.input_bounds.get_upper())

        relaxation[equal_bounds_idx, 1] = self.activation(self.input_bounds.get_lower()[equal_bounds_idx])

        return relaxation


class LinearizeSigmoid(LinearizeSLikeActivation):
    def __init__(self, input_hyper_rect: HyperRectangleBounds, num_iterations: int = 2):
        super().__init__(input_hyper_rect, num_iterations)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        sig = self.activation(x)
        return sig * (1 - sig)

    def compute_split_point(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        mid = (self.activation(lower) + self.activation(upper)) / 2
        return -torch.log((1 / mid) - 1)

    def _update_xi(self, xi: torch.Tensor, x_bound: torch.Tensor, upper: bool) -> torch.Tensor:
        inner = 1 - 4 * (self.activation(xi) - self.activation(x_bound)) / (xi - x_bound)
        root = torch.sqrt(inner) / 2.

        if upper:
            sxi = 0.5 + root
        else:
            sxi = 0.5 - root
        new_xi = -torch.log(1 / sxi - 1)

        non_valid = torch.isnan(new_xi) + torch.isinf(new_xi)
        new_xi[non_valid] = xi[non_valid]  # Rounding error, use last valid relaxation.

        return new_xi


class LinearizeTanh(LinearizeSLikeActivation):
    def __init__(self, input_hyper_rect: HyperRectangleBounds, num_iterations: int = 2):
        super().__init__(input_hyper_rect, num_iterations)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        tanh = self.activation(x)
        return 1 - tanh ** 2

    def compute_split_point(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        mid = (self.activation(lower) + self.activation(upper)) / 2
        return 0.5 * torch.log((1 + mid) / (1 - mid))

    def _update_xi(self, xi: torch.Tensor, x_bound: torch.Tensor, upper: bool) -> torch.Tensor:
        inner = 1 - (self.activation(xi) - self.activation(x_bound)) / (xi - x_bound)
        root = torch.sqrt(inner)
        root[inner < 0] = xi[inner < 0]  # Rounding error, use last valid upper relaxation.

        if upper:
            sxi = root
        else:
            sxi = -root
        new_xi = 0.5 * torch.log((1 + sxi) / (1 - sxi))
        new_xi[torch.isnan(new_xi)] = xi[torch.isnan(new_xi)]  # Rounding error, use last valid relaxation.

        return new_xi
