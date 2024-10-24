"""
This file contains specialized methods that provide
the linearization of non-linear activation functions

"""
import copy

import numpy as np

from pynever import nodes, tensors
from pynever.exceptions import FixedConflictWithBounds
from pynever.strategies.bounds_propagation.bounds import SymbolicLinearBounds, HyperRectangleBounds, PRECISION_GUARD
from pynever.strategies.bounds_propagation.linearfunctions import LinearFunctions
from pynever.strategies.verification.ssbp.constants import NeuronSplit
from pynever.tensors import Tensor


# noinspection PyUnresolvedReferences
def atleast1d(x: Tensor) -> Tensor:
    return x.reshape(1) if tensors.dim(x) == 0 else x


class LinearizeReLU:
    """
    This class provides the linearization for the ReLU function enhanced by information
    about currently active and inactive neurons

    Attributes
    ----------
    fixed_neurons : dict[tuple[int, int], NeuronSplit]
        The fixed neurons for this layer, active or inactive
    input_hyper_rect : HyperRectangleBounds
        The input numeric bounds for this layer

    """

    USE_FIXED_NEURONS = True

    def __init__(self, fixed_neurons: dict[tuple[int, int], NeuronSplit], input_hyper_rect: HyperRectangleBounds):
        self.fixed_neurons = fixed_neurons
        self.input_hyper_rect = input_hyper_rect

    def compute_output_linear_bounds(self, input_eq: SymbolicLinearBounds) -> SymbolicLinearBounds:
        """
        Set the equations to zero for the neurons that have been fixed to 0
        This does not work well, at least for ACAS_XU.
        It seems to mess up the equations in a strange way.
        For instance, when there are no stable neurons, the equations are different from
        what we get with abstract propagation.
        Not sure if there is a problem is with abstract propagation or here.
        Could be abstract propagation as the bug I was getting was because
        the counter-example after using abstract propagation was not valid.
        However, the bug does not appear when we don't incorporate info from the fixed neurons.

        """

        lower_l, lower_u, upper_l, upper_u = input_eq.get_all_bounds(self.input_hyper_rect)

        k_lower, b_lower = LinearizeReLU.get_array_lin_lower_bound_coefficients(lower_l, lower_u)
        k_upper, b_upper = LinearizeReLU.get_array_lin_upper_bound_coefficients(upper_l, upper_u)

        lower_matrix = LinearizeReLU.get_transformed_matrix(input_eq.get_lower().get_matrix(), k_lower)
        upper_matrix = LinearizeReLU.get_transformed_matrix(input_eq.get_upper().get_matrix(), k_upper)

        lower_offset = LinearizeReLU.get_transformed_offset(input_eq.get_lower().get_offset(), k_lower, b_lower)
        upper_offset = LinearizeReLU.get_transformed_offset(input_eq.get_upper().get_offset(), k_upper, b_upper)

        lower = LinearFunctions(lower_matrix, lower_offset)
        upper = LinearFunctions(upper_matrix, upper_offset)

        return SymbolicLinearBounds(lower, upper)

    def compute_output_numeric_bounds(self, relu: nodes.ReLUNode, cur_numeric_bounds: HyperRectangleBounds,
                                      cur_symbolic_bounds: SymbolicLinearBounds) -> HyperRectangleBounds:
        """
        Compute the numeric post-activation bounds of the linearized ReLU function
        using the information about currently inactive neurons

        """

        layer_id = relu.identifier

        current_layer_inactive = LinearizeReLU.extract_layer_inactive_from_fixed_neurons(self.fixed_neurons, layer_id)

        cur_layer_output_num_bounds = HyperRectangleBounds(
            np.maximum(cur_numeric_bounds.get_lower(), 0),
            np.maximum(cur_numeric_bounds.get_upper(), 0))

        if LinearizeReLU.USE_FIXED_NEURONS:
            LinearizeReLU.force_inactive_neurons(cur_symbolic_bounds, cur_layer_output_num_bounds,
                                                 current_layer_inactive)

        return cur_layer_output_num_bounds

    @staticmethod
    def compute_relu_equation(preact_num_lower, preact_num_upper):
        lower_relu_eq, postact_lower = LinearizeReLU.get_relu_relax_lower_bound_equation(preact_num_lower,
                                                                                         preact_num_upper)
        upper_relu_eq, postact_upper = LinearizeReLU.get_relu_relax_upper_bound_equation(preact_num_lower,
                                                                                         preact_num_upper)

        return SymbolicLinearBounds(lower_relu_eq, upper_relu_eq), HyperRectangleBounds(postact_lower, postact_upper)

    @staticmethod
    def get_relu_relax_lower_bound_equation(preact_lower_bounds, preact_upper_bounds):
        """
        The lower bound of unstable nodes is either 0, or
        the linear relaxation of the preactivation (hence, the slope).

        The latter is the case when the upper bound is greater than or equal to the absolute value of the lower bound,
        thus resulting in a triangle of smaller area than the one formed by 0.

        The former is the case when the absolute value of the lower bound is greater than the upper bound,
        thus resulting is a triangle of smaller area than the one formed by the slope.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = np.identity(size)
        offset = np.zeros(size)

        postact_lower_bounds = np.array(preact_lower_bounds)

        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the lower bound is exactly the preactivation
                # it remains 1
                pass

            elif preact_upper_bounds[i] >= -preact_lower_bounds[i]:
                # Unstable node, lower bound is linear relaxation of the equation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i][i] = k
                postact_lower_bounds[i] *= k

            else:  # upper[i] <= 0 (inactive node)
                # or
                # -lower[i] > upper[i]
                # lower bound is 0
                matrix[i][i] = 0
                postact_lower_bounds[i] = 0

        return LinearFunctions(matrix, offset), postact_lower_bounds

    @staticmethod
    def get_relu_relax_upper_bound_equation(preact_lower_bounds, preact_upper_bounds):
        """
        Compute the resulting upper bound equation after relaxing ReLU,
        qiven a preactivation upper bound equation.

        input_bounds are required for computing the concrete bounds.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = np.identity(size)
        offset = np.zeros(size)

        postact_upper_bounds = np.array(preact_upper_bounds)
        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the upper bound is exactly the preactivation
                # it remains 1
                pass

            elif preact_upper_bounds[i] >= 0:
                # Unstable node - linear relaxation of preactivation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i][i] = k
                offset[i] = - preact_lower_bounds[i] * k

            else:  # preact_upper_bounds[i] <= 0 (inactive node)
                # The upper bound is 0
                matrix[i][i] = 0
                postact_upper_bounds[i] = 0

        return LinearFunctions(matrix, offset), postact_upper_bounds

    @staticmethod
    def check_and_enforce_fixed_constraints(relu_input_eq, preact_bounds, fixed_neurons, layer_id):
        """
        We need to check if the bounds do not conflict with the currently fixed neurons.
        That could happen if we haven't detected that the current branch is infeasible.
        That could happen because we are dealing with approximated bounds.

        """

        current_layer_inactive = LinearizeReLU.extract_layer_inactive_from_fixed_neurons(fixed_neurons, layer_id)
        current_layer_active = LinearizeReLU.extract_layer_active_from_fixed_neurons(fixed_neurons, layer_id)

        new_bounds = preact_bounds.clone()

        new_eq = SymbolicLinearBounds(relu_input_eq.lower.clone(), relu_input_eq.upper.clone())

        for neuron_n in current_layer_active:
            if preact_bounds.upper[neuron_n] < 0:
                raise FixedConflictWithBounds("A neuron has been fixed to be positive, "
                                              "but the bounds are negative. The current branch is not viable.")

        for neuron_n in current_layer_inactive:
            if preact_bounds.lower[neuron_n] > 0:
                raise FixedConflictWithBounds("A neuron has been fixed to be negative, "
                                              "but the bounds are positive. The current branch is not viable.")

            if preact_bounds.upper[neuron_n] > 0:
                new_eq.lower.matrix[neuron_n] = 0 * new_eq.lower.matrix[neuron_n]
                new_eq.lower.offset[neuron_n] = 0

                new_eq.upper.matrix[neuron_n] = 0 * new_eq.upper.matrix[neuron_n]
                new_eq.upper.offset[neuron_n] = 0

                new_bounds.lower[neuron_n] = -PRECISION_GUARD
                new_bounds.upper[neuron_n] = -PRECISION_GUARD  # TODO is the sign correct?

        return new_eq, new_bounds

    @staticmethod
    def extract_layer_active_from_fixed_neurons(fixed_neurons: dict, layer_id: str) -> list[int]:
        return [neuron_n for ((lay_n, neuron_n), value) in fixed_neurons.items()
                if lay_n == layer_id and value == 1]

    @staticmethod
    def extract_layer_inactive_from_fixed_neurons(fixed_neurons: dict, layer_id: str) -> list[int]:
        return [neuron_n for ((lay_n, neuron_n), value) in fixed_neurons.items()
                if lay_n == layer_id and value == 0]

    @staticmethod
    def force_inactive_neurons(relu_eq, postact_bounds, current_layer_inactive):
        for neuron_n in current_layer_inactive:
            if postact_bounds.lower[neuron_n] > 0:
                raise Exception("A neuron is supposed to be fixed to be negative, "
                                "but the bounds are positive. A conflict must have been detected before.")

            if postact_bounds.upper[neuron_n] > 0:
                relu_eq.lower.matrix[neuron_n] = 0 * relu_eq.lower.matrix[neuron_n]
                relu_eq.lower.offset[neuron_n] = 0
                relu_eq.upper.matrix[neuron_n] = 0 * relu_eq.upper.matrix[neuron_n]
                relu_eq.upper.offset[neuron_n] = 0
                postact_bounds.lower[neuron_n] = 0
                postact_bounds.upper[neuron_n] = 0

    @staticmethod
    def get_transformed_matrix(matrix, k):
        return matrix * k[:, None]

    @staticmethod
    def get_transformed_offset(offset, k, b):
        return offset * k + b

    @staticmethod
    def get_array_lin_lower_bound_coefficients(lower, upper):
        ks = np.zeros(len(lower))
        bs = np.zeros(len(lower))

        for i in range(len(lower)):
            k, b = LinearizeReLU.get_lin_lower_bound_coefficients(lower[i], upper[i])
            ks[i] = k
            bs[i] = b

        return ks, bs

    @staticmethod
    def get_array_lin_upper_bound_coefficients(lower, upper):
        ks = np.zeros(len(lower))
        bs = np.zeros(len(lower))

        for i in range(len(lower)):
            k, b = LinearizeReLU.get_lin_upper_bound_coefficients(lower[i], upper[i])
            ks[i] = k
            bs[i] = b

        return ks, bs

    @staticmethod
    def get_lin_lower_bound_coefficients(lower, upper):
        if lower >= 0:
            return 1, 0

        if upper >= - lower:
            mult = upper / (upper - lower)
            return mult, 0

        # upper <= 0:
        # or
        # -lower > upper, i.e., 0 is a tighter lower bound that the slope mult above
        return 0, 0

    @staticmethod
    def get_lin_upper_bound_coefficients(lower, upper):
        if lower >= 0:
            return 1, 0

        if upper <= 0:
            return 0, 0

        mult = upper / (upper - lower)
        add = -mult * lower

        return mult, add


class LinearizeLeakyReLU:
    def compute_output_linear_bounds(self, input_eq: SymbolicLinearBounds) -> SymbolicLinearBounds:
        pass


class LinearizeSLikeActivation:
    """
    This class factorizes common initializations and procedures to compute
    a linearization of s-like functions such as sigmoid and tanh.
    Specialized, activation-specific components are implemented in the
    corresponding child classes

    Attributes
    ----------
    input_bounds : HyperRectangleBounds
        The input numeric bounds for this layer
    num_iterations : int
        The number of iterations for the search of the optimal midpoint

    """

    def __init__(self, input_hyper_rect: HyperRectangleBounds, num_iterations: int = 2):
        self.input_bounds = input_hyper_rect
        self.num_iterations = num_iterations

    def activation(self, x: Tensor) -> Tensor:
        """
        Compute the activation function for the input x

        """

        raise NotImplementedError

    def derivative(self, x: Tensor) -> Tensor:
        """
        Compute the first order derivative for the input x

        """

        raise NotImplementedError

    def compute_split_point(self, lower: Tensor, upper: Tensor) -> Tensor:
        """
        Compute the optimal split point for the linearization

        """

        raise NotImplementedError

    def __update_xi(self, xi: Tensor, x_bound: Tensor, upper: bool) -> Tensor:
        """
        Calculates the new xi for the iterative tangent method as described in the
        paper 'Efficient Neural Network Verification via Adaptive Refinement and
        Adversarial Search'.

        Parameters
        ----------
        xi : Tensor
            The last tangent point calculated.
        x_bound : Tensor
            The lower/upper input bound for calculating upper/lower relaxation
            respectively.
        upper : bool
            If true the upper tangent is calculated, else the lower tangent is
            calculated.

        Returns
        -------
            The new xi

        """

        raise NotImplementedError

    # noinspection PyUnresolvedReferences
    def get_intercepting_lines(self, lower_bounds: Tensor, upper_bounds: Tensor) -> Tensor:
        """
        This method computes the lines intercepting the activation function in all the
        lower and upper bounds, and builds a Nx2 tensor with the a and b coefficients
        of the lines equation y = ax + b

        Parameters
        ----------
        lower_bounds : Tensor
            The concrete lower bounds
        upper_bounds : Tensor
            The concrete upper bounds

        Returns
        -------
            The tensor of the lines coefficients

        """

        xl = atleast1d(lower_bounds)
        xu = atleast1d(upper_bounds)

        a = (self.activation(xu) - self.activation(xl)) / xu - xl
        b = self.activation(xu) - a * xu

        return tensors.stack([a, b], axis=1)

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def get_tangent_lines(self, lower_bounds: Tensor, upper_bounds: Tensor, tg_x: Tensor = None) -> Tensor:
        """
        This method computes the lines tangent to the activation function in all the
        lower and upper bounds, and builds a Nx2 tensor with the a and b coefficients
        of the lines equation y = ax + b

        Parameters
        ----------
        lower_bounds : Tensor
            The concrete lower bounds
        upper_bounds : Tensor
            The concrete upper bounds
        tg_x : Tensor, optional
            The tangent point. If None, it is computed by the function

        Returns
        -------
            The tensor of the lines coefficients

        """

        xl = atleast1d(lower_bounds)
        xu = atleast1d(upper_bounds)

        if tg_x is None:
            tg_x = (xu + xl) / 2

        a = self.derivative(tg_x)
        b = self.activation(tg_x) - a * tg_x

        return tensors.stack([a, b], axis=1)

    def get_iterative_tangent_lines(self, lower_bounds: Tensor, upper_bounds: Tensor, upper: bool) -> Tensor:
        """
        This method computes the lines tangent to the activation function in all the
        lower and upper bounds, using the iterative search of the tangent point
        and builds a Nx2 tensor with the a and b coefficients of the lines equation y = ax + b

        Parameters
        ----------
        lower_bounds : Tensor
            The concrete lower bounds
        upper_bounds : Tensor
            The concrete upper bounds
        upper : bool
            Flag for the upper or lower bound

        Returns
        -------
            The tensor of the lines coefficients

        """

        if upper:
            x_bound = lower_bounds
            xi = upper_bounds
        else:
            x_bound = upper_bounds
            xi = lower_bounds

        for i in range(self.num_iterations):
            xi = self.__update_xi(xi, x_bound, upper)

        return self.get_tangent_lines(lower_bounds, upper_bounds, xi)

    def compute_output_numeric_bounds(self, cur_numeric_bounds: HyperRectangleBounds) -> HyperRectangleBounds:
        return HyperRectangleBounds(
            self.activation(cur_numeric_bounds.get_lower()),
            self.activation(cur_numeric_bounds.get_upper())
        )

    @staticmethod
    def compute_output_linear_bounds(input_eq: SymbolicLinearBounds, lower_relax: Tensor,
                                     upper_relax: Tensor) -> SymbolicLinearBounds:
        """
        Propagates the given symbolic equations through the linear relaxations.

        Parameters
        ----------
        input_eq : SymbolicLinearBounds
            The input bounds to this layer
        lower_relax : Tensor
            The lower bound relaxations
        upper_relax : Tensor
            The upper bound relaxations

        Returns
        -------
            The symbolic bounds after the layer

        """

        out_symbolic = copy.deepcopy(input_eq)

        out_symbolic.lower *= lower_relax[:, 0:1]
        out_symbolic.upper *= upper_relax[:, 0:1]

        out_symbolic.lower[:, -1] += lower_relax[:, 1]
        out_symbolic.upper[:, -1] += upper_relax[:, 1]

        return out_symbolic

    def compute_linear_relaxation(self) -> tuple[Tensor, Tensor]:
        """
        This method computes the linear relaxation of the s-like activation function
        and returns the lower and upper linearization

        Returns
        -------
            The two relaxation tensors, lower and upper

        """

        # Return the lower and upper relaxations
        return self.__single_linear_relaxation(upper=False), self.__single_linear_relaxation(upper=True)

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def __single_linear_relaxation(self, upper: bool) -> Tensor:
        """
        This method computes the actual linear relaxation for both the lower
        and upper bound

        Parameters
        ----------
        upper : bool
            Flag to signal the upper or lower relaxation

        Returns
        -------
            The relaxation as a 2xN Tensor

        """

        layer_size = self.input_bounds.get_size()
        lower_bounds = self.input_bounds.get_lower_bounds()
        upper_bounds = self.input_bounds.get_upper_bounds()

        # Initialize the relaxation matrix and the unstable bounds
        relaxation = self.__init_relaxation()
        solved = tensors.zeros((layer_size,))

        unstable_idx = tensors.nonzero(lower_bounds != upper_bounds).squeeze()
        unstable_idx = atleast1d(unstable_idx)

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
        valid = tensors.nonzero(lines[:, 0] <= derivative)

        # Save the valid lines
        relaxation[unstable_idx[valid]] = lines[valid]
        solved[valid] = 1

        if not all(solved):
            # Try 2: the optimal tangent line
            lines = self.get_tangent_lines(unstable_lbs[solved != 1], unstable_ubs[solved != 1])

            if upper:
                valid = tensors.nonzero(lines[:, 0] * unstable_lbs + lines[:, 1] >= activation)
            else:
                valid = tensors.nonzero(lines[:, 0] * unstable_ubs + lines[:, 1] <= activation)

            relaxation[unstable_idx[valid]] = lines[valid]
            solved[valid] = 1

            if not all(solved):
                # Try 3: iterative method
                lines = self.get_iterative_tangent_lines(unstable_lbs[solved != 1],
                                                         unstable_ubs[solved != 1], upper)
                relaxation[unstable_idx[solved != 1]] = lines  # this method is always valid

        return relaxation

    # noinspection PyUnresolvedReferences
    def __init_relaxation(self) -> Tensor:
        """
        Relaxation that identifies whether there are some bounds with lb = ub.
        In that case, uses the linear relaxation y = ax + b where a = 0 and
        b = activation(lb) = activation(ub), otherwise sets the relaxation to zero.

        Returns
        -------
            The relaxations tensor initialized

        """

        # See https://github.com/vas-group-imperial/VeriNet/blob/main/verinet/sip_torch/operations/abstract_operation.py
        relaxation = tensors.zeros((self.input_bounds.size, 2))
        equal_bounds_idx = tensors.nonzero(self.input_bounds.get_lower_bounds() == self.input_bounds.get_upper_bounds())

        relaxation[equal_bounds_idx, 2] = self.activation(self.input_bounds.get_lower_bounds()[equal_bounds_idx])

        return relaxation


class LinearizeSigmoid(LinearizeSLikeActivation):
    def __init__(self, input_hyper_rect: HyperRectangleBounds, num_iterations: int = 2):
        super().__init__(input_hyper_rect, num_iterations)

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def activation(self, x: Tensor) -> Tensor:
        return 1 / (1 + np.exp(-x))

    # noinspection PyTypeChecker
    def derivative(self, x: Tensor) -> Tensor:
        sig = self.activation(x)
        return sig * (1 - sig)

    # noinspection PyUnresolvedReferences
    def compute_split_point(self, lower: Tensor, upper: Tensor) -> Tensor:
        mid = (self.activation(lower) + self.activation(upper)) / 2
        return -tensors.log((1 / mid) - 1)

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def __update_xi(self, xi: Tensor, x_bound: Tensor, upper: bool) -> Tensor:
        inner = 1 - 4 * (self.activation(xi) - self.activation(x_bound)) / (xi - x_bound)
        root = tensors.sqrt(inner) / 2.

        if upper:
            sxi = 0.5 + root
        else:
            sxi = 0.5 - root
        new_xi = -tensors.log(1 / sxi - 1)

        non_valid = tensors.isnan(new_xi) + tensors.isinf(new_xi)
        new_xi[non_valid] = xi[non_valid]  # Rounding error, use last valid relaxation.

        return new_xi


class LinearizeTanh(LinearizeSLikeActivation):
    def __init__(self, input_hyper_rect: HyperRectangleBounds, num_iterations: int = 2):
        super().__init__(input_hyper_rect, num_iterations)

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def activation(self, x: Tensor) -> Tensor:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # noinspection PyTypeChecker
    def derivative(self, x: Tensor) -> Tensor:
        tanh = self.activation(x)
        return 1 - tanh ** 2

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def compute_split_point(self, lower: Tensor, upper: Tensor) -> Tensor:
        mid = (self.activation(lower) + self.activation(upper)) / 2
        return 0.5 * tensors.log((1 + mid) / (1 - mid))

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def __update_xi(self, xi: Tensor, x_bound: Tensor, upper: bool) -> Tensor:
        inner = 1 - (self.activation(xi) - self.activation(x_bound)) / (xi - x_bound)
        root = tensors.sqrt(inner)
        root[inner < 0] = xi[inner < 0]  # Rounding error, use last valid upper relaxation.

        if upper:
            sxi = root
        else:
            sxi = - root
        new_xi = 0.5 * tensors.log((1 + sxi) / (1 - sxi))
        new_xi[tensors.isnan(new_xi)] = xi[tensors.isnan(new_xi)]  # Rounding error, use last valid relaxation.

        return new_xi
