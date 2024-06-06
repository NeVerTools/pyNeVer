from __future__ import annotations

import abc
import time
import uuid

import numpy as np
import numpy.linalg as la
from ortools.linear_solver import pywraplp

import pynever.tensors as tensors
from pynever.exceptions import InvalidDimensionError, NonOptimalLPError
from pynever.strategies.abstraction import LOGGER_EMPTY, LOGGER_LP, LOGGER_LB, LOGGER_UB
from pynever.strategies.bp.bounds import AbstractBounds
from pynever.tensors import Tensor


class AbsElement(abc.ABC):
    """
    An abstract class used for our internal representation of a generic Abstract Element (e.g., interval, zonotope,
    polyhedra etc.)

    Attributes
    ----------
    identifier : str
        Identifier of the AbsElement.
    """

    def __init__(self, identifier: str = None):
        if identifier is None:
            self.identifier = uuid.uuid4()
        else:
            self.identifier = identifier


class Star:
    """
    A concrete class used for our internal representation of a Star.
    The Star is defined as {x | x = c + Va such that Ca <= d}
    where c is a n-dimensional vector corresponding to the center of the Star.
    V is the n-by-m matrix composed by the basis vectors.
    a is the vector of m variables, C (p-by-m) and d (p-dim) are the matrix and the biases
    defining a set of constraints.

    We refer to <Star-Based Reachability Analysis of Deep Neural Networks>
    (https://link.springer.com/chapter/10.1007/978-3-030-30942-8_39) for details.

    Attributes
    ----------
    center : Tensor
        Center of the Star.
    basis_matrix : Tensor
        Matrix composed by the basis vectors of the Star
    predicate_matrix : Tensor
        Matrix of the Predicate.
    predicate_bias : Tensor
        Bias of the Predicate.

    Methods
    ----------
    get_bounds()
        Function used to get the upper and lower bounds of the n variables of the star.
    check_if_empty()
        Function used to check if the star corresponds to an empty set.

    """

    def __init__(self, predicate_matrix: Tensor, predicate_bias: Tensor, center: Tensor = None,
                 basis_matrix: Tensor = None):

        predicate_dim_message = f"Error: the first dimension of the predicate_matrix ({predicate_matrix.shape[0]}) " \
                                f"must be equal to the dimension of the predicate_bias ({predicate_bias.shape[0]})."
        if predicate_matrix.shape[0] != predicate_bias.shape[0]:
            raise InvalidDimensionError(predicate_dim_message)

        self.predicate_matrix: Tensor = predicate_matrix
        self.predicate_bias: Tensor = predicate_bias

        if center is None and basis_matrix is None:
            self.center: Tensor = np.zeros((predicate_matrix.shape[1], 1))
            self.basis_matrix: Tensor = np.identity(predicate_matrix.shape[1])

        else:
            center_dim_message = f"Error: the first dimension of the basis_matrix ({basis_matrix.shape[0]}) " \
                                 f"must be equal to the dimension of the center ({center.shape[0]})."
            if center.shape[0] != basis_matrix.shape[0]:
                raise InvalidDimensionError(center_dim_message)

            basis_dim_message = f"Error: the second dimension of the basis_matrix ({basis_matrix.shape[1]}) " \
                                f"must be equal to the second dimension of the predicate_matrix " \
                                f"({predicate_matrix.shape[1]})."
            if basis_matrix.shape[1] != predicate_matrix.shape[1]:
                raise InvalidDimensionError(basis_dim_message)

            self.center: Tensor = center
            self.basis_matrix: Tensor = basis_matrix

        self.n_neurons: int = self.center.shape[0]

        # Reference layer of the star (where it comes from)
        self.ref_layer: int = 0

        # Starting number of predicates (used in search verification)
        self.ref_neuron: int = 0

        # Private Attributes used for the sampling of the star.
        self.__auxiliary_points: list[Tensor] | None = None
        self.__current_point: Tensor | None = None

    def check_if_empty(self) -> bool:
        """
        Function used to check if the set of points defined by the star is empty.

        Return
        ---------
        bool
            True if the star defines an empty set of points, False otherwise.

        """

        start_time = time.perf_counter()

        solver, alphas, constraints = self.__get_predicate_lp_solver()
        objective = solver.Objective()
        for j in range(self.predicate_matrix.shape[1]):
            objective.SetCoefficient(alphas[j], 0)
        objective.SetOffset(0)

        objective.SetMinimization()
        status = solver.Solve()

        is_empty = False
        if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
            is_empty = True

        end_time = time.perf_counter()
        LOGGER_EMPTY.debug(f"{end_time - start_time},")

        return is_empty

    def get_bounds(self, i: int) -> tuple[float | None, float | None]:
        """
        Function used to get the upper and lower bounds of the n variables of the star.

        Return
        ---------
        (float, float)
            Tuple containing the lower and upper bounds of the variable i of the star

        """

        start_time = time.perf_counter()

        solver, alphas, constraints = self.__get_predicate_lp_solver()
        objective = solver.Objective()
        for j in range(self.basis_matrix.shape[1]):
            objective.SetCoefficient(alphas[j], self.basis_matrix[i, j])
        objective.SetOffset(self.center[i, 0])

        objective.SetMinimization()

        lb_start = time.perf_counter()
        status = solver.Solve()
        lb_end = time.perf_counter()

        if status != pywraplp.Solver.OPTIMAL:
            raise NonOptimalLPError()

        lb = None
        ub = None
        ub_start = 0
        ub_end = 0
        if status != pywraplp.Solver.INFEASIBLE and status != pywraplp.Solver.ABNORMAL:
            lb = solver.Objective().Value()
            objective.SetMaximization()
            ub_start = time.perf_counter()
            status = solver.Solve()
            ub_end = time.perf_counter()
            ub = solver.Objective().Value()

        end_time = time.perf_counter()

        LOGGER_LP.debug(f"{end_time - start_time},")
        LOGGER_LB.debug(f"{ub_end - ub_start},")
        LOGGER_UB.debug(f"{lb_end - lb_start},")

        return lb, ub

    def check_alpha_inside(self, alpha_point: Tensor) -> bool:
        """
        Function which checks if the alpha point passed as input is valid with respect to the constraints defined by the
        predicate matrix and bias of the star.

        Parameters
        ----------
        alpha_point : Tensor
            Point (with respect ot the predicate variables) whose validity is to test.

        Returns
        -------
        bool
            The result of the check as a boolean (True if the point is valid, False otherwise)

        """

        dim_error_msg = f"Wrong dimensionality for alpha_point: it should be {self.predicate_matrix.shape[1]} by one."
        if alpha_point.shape[0] != self.predicate_matrix.shape[1]:
            raise InvalidDimensionError(dim_error_msg)

        tests = np.matmul(self.predicate_matrix, alpha_point) <= self.predicate_bias
        test = np.all(tests)

        return test

    def check_point_inside(self, point: Tensor, epsilon: float) -> bool:
        """
        Function which checks if the point passed as input is valid with respect to the constraints defined by the
        predicate matrix and bias of the star.

        Parameters
        ----------
        point : Tensor
            Point whose validity is to test.
        epsilon : float
            Acceptable deviation from real point.

        Returns
        -------
        bool
            The result of the check as a boolean (True if the point is valid, False otherwise)

        """

        solver, alphas, constraints = self.__get_predicate_lp_solver()

        for i in range(self.basis_matrix.shape[0]):
            lb = point[i][0] - self.center[i][0] - epsilon
            ub = point[i][0] - self.center[i][0] + epsilon
            new_constraint = solver.Constraint(lb, ub)
            for j in range(self.basis_matrix.shape[1]):
                new_constraint.SetCoefficient(alphas[j], self.basis_matrix[i, j])
            constraints.append(new_constraint)

        objective = solver.Objective()
        for j in range(self.predicate_matrix.shape[1]):
            objective.SetCoefficient(alphas[j], 0)
        objective.SetOffset(0)

        objective.SetMinimization()
        status = solver.Solve()

        return status == pywraplp.Solver.FEASIBLE or status == pywraplp.Solver.OPTIMAL

    def get_samples(self, num_samples: int, reset_auxiliary: bool = False, new_start: bool = False) -> list[Tensor]:

        # As first thing we need to get a valid starting point:
        if self.check_if_empty():
            return []

        if self.__auxiliary_points is None or reset_auxiliary:
            auxiliary_points = self.__get_auxiliary_points()
        else:
            auxiliary_points = self.__auxiliary_points

        if self.__current_point is None or new_start:
            starting_point = self.__get_starting_point()
            current_point = np.array(starting_point)
        else:
            current_point = self.__current_point

        # We begin the iterative process to generate the samples of interest.
        samples = []
        while len(samples) < num_samples:

            direction = np.random.randn(self.predicate_matrix.shape[1], 1)
            direction = direction / la.norm(direction)
            lambdas = []
            for i in range(self.predicate_matrix.shape[0]):

                if not np.isclose(np.matmul(self.predicate_matrix[i, :], direction), 0):
                    temp = auxiliary_points[i] - current_point
                    lam = np.matmul(self.predicate_matrix[i, :], temp) / (np.matmul(self.predicate_matrix[i, :],
                                                                                    direction))
                    lambdas.append(lam)

            lambdas = np.array(lambdas)

            try:
                lam_upper = np.min(lambdas[lambdas >= 0])
                lam_lower = np.max(lambdas[lambdas < 0])
            except Exception:
                raise RuntimeError("The current direction does not intersect"
                                   "any of the hyperplanes.")

            increment = np.random.uniform(low=lam_lower, high=lam_upper)
            next_point = current_point + increment * direction
            if self.check_alpha_inside(next_point):
                current_point = next_point
                star_point = self.center + np.matmul(self.basis_matrix, current_point)
                samples.append(star_point)
                self.__current_point = current_point

        return samples

    def create_negative_stable(self, index: int) -> Star:
        """
        Function to build the negative stable processing of this star throughout
        a single ReLU neuron

        """

        mask = tensors.identity(self.center.shape[0])
        mask[index, index] = 0

        new_c = tensors.matmul(mask, self.center)
        new_b = tensors.matmul(mask, self.basis_matrix)
        new_pred = self.predicate_matrix
        new_bias = self.predicate_bias

        return Star(new_pred, new_bias, new_c, new_b)

    def create_approx(self, index: int, lb: float, ub: float) -> Star:
        """
        Function to build the approximate star for the given ReLU neuron

        """

        mask = tensors.identity(self.center.shape[0])
        mask[index, index] = 0

        # Build all components of the approximate star
        col_c_mat = self.predicate_matrix.shape[1]
        row_c_mat = self.predicate_matrix.shape[0]

        c_mat_1 = tensors.zeros((1, col_c_mat + 1))
        c_mat_1[0, col_c_mat] = -1
        c_mat_2 = tensors.hstack((tensors.array([self.basis_matrix[index, :]]), -tensors.ones((1, 1))))
        coef_3 = - ub / (ub - lb)
        c_mat_3 = tensors.hstack((tensors.array([coef_3 * self.basis_matrix[index, :]]), tensors.ones((1, 1))))
        c_mat_0 = tensors.hstack((self.predicate_matrix, tensors.zeros((row_c_mat, 1))))

        d_0 = self.predicate_bias
        d_1 = tensors.zeros((1, 1))
        d_2 = -self.center[index] * tensors.ones((1, 1))
        d_3 = tensors.array([(ub / (ub - lb)) * (self.center[index] - lb)])

        new_pred_mat = tensors.vstack((c_mat_0, c_mat_1, c_mat_2, c_mat_3))
        new_pred_bias = tensors.vstack((d_0, d_1, d_2, d_3))

        new_center = tensors.matmul(mask, self.center)
        temp_basis_mat = tensors.matmul(mask, self.basis_matrix)
        temp_vec = tensors.zeros((self.basis_matrix.shape[0], 1))
        temp_vec[index, 0] = 1
        new_basis_mat = tensors.hstack((temp_basis_mat, temp_vec))

        return Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat)

    def split(self, index: int) -> tuple[Star, Star]:
        """
        Function to build the two stars obtained by splitting the current star
        throughout a ReLU neuron

        """

        mask = tensors.identity(self.center.shape[0])
        mask[index, index] = 0

        # Lower star
        lower_c = tensors.matmul(mask, self.center)
        lower_b = tensors.matmul(mask, self.basis_matrix)
        lower_pred = tensors.vstack((self.predicate_matrix, self.basis_matrix[index, :]))
        lower_bias = tensors.vstack((self.predicate_bias, -self.center[index]))
        lower_star = Star(lower_pred, lower_bias, lower_c, lower_b)

        # Upper star
        upper_c = self.center
        upper_b = self.basis_matrix
        upper_pred = tensors.vstack((self.predicate_matrix, -self.basis_matrix[index, :]))
        upper_bias = tensors.vstack((self.predicate_bias, self.center[index]))
        upper_star = Star(upper_pred, upper_bias, upper_c, upper_b)

        return lower_star, upper_star

    def __get_auxiliary_points(self) -> list[Tensor]:
        """
        Function which returns the auxiliary points for each plane of the predicate.

        Returns
        -------
        List[Tensor]
            List of the auxiliary points: one for each plane of the predicate.

        """

        aux_points = []
        for i in range(self.predicate_matrix.shape[0]):
            p = np.zeros((self.predicate_matrix.shape[1], 1))
            plane = self.predicate_matrix[i, :]
            max_nonzero_index = np.argmax(np.where(plane != 0, plane, -np.inf))
            p[max_nonzero_index] = self.predicate_bias[i] / plane[max_nonzero_index]
            aux_points.append(p)

        return aux_points

    def __get_starting_point_by_bounds(self) -> Tensor:
        """
        Function used to get the starting point for the hit and run algorithm.

        Return
        ---------
        (float, float)
            Tuple containing the lower and upper bounds of the variables of the predicate

        """

        starting_point = []
        for i in range(self.predicate_matrix.shape[1]):

            solver, alphas, constraints = self.__get_predicate_lp_solver()
            objective = solver.Objective()
            for j in range(self.predicate_matrix.shape[1]):
                if j == i:
                    objective.SetCoefficient(alphas[j], 1)
                else:
                    objective.SetCoefficient(alphas[j], 0)

            objective.SetMinimization()
            status = solver.Solve()

            if status != pywraplp.Solver.OPTIMAL:
                raise NonOptimalLPError()

            lb = solver.Objective().Value()

            objective.SetMaximization()
            status = solver.Solve()

            if status != pywraplp.Solver.OPTIMAL:
                raise NonOptimalLPError()

            ub = solver.Objective().Value()

            starting_point.append([(lb + ub) / 2.0])

        starting_point = np.array(starting_point)
        return starting_point

    def __get_starting_point(self) -> Tensor:
        """
        Function used to get the starting point for the hit and run algorithm.

        Return
        ---------
        (float, float)
            Tuple containing the lower and upper bounds of the variables of the predicate

        """

        starting_point = []

        solver = pywraplp.Solver.CreateSolver('GLOP')
        alphas = []
        for j in range(self.basis_matrix.shape[1]):
            new_alpha = solver.NumVar(-solver.infinity(), solver.infinity(), f'alpha_{j}')
            alphas.append(new_alpha)
        radius = solver.NumVar(0, solver.infinity(), 'radius')

        constraints = []
        for k in range(self.predicate_matrix.shape[0]):
            new_constraint = solver.Constraint(-solver.infinity(), self.predicate_bias[k, 0])
            for j in range(self.predicate_matrix.shape[1]):
                new_constraint.SetCoefficient(alphas[j], self.predicate_matrix[k, j])
            new_constraint.SetCoefficient(radius, np.linalg.norm(self.predicate_matrix[k, :], 2))
            constraints.append(new_constraint)

        objective = solver.Objective()
        for j in range(self.predicate_matrix.shape[1]):
            objective.SetCoefficient(alphas[j], 0)
        objective.SetCoefficient(radius, 1)

        objective.SetMaximization()
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            raise NonOptimalLPError('It was impossible to compute the Chebyshev center of the predicate.')

        for alpha in alphas:
            # print(alpha.solution_value())
            starting_point.append([alpha.solution_value()])
        # print(radius.solution_value())

        starting_point = np.array(starting_point)

        return starting_point

    def __get_predicate_lp_solver(self) -> tuple[pywraplp.Solver, list, list]:
        """
        Creates an lp solver using pywraplp and adds the variables and constraints
        corresponding to the predicate of the star.

        Returns
        ---------
        (pywraplp.Solver, list, list)
            Respectively the lp solver, the variables and the constraints.

        """

        solver = pywraplp.Solver.CreateSolver('GLOP')
        alphas = []
        for j in range(self.basis_matrix.shape[1]):
            new_alpha = solver.NumVar(-solver.infinity(), solver.infinity(), f'alpha_{j}')
            alphas.append(new_alpha)

        constraints = []
        for k in range(self.predicate_matrix.shape[0]):
            new_constraint = solver.Constraint(-solver.infinity(), self.predicate_bias[k, 0])
            for j in range(self.predicate_matrix.shape[1]):
                new_constraint.SetCoefficient(alphas[j], self.predicate_matrix[k, j])
            constraints.append(new_constraint)

        return solver, alphas, constraints

    def intersect_with_halfspace(self, coef_mat: Tensor, bias_mat: Tensor) -> Star:
        """
        Function which takes as input a Star and a halfspace defined by its coefficient matrix and bias vector
        and returns the Star resulting from the intersection of the input Star with the halfspace.

        """

        new_center = self.center
        new_basis_matrix = self.basis_matrix
        hs_pred_matrix = tensors.matmul(coef_mat, self.basis_matrix)
        hs_pred_bias = bias_mat - tensors.matmul(coef_mat, self.center)
        new_pred_matrix = tensors.vstack((self.predicate_matrix, hs_pred_matrix))
        new_pred_bias = tensors.vstack((self.predicate_bias, hs_pred_bias))

        new_star: Star = Star(new_pred_matrix, new_pred_bias, new_center, new_basis_matrix)

        return new_star


class StarSet(AbsElement):
    """
    Concrete class for our internal representation of a StarSet abstract element. A StarSet consist in a set
    of Star objects.

    Attributes
    ----------
    stars : Set[Star]
        Set of Star objects.

    """

    def __init__(self, stars: set[Star] = None, identifier: str = None):
        super().__init__(identifier)
        if stars is None:
            self.stars = set()
        else:
            self.stars = stars
