import abc
import itertools
import logging
import math
import multiprocessing
import time
import uuid
from typing import Set, List, Union, Tuple

import numpy as np
import numpy.linalg as la
from ortools.linear_solver import pywraplp

import pynever.nodes as nodes
from pynever.tensor import Tensor

logger_empty = logging.getLogger("pynever.strategies.abstraction.empty_times")
logger_lp = logging.getLogger("pynever.strategies.abstraction.lp_times")
logger_lb = logging.getLogger("pynever.strategies.abstraction.lb_times")
logger_ub = logging.getLogger("pynever.strategies.abstraction.ub_times")

# save_times = False
propagate_bounds = False
parallel = True


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
    ubs : list
        Upper bounds of the points defined by the Star.
    lbs : list
        Lower bounds of the points defined by the Star.
    is_empty : bool
        Boolean flag: True if the Star defines an empty set of points, False otherwise

    Methods
    ----------
    get_bounds()
        Function used to get the the upper and lower bounds of the n variables of the star.
    check_if_empty()
        Function used to check if the star corresponds to an empty set.


    """

    def __init__(self, predicate_matrix: Tensor, predicate_bias: Tensor, center: Tensor = None,
                 basis_matrix: Tensor = None, lbs: list = None, ubs: list = None, is_empty: bool = None):

        predicate_dim_message = f"Error: the first dimension of the predicate_matrix ({predicate_matrix.shape[0]}) " \
                                f"must be equal to the dimension of the predicate_bias ({predicate_bias.shape[0]})."
        assert predicate_matrix.shape[0] == predicate_bias.shape[0], predicate_dim_message

        self.predicate_matrix = predicate_matrix
        self.predicate_bias = predicate_bias

        if center is None and basis_matrix is None:
            self.center = np.zeros((predicate_matrix.shape[1], 1))
            self.basis_matrix = np.identity(predicate_matrix.shape[1])

        else:
            center_dim_message = f"Error: the first dimension of the basis_matrix ({basis_matrix.shape[0]}) " \
                                 f"must be equal to the dimension of the center ({center.shape[0]})."
            assert center.shape[0] == basis_matrix.shape[0], center_dim_message

            basis_dim_message = f"Error: the second dimension of the basis_matrix ({basis_matrix.shape[1]}) " \
                                f"must be equal to the second dimension of the predicate_matrix " \
                                f"({predicate_matrix.shape[1]})."
            assert basis_matrix.shape[1] == predicate_matrix.shape[1], basis_dim_message

            self.center = center
            self.basis_matrix = basis_matrix

        if lbs is None:
            lbs = [None for i in range(self.center.shape[0])]

        if ubs is None:
            ubs = [None for i in range(self.center.shape[0])]

        self.lbs = lbs
        self.ubs = ubs
        self.is_empty = is_empty

        # Private Attributes used for the sampling of the star.
        self.__auxiliary_points = None
        self.__current_point = None

    def check_if_empty(self) -> bool:
        """
        Function used to check if the set of points defined by the star is empty.

        Return
        ---------
        bool
            True if the star defines an empty set of points, False otherwise.
        """

        start_time = time.perf_counter()
        if self.is_empty is None:

            solver, alphas, constraints = self.__get_predicate_lp_solver()
            objective = solver.Objective()
            for j in range(self.predicate_matrix.shape[1]):
                objective.SetCoefficient(alphas[j], 0)
            objective.SetOffset(0)

            objective.SetMinimization()
            status = solver.Solve()
            if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
                self.is_empty = True
            else:
                self.is_empty = False

        end_time = time.perf_counter()
        logger_empty.debug(f"{end_time - start_time},")

        return self.is_empty

    def get_bounds(self, i) -> Tuple[float, float]:
        """
        Function used to get the the upper and lower bounds of the n variables of the star.

        Return
        ---------
        (float, float)
            Tuple containing the lower and upper bounds of the variable i of the star
        """

        if self.lbs[i] is None or self.ubs[i] is None or self.is_empty is None:

            # print("Computing bounds")
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

            assert status == pywraplp.Solver.OPTIMAL, "The LP problem was not Optimal"

            if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
                self.is_empty = True
                self.lbs[i] = None
                self.ubs[i] = None
                ub_end = 0
                ub_start = 0
            else:
                self.is_empty = False

                lb = solver.Objective().Value()
                objective.SetMaximization()
                ub_start = time.perf_counter()
                status = solver.Solve()
                ub_end = time.perf_counter()
                ub = solver.Objective().Value()

                self.lbs[i] = lb
                self.ubs[i] = ub

            end_time = time.perf_counter()

            logger_lp.debug(f"{end_time - start_time},")
            logger_lb.debug(f"{ub_end - ub_start},")
            logger_ub.debug(f"{lb_end - lb_start},")

        return self.lbs[i], self.ubs[i]

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
        assert alpha_point.shape[0] == self.predicate_matrix.shape[1], dim_error_msg
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

    def get_samples(self, num_samples: int, reset_auxiliary: bool = False, new_start: bool = False) -> List[Tensor]:

        # As first thing we need to get a valid starting point:
        # assert not self.check_if_empty(), "Empty Set: impossible to sample."
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
                if np.isclose(np.matmul(self.predicate_matrix[i, :], direction), 0):
                    lambdas.append(np.nan)
                else:
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

    def __get_auxiliary_points(self) -> List[Tensor]:
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

            assert status == pywraplp.Solver.OPTIMAL, "The LP problem was not Optimal"

            lb = solver.Objective().Value()

            objective.SetMaximization()
            status = solver.Solve()

            assert status == pywraplp.Solver.OPTIMAL, "The LP problem was not Optimal"

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

        assert status == pywraplp.Solver.OPTIMAL, "It was impossible to compute the Chebyshev center of the predicate."

        for alpha in alphas:
            # print(alpha.solution_value())
            starting_point.append([alpha.solution_value()])
        # print(radius.solution_value())

        starting_point = np.array(starting_point)

        return starting_point

    def __get_predicate_lp_solver(self) -> Tuple[pywraplp.Solver, list, list]:
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


class StarSet(AbsElement):
    """
    Concrete class for our internal representation of a StarSet abstract element. A StarSet consist in a set
    of Star objects.

    Attributes
    ----------
    stars : Set[Star]
        Set of Star objects.
    """

    def __init__(self, stars: Set[Star] = None, identifier: str = None):
        super().__init__(identifier)
        if stars is None:
            self.stars = set()
        else:
            self.stars = stars


def intersect_with_halfspace(star: Star, coef_mat: Tensor, bias_mat: Tensor) -> Star:
    """
    Function which takes as input a Star and a halfspace defined by its coefficient matrix and bias vector and returns
    the Star resulting from the intesection of the input Star with the halfspace.
    """

    new_center = star.center
    new_basis_matrix = star.basis_matrix
    hs_pred_matrix = np.matmul(coef_mat, star.basis_matrix)
    hs_pred_bias = bias_mat - np.matmul(coef_mat, star.center)
    new_pred_matrix = np.vstack((star.predicate_matrix, hs_pred_matrix))
    new_pred_bias = np.vstack((star.predicate_bias, hs_pred_bias))

    new_star = Star(new_pred_matrix, new_pred_bias, new_center, new_basis_matrix)
    return new_star


def __mixed_step_relu(abs_input: Set[Star], var_index: int, refinement_flag: bool) -> Set[Star]:
    abs_input = list(abs_input)
    abs_output = set()

    ref_flags = [refinement_flag for i in range(len(abs_input))]

    for i in range(len(abs_input)):

        star = abs_input[i]
        lb, ub = star.get_bounds(var_index)

        if not star.is_empty:

            mask = np.identity(star.center.shape[0])
            mask[var_index, var_index] = 0

            if lb >= 0:
                abs_output = abs_output.union({star})

            elif ub <= 0:
                new_center = np.matmul(mask, star.center)
                new_basis_mat = np.matmul(mask, star.basis_matrix)
                new_pred_mat = star.predicate_matrix
                new_pred_bias = star.predicate_bias
                if propagate_bounds:
                    lbs = []
                    lbs.extend(star.lbs)
                    ubs = []
                    ubs.extend(star.ubs)
                    lbs[var_index] = 0
                    ubs[var_index] = 0
                else:
                    lbs = None
                    ubs = None
                new_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat, lbs, ubs)
                abs_output = abs_output.union({new_star})

            else:

                if ref_flags[i]:

                    # Creating lower bound star.
                    lower_star_center = np.matmul(mask, star.center)
                    lower_star_basis_mat = np.matmul(mask, star.basis_matrix)
                    # Adding x <= 0 constraints to the predicate.
                    lower_predicate_matrix = np.vstack((star.predicate_matrix, star.basis_matrix[var_index, :]))
                    # Possibile problema sulla dimensionalita' di star.center[var_index]
                    lower_predicate_bias = np.vstack((star.predicate_bias, -star.center[var_index]))
                    if propagate_bounds:
                        lbs = []
                        lbs.extend(star.lbs)
                        ubs = []
                        ubs.extend(star.ubs)
                        lbs[var_index] = 0
                        ubs[var_index] = 0
                    else:
                        lbs = None
                        ubs = None
                    lower_star = Star(lower_predicate_matrix, lower_predicate_bias, lower_star_center,
                                      lower_star_basis_mat, lbs, ubs)

                    # Creating upper bound star.
                    upper_star_center = star.center
                    upper_star_basis_mat = star.basis_matrix
                    # Adding x >= 0 constraints to the predicate.
                    upper_predicate_matrix = np.vstack((star.predicate_matrix, -star.basis_matrix[var_index, :]))
                    # Possibile problema sulla dimensionalita' di star.center[var_index]
                    upper_predicate_bias = np.vstack((star.predicate_bias, star.center[var_index]))
                    if propagate_bounds:
                        lbs = []
                        lbs.extend(star.lbs)
                        ubs = []
                        ubs.extend(star.ubs)
                        lbs[var_index] = 0
                        ubs[var_index] = star.ubs[var_index]
                    else:
                        lbs = None
                        ubs = None
                    upper_star = Star(upper_predicate_matrix, upper_predicate_bias, upper_star_center,
                                      upper_star_basis_mat, lbs, ubs)

                    abs_output = abs_output.union({lower_star, upper_star})

                else:

                    col_c_mat = star.predicate_matrix.shape[1]
                    row_c_mat = star.predicate_matrix.shape[0]

                    c_mat_1 = np.zeros((1, col_c_mat + 1))
                    c_mat_1[0, col_c_mat] = -1
                    c_mat_2 = np.hstack((np.array([star.basis_matrix[var_index, :]]), -np.ones((1, 1))))
                    coef_3 = - ub / (ub - lb)
                    c_mat_3 = np.hstack((np.array([coef_3 * star.basis_matrix[var_index, :]]), np.ones((1, 1))))
                    c_mat_0 = np.hstack((star.predicate_matrix, np.zeros((row_c_mat, 1))))

                    d_0 = star.predicate_bias
                    d_1 = np.zeros((1, 1))
                    d_2 = -star.center[var_index] * np.ones((1, 1))
                    d_3 = np.array([(ub / (ub - lb)) * (star.center[var_index] - lb)])

                    new_pred_mat = np.vstack((c_mat_0, c_mat_1, c_mat_2, c_mat_3))
                    new_pred_bias = np.vstack((d_0, d_1, d_2, d_3))

                    new_center = np.matmul(mask, star.center)
                    temp_basis_mat = np.matmul(mask, star.basis_matrix)
                    temp_vec = np.zeros((star.basis_matrix.shape[0], 1))
                    temp_vec[var_index, 0] = 1
                    new_basis_mat = np.hstack((temp_basis_mat, temp_vec))
                    if propagate_bounds:
                        lbs = []
                        lbs.extend(star.lbs)
                        ubs = []
                        ubs.extend(star.ubs)
                        lbs[var_index] = 0
                        ubs[var_index] = star.ubs[var_index]
                    else:
                        lbs = None
                        ubs = None
                    new_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat, lbs, ubs)

                    abs_output = abs_output.union({new_star})

    return abs_output


def mixed_single_relu_forward(star: Star, heuristic: str, params: List) -> Tuple[Set[Star], np.ndarray]:
    """
    Utility function for the management of the forward for AbsReLUNode. It is outside
    the class scope since multiprocessing does not support parallelization with
    function internal to classes.
    """

    assert heuristic == "given_flags" or heuristic == "best_n_neurons" or heuristic == "best_n_neurons_rel", \
        "Heuristic Selected is not valid"

    temp_abs_input = {star}
    if star.check_if_empty():
        return set(), None
    else:

        n_areas = []
        for i in range(star.center.shape[0]):
            lb, ub = star.get_bounds(i)
            if lb < 0 and ub > 0:
                n_areas.append(-lb * ub / 2.0)
            else:
                n_areas.append(0)

        n_areas = np.array(n_areas)

        if heuristic == "best_n_neurons" or heuristic == "best_n_neurons_rel":

            n_neurons = params[0]

            if n_neurons > 0:

                # We compute the ordered indexes of the neurons with decreasing values of the areas.
                # Our idea is that a greater value for the area correspond to greater loss of precision if the
                # star is not refined for the corresponding neuron.
                if heuristic == "best_n_neurons_rel":
                    relevances = params[1]
                    n_areas = n_areas * relevances

                sorted_indexes = np.flip(np.argsort(n_areas))
                index_to_refine = sorted_indexes[:n_neurons]
            else:
                index_to_refine = []

            refinement_flags = []
            for i in range(star.center.shape[0]):
                if i in index_to_refine:
                    refinement_flags.append(True)
                else:
                    refinement_flags.append(False)

        elif heuristic == "given_flags":
            refinement_flags = params

        else:
            raise NotImplementedError

        for i in range(star.center.shape[0]):
            temp_abs_input = __mixed_step_relu(temp_abs_input, i, refinement_flags[i])

        return temp_abs_input, n_areas


def single_fc_forward(star: Star, weight: Tensor, bias: Tensor) -> Set[Star]:
    """
    Utility function for the management of the forward for AbsFullyConnectedNode. It is outside
    the class scope since multiprocessing does not support parallelization with
    function internal to classes.
    """
    assert (weight.shape[1] == star.basis_matrix.shape[0])

    new_basis_matrix = np.matmul(weight, star.basis_matrix)
    new_center = np.matmul(weight, star.center) + bias
    new_predicate_matrix = star.predicate_matrix
    new_predicate_bias = star.predicate_bias

    if propagate_bounds:
        lbs = []
        ubs = []
        for i in range(new_center.shape[0]):

            w = weight[i, :]
            b = bias[i, 0]
            lb = b
            ub = b
            for j in range(len(w)):
                if star.lbs[j] is None or star.ubs[j] is None:
                    star.get_bounds(j)

                lb = lb + min(star.lbs[j] * w[j], star.ubs[j] * w[j])
                ub = ub + max(star.lbs[j] * w[j], star.ubs[j] * w[j])
            lbs.append(lb)
            ubs.append(ub)
    else:
        lbs = None
        ubs = None

    new_star = Star(new_predicate_matrix, new_predicate_bias, new_center, new_basis_matrix, lbs, ubs)

    return {new_star}


def sig(x: float) -> float:
    """
    Utility function computing the logistic function of the input.
    """
    return 1.0 / (1.0 + math.exp(-x))


def sig_fod(x: float) -> float:
    """
    Utility function computing the first order derivative of the logistic function of the input.
    """
    return math.exp(-x) / math.pow(1 + math.exp(-x), 2)


def area_sig_triangle(lb: float, ub: float) -> float:
    """
    Utility function computing the area of the triangle defined by an upper bound and a lower bound on the
    logistic function. In particular is the triangle composed by the two tangents and line passing by the two bounds.
    """

    x_p = (ub * sig_fod(ub) - lb * sig_fod(lb)) / (sig_fod(ub) - sig_fod(lb)) - \
          (sig(ub) - sig(lb)) / (sig_fod(ub) - sig_fod(lb))

    y_p = sig_fod(ub) * (x_p - ub) + sig(ub)

    height = abs(y_p - (sig(ub) - sig(lb)) / (ub - lb) * x_p + sig(lb) - lb * (sig(ub) - sig(lb)) / (ub - lb)) / \
             math.sqrt(1 + math.pow((sig(ub) - sig(lb)) / (ub - lb), 2))

    base = math.sqrt(math.pow(ub - lb, 2) + math.pow(sig(ub) - sig(lb), 2))

    return base * height / 2.0


def __recursive_step_sigmoid(star: Star, var_index: int, approx_level: int, lb: float, ub: float, tolerance: float) -> \
        Set[Star]:
    assert approx_level >= 0

    if abs(ub - lb) < tolerance:

        if ub <= 0:
            if ub + tolerance > 0:
                ub = 0
            else:
                ub = ub + tolerance
            lb = lb - tolerance
        else:
            if lb - tolerance < 0:
                lb = 0
            else:
                lb = lb - tolerance
            ub = ub + tolerance

    assert (lb <= 0 and ub <= 0) or (lb >= 0 and ub >= 0)

    mask = np.identity(star.center.shape[0])
    mask[var_index, var_index] = 0

    if approx_level == 0:

        if lb < 0 and ub <= 0:

            c_mat_1 = np.hstack((np.array([sig_fod(lb) * star.basis_matrix[var_index, :]]), -np.ones((1, 1))))
            c_mat_2 = np.hstack((np.array([sig_fod(ub) * star.basis_matrix[var_index, :]]), -np.ones((1, 1))))
            coef_3 = - (sig(ub) - sig(lb)) / (ub - lb)
            c_mat_3 = np.hstack((np.array([coef_3 * star.basis_matrix[var_index, :]]), np.ones((1, 1))))

            d_1 = np.array([-sig_fod(lb) * (star.center[var_index] - lb) - sig(lb)])
            d_2 = np.array([-sig_fod(ub) * (star.center[var_index] - ub) - sig(ub)])
            d_3 = np.array([-coef_3 * (star.center[var_index] - lb) + sig(lb)])

        else:

            c_mat_1 = np.hstack((np.array([-sig_fod(lb) * star.basis_matrix[var_index, :]]), np.ones((1, 1))))
            c_mat_2 = np.hstack((np.array([-sig_fod(ub) * star.basis_matrix[var_index, :]]), np.ones((1, 1))))
            coef_3 = (sig(ub) - sig(lb)) / (ub - lb)
            c_mat_3 = np.hstack((np.array([coef_3 * star.basis_matrix[var_index, :]]), -np.ones((1, 1))))

            d_1 = np.array([sig_fod(lb) * (star.center[var_index] - lb) + sig(lb)])
            d_2 = np.array([sig_fod(ub) * (star.center[var_index] - ub) + sig(ub)])
            d_3 = np.array([-coef_3 * (star.center[var_index] - lb) - sig(lb)])

        col_c_mat = star.predicate_matrix.shape[1]

        # Adding lb and ub bounds to enhance stability
        c_mat_lb = np.zeros((1, col_c_mat + 1))
        c_mat_lb[0, col_c_mat] = -1
        d_lb = -sig(lb) * np.ones((1, 1))

        c_mat_ub = np.zeros((1, col_c_mat + 1))
        c_mat_ub[0, col_c_mat] = 1
        d_ub = sig(ub) * np.ones((1, 1))

        row_c_mat = star.predicate_matrix.shape[0]
        c_mat_0 = np.hstack((star.predicate_matrix, np.zeros((row_c_mat, 1))))
        d_0 = star.predicate_bias

        new_pred_mat = np.vstack((c_mat_0, c_mat_1, c_mat_2, c_mat_3, c_mat_lb, c_mat_ub))
        new_pred_bias = np.vstack((d_0, d_1, d_2, d_3, d_lb, d_ub))

        new_center = np.matmul(mask, star.center)
        temp_basis_mat = np.matmul(mask, star.basis_matrix)
        temp_vec = np.zeros((star.basis_matrix.shape[0], 1))
        temp_vec[var_index, 0] = 1
        new_basis_mat = np.hstack((temp_basis_mat, temp_vec))

        new_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat)

        return {new_star}

    else:

        # We need to select the boundary between lb and ub. The optimal boundary is the one which minimizes the
        # area of the two resulting triangle. Since computing the optimal is too slow we do an approximate search
        # between lb and ub considering s search points.

        num_search_points = 10
        boundaries = np.linspace(lb, ub, num_search_points, endpoint=False)
        boundaries = boundaries[1:]

        best_boundary = None
        smallest_area = 99999999
        for boundary in boundaries:
            area_1 = area_sig_triangle(lb, boundary)
            area_2 = area_sig_triangle(boundary, ub)
            if area_1 + area_2 < smallest_area:
                smallest_area = area_1 + area_2
                best_boundary = boundary

        star_set = set()
        star_set = star_set.union(
            __recursive_step_sigmoid(star, var_index, approx_level - 1, lb, best_boundary, tolerance))
        star_set = star_set.union(
            __recursive_step_sigmoid(star, var_index, approx_level - 1, best_boundary, ub, tolerance))

        return star_set


def __approx_step_sigmoid(abs_input: Set[Star], var_index: int, approx_level: int, tolerance: float) -> Set[Star]:
    abs_output = set()
    for star in abs_input:

        if not star.check_if_empty():
            lb, ub = star.get_bounds(var_index)

            if (lb < 0) and (ub > 0):
                abs_output = abs_output.union(__recursive_step_sigmoid(star, var_index, approx_level, lb, 0,
                                                                       tolerance))
                abs_output = abs_output.union(__recursive_step_sigmoid(star, var_index, approx_level, 0, ub,
                                                                       tolerance))
            else:
                abs_output = abs_output.union(__recursive_step_sigmoid(star, var_index, approx_level, lb,
                                                                       ub, tolerance))

    return abs_output


def single_sigmoid_forward(star: Star, approx_levels: List[int]) -> Set[Star]:
    """
    Utility function for the management of the forward for AbsSigmoidNode. It is outside
    the class scope since multiprocessing does not support parallelization with
    function internal to classes.
    """
    tolerance = 0.01
    temp_abs_input = {star}
    for i in range(star.center.shape[0]):
        temp_abs_input = __approx_step_sigmoid(temp_abs_input, i, approx_levels[i], tolerance)
        print(f"Index {i}, NumStar: {len(temp_abs_input)}")
    return temp_abs_input


class RefinementState(abc.ABC):
    """
    A class used for the internal control of the refinement strategies/heuristics applied in the abstraction refinement
    step. At present is not still used and it is just an abstract placeholder. It will be used in future
    implementations.
    """
    pass


class AbsLayerNode(abc.ABC):
    """
    An abstract class used for our internal representation of a generic Abstract Transformer Layer of an
    AbsNeural Network. Its concrete children correspond to real abstract interpretation network layers.

    Attributes
    ----------
    identifier : str
        Identifier of the AbsLayerNode.

    ref_node : LayerNode
        LayerNode di riferimento per l'abstract transformer.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction. At present the function is just a
        placeholder for future implementations.

    """

    def __init__(self, identifier: str, ref_node: nodes.LayerNode):
        self.identifier = identifier
        self.ref_node = ref_node

    @abc.abstractmethod
    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        pass

    @abc.abstractmethod
    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass


class AbsReLUNode(AbsLayerNode):
    """
    A class used for our internal representation of a ReLU Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.

    ref_node : ReLUNode
        Reference LayerNode for the abstract transformer.

    heuristic : str
        Heuristic used to decide the refinement level of the abstraction.
        At present can be only one of the following:
        - given_flags: the neuron to be refined are selected referring to the list given in params
        - best_n_neurons: for each star the n best neuron to refine are selected based on the loss of precision
          the abstraction would incur using the coarse over_approximation.
        - best_n_neurons_rel: for each star the n best neuron to refine are selected based on the loss of precision
          the abstraction would incur using the coarse over_approximation together with the neuron relevances values.

    params : List
        Parameters for the heuristic of interest.
        If the heuristic is given_flags then params is a List whose first element is the list of refinement flags.
        If the heuristic is best_n_neurons then params is a List whose first element is the number of neurons to refine.
        If the heuristic is best_n_neurons_rel then params is a List whose first element is a tuple containing
        the number of neurons to refine and a list containing the relevances of the neurons.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction. At present the function is just a
        placeholder for future implementations.
    """

    def __init__(self, identifier: str, ref_node: nodes.ReLUNode, heuristic: str, params: List):

        super().__init__(identifier, ref_node)

        self.heuristic = heuristic
        self.params = params
        self.n_areas = None

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """

        if isinstance(abs_input, StarSet):
            if parallel:
                return self.__parallel_starset_forward(abs_input)
            else:
                return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass

    def __parallel_starset_forward(self, abs_input: StarSet) -> StarSet:

        my_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        parallel_results = my_pool.starmap(mixed_single_relu_forward, zip(abs_input.stars,
                                                                          itertools.repeat(self.heuristic),
                                                                          itertools.repeat(self.params)))
        my_pool.close()

        abs_output = StarSet()

        tot_areas = np.zeros(self.ref_node.in_dim)
        num_areas = 0
        for star_set, areas in parallel_results:
            if star_set != set():
                num_areas = num_areas + 1
                tot_areas = tot_areas + areas
            abs_output.stars = abs_output.stars.union(star_set)

        self.n_areas = tot_areas / num_areas

        return abs_output

    def __starset_forward(self, abs_input: StarSet) -> StarSet:
        """
        Forward function specialized for the concrete AbsElement StarSet.
        """

        abs_output = StarSet()
        tot_areas = np.zeros(self.ref_node.in_dim)
        num_areas = 0
        for star in abs_input.stars:
            result, areas = mixed_single_relu_forward(star, self.heuristic, self.params)
            abs_output.stars = abs_output.stars.union(result)
            tot_areas = tot_areas + areas
            num_areas = num_areas + 1

        self.n_areas = tot_areas / num_areas

        return abs_output


class AbsFullyConnectedNode(AbsLayerNode):
    """
    A class used for our internal representation of a Fully Connected Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.

    ref_node : FullyConnectedNode
        LayerNode di riferimento per l'abstract transformer.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction. At present the function is just a
        placeholder for future implementations.
    """

    def __init__(self, identifier: str, ref_node: nodes.FullyConnectedNode):
        super().__init__(identifier, ref_node)

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        if isinstance(abs_input, StarSet):
            return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def __starset_forward(self, abs_input: StarSet) -> StarSet:

        my_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # Need to expand bias since they are memorized like one-dimensional vector in fullyconnectednodes.
        if self.ref_node.bias.shape != (self.ref_node.weight.shape[0], 1):
            bias = np.expand_dims(self.ref_node.bias, 1)
        else:
            bias = self.ref_node.bias
        parallel_results = my_pool.starmap(single_fc_forward, zip(abs_input.stars,
                                                                  itertools.repeat(self.ref_node.weight),
                                                                  itertools.repeat(bias)))
        my_pool.close()
        abs_output = StarSet()
        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass


class AbsSigmoidNode(AbsLayerNode):
    """
    A class used for our internal representation of a Sigmoid transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.

    ref_node : SigmoidNode
        LayerNode di riferimento per l'abstract transformer.

    refinement_level : Union[int, List[int]]
        Refinement level for the sigmoid nodes: if it is a single int then that refinement level is applied to all
        the neurons of the layers, otherwise it is a list containing the refinement levels for each layers.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction. At present the function is just a
        placeholder for future implementations.
    """

    def __init__(self, identifier: str, ref_node: nodes.SigmoidNode, approx_levels: Union[int, List[int]] = None):
        super().__init__(identifier, ref_node)

        if approx_levels is None:
            approx_levels = [0 for i in range(ref_node.in_dim[-1])]
        elif isinstance(approx_levels, int):
            approx_levels = [approx_levels for i in range(ref_node.in_dim[-1])]

        self.approx_levels = approx_levels

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        if isinstance(abs_input, StarSet):
            return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def __starset_forward(self, abs_input: StarSet) -> StarSet:

        # parallel = True
        if parallel:
            abs_output = StarSet()
            my_pool = multiprocessing.Pool(1)
            parallel_results = my_pool.starmap(single_sigmoid_forward, zip(abs_input.stars,
                                                                           itertools.repeat(self.approx_levels)))
            my_pool.close()
            for star_set in parallel_results:
                abs_output.stars = abs_output.stars.union(star_set)
        else:
            abs_output = StarSet()
            for star in abs_input.stars:
                abs_output.stars = abs_output.stars.union(single_sigmoid_forward(star, self.approx_levels))

        return abs_output

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass


class AbsNeuralNetwork(abc.ABC):
    """
    An abstract class used for our internal representation of a generic NeuralNetwork for Abstract Interpretation.
    It consists of a graph of AbsLayerNodes. The properties of the computational graph are specialized in the
    concrete classes. The method forward and backward calls the corresponding methods in the AbsLayerNodes following the
    correct order to compute the output AbsElement.

    Attributes
    ----------
    nodes : dict <str, LayerNode>
        Dictionary containing str keys and AbsLayerNodes values. It contains the nodes of the graph,
        the identifier of the node of interest is used as a key in the nodes dictionary.

    edges : dict <str, list <str>>
        Dictionary of identifiers of AbsLayerNodes, it contains for each nodes identified by the keys, the list of nodes
        connected to it.

    Methods
    ----------
    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the AbsLayerNode
        of the network.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the AbsLayerNodes
        to control the refinement component of the abstraction. At present the function is just a placeholder
        for future implementations.

    """

    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformers.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """
        pass

    @abc.abstractmethod
    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass


class AbsSeqNetwork(AbsNeuralNetwork):
    """
    Concrete children of AbsNeuralNetwork representing a sequential AbsNeuralNetwork.
    It consists of a graph of LayerNodes. The computational graph of a SequentialNetwork must
    correspond to a standard list. The method forward and backward calls the corresponding methods
    in the AbsLayerNodes following the correct order to compute the output AbsElement.

    Attributes
    ----------
    identifier : str
        Identifier of the Sequential AbsNeuralNetwork.

    Methods
    -------
    add_node(LayerNode)
        Procedure to add a new AbsLayerNode to the sequential AbsNeuralNetwork.

    get_first_node()
        Procedure to extract the first AbsLayerNode of the sequential AbsNeuralNetwork.

    get_next_node(LayerNode)
        Procedure to get the next AbsLayerNode of the AbsNeuralNetwork given an input AbsLayerNode

    get_last_node()
        Procedure to extract the last AbsLayerNode of the sequential AbsNeuralNetwork.

    forward(AbsElement)
        Function which takes an AbsElement and compute the corresponding output AbsElement based on the AbsLayerNode
        of the network.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the AbsLayerNodes
        to control the refinement component of the abstraction. At present the function is just a placeholder for
        future implementations.

    """

    def __init__(self, identifier: str):

        super().__init__()
        self.identifier = identifier

    def add_node(self, node: AbsLayerNode):
        """
        Procedure to add a new AbsLayerNode. In sequential network the new node must be connected directly to the
        previous node forming a list.

        Parameters
        ----------
        node : AbsLayerNode
            New node to add to the Sequential network.

        """

        if len(self.nodes.keys()) == 0:
            self.nodes[node.identifier] = node
            self.edges[node.identifier] = []
        else:
            previous_node_key = self.get_last_node().identifier
            self.nodes[node.identifier] = node
            self.edges[previous_node_key].append(node.identifier)
            self.edges[node.identifier] = []

    def get_first_node(self) -> AbsLayerNode:
        """
        Procedure to get the first AbsLayerNode of the network.

        Return
        ---------
        AbsLayerNode
            The first node of the network.

        """
        assert self.nodes

        keys = [key for key in self.nodes.keys()]
        for key in self.nodes.keys():
            for sub_key in self.nodes.keys():
                if sub_key in self.edges[key]:
                    keys.remove(sub_key)

        return self.nodes[keys[0]]

    def get_next_node(self, node: AbsLayerNode) -> AbsLayerNode:
        """
        Procedure to get the next AbsLayerNode of the network given an input AbsLayerNode.

        Return
        ---------
        LayerNode
            The next node of the network.

        """

        assert self.nodes

        next_node = None
        if node is not None:
            current_key = node.identifier
            if len(self.edges[current_key]) != 0:
                next_key = self.edges[current_key][0]
                next_node = self.nodes[next_key]
        else:
            next_node = self.get_first_node()

        return next_node

    def get_last_node(self) -> AbsLayerNode:
        """
        Procedure to get the last AbsLayerNode of the network.

        Return
        ---------
        AbsLayerNode
            The last node of the network.

        """

        assert self.nodes

        current_node = self.get_first_node()
        while self.get_next_node(current_node) is not None:
            current_node = self.get_next_node(current_node)

        return current_node

    def forward(self, abs_input: AbsElement) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformers.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """

        current_node = self.get_first_node()
        while current_node is not None:
            abs_input = current_node.forward(abs_input)
            current_node = self.get_next_node(current_node)

        return abs_input

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass
