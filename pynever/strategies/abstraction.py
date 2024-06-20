import abc
import copy
import itertools
import logging
import math
import multiprocessing
import time
import uuid
from typing import Set, List, Union, Tuple, Optional

import numpy as np
import numpy.linalg as la
from ortools.linear_solver import pywraplp

import pynever.nodes as nodes
from pynever.strategies.bp.bounds import AbstractBounds
from pynever.tensors import Tensor

logger_empty = logging.getLogger("pynever.strategies.abstraction.empty_times")
logger_lp = logging.getLogger("pynever.strategies.abstraction.lp_times")
logger_lb = logging.getLogger("pynever.strategies.abstraction.lb_times")
logger_ub = logging.getLogger("pynever.strategies.abstraction.ub_times")

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
    is_empty : bool
        Boolean flag: True if the Star defines an empty set of points, False otherwise

    Methods
    ----------
    get_bounds()
        Function used to get the upper and lower bounds of the n variables of the star.
    check_if_empty()
        Function used to check if the star corresponds to an empty set.

    """

    def __init__(self, predicate_matrix: Tensor, predicate_bias: Tensor, center: Tensor = None,
                 basis_matrix: Tensor = None, is_empty: bool = None,
                 ref_layer: int = None, ref_neuron: int = 0, ref_unstable_neurons: set = None,
                 fixed_neurons: dict = dict()):

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

        self.is_empty = is_empty

        # Reference layer of the star (where it comes from)
        self.ref_layer = ref_layer

        # Reference neuron of the star (last fixed one)
        self.ref_neuron = ref_neuron

        # Unstable neurons in the reference layer
        self.ref_unstable_neurons = ref_unstable_neurons

        # The neurons fixed so far
        self.fixed_neurons = fixed_neurons

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
        Function used to get the upper and lower bounds of the n variables of the star.

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


def check_stable(var_index: int, bounds: AbstractBounds) -> int:
    """

    Parameters
    ----------
    var_index
    bounds

    Returns
    -------
    0 if unstable, 1 if positive stable, -1 if negative stable

    """

    precision_guard = 10e-15

    lb = bounds.get_lower()[var_index]
    ub = bounds.get_upper()[var_index]

    # Positive stable
    if lb >= precision_guard:
        return 1

    # Negative stable
    elif ub <= -precision_guard:
        return -1

    # Unstable
    else:
        return 0


def intersect_with_halfspace(star: Star, coef_mat: Tensor, bias_mat: Tensor) -> Star:
    """
    Function which takes as input a Star and a halfspace defined by its coefficient matrix and bias vector
    and returns the Star resulting from the intersection of the input Star with the halfspace.

    """

    new_center = star.center
    new_basis_matrix = star.basis_matrix
    hs_pred_matrix = np.matmul(coef_mat, star.basis_matrix)
    hs_pred_bias = bias_mat - np.matmul(coef_mat, star.center)
    new_pred_matrix = np.vstack((star.predicate_matrix, hs_pred_matrix))
    new_pred_bias = np.vstack((star.predicate_bias, hs_pred_bias))

    new_star = Star(new_pred_matrix, new_pred_bias, new_center, new_basis_matrix)

    return new_star


def __mixed_step_relu(abs_input: Set[Star], var_index: int, refinement_flag: bool,
                      symb_lb: float = None, symb_ub: float = None) -> Set[Star]:
    abs_input = list(abs_input)
    abs_output = set()

    guard = 10e-15

    if symb_lb is None:
        symb_lb = -100

    if symb_ub is None:
        symb_ub = 100

    for i in range(len(abs_input)):

        is_pos_stable = False
        is_neg_stable = False
        lb, ub = None, None

        star = abs_input[i]

        # Check abstract bounds for stability
        if symb_lb >= guard:
            is_pos_stable = True
        elif symb_ub <= -guard:
            is_neg_stable = True
        else:
            lb, ub = star.get_bounds(var_index)

        if not star.is_empty:

            mask = np.identity(star.center.shape[0])
            mask[var_index, var_index] = 0

            if is_pos_stable or (lb is not None and lb >= 0):
                abs_output = abs_output.union({star})

            elif is_neg_stable or (ub is not None and ub <= 0):
                new_center = np.matmul(mask, star.center)
                new_basis_mat = np.matmul(mask, star.basis_matrix)
                new_pred_mat = star.predicate_matrix
                new_pred_bias = star.predicate_bias
                new_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat)
                abs_output = abs_output.union({new_star})

            else:

                if refinement_flag:

                    # Creating lower bound star.
                    lower_star_center = np.matmul(mask, star.center)
                    lower_star_basis_mat = np.matmul(mask, star.basis_matrix)
                    # Adding x <= 0 constraints to the predicate.
                    lower_predicate_matrix = np.vstack((star.predicate_matrix, star.basis_matrix[var_index, :]))

                    lower_predicate_bias = np.vstack((star.predicate_bias, -star.center[var_index]))
                    lower_star = Star(lower_predicate_matrix, lower_predicate_bias, lower_star_center,
                                      lower_star_basis_mat)

                    # Creating upper bound star.
                    upper_star_center = star.center
                    upper_star_basis_mat = star.basis_matrix
                    # Adding x >= 0 constraints to the predicate.
                    upper_predicate_matrix = np.vstack((star.predicate_matrix, -star.basis_matrix[var_index, :]))

                    upper_predicate_bias = np.vstack((star.predicate_bias, star.center[var_index]))
                    upper_star = Star(upper_predicate_matrix, upper_predicate_bias, upper_star_center,
                                      upper_star_basis_mat)

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
                    new_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat)

                    abs_output = abs_output.union({new_star})

    return abs_output


def mixed_single_relu_forward(star: Star, heuristic: str, params: List, layer_bounds: AbstractBounds) \
        -> Tuple[Set[Star], Optional[np.ndarray]]:
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
            if layer_bounds is not None and (layer_bounds.get_lower()[i] >= 0 or layer_bounds.get_upper()[i] < 0):
                n_areas.append(0)
            else:
                lb, ub = star.get_bounds(i)
                n_areas.append(-lb * ub / 2.0)

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

        if layer_bounds is None:
            for i in range(star.center.shape[0]):
                temp_abs_input = __mixed_step_relu(temp_abs_input, i, refinement_flags[i])
        else:
            for i in range(star.center.shape[0]):
                temp_abs_input = __mixed_step_relu(temp_abs_input, i, refinement_flags[i],
                                                   layer_bounds.get_lower()[i], layer_bounds.get_upper()[i])

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

    new_star = Star(new_predicate_matrix, new_predicate_bias, new_center, new_basis_matrix,
                    fixed_neurons=star.fixed_neurons)

    return {new_star}


def approx_relu_forward(star: Star, bounds: AbstractBounds, dim: int, start_idx: int = 0, layer_n: int = 1) -> Star:
    """
    Approximate abstract propagation for a ReLU layer starting from a
    specific index

    Parameters
    ----------
    star : Star
        The star to propagate in this layer
    bounds : AbstractBounds
        The bounds of this layer
    dim : int
        The number of neurons in this layer
    start_idx : int
        The neuron to start the propagation from

    Returns
    ----------
    Star
        The abstract star result from the propagation

    """

    # TODO: not using start_idx anymore. Check that it does not interfere with old implementations.
    #  In particular, need to check that fixed_neurons is properly passed and updated
    fixed_neurons = star.fixed_neurons

    # Set the transformation for inactive neurons to 0
    inactive = [i for i in range(dim) if check_stable(i, bounds) == -1]

    # Compute the set of unstable neurons.
    # Neuron i has been fixed before, so we don't need to
    # approximate it (as it might still appear unstable according to the bounds)
    unstable = [i for i in range(dim) if check_stable(i, bounds) == 0 and not (layer_n, i) in fixed_neurons]

    # Return if there are no unstable neurons
    if len(unstable) == 0:
        from pynever.strategies.search import mask_transformation_for_inactive_neurons
        new_basis_matrix, new_center = mask_transformation_for_inactive_neurons(inactive, star.basis_matrix, star.center)
        return Star(star.predicate_matrix, star.predicate_bias, new_center, new_basis_matrix, fixed_neurons=fixed_neurons)


    ## The rest is mostly about approximating unstable neurons
    unstable_count = len(unstable)
    lower_bounds = [bounds.get_lower()[neuron_n] for neuron_n in unstable]
    upper_bounds = [bounds.get_upper()[neuron_n] for neuron_n in unstable]

    ### =================== First, update the predicate ===================
    ## For every unstable neuron y we introduce a fresh variable z and
    # relate it to the input variables x via 3 constraints.
    #
    # (1)  z >= 0
    # (2)  z >= y = eq(x)                      // eq(x) is the equation that defines y from x,
    #                                          // it is stored in the basis of the star
    # (3)  z <= relu_slope * y + relu_shift

    ## For every unstable neuron we add 3 rows to lower_left_matrix
    # that correspond to the original x variables
    #
    # (1) zeros
    # (2) equation for the neuron in the basis matrix
    # (3) - the upper triangular relaxation, that is    - ub / (ub - lb) * equation
    def _get_left_matrix_for_unstable_neuron(neuron_n, lb, ub, star):
        from pynever.strategies.search import get_neuron_equation
        first_row = np.zeros(star.predicate_matrix.shape[1])
        second_row = get_neuron_equation(star, neuron_n)[0]
        third_row = - ub / (ub - lb) * get_neuron_equation(star, neuron_n)[0]
        return [first_row, second_row, third_row]

    lower_left_matrix = [
        _get_left_matrix_for_unstable_neuron(unstable[i], lower_bounds[i], upper_bounds[i], star)
        for i in range(unstable_count)
    ]
    lower_left_matrix = np.array(lower_left_matrix).reshape(3 * unstable_count, -1)

    ## For every unstable neuron we add a column [-1, -1, 1]^T to lower_right_matrix
    # that corresponds to the fresh variable z
    new_dimension_column = [[-1], [-1], [1]]
    zero_column = [[0], [0], [0]]
    lower_right_matrix = [
        [zero_column for _ in range(i)] + [new_dimension_column] + [zero_column for _ in range(i + 1, unstable_count)]
        for i in range(unstable_count)
    ]
    lower_right_matrix = np.array(lower_right_matrix).reshape(unstable_count, -1).transpose()

    ## The new predicate matrix is made of 4 blocks, [[1, 2], [3, 4]], where
    # 1 is the original predicate matrix, 2 is zeros,
    # 3 is lower_left_matrix and 4 is lower_right_matrix
    new_pred_matrix = np.block([
        [star.predicate_matrix, np.zeros((star.predicate_matrix.shape[0], unstable_count))],
        [lower_left_matrix, lower_right_matrix]
    ])

    ## The new predicate bias adds the shifts from the above constraints.
    # So for each unstable neuron we append a vector
    #           [0, -c, relu_slope * (c - lower_bound)]
    additional_bias = [[[0],
                        -star.center[unstable[i]],
                        (upper_bounds[i] / (upper_bounds[i] - lower_bounds[i])) * (star.center[unstable[i]] - lower_bounds[i])
                       ] for i in range(unstable_count)]
    additional_bias = np.array(additional_bias).reshape(-1, 1)
    # Stack the new values
    new_pred_bias = np.vstack([star.predicate_bias, additional_bias])

    ### =================== Second, update the basis ===================
    ## The new basis sets to 0 all unstable neurons and adds a 1 for the fresh variable z
    # Set the transformation for inactive and unstable neurons to 0
    from pynever.strategies.search import mask_transformation_for_inactive_neurons
    new_basis_matrix, new_center = mask_transformation_for_inactive_neurons(
        inactive + unstable, star.basis_matrix, star.center)

    # Add a 1 for each fresh variable z
    basis_height = star.basis_matrix.shape[0]
    additional_basis_columns = np.zeros((basis_height, unstable_count))
    for i in range(unstable_count):
        additional_basis_columns[unstable[i]][i] = 1

    new_basis_matrix = np.hstack((new_basis_matrix, additional_basis_columns))

    return Star(new_pred_matrix, new_pred_bias, new_center, new_basis_matrix, fixed_neurons=fixed_neurons)

def approx_relu_forward_orig(star: Star, bounds: AbstractBounds, dim: int, start_idx: int = 0, layer_n: int = 1) -> Star:
    """
    Approximate abstract propagation for a ReLU layer starting from a
    specific index

    Parameters
    ----------
    star : Star
        The star to propagate in this layer
    bounds : AbstractBounds
        The bounds of this layer
    dim : int
        The number of neurons in this layer
    start_idx : int
        The neuron to start the propagation from

    Returns
    ----------
    Star
        The abstract star result from the propagation

    """

    # TODO: not using start_idx anymore. Check that it does not interfere with old implementations.
    #  In particular, need to check that fixed_neurons is properly passed and updated
    fixed_neurons = star.fixed_neurons

    # Set the transformation for inactive neurons to 0
    inactive = [i for i in range(dim) if check_stable(i, bounds) == -1]
    #
    from pynever.strategies.search import mask_transformation_for_inactive_neurons
    new_basis_matrix, new_center = mask_transformation_for_inactive_neurons(inactive, star.basis_matrix, star.center)
    out_star = Star(star.predicate_matrix, star.predicate_bias, new_center, new_basis_matrix,
                    fixed_neurons=fixed_neurons)

    # Compute the set of unstable neurons.
    # Neuron i has been fixed before, so we don't need to
    # approximate it (as it might still appear unstable according to the bounds)
    unstable = [i for i in range(dim) if check_stable(i, bounds) == 0 and not (layer_n, i) in fixed_neurons]

    # Approximate unstable neurons
    for neuron_n in unstable:
        lb = bounds.get_lower()[neuron_n]
        ub = bounds.get_upper()[neuron_n]
        mask = np.identity(out_star.center.shape[0])
        mask[neuron_n, neuron_n] = 0

        col_c_mat = out_star.predicate_matrix.shape[1]
        row_c_mat = out_star.predicate_matrix.shape[0]

        c_mat_1 = np.zeros((1, col_c_mat + 1))
        c_mat_1[0, col_c_mat] = -1
        c_mat_2 = np.hstack((np.array([out_star.basis_matrix[neuron_n, :]]), -np.ones((1, 1))))
        coef_3 = - ub / (ub - lb)
        c_mat_3 = np.hstack((np.array([coef_3 * out_star.basis_matrix[neuron_n, :]]), np.ones((1, 1))))
        c_mat_0 = np.hstack((out_star.predicate_matrix, np.zeros((row_c_mat, 1))))

        d_0 = out_star.predicate_bias
        d_1 = np.zeros((1, 1))
        d_2 = -out_star.center[neuron_n] * np.ones((1, 1))
        d_3 = np.array([(ub / (ub - lb)) * (out_star.center[neuron_n] - lb)])

        new_pred_mat = np.vstack((c_mat_0, c_mat_1, c_mat_2, c_mat_3))
        new_pred_bias = np.vstack((d_0, d_1, d_2, d_3))

        new_center = np.matmul(mask, out_star.center)
        temp_basis_mat = np.matmul(mask, out_star.basis_matrix)
        temp_vec = np.zeros((out_star.basis_matrix.shape[0], 1))
        temp_vec[neuron_n, 0] = 1
        new_basis_mat = np.hstack((temp_basis_mat, temp_vec))

        out_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat, fixed_neurons=fixed_neurons)

    return out_star


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


def __recursive_step_sigmoid(star: Star, var_index: int, approx_level: int, lb: float, ub: float,
                             tolerance: float) -> \
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


def single_concat_forward(first_star: Star, second_star: Star) -> Set[Star]:
    """
    Utility function for the management of the forward for AbsConcatNode. It is outside
    the class scope since multiprocessing does not support parallelization with
    function internal to classes.

    """

    new_basis_matrix = np.zeros((first_star.basis_matrix.shape[0] + second_star.basis_matrix.shape[0],
                                 first_star.basis_matrix.shape[1] + second_star.basis_matrix.shape[1]))
    new_basis_matrix[0:first_star.basis_matrix.shape[0],
    0:first_star.basis_matrix.shape[1]] = first_star.basis_matrix
    new_basis_matrix[first_star.basis_matrix.shape[0]:,
    first_star.basis_matrix.shape[1]:] = second_star.basis_matrix

    new_center = np.vstack((first_star.center, second_star.center))

    new_predicate_matrix = np.zeros((first_star.predicate_matrix.shape[0] + second_star.predicate_matrix.shape[0],
                                     first_star.predicate_matrix.shape[1] + second_star.predicate_matrix.shape[1]))
    new_predicate_matrix[0:first_star.predicate_matrix.shape[0], 0:first_star.predicate_matrix.shape[1]] = \
        first_star.predicate_matrix
    new_predicate_matrix[first_star.predicate_matrix.shape[0]:, first_star.predicate_matrix.shape[1]:] = \
        second_star.predicate_matrix

    new_predicate_bias = np.vstack((first_star.predicate_bias, second_star.predicate_bias))

    new_star = Star(new_predicate_matrix, new_predicate_bias, new_center, new_basis_matrix)

    return {new_star}


def single_sum_forward(first_star: Star, second_star: Star) -> Set[Star]:
    """
    Utility function for the management of the forward for AbsSumNode. It is outside
    the class scope since multiprocessing does not support parallelization with
    function internal to classes.

    """

    new_basis_matrix = np.hstack((first_star.basis_matrix, second_star.basis_matrix))
    new_center = first_star.center + second_star.center

    new_predicate_matrix = np.zeros((first_star.predicate_matrix.shape[0] + second_star.predicate_matrix.shape[0],
                                     first_star.predicate_matrix.shape[1] + second_star.predicate_matrix.shape[1]))
    new_predicate_matrix[0:first_star.predicate_matrix.shape[0], 0:first_star.predicate_matrix.shape[1]] = \
        first_star.predicate_matrix
    new_predicate_matrix[first_star.predicate_matrix.shape[0]:, first_star.predicate_matrix.shape[1]:] = \
        second_star.predicate_matrix

    new_predicate_bias = np.vstack((first_star.predicate_bias, second_star.predicate_bias))

    new_star = Star(new_predicate_matrix, new_predicate_bias, new_center, new_basis_matrix)

    return {new_star}


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

    ref_node : SingleInputLayerNode
        Reference SingleInputLayerNode for the abstract transformer.

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


class AbsSingleInputLayerNode(AbsLayerNode):
    """
    An abstract class used for our internal representation of a Single Input Abstract Transformer Layer of an
    AbsNeural Network. Its concrete children correspond to real abstract interpretation network layers.

    Attributes
    ----------
    identifier : str
        Identifier of the AbsSingleInputLayerNode.

    ref_node : SingleInputLayerNode
        Reference SingleInputLayerNode for the abstract transformer.

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

    def __init__(self, identifier: str, ref_node: nodes.SingleInputLayerNode):
        super().__init__(identifier, ref_node)

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


class AbsMultiInputLayerNode(AbsLayerNode):
    """
    An abstract class used for our internal representation of a generic Abstract Transformer Multi Input Layer of an
    AbsNeural Network. Its concrete children correspond to real abstract interpretation network layers.

    Attributes
    ----------
    identifier : str
        Identifier of the AbsMultiInputLayerNode.

    ref_node : MultiInputLayerNode
        Reference MultiInputLayerNode for the abstract transformer.

    Methods
    ----------
    forward(List[AbsElement])
        Function which takes a list of AbsElement and compute the corresponding output AbsElement based on the abstract
        transformer.

    backward(RefinementState)
        Function which takes a reference to the refinement state and update both it and the state of the abstract
        transformer to control the refinement component of the abstraction. At present the function is just a
        placeholder for future implementations.

    """

    def __init__(self, identifier: str, ref_node: nodes.MultiInputLayerNode):
        super().__init__(identifier, ref_node)

    @abc.abstractmethod
    def forward(self, abs_inputs: List[AbsElement]) -> AbsElement:
        """
        Compute the output AbsElement based on the inputs AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_inputs : List[AbsElement]
            The input abstract elements.

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


class AbsFullyConnectedNode(AbsSingleInputLayerNode):
    """
    A class used for our internal representation of a Fully Connected Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the SingleInputLayerNode.

    ref_node : FullyConnectedNode
        SingleInputLayerNode di riferimento per l'abstract transformer.

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

        # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
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


class AbsReLUNode(AbsSingleInputLayerNode):
    """
    A class used for our internal representation of a ReLU Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the SingleInputLayerNode.

    ref_node : ReLUNode
        Reference SingleInputLayerNode for the abstract transformer.

    heuristic : str
        Heuristic used to decide the refinement level of the abstraction.
        At present can be only one of the following:
        - complete: for each star all the neurons are processed with a precise abstraction
        - mixed: for each star a given number of neurons is processed with a precise abstraction
        - overapprox: for each star all the neurons are processed with a coarse abstraction

    params : List
        Parameters for the heuristic of interest.
        It is a List with the number of neurons to process with a precise abstraction in this layer.

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
        self.layer_bounds = None
        self.n_areas = None

    def forward(self, abs_input: AbsElement, bounds: AbstractBounds = None) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.

        bounds : dict
            Optional bounds for this layer as computed by the previous

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.

        """

        if bounds is not None:
            self.layer_bounds = bounds

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
                                                                          itertools.repeat(self.params),
                                                                          itertools.repeat(self.layer_bounds)))
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
            result, areas = mixed_single_relu_forward(star, self.heuristic, self.params, self.layer_bounds)
            abs_output.stars = abs_output.stars.union(result)
            tot_areas = tot_areas + areas
            num_areas = num_areas + 1

        self.n_areas = tot_areas / num_areas

        return abs_output


class AbsSigmoidNode(AbsSingleInputLayerNode):
    """
    A class used for our internal representation of a Sigmoid transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the SingleInputLayerNode.

    ref_node : SigmoidNode
        Reference SingleInputLayerNode for the abstract transformer.

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


class AbsConcatNode(AbsMultiInputLayerNode):
    """
    A class used for our internal representation of a Concat Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the SingleInputLayerNode.

    ref_node : ConcatNode
        Reference SingleInputLayerNode for the abstract transformer.

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

    def __init__(self, identifier: str, ref_node: nodes.ConcatNode):
        super().__init__(identifier, ref_node)

    def forward(self, abs_inputs: List[AbsElement]) -> AbsElement:
        """
        Compute the output AbsElement based on the inputs AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_inputs : list[AbsElement]
            The input abstract elements.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.

        """

        all_starset = True
        for abs_input in abs_inputs:
            if not isinstance(abs_input, StarSet):
                all_starset = False

        if all_starset:
            return self.__starset_list_forward(abs_inputs)
        else:
            raise NotImplementedError

    def __starset_list_forward(self, abs_inputs: List[StarSet]) -> StarSet:

        # If we have to concatenate a list of starset we need to concatenate them in order:
        # the first one with the second one, the result with the third one and so on and so forth.

        abs_output = StarSet()
        for i in range(len(abs_inputs) - 1):
            if parallel:
                temp_starset = self.__parallel_concat_starset(abs_inputs[i], abs_inputs[i + 1])
            else:
                temp_starset = self.__concat_starset(abs_inputs[i], abs_inputs[i + 1])

            abs_output.stars = abs_output.stars.union(temp_starset.stars)

        return abs_output

    @staticmethod
    def __parallel_concat_starset(first_starset: StarSet, second_starset: StarSet) -> StarSet:

        my_pool = multiprocessing.Pool(multiprocessing.cpu_count())

        # We build the list of combination of stars between the two starset.
        unique_combination = []
        for first_star in first_starset.stars:
            for second_star in second_starset.stars:
                unique_combination.append((first_star, second_star))

        parallel_results = my_pool.starmap(single_concat_forward, unique_combination)

        my_pool.close()
        abs_output = StarSet()
        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    @staticmethod
    def __concat_starset(first_starset: StarSet, second_starset: StarSet) -> StarSet:

        abs_output = StarSet()
        for first_star in first_starset.stars:
            for second_star in second_starset.stars:
                abs_output.stars = abs_output.stars.union(single_concat_forward(first_star, second_star))

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


class AbsSumNode(AbsMultiInputLayerNode):
    """
    A class used for our internal representation of a Sum Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the SingleInputLayerNode.

    ref_node : SumNode
        Reference SingleInputLayerNode for the abstract transformer.

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

    def __init__(self, identifier: str, ref_node: nodes.SumNode):
        super().__init__(identifier, ref_node)

    def forward(self, abs_inputs: List[AbsElement]) -> AbsElement:
        """
        Compute the output AbsElement based on the inputs AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_inputs : list[AbsElement]
            The input abstract elements.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.

        """

        all_starset = True
        for abs_input in abs_inputs:
            if not isinstance(abs_input, StarSet):
                all_starset = False

        if all_starset:
            return self.__starset_list_forward(abs_inputs)
        else:
            raise NotImplementedError

    def __starset_list_forward(self, abs_inputs: List[StarSet]) -> StarSet:

        # If we have to concatenate a list of starset we need to concatenate them in order:
        # the first one with the second one, the result with the third one and so on and so forth.

        abs_output = StarSet()
        for i in range(len(abs_inputs) - 1):
            if parallel:
                temp_starset = self.__parallel_sum_starset(abs_inputs[i], abs_inputs[i + 1])
            else:
                temp_starset = self.__sum_starset(abs_inputs[i], abs_inputs[i + 1])

            abs_output.stars = abs_output.stars.union(temp_starset.stars)

        return abs_output

    @staticmethod
    def __parallel_sum_starset(first_starset: StarSet, second_starset: StarSet) -> StarSet:

        my_pool = multiprocessing.Pool(multiprocessing.cpu_count())

        # We build the list of combination of stars between the two starset.
        unique_combination = []
        for first_star in first_starset.stars:
            for second_star in second_starset.stars:
                unique_combination.append((first_star, second_star))

        parallel_results = my_pool.starmap(single_sum_forward, unique_combination)

        my_pool.close()
        abs_output = StarSet()
        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    @staticmethod
    def __sum_starset(first_starset: StarSet, second_starset: StarSet) -> StarSet:

        abs_output = StarSet()
        for first_star in first_starset.stars:
            for second_star in second_starset.stars:
                abs_output.stars = abs_output.stars.union(single_sum_forward(first_star, second_star))

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
    nodes : dict <str, AbsLayerNode>
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

    def get_children(self, node: AbsLayerNode) -> List[AbsLayerNode]:

        child_nodes = [self.nodes[child_node_id] for child_node_id in self.edges[node.identifier]]
        return child_nodes

    def get_parents(self, node: AbsLayerNode) -> List[AbsLayerNode]:

        parent_nodes = [self.nodes[parent_node_id] for parent_node_id, end_nodes_ids in self.edges.items() if
                        node.identifier in end_nodes_ids]

        return parent_nodes

    def has_parents(self, node: AbsLayerNode) -> bool:
        return len(self.get_parents(node)) != 0

    def has_children(self, node: AbsLayerNode) -> bool:
        return len(self.get_children(node)) != 0

    def get_roots(self) -> List[AbsLayerNode]:

        root_nodes = [root_node for root_node_id, root_node in self.nodes.items() if
                      not self.has_parents(root_node)]
        return root_nodes

    def get_leaves(self) -> List[AbsLayerNode]:

        leaf_nodes = [leaf_node for leaf_node_id, leaf_node in self.nodes.items() if
                      not self.has_children(leaf_node)]
        return leaf_nodes

    def remove_node(self, node: AbsLayerNode):

        for parent_node in self.get_parents(node):
            self.edges[parent_node.identifier].remove(node.identifier)

        self.edges.pop(node.identifier)
        self.nodes.pop(node.identifier)

        return

    def generic_add_node(self, node: AbsLayerNode, parents: Optional[List[AbsLayerNode]] = None,
                         children: Optional[List[AbsLayerNode]] = None):

        if parents is None:
            parents = []

        if children is None:
            children = []

        for parent_node in parents:

            if parent_node.identifier not in self.nodes.keys():
                raise Exception(f"Parent Node {parent_node.identifier} is not a node of the Network.")

        for child_node in children:

            if child_node.identifier not in self.nodes.keys():
                raise Exception(f"Child Node {child_node.identifier} is not a node of the Network.")

        self.nodes[node.identifier] = node
        self.edges[node.identifier] = [c_node.identifier for c_node in children]

        for parent in parents:
            self.edges[parent.identifier].append(node.identifier)

    def is_acyclic(self):

        aux_network = copy.deepcopy(self)
        root_nodes = aux_network.get_roots()
        topologically_sorted = []

        while len(root_nodes) > 0:

            temp_node = root_nodes[0]
            root_nodes.remove(temp_node)

            topologically_sorted.append(temp_node)
            for child_node in aux_network.get_children(temp_node):
                aux_network.edges[temp_node.identifier].remove(child_node.identifier)
                if not aux_network.has_parents(child_node):
                    root_nodes.append(child_node)

        has_edges = False
        for start_node_id, end_nodes_ids in aux_network.edges.items():
            if len(end_nodes_ids) > 0:
                has_edges = True

        return not has_edges

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
    add_node(SingleInputLayerNode)
        Procedure to add a new AbsLayerNode to the sequential AbsNeuralNetwork.

    get_first_node()
        Procedure to extract the first AbsLayerNode of the sequential AbsNeuralNetwork.

    get_next_node(SingleInputLayerNode)
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
        SingleInputLayerNode
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


class AbsAcyclicNetwork(AbsNeuralNetwork):

    def __init__(self, identifier: str, input_ids: List[str], input_edges: dict):
        super().__init__()
        self.identifier = identifier
        self.input_ids = input_ids
        self.input_edges = input_edges

    def add_node(self, node: AbsLayerNode, parents: Optional[List[AbsLayerNode]] = None,
                 children: Optional[List[AbsLayerNode]] = None):

        self.generic_add_node(node, parents, children)
        if not self.is_acyclic():
            self.remove_node(node)
            raise Exception(f"Adding {node.identifier} with the provided parents and children would create a cycle"
                            f" in the Network!")

    def get_node_inputs(self, node: AbsLayerNode):

        node_input_ids = [key for key, value in self.input_edges.items() if node.identifier in value]
        return node_input_ids

    def forward(self, abs_inputs: List[AbsElement]) -> List[AbsElement]:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformers.

        Parameters
        ----------
        abs_inputs : List[AbsElement]
            The input abstract element.

        Returns
        ----------
        List[AbsElement]
            The AbsElements resulting from the computation corresponding to the abstract transformer.
        """

        abs_input_ids = [abs_elem.identifier for abs_elem in abs_inputs]
        if set(abs_input_ids) != set(self.input_ids):
            raise Exception("The IDs of the Abstract Elements do not corresponds to the expected Input IDs!")

        if set(abs_input_ids) != set(self.input_edges.keys()):
            raise Exception(
                "The IDs of the Abstract Elements do not corresponds to the Keys of the Input Edges Dict!")

        if [] in self.input_edges.values():
            raise Exception("Every Input in the Input Edges Dictionary should have at least an Edge!")

        nodes_stack = self.get_roots()
        temp_abs_inputs = copy.deepcopy(abs_inputs)

        while nodes_stack.__len__() != 0:

            current_node = nodes_stack.pop(0)
            if not self.has_parents(current_node):
                input_ids = self.get_node_inputs(current_node)
            else:
                input_ids = [parent.identifier for parent in self.get_parents(current_node)]

            current_node_inputs = [a_input for a_input in temp_abs_inputs if a_input.identifier in input_ids]

            # TODO: At this time we need to check the difference between the inputs for multinputlayernodes and single
            # input layer nodes. Once nodes refactor is done it can be simplified.
            if isinstance(current_node, AbsSingleInputLayerNode):

                if len(current_node_inputs) > 1:
                    raise Exception(f"{current_node.__class__} should have a single input!")
                else:
                    current_abs_output = current_node.forward(current_node_inputs[0])

            elif isinstance(current_node, AbsMultiInputLayerNode):
                current_abs_output = current_node.forward(current_node_inputs)
            else:
                raise NotImplementedError

            current_abs_output.identifier = current_node.identifier
            temp_abs_inputs.append(current_abs_output)

            current_children = self.get_children(current_node)
            for child in current_children:
                if child not in nodes_stack:
                    nodes_stack.append(child)

        leaves_ids = [leaf.identifier for leaf in self.get_leaves()]
        final_outputs = [final_output for final_output in temp_abs_inputs if final_output.identifier in leaves_ids]

        return final_outputs

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        pass
