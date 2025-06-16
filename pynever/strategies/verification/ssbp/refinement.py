"""
This file contains the branching refinement algorithms for the
bounds propagation over ReLU layers

"""
import numpy as np
import torch
from ortools.linear_solver import pywraplp

from pynever.networks import SequentialNetwork
from pynever.strategies.abstraction.bounds_propagation import BOUNDS_LOGGER
from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.abstraction.bounds_propagation.manager import BoundsManager
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.strategies.verification.ssbp.constants import RefinementTarget, NeuronSplit, BoundsDirection
from pynever.strategies.verification.statistics import VerboseBounds


class BoundsRefinement:
    """
    This class handles the refinement of the input bounds after a branch split
    """
    INPUT_DIMENSIONS_TO_REFINE = 50

    def __init__(self, direction: BoundsDirection):
        self.direction = direction
        self.logger = BOUNDS_LOGGER

    def branch_update_bounds(self, pre_branch_bounds: VerboseBounds, nn: SequentialNetwork, target: RefinementTarget,
                             fixed_neurons: dict) -> tuple[VerboseBounds, VerboseBounds]:
        """
        Update the bounds for after splitting the target neuron.
        Attempts to refine the input bounds for each of the two splits.
        If the input bounds have been updated, recomputes the bounds.
        """
        self.logger.debug("\tTarget {} "
                          "Overapprox. area {:10.4}".format(target,
                                                            pre_branch_bounds.statistics.approximation_info[
                                                                target.to_pair()]))

        input_bounds = pre_branch_bounds.numeric_pre_bounds[nn.get_id_from_index(0)]

        """
        NEGATIVE BRANCH
        """
        negative_branch_input = self.refine_input_bounds_after_split(
            pre_branch_bounds, nn, target, NeuronSplit.NEGATIVE, fixed_neurons
        )

        negative_bounds = None if negative_branch_input is None else (
            pre_branch_bounds if negative_branch_input == input_bounds else
            BoundsManager(nn, input_bounds=negative_branch_input).compute_bounds()
        )

        # self.logger.debug("\tNegative Stable count  {}  Volume {} --- {}".format(
        #     None if negative_bounds is None else "{:4}".format(
        #         negative_bounds.statistics.stability_info['stable_count']),
        #     None if negative_bounds is None else "{:10.4}".format(negative_bounds.statistics.overapprox_area['volume']),
        #     negative_branch_input))

        """
        POSITIVE BRANCH
        """
        positive_branch_input = self.refine_input_bounds_after_split(
            pre_branch_bounds, nn, target, NeuronSplit.POSITIVE, fixed_neurons
        )

        positive_bounds = None if positive_branch_input is None else (
            pre_branch_bounds if positive_branch_input == input_bounds else
            BoundsManager(nn, input_bounds=positive_branch_input).compute_bounds()
        )

        # self.logger.debug("\tPositive Stable count  {}  Volume {} --- {}".format(
        #     None if positive_bounds is None else "{:4}".format(
        #         positive_bounds.statistics.stability_info['stable_count']),
        #     None if positive_bounds is None else "{:10.4}".format(positive_bounds.statistics.overapprox_area['volume']),
        #     positive_branch_input))

        return negative_bounds, positive_bounds

    def refine_input_bounds_after_split(self, pre_branch_bounds: VerboseBounds, nn: SequentialNetwork,
                                        target: RefinementTarget, status: NeuronSplit, fixed_neurons: dict) \
            -> HyperRectangleBounds:
        """
        Given an unstable neuron y that we are going to constrain
        to be negative or positive according to the status,
        we recompute tighter input bounds of x=(x1,...,xn)
        for the solution space induced by this split.

        If y is set to be negative, we take its lower bound equation from the input variables:
            y >= c * x + b
        If y is set to be positive, we take its upper bound equation from the input variables:
            y <= c * x + b

        for all x coming from the hyper-rectangle [l,u] (i.e., li <= xi <=ui).

        If we are constraining y to be negative, we have the constraint
            c * x + b <= 0.
        If we are constraining y to be positive, we have the constraint
            c * x + b >= 0 or, in the normal form, - c * x - b <= 0.

        We recompute the bounds for x using the constraint.

        Parameters
        ----------
        pre_branch_bounds : VerboseBounds
            The bounds before the split
        nn : SequentialNetwork
            The neural network
        target : RefinementTarget
            The neuron to be split
        status : NeuronSplit
            The status of the neuron
        fixed_neurons : dict
            The dictionary of fixed neurons so far

        Returns
        -------
        Tighter input bounds induced by the split
        """
        # the bounds for the input layer that we try to refine
        input_bounds = pre_branch_bounds.numeric_pre_bounds[nn.get_first_node().identifier]

        # If the bounds have not been refined,
        # try to use constraints from all the fixed neurons
        # fixed_neurons = compute_fixed_but_unstable_wrt_bounds(pre_branch_bounds, fixed_neurons)
        if len(fixed_neurons) > 0:
            refined_bounds = BoundsRefinement.optimise_input_bounds_for_branch(
                fixed_neurons | {target.to_pair(): status.value}, pre_branch_bounds, nn
            )

        else:
            coef, shift = BoundsRefinement._get_equation_from_fixed_neuron(target, status.value, pre_branch_bounds, nn)
            refined_bounds = self._refine_input_bounds_for_equation(coef, shift, input_bounds)

        return refined_bounds

    @staticmethod
    def _choose_dimensions_to_consider(coef: torch.Tensor) -> list[int]:
        """
        This method performs an optimisation for a high-dimensional input
        """
        n_input_dimensions = len(coef)

        dimensions_to_consider = torch.Tensor(range(n_input_dimensions))

        if n_input_dimensions > BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE:
            # we will only consider the dimensions
            # with the coefficient that is large enough in absolute terms
            # and at most BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE
            percentage = 1 - BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE / n_input_dimensions
            cutoff_c = torch.quantile(abs(coef), percentage)
            mask = (abs(coef) > cutoff_c)
            dimensions_to_consider = torch.Tensor(dimensions_to_consider[mask])

        return [int(i.item()) for i in dimensions_to_consider]

    def _refine_input_bounds_for_equation(self, coef: torch.Tensor, shift: torch.Tensor,
                                          input_bounds: HyperRectangleBounds) \
            -> HyperRectangleBounds | None:
        """
        We have a constraint from the input variables
            c1 * x1 + ... + cn * xn + b <= 0

        for x1,...,xn coming from the hyper-rectangle [l,u] (i.e., li <= xi <=ui).

        We refine the bounds for x to the imposed solution space.
        For instance, for x1 we have

            x1 <= (-c2 * x2 - ... -cn * xn - b)/c1 if c1 is positive
            x1 >= (-c2 * x2 - ... -cn * xn - b)/c1 if c1 is negative

        Thus, when c1 > 0, we can compute a new upper bound of x1,
        and when c1 < 0, we can compute a new lower bound of x1.
        We do it using the standard interval arithmetics.

        If the new bound is inconsistent, e.g., the new upper bound is below the existing lower bound,
        it means the corresponding split/branch is not feasible. We return None.

        We only update the bound if it improves the previous one.
        """
        refined_input_bounds = input_bounds

        for i in BoundsRefinement._choose_dimensions_to_consider(coef):
            # Refine the bounds for each input dimension
            i_bounds = BoundsRefinement.refine_input_dimension(refined_input_bounds, coef, shift, i)

            if i_bounds is None:
                self.logger.info("!! Split is infeasible !!")
                # The split is infeasible
                return None

            elif i_bounds == 0:
                # No changes
                pass

            else:
                # self.logger.info(f"!! Bounds refined for branch !!")
                # Bounds have been refined

                if refined_input_bounds == input_bounds:
                    # Only create a new copy of the bounds if there was a change
                    refined_input_bounds = input_bounds.clone()
                # Update the bounds
                refined_input_bounds.get_lower()[i] = i_bounds[0]
                refined_input_bounds.get_upper()[i] = i_bounds[1]

        return refined_input_bounds

    def _refine_input_bounds_for_branch(self, fixed_neurons: dict, target: RefinementTarget, value: NeuronSplit,
                                        input_bounds: HyperRectangleBounds, nn: SequentialNetwork,
                                        pre_branch_bounds: VerboseBounds) -> HyperRectangleBounds | None:
        """
        We assume that the refinement is done when setting the equations to be <= 0
        """
        # Collecting the equations in normal form (<= 0) from all the fixes, including the latest
        # If value is 0, we take the lower bound.
        # Otherwise, we take the negation of the upper bound.
        equations = BoundsRefinement.get_equations_from_fixed_neurons(fixed_neurons, pre_branch_bounds, nn)
        coef, shift = BoundsRefinement._get_equation_from_fixed_neuron(target, value.value, pre_branch_bounds, nn)

        input_bounds = self._refine_input_bounds_for_equation(coef, shift, input_bounds)
        if input_bounds is None:
            return None

        # The rest is similar to _refine_input_bounds,
        # but we get two different equations for each input dimension i,
        # obtained as the sum of the equations where i appears with the same sign
        refined_input_bounds = input_bounds
        for i in BoundsRefinement._choose_dimensions_to_consider(coef):
            i_bounds = self._refine_input_dimension_for_neuron_and_branch(input_bounds, equations,
                                                                          coef, shift, i)

            if i_bounds is None:
                self.logger.info("!! Split is infeasible !!")
                # The split is infeasible
                return None

            elif i_bounds == 0:
                # No changes
                pass

            else:
                # self.logger.info(f"!! Bounds refined for branch !!")
                # Bounds have been refined

                if refined_input_bounds == input_bounds:
                    # Only create a new copy of the bounds if there was a change
                    refined_input_bounds = input_bounds.clone()
                # Update the bounds
                refined_input_bounds.get_lower()[i] = i_bounds[0]
                refined_input_bounds.get_upper()[i] = i_bounds[1]

        return refined_input_bounds

    def _refine_input_bounds_for_branch_naive(self, branch: dict, input_bounds: HyperRectangleBounds,
                                              nn: SequentialNetwork, pre_branch_bounds: VerboseBounds) \
            -> HyperRectangleBounds | None:
        """
        We assume that the refinement is done when setting the equations to be <= 0
        """
        # Collecting the equations in normal form (<= 0) from all the fixes, including the latest
        # If value is 0, we take the lower bound.
        # Otherwise, we take the negation of the upper bound.
        equations = self.get_equations_from_fixed_neurons(branch, pre_branch_bounds, nn)
        coefs = equations.get_matrix()
        shifts = equations.get_offset()

        # The rest is similar to _refine_input_bounds,
        # but we get two different equations for each input dimension i,
        # obtained as the sum of the equations where i appears with the same sign
        n_input_dimensions = len(coefs[0])

        all_dimensions = torch.Tensor(range(n_input_dimensions))
        dimensions_to_consider = []
        # An optimisation for very high-dimensional inputs
        if n_input_dimensions > BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE:
            # we will only consider the dimensions
            # with the aggregated coefficient that is large enough in absolute terms
            # and at most BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE
            percentage = 1 - BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE / n_input_dimensions

            filtering_pos_coefs = abs(
                torch.Tensor([coefs[(coefs[:, i] > 0), i].sum() for i in range(n_input_dimensions)]))
            cutoff_c = torch.quantile(filtering_pos_coefs, percentage)
            dimensions_to_consider.append(all_dimensions[(filtering_pos_coefs > cutoff_c)])

            filtering_neg_coefs = abs(
                torch.Tensor([coefs[(coefs[:, i] < 0), i].sum() for i in range(n_input_dimensions)]))
            cutoff_c = torch.quantile(filtering_neg_coefs, percentage)
            dimensions_to_consider.append(all_dimensions[(filtering_neg_coefs > cutoff_c)])

        else:
            dimensions_to_consider.extend([all_dimensions, all_dimensions])

        refined_input_bounds = input_bounds
        for dimensions, sign in zip(dimensions_to_consider, ["pos", "neg"]):
            for i in dimensions:
                # For each input dimension i, we select the subset of the equations
                # with the same coefficient sign for i and optimise for the sum of those equations
                if sign == "pos":
                    mask = (coefs[:, i] > 0)
                else:
                    mask = (coefs[:, i] < 0)

                coef_i = coefs[mask, :]
                shift_i = shifts[mask].sum()

                if len(coef_i) <= 1:
                    # none or one equation have been selected. We want to combine at least two equations.
                    # Nothing to be done
                    continue

                coef_i = coef_i.sum(dim=0)
                i_bounds = self.refine_input_dimension(refined_input_bounds, coef_i, shift_i, i)

                if i_bounds is None:
                    self.logger.info(f"!! Split is infeasible !! {coef_i[i]}")
                    # The split is infeasible
                    return None

                elif i_bounds == 0:
                    # No changes
                    pass

                else:
                    self.logger.info(f"!! Bounds refined for branch !! {coef_i[i]}")
                    # Bounds have been refined
                    if refined_input_bounds == input_bounds:
                        # Only create a new copy of the bounds if there was a change
                        refined_input_bounds = input_bounds.clone()
                    # Update the bounds
                    refined_input_bounds.get_lower()[i] = i_bounds[0]
                    refined_input_bounds.get_upper()[i] = i_bounds[1]

        return refined_input_bounds

    def _refine_input_dimension_for_neuron_and_branch(self, input_bounds: HyperRectangleBounds,
                                                      equations: LinearFunctions, coef: torch.Tensor,
                                                      shift: torch.Tensor,
                                                      i: int) -> tuple | int | None:

        coefs = equations.get_matrix()
        shifts = equations.get_offset()

        # Find equations where the coefficient is the same sign as coef[i]
        if coef[i] > 0:
            mask = (coefs[:, i] > 0)
        else:  # coef[i] < 0:
            mask = (coefs[:, i] < 0)

        coefs1 = coefs[mask, :]
        shifts1 = shifts[mask]

        # If no equations have been selected, still can try to refine for the new equation
        if len(coefs1) == 0:
            return 0

        best_i_bounds = input_bounds.get_dimension_bounds(i)

        # For every other dimension j, choose an equation eq2 where
        # coefficient j is the opposite sign of coef[j].
        # The idea is to combine eq1 and eq2 so that
        # coefficient i is 1 and
        # coefficient j is 0.

        n_input_dimensions = len(coef)

        for j in [h for h in range(n_input_dimensions) if h != i]:
            if coef[j] > 0:
                mask1 = (coefs1[:, j] < 0)

            elif coef[j] < 0:
                mask1 = (coefs1[:, j] > 0)

            else:
                # Maybe try to refine using the equation eq1?
                continue

            coefs2 = coefs1[mask1, :]
            shifts2 = shifts1[mask1]

            for n in range(len(coefs2)):
                eq2_coef = coefs2[n]
                eq2_shift = shifts2[n]

                k = -coef[j] / eq2_coef[j]

                # in this equation coefficient j is 0
                combined_coef = coef + k * eq2_coef
                combined_shift = shift + k * eq2_shift

                i_bounds = self.refine_input_dimension(input_bounds, combined_coef, combined_shift, i)
                if i_bounds is None:
                    # The split is infeasible
                    return None

                elif i_bounds != 0:
                    best_i_bounds = max(best_i_bounds[0], i_bounds[0]), min(best_i_bounds[1], i_bounds[1])

        if best_i_bounds != input_bounds.get_dimension_bounds(i):
            return best_i_bounds

        return 0

    def compute_refines_input_by(self, unstable: list, fixed_neurons: dict, bounds: VerboseBounds, network) \
            -> list[tuple[tuple[str, int], int]]:

        input_bounds = bounds.numeric_pre_bounds[network.get_first_node().identifier]

        differences = list()
        for (layer_id, neuron_n) in unstable:
            negative_branch_input = self.refine_input_bounds_after_split(
                bounds, network, RefinementTarget(layer_id, neuron_n), NeuronSplit.NEGATIVE, fixed_neurons)
            positive_branch_input = self.refine_input_bounds_after_split(
                bounds, network, RefinementTarget(layer_id, neuron_n), NeuronSplit.POSITIVE, fixed_neurons)

            if negative_branch_input is not None and positive_branch_input is not None:
                diff = \
                    ((negative_branch_input.get_lower() - input_bounds.get_lower()).sum() +
                     (input_bounds.get_upper() - negative_branch_input.get_upper()).sum() +
                     (input_bounds.get_upper() - positive_branch_input.get_upper()).sum() +
                     (positive_branch_input.get_lower() - input_bounds.get_lower()).sum())
            else:
                diff = 100

            if diff != 0:
                differences.append(((layer_id, neuron_n), diff))

        differences = sorted(differences, key=lambda x: x[1], reverse=True)

        return differences

    def branch_bisect_input(self, bounds: VerboseBounds, nn: SequentialNetwork, fixed_neurons: dict) \
            -> tuple[VerboseBounds, VerboseBounds]:

        input_bounds = bounds.numeric_pre_bounds[nn.get_first_node().identifier]

        lower_half, upper_half = BoundsRefinement.bisect_an_input_dimension(input_bounds)

        negative_bounds = BoundsManager(nn, input_bounds=lower_half).compute_bounds()
        positive_bounds = BoundsManager(nn, input_bounds=upper_half).compute_bounds()

        # self.logger.debug("\tBisect1 Stable count  {}  Volume {} --- {}".format(
        #     None if negative_bounds is None else "{:4}".format(
        #         negative_bounds.statistics.stability_info['stable_count']),
        #     None if negative_bounds is None else "{:10.4}".format(negative_bounds.statistics.overapprox_area['volume']),
        #     lower_half))

        # self.logger.debug("\tBisect2 Stable count  {}  Volume {} --- {}".format(
        #     None if positive_bounds is None else "{:4}".format(
        #         positive_bounds.statistics.stability_info['stable_count']),
        #     None if positive_bounds is None else "{:10.4}".format(positive_bounds.statistics.overapprox_area['volume']),
        #     upper_half))

        return negative_bounds, positive_bounds

    @staticmethod
    def bisect_an_input_dimension(input_bounds: HyperRectangleBounds) -> tuple[
        HyperRectangleBounds, HyperRectangleBounds]:

        diff = input_bounds.get_upper() - input_bounds.get_lower()
        widest_dim = torch.argmax(diff)
        mid = diff[widest_dim] / 2

        lower_half = input_bounds.clone()
        upper_half = input_bounds.clone()

        lower_half.upper[widest_dim] = lower_half.lower[widest_dim] + mid
        upper_half.lower[widest_dim] = lower_half.upper[widest_dim]

        return lower_half, upper_half

    @staticmethod
    def get_equations_from_fixed_neurons(fixed_neurons: dict, bounds: VerboseBounds,
                                         nn: SequentialNetwork) -> LinearFunctions:
        """
        Extract the constraints in the normal from
            equation <= 0
        imposed by fixing neurons, given their symbolic preactivation bounds.

        The assumption is that if a neuron y is constrained to be negative, then
            lower_bound <= y <= 0
        gives as the constraint
            lower_bound <= 0.
        Conversely, if y is constrained to be positive, then we have that upper_bound >= y >= 0.
        In the normal form it gives us
            -upper_bound <= 0.
        """
        coefs = []
        shifts = []
        for ((layer_id, neuron_n), value) in fixed_neurons.items():
            coef, shift = BoundsRefinement._get_equation_from_fixed_neuron(
                RefinementTarget(layer_id, neuron_n), value, bounds, nn
            )
            coefs.append(coef)
            shifts.append(shift)

        return LinearFunctions(torch.stack(coefs), torch.stack(shifts))

    @staticmethod
    def _get_equation_from_fixed_neuron(target: RefinementTarget, value: int, bounds: VerboseBounds, nn) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """
        See _get_equations_from_fixed_neurons
        """
        symbolic_preact_bounds = \
            BoundsManager.get_symbolic_preactivation_bounds_at(bounds, nn.nodes[target.layer_id], nn)[0]

        if value == 0:
            # The linear equation for the upper bound of the target neuron
            coef = symbolic_preact_bounds.get_lower().get_matrix()[target.neuron_idx]
            shift = symbolic_preact_bounds.get_lower().get_offset()[target.neuron_idx]

        else:  # sign == NeuronSplit.POSITIVE:
            # The negated linear equation for the lower bound of the target neuron
            coef = -symbolic_preact_bounds.get_upper().get_matrix()[target.neuron_idx]
            shift = -symbolic_preact_bounds.get_upper().get_offset()[target.neuron_idx]

        return coef, shift

    @staticmethod
    def refine_input_dimension(input_bounds: HyperRectangleBounds, coef: torch.Tensor, shift: torch.Tensor,
                               i: int) -> tuple[float, float] | int | None:
        """
        We are given the constraint
            coef * (x1,...,xn) + shift <= 0

        See _refine_input_bounds for more information.

        We are refining the bounds for the input dimension i:

            xi <= (-c1 * x1 - ... -cn * xn - b)/ci if ci is positive -- we refine the upper bound
            xi >= (-c1 * x1 - ... -cn * xn - b)/ci if ci is negative -- we refine the lower bound

        Returns
        -------
        None    if the constraint is infeasible
        0       if no changes
        (l, u)  the new bounds for input dimension i
        """
        c = coef[i]

        if c == 0:
            return None

        # the rest is moved to the other side, so we have the minus and divided by c
        negated_rem_coef = - torch.Tensor(list(coef[:i]) + list(coef[i + 1:])) / c
        shift_div_c = - shift / c
        pos_rem_coef = torch.max(torch.zeros(len(coef) - 1), negated_rem_coef)
        neg_rem_coef = torch.min(torch.zeros(len(coef) - 1), negated_rem_coef)

        rem_lower_input_bounds = torch.Tensor(
            list(input_bounds.get_lower()[:i]) + list(input_bounds.get_lower()[i + 1:]))
        rem_upper_input_bounds = torch.Tensor(
            list(input_bounds.get_upper()[:i]) + list(input_bounds.get_upper()[i + 1:]))

        if c > 0:
            # compute maximum of xi, xi <= (-coefi * rem_xi - b)/c
            new_upper_i = pos_rem_coef.dot(rem_upper_input_bounds) + \
                          neg_rem_coef.dot(rem_lower_input_bounds) + shift_div_c

            if new_upper_i < input_bounds.get_lower()[i]:
                # infeasible branch
                return None

            elif new_upper_i < input_bounds.get_upper()[i]:
                # from pynever.strategies.verification.ssbp.intersection import compute_input_new_max
                # new_upper = compute_input_new_max(coef, shift, input_bounds, i)
                # if new_upper != new_upper_i:
                #     print("Different new upper", new_upper_i, new_upper)
                return input_bounds.get_lower()[i].item(), new_upper_i

        elif c < 0:
            # compute minimum of xi, xi >= (-coefi * rem_xi - b)/c
            new_lower_i = pos_rem_coef.dot(rem_lower_input_bounds) + \
                          neg_rem_coef.dot(rem_upper_input_bounds) + shift_div_c

            if new_lower_i > input_bounds.get_upper()[i]:
                # infeasible branch
                return None

            elif new_lower_i > input_bounds.get_lower()[i]:
                # from pynever.strategies.verification.ssbp.intersection import compute_input_new_min
                # new_lower = compute_input_new_min(coef, shift, input_bounds, i)
                # if new_lower != new_lower_i:
                #     print("Different new lower", new_lower_i, new_lower)
                return new_lower_i, input_bounds.get_upper()[i].item()

        return 0

    @staticmethod
    def optimise_input_bounds_for_branch(fixed_neurons: dict, bounds: VerboseBounds, nn) -> VerboseBounds | None:
        """
        Optimises input bounds by building a MILP that has
        input variables and, for each fixed neuron, a constraint using its symbolic lower or upper bound.
        The solves for each input variable two optimisation problems: minimising and maximising it.
        """
        input_bounds = bounds.numeric_pre_bounds[nn.get_first_node().identifier]
        n_input_dimensions = input_bounds.get_size()

        solver = pywraplp.Solver("", pywraplp.Solver.CLP_LINEAR_PROGRAMMING)

        input_vars = np.array([
            solver.NumVar(input_bounds.get_lower()[j].item(), input_bounds.get_upper()[j].item(), f'alpha_{j}')
            for j in range(n_input_dimensions)])

        # The constraints from fixing the neurons
        equations = BoundsRefinement.get_equations_from_fixed_neurons(fixed_neurons, bounds, nn)

        # This way of encoding allows to access the dual solution
        worker_constraints = {}
        infinity = solver.infinity()
        for constr_n in range(len(equations.matrix)):
            # solver.Add(input_vars.dot(equations.matrix[i]) + equations.offset[i] <= 0)
            # -infinity <= eq <= 0
            worker_constraints[constr_n] = solver.Constraint(-infinity, -equations.offset[constr_n].item(),
                                                             'c[%i]' % constr_n)
            for input_var_n in range(n_input_dimensions):
                worker_constraints[constr_n].SetCoefficient(input_vars[input_var_n],
                                                            equations.matrix[constr_n][input_var_n].item())

        ## The actual optimisation part
        new_input_bounds = input_bounds.clone()
        bounds_improved = False

        dimensions_to_consider = torch.Tensor(range(n_input_dimensions))
        # An optimisation for very high-dimensional inputs
        if n_input_dimensions > BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE:
            # we will only consider the dimensions
            # with the coefficient that is large enough in absolute terms
            # and at most BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE

            # This part needs checking
            percentage = 1 - BoundsRefinement.INPUT_DIMENSIONS_TO_REFINE / n_input_dimensions
            max_coefs = abs(equations.matrix).max(axis=0)
            cutoff_c = torch.quantile(max_coefs, percentage)
            all_dimensions = torch.Tensor(range(n_input_dimensions))
            dimensions_to_consider = all_dimensions[(max_coefs > cutoff_c)]

        for idx in dimensions_to_consider:
            i_dim = int(idx)
            solver.Maximize(input_vars[i_dim])
            status = solver.Solve()

            new_lower, new_upper = input_bounds.get_dimension_bounds(i_dim)
            if status == pywraplp.Solver.INFEASIBLE:
                return None

            elif status == pywraplp.Solver.OPTIMAL:
                if input_vars[i_dim].solution_value() < new_upper:
                    # dual_sol = [worker_constraints[i].dual_value() for i in worker_constraints]
                    # self.logger.debug(f"Dual solution: {dual_sol}")

                    # eq_mult = torch.Tensor([worker_constraints[i].dual_value() for i in worker_constraints])
                    # coef = -(eq_mult.reshape(-1, 1) * equations.matrix).sum(axis=0)
                    # shift = -(eq_mult * equations.offset).sum()
                    # print("Equation", list(coef), shift)

                    new_upper = input_vars[i_dim].solution_value()
                    bounds_improved = True

            solver.Minimize(input_vars[i_dim])
            status = solver.Solve()

            if status == pywraplp.Solver.INFEASIBLE:
                return None

            elif status == pywraplp.Solver.OPTIMAL:
                if input_vars[i_dim].solution_value() > new_lower:
                    # dual_sol = [worker_constraints[i].dual_value() for i in worker_constraints]
                    # self.logger.debug(f"Dual solution: {dual_sol}")

                    # eq_mult = torch.Tensor([worker_constraints[i].dual_value() for i in worker_constraints])
                    # coef = -(eq_mult.reshape(-1, 1) * equations.matrix).sum(axis=0)
                    # shift = -(eq_mult * equations.offset).sum()
                    # print("Equation", list(coef), shift)

                    new_lower = input_vars[i_dim].solution_value()
                    bounds_improved = True

            new_input_bounds.get_lower()[i_dim] = new_lower
            new_input_bounds.get_upper()[i_dim] = new_upper

        if bounds_improved:
            return new_input_bounds

        return input_bounds
