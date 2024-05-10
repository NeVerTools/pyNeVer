import abc
import itertools
import math
import multiprocessing

import numpy as np

import pynever.nodes as nodes
from pynever.strategies.abstraction import PARALLEL
from pynever.strategies.abstraction.star import AbsElement, Star, StarSet
from pynever.strategies.bp.bounds import AbstractBounds
from pynever.tensors import Tensor


# TODO change assert to exceptions


class RefinementState(abc.ABC):
    """
    A class used for the internal control of the refinement strategies/heuristics applied in the abstraction refinement
    step. At present is not still used and it is just an abstract placeholder. It will be used in future
    implementations.

    """

    raise NotImplementedError


class AbsLayerNode(nodes.LayerNode):
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

    def __init__(self, identifier: str, ref_node: nodes.ConcreteLayerNode):
        super().__init__(identifier)
        self.ref_node = ref_node

    @abc.abstractmethod
    def forward(self, abs_input: AbsElement | list[AbsElement]) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement | list[AbsElement]
            The input abstract element or a list of inputs.

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """

        raise NotImplementedError


class AbsFullyConnectedNode(AbsLayerNode):
    """
    A class used for our internal representation of a Fully Connected Abstract transformer.

    Attributes
    ----------
    identifier : str
        Identifier of the LayerNode.

    ref_node : FullyConnectedNode
        Reference LayerNode for the abstract transformer.

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

    # TODO move __starset_forward here?
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

        # TODO move expansion
        # Need to expand bias since they are memorized like one-dimensional vectors in FC nodes.
        if self.ref_node.bias.shape != (self.ref_node.weight.shape[0], 1):
            bias = np.expand_dims(self.ref_node.bias, 1)
        else:
            bias = self.ref_node.bias
        # TODO use with and remove repeat parameters
        parallel_results = my_pool.starmap(AbsFullyConnectedNode.single_fc_forward, zip(abs_input.stars,
                                                                                        itertools.repeat(
                                                                                            self.ref_node.weight),
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
        raise NotImplementedError

    @staticmethod
    def single_fc_forward(star: Star, weight: Tensor, bias: Tensor) -> set[Star]:
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

        new_star = Star(new_predicate_matrix, new_predicate_bias, new_center, new_basis_matrix)

        return {new_star}


class AbsReLUNode(AbsLayerNode):
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

    def __init__(self, identifier: str, ref_node: nodes.ReLUNode, heuristic: str, params: list):

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
            if PARALLEL:
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

        raise NotImplementedError

    def __parallel_starset_forward(self, abs_input: StarSet) -> StarSet:

        my_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        parallel_results = my_pool.starmap(AbsReLUNode.mixed_single_relu_forward, zip(abs_input.stars,
                                                                                      itertools.repeat(self.heuristic),
                                                                                      itertools.repeat(self.params),
                                                                                      itertools.repeat(
                                                                                          self.layer_bounds)))
        my_pool.close()

        abs_output = StarSet()

        tot_areas = np.zeros(self.ref_node.get_input_dim())
        num_areas = 0
        for star_set, areas in parallel_results:
            if star_set != set():
                num_areas = num_areas + 1
                tot_areas = tot_areas + areas
            abs_output.stars = abs_output.stars.union(star_set)

        self.n_areas = tot_areas / num_areas

        return abs_output

    @staticmethod
    def mixed_single_relu_forward(star: Star, heuristic: str, params: list, layer_bounds: AbstractBounds) \
            -> tuple[set[Star], np.ndarray | None]:
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
                    temp_abs_input = AbsReLUNode.__mixed_step_relu(temp_abs_input, i, refinement_flags[i])
            else:
                for i in range(star.center.shape[0]):
                    temp_abs_input = AbsReLUNode.__mixed_step_relu(temp_abs_input, i, refinement_flags[i],
                                                                   layer_bounds.get_lower()[i],
                                                                   layer_bounds.get_upper()[i])

            return temp_abs_input, n_areas

    @staticmethod
    def __mixed_step_relu(abs_input: set[Star], var_index: int, refinement_flag: bool,
                          symb_lb: float = None, symb_ub: float = None) -> set[Star]:
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

            if not star.check_if_empty():

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

    def __starset_forward(self, abs_input: StarSet) -> StarSet:
        """
        Forward function specialized for the concrete AbsElement StarSet.
        """

        abs_output = StarSet()
        tot_areas = np.zeros(self.ref_node.get_input_dim())
        num_areas = 0
        for star in abs_input.stars:
            result, areas = AbsReLUNode.mixed_single_relu_forward(star, self.heuristic, self.params, self.layer_bounds)
            abs_output.stars = abs_output.stars.union(result)
            tot_areas = tot_areas + areas
            num_areas = num_areas + 1

        self.n_areas = tot_areas / num_areas

        return abs_output


class AbsSigmoidNode(AbsLayerNode):
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

    def __init__(self, identifier: str, ref_node: nodes.SigmoidNode, approx_levels: int | list[int] | None = None):
        super().__init__(identifier, ref_node)

        if approx_levels is None:
            approx_levels = [0 for _ in range(ref_node.get_input_dim()[-1])]
        elif isinstance(approx_levels, int):
            approx_levels = [approx_levels for _ in range(ref_node.get_input_dim()[-1])]

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

        if PARALLEL:
            abs_output = StarSet()
            # TODO ???
            my_pool = multiprocessing.Pool(1)
            parallel_results = my_pool.starmap(AbsSigmoidNode.single_sigmoid_forward, zip(abs_input.stars,
                                                                                          itertools.repeat(
                                                                                              self.approx_levels)))
            my_pool.close()
            for star_set in parallel_results:
                abs_output.stars = abs_output.stars.union(star_set)
        else:
            abs_output = StarSet()
            for star in abs_input.stars:
                abs_output.stars = abs_output.stars.union(AbsSigmoidNode.single_sigmoid_forward(star,
                                                                                                self.approx_levels))

        return abs_output

    @staticmethod
    def single_sigmoid_forward(star: Star, approx_levels: list[int]) -> set[Star]:
        """
        Utility function for the management of the forward for AbsSigmoidNode. It is outside
        the class scope since multiprocessing does not support parallelization with
        function internal to classes.

        """

        tolerance = 0.01
        temp_abs_input = {star}
        for i in range(star.center.shape[0]):
            temp_abs_input = AbsSigmoidNode.__approx_step_sigmoid(temp_abs_input, i, approx_levels[i], tolerance)
            print(f"Index {i}, NumStar: {len(temp_abs_input)}")
        return temp_abs_input

    @staticmethod
    def __approx_step_sigmoid(abs_input: set[Star], var_index: int, approx_level: int, tolerance: float) -> set[Star]:
        abs_output = set()
        for star in abs_input:

            if not star.check_if_empty():
                lb, ub = star.get_bounds(var_index)

                if (lb < 0) and (ub > 0):
                    abs_output = abs_output.union(AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level, lb, 0,
                                                                           tolerance))
                    abs_output = abs_output.union(AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level, 0, ub,
                                                                           tolerance))
                else:
                    abs_output = abs_output.union(AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level, lb,
                                                                           ub, tolerance))

        return abs_output

    @staticmethod
    def __recursive_step_sigmoid(star: Star, var_index: int, approx_level: int, lb: float, ub: float,
                                 tolerance: float) -> set[Star]:
        # TODO y/n?
        sig_fod = AbsSigmoidNode.sig_fod
        sig = AbsSigmoidNode.sig
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
                area_1 = AbsSigmoidNode.area_sig_triangle(lb, boundary)
                area_2 = AbsSigmoidNode.area_sig_triangle(boundary, ub)
                if area_1 + area_2 < smallest_area:
                    smallest_area = area_1 + area_2
                    best_boundary = boundary

            star_set = set()
            star_set = star_set.union(
                AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level - 1, lb, best_boundary, tolerance))
            star_set = star_set.union(
                AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level - 1, best_boundary, ub, tolerance))

            return star_set

    @staticmethod
    def sig_fod(x: float) -> float:
        """
        Utility function computing the first order derivative of the logistic function of the input.

        """

        return math.exp(-x) / math.pow(1 + math.exp(-x), 2)

    @staticmethod
    def sig(x: float) -> float:
        """
        Utility function computing the logistic function of the input.

        """

        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def area_sig_triangle(lb: float, ub: float) -> float:
        """
        Utility function computing the area of the triangle defined by an upper bound and a lower bound on the
        logistic function. In particular is the triangle composed by the two tangents and line passing by the two bounds.

        """
        # TODO y/n?
        sig_fod = AbsSigmoidNode.sig_fod
        sig = AbsSigmoidNode.sig

        x_p = (ub * sig_fod(ub) - lb * sig_fod(lb)) / (sig_fod(ub) - sig_fod(lb)) - \
              (sig(ub) - sig(lb)) / (sig_fod(ub) - sig_fod(lb))

        y_p = sig_fod(ub) * (x_p - ub) + sig(ub)

        height = (abs(y_p - (sig(ub) - sig(lb)) / (ub - lb) * x_p + sig(lb) - lb * (sig(ub) - sig(lb)) / (ub - lb)) / \
                  math.sqrt(1 + math.pow((sig(ub) - sig(lb)) / (ub - lb), 2)))

        base = math.sqrt(math.pow(ub - lb, 2) + math.pow(sig(ub) - sig(lb), 2))

        return base * height / 2.0

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        raise NotImplementedError


class AbsConcatNode(AbsLayerNode):
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

    def forward(self, abs_inputs: list[AbsElement]) -> AbsElement:
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

    def __starset_list_forward(self, abs_inputs: list[StarSet]) -> StarSet:

        # If we have to concatenate a list of starset we need to concatenate them in order:
        # the first one with the second one, the result with the third one and so on and so forth.

        abs_output = StarSet()
        for i in range(len(abs_inputs) - 1):
            if PARALLEL:
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

        parallel_results = my_pool.starmap(AbsConcatNode.single_concat_forward, unique_combination)

        my_pool.close()
        abs_output = StarSet()
        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    @staticmethod
    def single_concat_forward(first_star: Star, second_star: Star) -> set[Star]:
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

    @staticmethod
    def __concat_starset(first_starset: StarSet, second_starset: StarSet) -> StarSet:

        abs_output = StarSet()
        for first_star in first_starset.stars:
            for second_star in second_starset.stars:
                abs_output.stars = abs_output.stars.union(AbsConcatNode.single_concat_forward(first_star, second_star))

        return abs_output

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        raise NotImplementedError


class AbsSumNode(AbsLayerNode):
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

    def forward(self, abs_inputs: list[AbsElement]) -> AbsElement:
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

    def __starset_list_forward(self, abs_inputs: list[StarSet]) -> StarSet:

        # If we have to concatenate a list of starset we need to concatenate them in order:
        # the first one with the second one, the result with the third one and so on and so forth.

        abs_output = StarSet()
        for i in range(len(abs_inputs) - 1):
            if PARALLEL:
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

        parallel_results = my_pool.starmap(AbsSumNode.single_sum_forward, unique_combination)

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
                abs_output.stars = abs_output.stars.union(AbsSumNode.single_sum_forward(first_star, second_star))

        return abs_output

    @staticmethod
    def single_sum_forward(first_star: Star, second_star: Star) -> set[Star]:
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

    def backward(self, ref_state: RefinementState):
        """
        Update the RefinementState. At present the function is just a placeholder for future implementations.

        Parameters
        ----------
        ref_state: RefinementState
            The RefinementState to update.
        """
        raise NotImplementedError
