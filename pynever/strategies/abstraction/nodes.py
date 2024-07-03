import abc
import math
import multiprocessing

import numpy as np

import pynever.nodes as nodes
from pynever.exceptions import InvalidDimensionError
from pynever.strategies.abstraction.star import AbsElement, Star, StarSet
from pynever.strategies.bounds_propagation.bounds import AbstractBounds
from pynever.strategies.verification.parameters import VerificationParameters


# TODO update method documentation


class RefinementState(abc.ABC):
    """
    A class used for the internal control of the refinement strategies/heuristics applied in the abstraction refinement
    step. At present is not still used and it is just an abstract placeholder. It will be used in future
    implementations.

    """
    pass


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

    def __init__(self, identifier: str, ref_node: nodes.ConcreteLayerNode,
                 parameters: VerificationParameters | None = None):
        super().__init__(identifier)
        self.ref_node = ref_node

    @abc.abstractmethod
    def forward(self, abs_input: AbsElement | list[AbsElement], bounds: AbstractBounds | None = None) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement | list[AbsElement]
            The input abstract element or a list of inputs.
        bounds: AbstractBounds | None
            The optional abstract bounds obtained by bound propagation

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

    def __init__(self, identifier: str, ref_node: nodes.FullyConnectedNode,
                 parameters: VerificationParameters | None = None):
        super().__init__(identifier, ref_node)
        self.ref_node = ref_node

    def forward(self, abs_input: AbsElement | list[AbsElement], bounds: AbstractBounds | None = None) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.
        bounds: AbstractBounds | None
            The optional abstract bounds obtained by bound propagation

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.

        """

        if isinstance(abs_input, list):
            if len(abs_input) != 1:
                raise Exception('There should only be one input element for this abstract node.')
            abs_input = abs_input[0]

        if isinstance(abs_input, StarSet):
            return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def __starset_forward(self, abs_input: StarSet) -> StarSet:

        with multiprocessing.Pool(multiprocessing.cpu_count()) as my_pool:
            parallel_results = my_pool.map(self._single_fc_forward, abs_input.stars)

        abs_output = StarSet()
        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    def _single_fc_forward(self, star: Star) -> set[Star]:
        """
        Utility function for the management of the forward for AbsFullyConnectedNode. It is outside
        the class scope since multiprocessing does not support parallelization with
        function internal to classes.

        """
        if self.ref_node.weight.shape[1] != star.basis_matrix.shape[0]:
            raise InvalidDimensionError("The shape of the weight matrix of the concrete node is different from the "
                                        "shape of the basis matrix")

        bias = self.ref_node.get_layer_bias_as_two_dimensional()

        new_basis_matrix = np.matmul(self.ref_node.weight, star.basis_matrix)
        new_center = np.matmul(self.ref_node.weight, star.center) + bias
        new_predicate_matrix = star.predicate_matrix
        new_predicate_bias = star.predicate_bias

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

    def __init__(self, identifier: str, ref_node: nodes.ReLUNode, parameters: VerificationParameters):

        super().__init__(identifier, ref_node)

        if not hasattr(parameters, 'heuristic') or not hasattr(parameters, 'neurons_to_refine'):
            raise Exception('Verification parameters must include attributes "heuristic" and "neurons_to_refine"')

        self.parameters = parameters
        self.layer_bounds = None
        self.n_areas = None

    def forward(self, abs_input: AbsElement | list[AbsElement], bounds: AbstractBounds | None = None) -> AbsElement:
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

        if isinstance(abs_input, list):
            if len(abs_input) != 1:
                raise Exception('There should only be one input element for this abstract node.')
            abs_input = abs_input[0]

        if bounds is not None:
            self.layer_bounds = bounds

        if isinstance(abs_input, StarSet):
            return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def __starset_forward(self, abs_input: StarSet) -> StarSet:

        with multiprocessing.Pool(multiprocessing.cpu_count()) as my_pool:
            parallel_results = my_pool.map(self._mixed_single_relu_forward, abs_input.stars)

        # Here we pop the first element of parameters.neurons_to_refine to preserve the layer ordering
        if hasattr(self.parameters, 'neurons_to_refine'):
            if self.parameters.neurons_to_refine is not None:
                self.parameters.neurons_to_refine.pop(0)
        else:
            # TODO check exception
            raise Exception('SSLP parameters must have "neurons_to_refine" attribute!')

        abs_output = StarSet()

        # This is used in mixed verification
        tot_areas = np.zeros(self.ref_node.get_input_dim())
        num_areas = 0

        for star_set, areas in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

            # Perform this code only if necessary
            if hasattr(self.parameters, 'compute_areas') and self.parameters.compute_areas:
                if star_set != set():
                    num_areas = num_areas + 1
                    tot_areas = tot_areas + areas

        if num_areas > 0:
            self.n_areas = tot_areas / num_areas

        return abs_output

    def _mixed_single_relu_forward(self, star: Star) -> tuple[set[Star], np.ndarray | None]:
        """
        Utility function for the management of the forward for AbsReLUNode. It is outside
        the class scope since multiprocessing does not support parallelization with
        function internal to classes.

        """

        temp_abs_input = {star}
        if star.check_if_empty():
            return set(), None

        n_areas = []

        # Perform this code only if necessary
        if hasattr(self.parameters, 'compute_areas') and self.parameters.compute_areas:

            for i in range(star.n_neurons):
                if (self.layer_bounds is not None
                        and (self.layer_bounds.get_lower()[i] >= 0 or self.layer_bounds.get_upper()[i] < 0)):
                    n_areas.append(0)
                else:
                    lb, ub = star.get_bounds(i)
                    n_areas.append(-lb * ub / 2.0)

            n_areas = np.array(n_areas)

        refinement_flags = []

        match self.parameters.heuristic:
            case 'complete':
                refinement_flags = [True for _ in range(star.n_neurons)]

            case 'overapprox':
                refinement_flags = [False for _ in range(star.n_neurons)]

            case 'mixed':
                # The first element corresponds to the current layer
                n_neurons = self.parameters.neurons_to_refine[0]

                if n_neurons > 0:
                    sorted_indexes = np.flip(np.argsort(n_areas))
                    index_to_refine = sorted_indexes[:n_neurons]
                else:
                    index_to_refine = []

                refinement_flags = []
                for i in range(star.n_neurons):
                    if i in index_to_refine:
                        refinement_flags.append(True)
                    else:
                        refinement_flags.append(False)

        for i in range(star.n_neurons):
            temp_abs_input = self.__mixed_step_relu(temp_abs_input, i, refinement_flags[i])

        return temp_abs_input, n_areas

    def __mixed_step_relu(self, abs_input: set[Star], var_index: int, refinement_flag: bool) -> set[Star]:
        symb_lb = None
        symb_ub = None
        if self.layer_bounds is not None:
            symb_lb = self.layer_bounds.get_lower()[var_index]
            symb_ub = self.layer_bounds.get_upper()[var_index]

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

                if is_pos_stable or (lb is not None and lb >= 0):
                    abs_output = abs_output.union({star})

                elif is_neg_stable or (ub is not None and ub <= 0):
                    abs_output = abs_output.union({star.create_negative_stable(var_index)})

                else:

                    if refinement_flag:

                        lower_star, upper_star = star.split(var_index)
                        abs_output = abs_output.union({lower_star, upper_star})

                    else:

                        abs_output = abs_output.union({star.create_approx(var_index, lb, ub)})

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

    def __init__(self, identifier: str, ref_node: nodes.SigmoidNode, parameters: VerificationParameters | None = None):
        super().__init__(identifier, ref_node)

        approx_levels = parameters.sigmoid_params.approx_levels

        if approx_levels is None:
            approx_levels = [0 for _ in range(ref_node.get_input_dim()[-1])]
        elif isinstance(approx_levels, int):
            approx_levels = [approx_levels for _ in range(ref_node.get_input_dim()[-1])]

        self.approx_levels = approx_levels

    def forward(self, abs_input: AbsElement, bounds: AbstractBounds | None = None) -> AbsElement:
        """
        Compute the output AbsElement based on the input AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_input : AbsElement
            The input abstract element.
        bounds: AbstractBounds | None
            The optional abstract bounds obtained by bound propagation

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.
        """

        if isinstance(abs_input, list):
            if len(abs_input) != 1:
                raise Exception('There should only be one input element for this abstract node.')
            abs_input = abs_input[0]

        if isinstance(abs_input, StarSet):
            return self.__starset_forward(abs_input)
        else:
            raise NotImplementedError

    def __starset_forward(self, abs_input: StarSet) -> StarSet:

        abs_output = StarSet()

        with multiprocessing.Pool(multiprocessing.cpu_count()) as my_pool:
            parallel_results = my_pool.map(self._single_sigmoid_forward, abs_input.stars)

        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    def _single_sigmoid_forward(self, star: Star) -> set[Star]:
        """
        Utility function for the management of the forward for AbsSigmoidNode. It is outside
        the class scope since multiprocessing does not support parallelization with
        function internal to classes.

        """

        tolerance = 0.01
        temp_abs_input = {star}
        for i in range(star.n_neurons):
            temp_abs_input = AbsSigmoidNode.__approx_step_sigmoid(temp_abs_input, i, self.approx_levels[i], tolerance)
            print(f"Index {i}, NumStar: {len(temp_abs_input)}")
        return temp_abs_input

    @staticmethod
    def __approx_step_sigmoid(abs_input: set[Star], var_index: int, approx_level: int, tolerance: float) -> set[Star]:
        abs_output = set()
        for star in abs_input:

            if not star.check_if_empty():
                lb, ub = star.get_bounds(var_index)

                if (lb < 0) and (ub > 0):
                    abs_output = abs_output.union(AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level,
                                                                                          lb, 0, tolerance))
                    abs_output = abs_output.union(AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level,
                                                                                          0, ub, tolerance))
                else:
                    abs_output = abs_output.union(AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level,
                                                                                          lb, ub, tolerance))

        return abs_output

    @staticmethod
    def __recursive_step_sigmoid(star: Star, var_index: int, approx_level: int, lb: float, ub: float,
                                 tolerance: float) -> set[Star]:
        sig_fod = AbsSigmoidNode.sig_fod
        sig = AbsSigmoidNode.sig

        if approx_level < 0:
            raise Exception("approx_level must be greater than or equal to 0")

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

        if not ((lb <= 0 and ub <= 0) or (lb >= 0 and ub >= 0)):
            raise Exception

        mask = np.identity(star.n_neurons)
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

            # TODO magic numbers?

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
                AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level - 1, lb, best_boundary,
                                                        tolerance))
            star_set = star_set.union(
                AbsSigmoidNode.__recursive_step_sigmoid(star, var_index, approx_level - 1, best_boundary, ub,
                                                        tolerance))

            return star_set

    @staticmethod
    def sig(x: float) -> float:
        """
        Utility function computing the logistic function of the input.

        """

        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def sig_fod(x: float) -> float:
        """
        Utility function computing the first order derivative of the logistic function of the input.

        """

        return math.exp(-x) / math.pow(1 + math.exp(-x), 2)

    @staticmethod
    def area_sig_triangle(lb: float, ub: float) -> float:
        """
        Utility function computing the area of the triangle defined by an upper bound and a lower bound on the
        logistic function. In particular is the triangle composed by the two tangents and line passing by the two
        bounds.

        """
        sig_fod = AbsSigmoidNode.sig_fod
        sig = AbsSigmoidNode.sig

        x_p = (ub * sig_fod(ub) - lb * sig_fod(lb)) / (sig_fod(ub) - sig_fod(lb)) - \
              (sig(ub) - sig(lb)) / (sig_fod(ub) - sig_fod(lb))

        y_p = sig_fod(ub) * (x_p - ub) + sig(ub)

        height = (abs(y_p - (sig(ub) - sig(lb)) / (ub - lb) * x_p + sig(lb) - lb * (sig(ub) - sig(lb)) / (ub - lb)) /
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

    def __init__(self, identifier: str, ref_node: nodes.ConcatNode, parameters: VerificationParameters | None = None):
        super().__init__(identifier, ref_node)

    def forward(self, abs_inputs: list[AbsElement], bounds: AbstractBounds | None = None) -> AbsElement:
        """
        Compute the output AbsElement based on the inputs AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_inputs : list[AbsElement]
            The input abstract elements.
        bounds: AbstractBounds | None
            The optional abstract bounds obtained by bound propagation

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.

        """

        if not isinstance(abs_inputs, list) or len(abs_inputs) < 2:
            raise Exception('There should be at least two input elements for this abstract node.')

        if all([isinstance(abs_input, StarSet) for abs_input in abs_inputs]):
            return self.__starset_list_forward(abs_inputs)
        else:
            raise NotImplementedError

    def __starset_list_forward(self, abs_inputs: list[StarSet]) -> StarSet:

        # If we have to concatenate a list of starset we need to concatenate them in order:
        # the first one with the second one, the result with the third one and so on and so forth.

        abs_output = StarSet()
        for i in range(len(abs_inputs) - 1):
            temp_starset = self.__concat_starset(abs_inputs[i], abs_inputs[i + 1])

            abs_output.stars = abs_output.stars.union(temp_starset.stars)

        return abs_output

    @staticmethod
    def __concat_starset(first_starset: StarSet, second_starset: StarSet) -> StarSet:

        with multiprocessing.Pool(multiprocessing.cpu_count()) as my_pool:
            # We build the list of combination of stars between the two starset.
            unique_combination = []
            for first_star in first_starset.stars:
                for second_star in second_starset.stars:
                    unique_combination.append((first_star, second_star))

            parallel_results = my_pool.starmap(AbsConcatNode._single_concat_forward, unique_combination)

        abs_output = StarSet()
        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

        return abs_output

    @staticmethod
    def _single_concat_forward(first_star: Star, second_star: Star) -> set[Star]:
        """
        Utility function for the management of the forward for AbsConcatNode. It is outside
        the class scope since multiprocessing does not support parallelization with
        function internal to classes.

        """

        new_basis_matrix = np.zeros((first_star.basis_matrix.shape[0] + second_star.basis_matrix.shape[0],
                                     first_star.basis_matrix.shape[1] + second_star.basis_matrix.shape[1]))
        new_basis_matrix[0:first_star.basis_matrix.shape[0], 0:first_star.basis_matrix.shape[1]] = \
            first_star.basis_matrix
        new_basis_matrix[first_star.basis_matrix.shape[0]:, first_star.basis_matrix.shape[1]:] = \
            second_star.basis_matrix

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

    def __init__(self, identifier: str, ref_node: nodes.SumNode, parameters: VerificationParameters | None = None):
        super().__init__(identifier, ref_node)

    def forward(self, abs_inputs: list[AbsElement], bounds: AbstractBounds | None = None) -> AbsElement:
        """
        Compute the output AbsElement based on the inputs AbsElement and the characteristics of the
        concrete abstract transformer.

        Parameters
        ----------
        abs_inputs : list[AbsElement]
            The input abstract elements.
        bounds: AbstractBounds | None
            The optional abstract bounds obtained by bound propagation

        Returns
        ----------
        AbsElement
            The AbsElement resulting from the computation corresponding to the abstract transformer.

        """

        if not isinstance(abs_inputs, list) or len(abs_inputs) < 2:
            raise Exception('There should be at least two input elements for this abstract node.')

        if all([isinstance(abs_input, StarSet) for abs_input in abs_inputs]):
            return self.__starset_list_forward(abs_inputs)
        else:
            raise NotImplementedError

    def __starset_list_forward(self, abs_inputs: list[StarSet]) -> StarSet:

        # If we have to concatenate a list of starset we need to concatenate them in order:
        # the first one with the second one, the result with the third one and so on and so forth.

        abs_output = StarSet()
        for i in range(len(abs_inputs) - 1):
            temp_starset = self.__sum_starset(abs_inputs[i], abs_inputs[i + 1])

            abs_output.stars = abs_output.stars.union(temp_starset.stars)

        return abs_output

    @staticmethod
    def __sum_starset(first_starset: StarSet, second_starset: StarSet) -> StarSet:

        with multiprocessing.Pool(multiprocessing.cpu_count()) as my_pool:
            # We build the list of combination of stars between the two starset.
            unique_combination = []
            for first_star in first_starset.stars:
                for second_star in second_starset.stars:
                    unique_combination.append((first_star, second_star))

            parallel_results = my_pool.starmap(AbsSumNode.single_sum_forward, unique_combination)

        abs_output = StarSet()
        for star_set in parallel_results:
            abs_output.stars = abs_output.stars.union(star_set)

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
