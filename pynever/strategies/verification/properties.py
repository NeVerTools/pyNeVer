from fractions import Fraction

import pynever.strategies.smt_reading as reading
from pynever import tensors
from pynever.exceptions import InvalidDimensionError
from pynever.strategies.abstraction.star import Star
from pynever.tensors import Tensor


class NeverProperty:
    """
    An abstract class used to represent a generic property for a NeuralNetwork.

    """

    def __init__(self, in_coef_mat: Tensor = None, in_bias_mat: Tensor = None,
                 out_coef_mat: list[Tensor] = None, out_bias_mat: list[Tensor] = None):
        self.in_coef_mat = in_coef_mat
        self.in_bias_mat = in_bias_mat
        self.out_coef_mat = out_coef_mat
        self.out_bias_mat = out_bias_mat

    def to_input_star(self) -> Star:
        """
        This method creates the input star based on the property specification

        Returns
        ----------
        Star
            The input star

        """

        return Star(self.in_coef_mat, self.in_bias_mat)

    def to_smt_file(self, filepath: str, input_id: str = 'X', output_id: str = 'Y'):
        """
        This method builds the SMT-LIB representation of the NeVerProperty, expressing
        the variables and the matrices as constraints in the corresponding logic

        Parameters
        ----------
        input_id : str, Optional
            Identifier of the input node (default: 'X')
        output_id : str, Optional
            Identifier of the output node (default: 'Y')
        filepath : str
            Path to the SMT-LIB file to create

        """

        with open(filepath, 'w+') as f:
            # Variables definition
            input_vars = [f"{input_id}_{i}" for i in range(self.in_coef_mat.shape[1])]
            if self.out_coef_mat:
                output_vars = [f"{output_id}_{i}" for i in range(self.out_coef_mat[0].shape[1])]
            else:
                output_vars = []

            f.write(';; --- INPUT VARIABLES ---\n')
            for v_name in input_vars:
                f.write(f"(declare-const {v_name} Real)\n")

            f.write('\n;; --- OUTPUT VARIABLES ---\n')
            for v_name in output_vars:
                f.write(f"(declare-const {v_name} Real)\n")

            # Constraints definition
            f.write('\n;; --- INPUT CONSTRAINTS ---\n')

            infix_in_constraints = self.__create_infix_constraints(input_vars, self.in_coef_mat, self.in_bias_mat)
            for c in infix_in_constraints:
                prefix_smt_row = reading.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                f.write(f"(assert {prefix_smt_row})\n")

            f.write('\n;; --- OUTPUT CONSTRAINTS ---\n')

            # Allow multiple output properties
            infix_output_properties = []
            for out_mat, out_bias in zip(self.out_coef_mat, self.out_bias_mat):
                infix_constraints = self.__create_infix_constraints(output_vars, out_mat, out_bias)
                infix_output_properties.append(infix_constraints)

            if len(infix_output_properties) == 1:
                for c in infix_output_properties[0]:
                    prefix_smt_row = reading.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                    f.write(f"(assert {prefix_smt_row})\n")
            else:
                s = '(assert (or '
                for p in infix_output_properties:
                    if len(p) == 1:
                        prefix_smt_row = reading.ExpressionTreeConverter().build_from_infix(p[0]).as_prefix()
                        s = s + '\n' + prefix_smt_row
                    else:
                        s = s + '(and '
                        for c in p:
                            prefix_smt_row = reading.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                            s = s + '\n' + prefix_smt_row
                        s = s + ')\n'
                s = s + '))'
                f.write(s)

    @staticmethod
    def __create_infix_constraints(variables: list, coef_mat: Tensor, bias_mat: Tensor) -> list:
        c_list = []

        for row in range(coef_mat.shape[0]):
            coef = coef_mat[row, :]
            bias = bias_mat[row][0]
            s = '('

            # Assign coefficients
            for k in range(len(coef)):
                c = coef[k]
                if c != 0:
                    s = s + f"({float(c)} * {variables[k]})"
                    if k < len(coef) - 1 and any(coef[k + 1:]):
                        s = s + ' + '

            # Add bias preventing exponential representation
            bias_repr = float(bias)

            if 'e' in str(bias_repr):
                bias_repr = Fraction(bias_repr)

            s = s + f") <= ({bias_repr})"
            c_list.append(s)

        return c_list


class VnnLibProperty(NeverProperty):
    """
    A concrete class used to represent a NeVer property for a NeuralNetwork. We assume that the hyperplane
    out_coef_mat * y <= out_bias_mat represent the unsafe region (i.e., the negation of the desired property).
    At present the input set must be defined as in_coef_mat * x <= in_bias_mat

    Attributes
    ----------
    in_coef_mat: Tensor
        Matrix of the coefficients for the input constraints.
    in_bias_mat: Tensor
        Matrix of the biases for the input constraints.
    out_coef_mat: List[Tensor]
        Matrices of the coefficients for the output constraints.
    out_bias_mat: List[Tensor]
        Matrices of the biases for the output constraints.

    """

    def __init__(self, filepath: str):
        smt_parser = reading.SmtPropertyParser(filepath)

        super().__init__(*smt_parser.parse_property())


class LocalRobustnessProperty(NeverProperty):
    """
    TODO

    sample : Tensor
    epsilon : float
    label : str
    max_output : bool

    """

    def __init__(self, sample: Tensor, epsilon: float, n_outputs: int, label: int, max_output: bool):
        super().__init__(*LocalRobustnessProperty.build_matrices(sample, epsilon, n_outputs, label, max_output))

    @staticmethod
    def build_matrices(sample: Tensor, epsilon: float, n_outputs: int, label: int, max_output: bool) -> tuple[
            Tensor, Tensor, list[Tensor], list[Tensor]]:

        if sample.shape[1] != 1:
            raise InvalidDimensionError('Wrong shape for the sample, should be mono-dimensional')

        # Input property
        n_dims = sample.shape[0]
        in_coef_mat = tensors.zeros((2 * n_dims, n_dims))
        in_bias_mat = tensors.zeros((2 * n_dims, 1))

        for i, x_i in enumerate(sample):

            if not isinstance(x_i, float):
                raise Exception

            in_coef_mat[2 * i, i] = 1
            in_coef_mat[2 * i + 1, i] = -1

            in_bias_mat[2 * i] = x_i + epsilon
            in_bias_mat[2 * i + 1] = -x_i + epsilon

        # Output property
        if label >= n_outputs:
            raise Exception

        out_coef_mat = tensors.zeros((n_outputs - 1, n_outputs))
        out_bias_mat = tensors.zeros((n_outputs - 1, 1))

        outputs = set(range(n_outputs))
        outputs.remove(label)
        for i in outputs:
            out_coef_mat[i, label] = 1
            out_coef_mat[i, i] = -1

        if not max_output:
            out_coef_mat = -out_coef_mat

        return in_coef_mat, in_bias_mat, [out_coef_mat], [out_bias_mat]
