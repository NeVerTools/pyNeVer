import abc
from fractions import Fraction

from pynever.strategies.abstraction.star import Star
import pynever.strategies.smt_reading as reading
from pynever.tensors import Tensor


class NeverProperty(abc.ABC):
    """
    An abstract class used to represent a generic property for a NeuralNetwork.
    """

    def __init__(self, in_coef_mat: Tensor = None, in_bias_mat: Tensor = None,
                 out_coef_mat: list[Tensor] = None, out_bias_mat: list[Tensor] = None):
        self.in_coef_mat = in_coef_mat
        self.in_bias_mat = in_bias_mat
        self.out_coef_mat = out_coef_mat
        self.out_bias_mat = out_bias_mat


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
        Matrixes of the coefficients for the output constraints.
    out_bias_mat: List[Tensor]
        Matrixes of the biases for the output constraints.

    """

    def __init__(self, filepath: str = '', input_name: str = 'X', output_name: str = 'Y'):
        smt_parser = reading.SmtPropertyParser(filepath, input_name, output_name)
        in_coef_mat, in_bias_mat, out_coef_mat, out_bias_mat = smt_parser.parse_property()

        super().__init__(in_coef_mat, in_bias_mat, out_coef_mat, out_bias_mat)

    def to_smt_file(self, input_id: str = 'X', output_id: str = 'Y', filepath: str = ''):
        """
        This method builds the SMT-LIB representation of the NeVerProperty, expressing
        the variables and the matrices as constraints in the corresponding logic

        Parameters
        ----------
        input_id : str, Optional
            Identifier of the input node (default: 'X')
        output_id : str, Optional
            Identifier of the output node (default: 'X')
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

    def to_input_star(self) -> Star:
        """
        This method creates the input star based on the property specification

        Returns
        ----------
        Star
            The input star

        """

        return Star(self.in_coef_mat, self.in_bias_mat)

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
                bias_repr = Fraction(bias)
            s = s + f") <= ({bias_repr})"
            c_list.append(s)

        return c_list
