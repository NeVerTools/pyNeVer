"""
This module defines neural network verification properties. The most general representation
defines a single input property expressed as a linear combination of input variables, while
the output property is a list of linear inequalities.
"""

from fractions import Fraction

import torch

from pynever.exceptions import InvalidDimensionError
from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.abstraction.linearfunctions import LinearFunctions
from pynever.strategies.abstraction.star import Star
from pynever.strategies.parser import expression, vnnlib


# TODO handle sets of properties?
# TODO specify convex properties

class NeverProperty:
    """
    An abstract class used to represent a generic property for a :class:`~pynever.networks.NeuralNetwork`.

    Attributes
    ----------
    in_matrix: torch.Tensor
        Matrix of the coefficients for the input constraints.
    in_bias: torch.Tensor
        Matrix of the biases for the input constraints.
    out_matrix_list: list[torch.Tensor]
        Matrices of the coefficients for the output constraints.
    out_bias_list: list[torch.Tensor]
        Matrices of the biases for the output constraints.
    """

    def __init__(self, input_matrix: torch.Tensor = None, input_bias: torch.Tensor = None,
                 out_matrix_list: list[torch.Tensor] = None, out_bias_list: list[torch.Tensor] = None):
        self.in_matrix = input_matrix
        self.in_bias = input_bias
        self.out_matrix_list = out_matrix_list
        self.out_bias_list = out_bias_list

    def to_numeric_bounds(self) -> HyperRectangleBounds:
        """
        This method creates a :class:`~pynever.strategies.abstraction.bounds_propagation.bounds.HyperRectangleBounds`
        object from the property specification.
        If the property is already a hyper rectangle it just initializes the object, otherwise
        it returns the hyper rectangle approximation of the input property.

        Returns
        -------
        HyperRectangleBounds
            The hyper rectangle approximation of the input property
        """
        # TODO approximate if not already a hyperrectangle
        lbs = []
        ubs = []

        for i in range(self.in_bias.shape[0]):
            if 1 in self.in_matrix[i, :]:
                ubs.append(self.in_bias[i, :])
            else:
                lbs.append(-self.in_bias[i, :])

        # debug
        assert len(lbs) == len(ubs) == self.in_bias.shape[0] // 2

        return HyperRectangleBounds(torch.Tensor(lbs), torch.Tensor(ubs))

    def to_star(self) -> Star:
        """
        This method creates the input star based on the property specification

        Returns
        -------
        Star
            The input star
        """
        return Star(self.in_matrix, self.in_bias)

    def to_smt_file(self, filepath: str, input_id: str = 'X', output_id: str = 'Y'):
        """
        This method builds the SMT-LIB representation of the :class:`~pynever.strategies.verification.properties.NeVerProperty`,
        expressing the variables and the matrices as constraints in the corresponding logic

        Parameters
        ----------
        input_id: str, Optional
            Identifier of the input node (default: 'X')
        output_id: str, Optional
            Identifier of the output node (default: 'Y')
        filepath: str
            Path to the SMT-LIB file to create
        """
        with open(filepath, 'w+') as f:
            # Variables definition
            input_vars = [f"{input_id}_{i}" for i in range(self.in_matrix.shape[1])]
            if self.out_matrix_list:
                output_vars = [f"{output_id}_{i}" for i in range(self.out_matrix_list[0].shape[1])]
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

            infix_in_constraints = NeverProperty.create_infix_constraints(input_vars,
                                                                          LinearFunctions(self.in_matrix, self.in_bias))
            for c in infix_in_constraints:
                prefix_smt_row = expression.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                f.write(f"(assert {prefix_smt_row})\n")

            f.write('\n;; --- OUTPUT CONSTRAINTS ---\n')

            # Allow multiple output properties
            infix_output_properties = []
            for out_mat, out_bias in zip(self.out_matrix_list, self.out_bias_list):
                infix_constraints = NeverProperty.create_infix_constraints(output_vars,
                                                                           LinearFunctions(out_mat, out_bias))
                infix_output_properties.append(infix_constraints)

            if len(infix_output_properties) == 1:
                for c in infix_output_properties[0]:
                    prefix_smt_row = expression.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                    f.write(f"(assert {prefix_smt_row})\n")
            else:
                s = '(assert (or '
                for p in infix_output_properties:
                    if len(p) == 1:
                        prefix_smt_row = expression.ExpressionTreeConverter().build_from_infix(p[0]).as_prefix()
                        s = s + '\n' + prefix_smt_row
                    else:
                        s = s + '(and '
                        for c in p:
                            prefix_smt_row = expression.ExpressionTreeConverter().build_from_infix(c).as_prefix()
                            s = s + '\n' + prefix_smt_row
                        s = s + ')\n'
                s = s + '))'
                f.write(s)

    @staticmethod
    def create_infix_constraints(variables: list[str], coefs: LinearFunctions) -> list[str]:
        """
        Procedure to create a list of string infix constraints starting from the coefficients and biases

        Parameters
        ----------
        variables: list[str]
            The list of variable names
        coefs: LinearFunctions
            The linear function coefficients

        Returns
        -------
        list[str]
            The list of infix constraints
        """
        c_list = []

        for row in range(coefs.matrix.shape[0]):
            coef = coefs.matrix[row, :]
            bias = coefs.offset[row][0]
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
    A class used to represent a VNN-LIB property. It directly loads
    the property from a `.vnnlib` file.
    """

    def __init__(self, filepath: str):
        smt_parser = vnnlib.VnnlibParser()
        super().__init__(*smt_parser.parse_property(filepath))


class LocalRobustnessProperty(NeverProperty):
    """
    TODO

    sample: torch.Tensor
    epsilon: float
    label: str
    max_output: bool
    """

    def __init__(self, sample: torch.Tensor, epsilon: float, n_outputs: int, label: int, max_output: bool):
        super().__init__(*LocalRobustnessProperty.build_matrices(sample, epsilon, n_outputs, label, max_output))

    @staticmethod
    def build_matrices(sample: torch.Tensor, epsilon: float, n_outputs: int, label: int, max_output: bool) -> tuple[
        torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:

        if sample.shape[1] != 1:
            raise InvalidDimensionError('Wrong shape for the sample, should be single-dimensional')

        # Input property
        n_dims = sample.shape[0]
        in_coef_mat = torch.zeros((2 * n_dims, n_dims))
        in_bias_mat = torch.zeros((2 * n_dims, 1))

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

        out_coef_mat = torch.zeros((n_outputs - 1, n_outputs))
        out_bias_mat = torch.zeros((n_outputs - 1, 1))

        outputs = set(range(n_outputs))
        outputs.remove(label)
        for i in outputs:
            out_coef_mat[i, label] = 1
            out_coef_mat[i, i] = -1

        if not max_output:
            out_coef_mat = -out_coef_mat

        return in_coef_mat, in_bias_mat, [out_coef_mat], [out_bias_mat]
