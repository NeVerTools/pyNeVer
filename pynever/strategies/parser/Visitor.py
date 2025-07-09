"""This module contains the visitor methods to create the data structures representing a property"""

import torch

from pynever.strategies.parser.tree import InfoNode
from pynever.strategies.parser.util import NodeType, Assertion, Operation


class Visitor:
    """

    Attributes
    ----------
    n_inputs: int
        The number of input variables
    n_outputs: int
        The number of output variables
    """

    def __init__(self, n_inputs: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def visit(self, nodes: list[InfoNode]) -> tuple[
        torch.Tensor, torch.Tensor,
        list[torch.Tensor], list[torch.Tensor]
    ]:
        """
        Procedure to visit the list of parsed nodes from a VNNLIB property and initialize the
        coefficients and biases of a property.

        Parameters
        ----------
        nodes: list[InfoNode]
            The list of parsed nodes

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
            The coefficient and bias matrices required to initialize a VNNLIB property
        """
        out_visited = False
        out_matrices = []
        out_biases = []
        rows = len(nodes)

        if self.n_outputs > 0:
            rows -= 1
        if rows <= 0:
            raise ValueError(f'Not enough input constraints to build a property. Expected at least 1, got {rows}')

        # Init input property
        in_matrix = torch.zeros((rows, self.n_inputs))
        in_bias = torch.zeros((rows, 1))
        in_cnt = 0  # Input assertion counter

        for node in nodes:
            if node.type != NodeType.ASSERTION:
                raise ValueError(f'Node {node} is not an assertion node, cannot evaluate it.')

            match node.value:
                # Simple input assertion
                case Assertion.INPUT:
                    op_node = node.children[0]
                    if op_node.type != NodeType.OPERATION:
                        raise ValueError(f'Node {node} is not an operation node, cannot evaluate it.')

                    coefs, bias = self.visit_input(op_node)
                    in_matrix[in_cnt, :] = torch.Tensor(coefs)
                    in_bias[in_cnt, 0] = bias
                    in_cnt += 1

                case Assertion.OUT_SIMPLE:
                    # Error
                    raise ValueError(f'Output assertion {node} must have an OR statement')

                case Assertion.OUT_COMPLEX:
                    if out_visited:
                        raise ValueError(f'Output assertion {node} have already been visited')
                    out_matrices, out_biases = self.visit_output(node)
                    out_visited = True

                case _:
                    raise ValueError(f'Unknown node type {node.type}')

        return in_matrix, in_bias, out_matrices, out_biases

    def visit_input(self, node: InfoNode) -> tuple[list[float], float]:
        """
        Procedure to visit a single input constraint defined by a node

        Parameters
        ----------
        node: InfoNode
            Node representing a single constraint

        Returns
        -------
        tuple[list[float], float]
            The list of coefficients and the bias term for the constraint
        """
        switch_sign = False  # Flag for >= constraints

        if node.type != NodeType.OPERATION:
            raise ValueError(f'Node {node} is not an operation node, cannot evaluate it.')
        if node.value not in [Operation.GE, Operation.LE]:
            raise ValueError(f'Node {node} is not a comparison operation, cannot evaluate it.')
        if node.value == Operation.GE:
            switch_sign = True

        lhs = node.children[0]
        rhs = node.children[1]

        coefs = [0] * self.n_inputs
        bias = 0.0

        # Check the left-hand side
        match lhs.type:
            case NodeType.IN_VAR:
                coefs[lhs.value] += 1

            case NodeType.CONST:
                bias -= lhs.value

            case NodeType.OPERATION:
                c, b, is_variable = self.visit_op(lhs)
                if is_variable:
                    for i in range(self.n_inputs):
                        coefs[i] += c[i]  # TODO check se può sostituire
                    bias -= b

            case _:
                raise ValueError(f'Left-hand side {lhs} is not a valid operand')

        # Check the right-hand side
        match rhs.type:
            case NodeType.IN_VAR:
                coefs[rhs.value] -= 1

            case NodeType.CONST:
                bias += rhs.value

            case NodeType.OPERATION:
                c, b, is_variable = self.visit_op(rhs)
                if is_variable:
                    for i in range(self.n_inputs):
                        coefs[i] -= c[i]
                    bias += b

            case _:
                raise ValueError(f'Right-hand side {rhs} is not a valid operand')

        if switch_sign:
            coefs = [-c for c in coefs]
            bias = -bias

        return coefs, bias

    def visit_output(self, node: InfoNode) -> tuple[list[list[float]], list[float]]:
        """

        Parameters
        ----------
        node: InfoNode
            Node representing multiple constraints

        Returns
        -------
        tuple[list[list[float]], list[float]]
            The list of output matrices and biases for the constraints
        """
        pass
