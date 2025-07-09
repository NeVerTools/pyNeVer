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

                    coefs, bias = self.visit_single(op_node)
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

    def visit_single(self, node: InfoNode, pre: bool = True) -> tuple[list[float], float]:
        """
        Procedure to visit a single constraint defined by a node

        Parameters
        ----------
        node: InfoNode
            Node representing a single constraint
        pre: bool
            Flag indicating if the constraint is on the input or the output

        Returns
        -------
        tuple[list[float], float]
            The list of coefficients and the bias term for the constraint
        """
        switch_sign = False  # Flag for >= constraints
        n_var = self.n_inputs if pre else self.n_outputs

        if node.type != NodeType.OPERATION:
            raise ValueError(f'Node {node} is not an operation node, cannot evaluate it.')
        if node.value not in [Operation.GE, Operation.LE]:
            raise ValueError(f'Node {node} is not a comparison operation, cannot evaluate it.')
        if node.value == Operation.GE:
            switch_sign = True

        lhs = node.children[0]
        rhs = node.children[1]

        coefs = [0] * n_var
        bias = 0

        # Check the left-hand side
        match lhs.type:
            case NodeType.IN_VAR | NodeType.OUT_VAR:
                coefs[lhs.value] += 1

            case NodeType.CONST:
                bias -= lhs.value

            case NodeType.OPERATION:
                c, b, is_variable = self.visit_op(lhs, pre)
                if is_variable:
                    for i in range(n_var):
                        coefs[i] += c[i]  # TODO check se può sostituire
                    bias -= b

            case _:
                raise ValueError(f'Left-hand side {lhs} is not a valid operand')

        # Check the right-hand side
        match rhs.type:
            case NodeType.IN_VAR | NodeType.OUT_VAR:
                coefs[rhs.value] -= 1

            case NodeType.CONST:
                bias += rhs.value

            case NodeType.OPERATION:
                c, b, is_variable = self.visit_op(rhs, pre)
                if is_variable:
                    for i in range(n_var):
                        coefs[i] -= c[i]
                    bias += b

            case _:
                raise ValueError(f'Right-hand side {rhs} is not a valid operand')

        if switch_sign:
            coefs = [-c for c in coefs]
            bias = -bias

        return coefs, bias

    def visit_output(self, node: InfoNode) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """

        Parameters
        ----------
        node: InfoNode
            Node representing multiple constraints

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor]]
            The list of output matrices and biases for the constraints
        """
        if len(node.children) != 1:
            raise ValueError(f'Expected node {node} to have a single OR statement, multiple found.')

        or_node = node.children[0]
        if or_node.value != Operation.OR:
            raise ValueError(f'Node {node} is not an OR node, cannot evaluate it.')

        out_matrices, out_biases = [], []
        disjuncts = len(or_node.children)

        match disjuncts:
            case 0:
                raise ValueError(f'Node {node} has no children.')

            case 1:  # TODO function
                # Single disjunct case
                single_constraint = or_node.children[0]

                coefs = torch.zeros((1, self.n_inputs))
                bias = torch.zeros((1, 1))

                match single_constraint.value:

                    case Operation.GE | Operation.LE:
                        # Visit single
                        c, b = self.visit_single(single_constraint, pre=False)
                        coefs[0, :] = torch.Tensor(c)
                        bias[0, 0] = b

                    case Operation.AND:
                        # Visit AND
                        coefs, bias = self.visit_and(single_constraint)

                    case _:
                        raise ValueError(f'Node {node} is not a constraint, cannot evaluate it.')

                out_matrices.append(coefs)
                out_biases.append(bias)

            case _:
                for child in or_node.children:
                    if len(child.children) == 0:
                        raise ValueError(f'Node {node} is not a constraint, cannot evaluate it.')

                    # Children are expected to be AND nodes
                    if child.value != Operation.AND:
                        raise ValueError(f'Node {node} is not an AND, cannot evaluate it.')

                    coefs, bias = self.visit_and(child)
                    out_matrices.append(coefs)
                    out_biases.append(bias)

        return out_matrices, out_biases

    def visit_and(self, node: InfoNode) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Procedure to visit an AND node and return the coefficients and biases.

        Parameters
        ----------
        node: InfoNode
            The AND node to visit

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The list of values for the coefficients and biases in the AND statement
        """
        rows = len(node.children)
        if rows == 0:
            raise ValueError(f'Node {node} is not a valid AND node, cannot evaluate it.')

        coefs = torch.zeros((rows, self.n_outputs))
        bias = torch.zeros((rows, 1))

        for i, child in enumerate(node.children):
            if child.type != NodeType.OPERATION:
                raise ValueError(f'Node {node} is not an assertion node, cannot evaluate it.')

            c, b = self.visit_single(child, pre=False)
            coefs[i, :] = torch.Tensor(c)
            bias[i, 0] = b

        return coefs, bias

    def visit_op(self, node: InfoNode, pre: bool = True) -> tuple[list[float], float, bool]:
        """

        Parameters
        ----------
        node
        pre

        Returns
        -------
        tuple[list[float], float, bool]
            The list of coefficients, the bias and a flag indicating if the operator is on a variable or not
        """
        n_var = self.n_inputs if pre else self.n_outputs
        coefs = [0] * n_var
        right_coefs = [0] * n_var
        bias = 0
        right_bias = 0

        if node.value not in [Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV]:
            raise ValueError(f'Node {node} is not a valid operator, cannot evaluate it.')

        if len(node.children) != 2:
            raise ValueError(f'Expected node {node} to have two children.')

        lhs = node.children[0]
        rhs = node.children[1]

        # Flag to signal there is a variable on either side
        var_left = False
        var_right = False

        match lhs.type:
            case NodeType.IN_VAR | NodeType.OUT_VAR:
                coefs[lhs.value] += 1
                var_left = True

            case NodeType.CONST:
                bias = lhs.value

            case NodeType.OPERATION:
                coefs, bias, var_left = self.visit_op(lhs, pre)

            case _:
                raise ValueError(f'LHS of node {node} is not valid, cannot evaluate it.')

        match rhs.type:
            case NodeType.IN_VAR | NodeType.OUT_VAR:
                right_coefs[rhs.value] += 1
                var_right = True

            case NodeType.CONST:
                right_bias = rhs.value

            case NodeType.OPERATION:
                right_coefs, right_bias, var_right = self.visit_op(rhs, pre)

            case _:
                raise ValueError(f'RHS of node {node} is not valid, cannot evaluate it.')

        # Now update the coefficients based on the LHS and RHS
        match node.value:
            case Operation.ADD:
                if var_left and var_right:
                    for i in range(n_var):
                        coefs[i] += right_coefs[i]
                elif var_right:
                    for i in range(n_var):
                        coefs[i] = right_coefs[i]
                bias += right_bias

            case Operation.SUB:
                if var_left and var_right:
                    for i in range(n_var):
                        coefs[i] -= right_coefs[i]
                elif var_right:
                    for i in range(n_var):
                        coefs[i] = -right_coefs[i]
                bias -= right_bias

            case Operation.MUL:
                if var_left and var_right:
                    raise ValueError('Multiplication of two variables not supported')
                elif var_left:
                    for i in range(n_var):
                        coefs[i] *= right_bias
                elif var_right:
                    for i in range(n_var):
                        coefs[i] = right_coefs[i] * bias
                bias *= right_bias

            case Operation.DIV:
                if var_left and var_right:
                    raise ValueError('Division of two variables not supported')
                elif var_left:
                    for i in range(n_var):
                        coefs[i] /= right_bias
                elif var_right:
                    raise ValueError('Division by a variables not supported')
                bias /= right_bias

            case _:
                raise ValueError(f'Node {node} is not valid, cannot evaluate it.')

        return coefs, bias, (var_left or var_right)
