"""This module defines the VNN-LIB parser"""

import torch

from pynever.strategies.parser.tokenizer import Token, Tokenizer
from pynever.strategies.parser.tree import InfoNode, OperationNode, AssertionNode, ConstantNode
from pynever.strategies.parser.util import Operation, Assertion
from pynever.utilities import xor


class VnnlibParser:
    """
    A class to parse a list of ``Token`` objects to create an AST of ``InfoNode``.
    Each node represents a declaration, an assertion or an expression.

    Attributes
    ----------
    input_name: str
        The name of the input variables
    output_name: str
        The name of the output variables
    input_len: int
        The number of input variables
    output_len: int
        The number of output variables
    __token_itr: Iterator[Token]
        The :class:`~pynever.strategies.parser.tokenizer.Token` iterator
    __itr_idx: int
        The current iteration counter
    """

    def __init__(self, input_name: str = 'X', output_name: str = 'Y'):
        self.input_name = input_name
        self.output_name = output_name
        self.input_len = 0
        self.output_len = 0

        self.__token_itr = None
        self.__itr_idx = 0

    def safe_next(self) -> Token | None:
        """
        Procedure to read the next token from the Iterator without raising an exception.

        Returns
        -------
        Token | None
            The next token in the iterator, if not exhausted. Returns None otherwise
        """
        try:
            self.__itr_idx += 1
            return next(self.__token_itr)
        except StopIteration:
            self.__itr_idx -= 1
            return None

    def read_par(self, start: bool = True, progress: bool = True) -> Token | None:
        """
        Procedure to read a parenthesis (opening or closing)

        Parameters
        ----------
        start: bool
            Flag indicating the opening or closing parenthesis
        progress: bool
            Flag indicating whether to return the next iteration or not

        Returns
        -------
        Token | None
            The next token in the iterator, if not exhausted. Returns None otherwise
        """
        token = self.safe_next()

        # All statements are inside '(' and ')'
        op = Operation.LP if start else Operation.RP
        if token.tag != op:
            raise SyntaxError(f'Expected "{op.value}" at line {token.line}, found "{token.tag.value}" instead')
        else:
            return self.safe_next() if progress else token

    def parse_property(self, filepath: str) -> tuple[
        torch.Tensor, torch.Tensor,
        list[torch.Tensor], list[torch.Tensor]
    ]:
        """
        This procedure is the entry point of the VNN-LIB parser. Starting from the property file
        it creates the data structures to represent the input and output coefficients.

        Parameters
        ----------
        filepath: str
            The path to the property file

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
            The input coefficients and biases and the lists of output coefficients and biases
        """

        # Step 1 - tokenizer
        tokens = Tokenizer(filepath).tokenize()

        # Step 2 - parser
        nodes = self.parse_tokens(tokens)

        # Step 3 - visitor

        pass

    def parse_tokens(self, tokens: list[Token]) -> list[InfoNode]:
        """
        Procedure to parse a list of ``Token`` objects to create the AST of ``InfoNode`` representing the property.

        Parameters
        ----------
        tokens: list[Token]
            The list of tokens to parse

        Returns
        -------
        list[InfoNode]
            The list of nodes created
        """
        nodes = []

        if len(tokens) == 0:
            return nodes

        # Build an iterator for the tokens to preserve the pointer
        stream_end = len(tokens)
        self.__token_itr = iter(tokens)
        self.__itr_idx = 0

        while self.__itr_idx < stream_end:
            token = self.read_par(start=True)

            # Determine statement kind
            match token.tag:
                case Operation.DEC:  # Declaration
                    self.parse_declaration()

                case Operation.ASS:  # Assertion
                    nodes.append(self.parse_assertion())

                case _:
                    raise SyntaxError(f'Expected declaration or assertion statement, found "{token.tag.value}" instead')

        return nodes

    def parse_declaration(self):
        """
        Procedure to parse a variable declaration and update the variables counter
        """
        token = self.safe_next()

        match token.tag:
            case Operation.IN:
                if int(token.value) == self.input_len:
                    self.input_len += 1
                else:
                    raise SyntaxError(f'Malformed input variable declaration at line {token.line}: '
                                      f'expected "{self.input_name}_{self.input_len}", '
                                      f'found "{self.input_name}_{token.value}" instead')

            case Operation.OUT:
                if int(token.value) == self.output_len:
                    self.output_len += 1
                else:
                    raise SyntaxError(f'Malformed output variable declaration at line {token.line}: '
                                      f'expected "{self.output_name}_{self.output_len}", '
                                      f'found "{self.output_name}_{token.value}" instead')

            case _:
                raise SyntaxError(f'Expected variable declaration at line {token.line}, '
                                  f'found "{token.tag.value}" instead')

        # Check type
        token = self.safe_next()
        if token.tag == Operation.REA:
            token = self.safe_next()

        if token.tag != Operation.RP:
            raise SyntaxError(f'Expected ")" at line {token.line}, found "{token.tag.value}" instead')

    def parse_assertion(self) -> InfoNode:
        """
        Procedure to parse an assertion tree.

        Returns
        -------
        InfoNode
            The new InfoNode tree representing the assertion
        """
        token = self.read_par()
        operation = token.tag

        # Two possible cases: OR as the root of output property...
        if operation == Operation.OR:
            assertion_node = AssertionNode(Assertion.OUT_COMPLEX)
            assertion_node.add_child(self.parse_or(token))

        # ...or a constraint
        else:
            token = self.safe_next()
            left_node, right_node, token, input_flag = self.parse_constraint(token)
            n_type = Assertion.INPUT if input_flag else Assertion.OUT_SIMPLE

            if token.tag != Operation.RP:
                raise SyntaxError(f'Expected ")" at line {token.line}, found "{token.tag.value}" instead')
            self.read_par(start=False, progress=False)

            # Build the constraint tree
            child_node = OperationNode(operation)
            child_node.add_child(left_node)
            child_node.add_child(right_node)

            assertion_node = AssertionNode(n_type)
            assertion_node.add_child(child_node)

        return assertion_node

    def parse_or(self, token: Token) -> InfoNode:
        """
        Procedure to parse the output OR tree from the property file.

        Parameters
        ----------
        token: Token
            The current token to parse

        Returns
        -------
        InfoNode
            The InfoNode tree representing the OR
        """
        and_nodes = []  # Potential operations in and

        l = token.line
        token = self.read_par()
        if token is None:
            raise SyntaxError(f'Expected output constraints, but the file has ended at line {l}')

        if token.tag != Operation.AND:
            # If no AND statement is found we interpret it as a single output constraint
            node, _ = self.parse_operation(input_flag=False, root=False)
            and_nodes.append(node)

            # Read next token as a closing parenthesis
            self.read_par(start=False)

        else:
            # Here we handle the AND statement
            while True:
                # Parse and add the AND
                token = self.safe_next()
                and_nodes.append(self.parse_and(token))

                # Loop until the closing parenthesis is found
                token = self.safe_next()
                if token.tag == Operation.LP:  # Another statement
                    continue
                elif token.tag == Operation.RP:
                    break

        # This is the end of the OR statement
        self.read_par(start=False, progress=False)

        # Build the OR tree
        or_node = OperationNode(Operation.OR)
        for and_node in and_nodes:
            or_node.add_child(and_node)

        return or_node

    def parse_and(self, token: Token) -> InfoNode:
        """
        Procedure to parse an AND tree inside the output property.

        Parameters
        ----------
        token: Token
            The current token to parse

        Returns
        -------
        InfoNode
            The InfoNode tree representing the AND
        """
        ops = []

        if token.tag != Operation.AND:
            raise SyntaxError(f'Expected "AND" at line {token.line}, found "{token.tag.value}" instead')

        token = self.safe_next()

        while True:
            if token.tag != Operation.LP:
                raise SyntaxError(f'Expected "(" at line {token.line}, found "{token.tag.value}" instead')

            operation, _ = self.parse_operation(input_flag=False, root=False)
            ops.append(operation)

            # Loop until the closing parenthesis is found
            token = self.safe_next()

            if token.tag == Operation.RP:  # End of the AND statement
                break
            elif token.tag == Operation.LP:  # Another AND operand
                continue
            else:
                raise SyntaxError(f'Expected "(" or ")" after AND statement at line {token.line}, '
                                  f'found "{token.tag.value}" instead')

        node = OperationNode(Operation.AND)
        for op in ops:
            node.add_child(op)

        return node

    def parse_constraint(self, token: Token, input_flag: bool = True) -> tuple[InfoNode, InfoNode, Token, bool]:
        """
        Procedure to parse a constraint assertion tree

        Parameters
        ----------
        token: Token
            The current token being parsed
        input_flag: bool
            Flag indicating if the constraint is expected to be on the input

        Returns
        -------
        tuple[InfoNode, InfoNode, Token, bool]
            The LHS and RHS nodes of the assertion, the new token and a flag
            to tell whether the constraint operates on the input variables or not
        """
        left, _, flag_left = self.parse_operand(token, input_flag)
        right, token, flag_right = self.parse_operand(token, input_flag)

        if xor(flag_left, flag_right):
            raise SyntaxError(f'Expected only input or output variables in constraint at line {token.line}')

        return left, right, token, flag_left

    def parse_operation(self, input_flag: bool = True, root: bool = False) -> tuple[InfoNode, bool]:
        """

        Parameters
        ----------
        input_flag: bool
            Flag indicating if the operation is expected to be on the input
        root: bool
            Flag indicating if the operation is a root one, i.e.,
            not part of a conjunction of AND statements

        Returns
        -------
        tuple[InfoNode, bool]
            The InfoNode corresponding to the operation and the updated flag
        """

        token = self.safe_next()

        if token.tag == Operation.RP:
            raise SyntaxError(f'Unexpected ")" at line {token.line}')

        if root:
            if token.tag not in [Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV]:
                raise SyntaxError(f'Expected operation in assertion at line {token.line}, '
                                  f'found "{token.tag.value}" instead')
        else:
            if token.tag not in [Operation.GE, Operation.LE]:
                raise SyntaxError(f'Expected "<=" or ">=" in assertion at line {token.line}, '
                                  f'found "{token.tag.value}" instead')

        operation = token.tag
        token = self.safe_next()
        left_node, right_node, token, input_flag = self.parse_constraint(token, input_flag)

        if token.tag != Operation.RP:
            raise SyntaxError(f'Expected ")" at line {token.line}, found "{token.tag.value}" instead')

        # Create and return the node
        node = OperationNode(operation)
        node.add_child(left_node)
        node.add_child(right_node)

        return node, input_flag

    def parse_operand(self, token: Token, input_flag: bool = True) -> tuple[InfoNode, Token, bool]:
        """
        Procedure to parse two operands from a constraint

        Parameters
        ----------
        token: Token
            The current token being parsed
        input_flag: bool
            Flag indicating if the constraint is expected to be on the input

        Returns
        -------
        tuple[InfoNode, Token, bool]
            The new node representing the operand, the new token to process and the updated flag
        """
        match token.tag:
            case Operation.LP:
                node, input_flag = self.parse_operation(input_flag, root=True)
                token = self.safe_next()

            case _ if token.tag == Operation.IN or token.tag == Operation.OUT:
                node, input_flag = self.parse_var(token)
                token = self.safe_next()

            case Operation.NUM:
                token, value = self.parse_const(token)
                node = ConstantNode(value)

            case Operation.SUB:
                token = self.safe_next()
                if token.tag == Operation.NUM:
                    token, value = self.parse_const(token)
                else:
                    raise SyntaxError(f'Expected numeric value at line {token.line} after "-", '
                                      f'found "{token.tag.value}" instead')
                node = ConstantNode(-value)

            case _:
                raise SyntaxError(f'Unexpected "{token.tag.value}" at line {token.line}, expected operand instead')

        return node, token, input_flag

    def parse_var(self, token: Token) -> tuple[InfoNode, bool]:
        """

        Parameters
        ----------
        token

        Returns
        -------

        """
        pass

    def parse_const(self, token: Token) -> tuple[Token, float]:
        """

        Parameters
        ----------
        token

        Returns
        -------

        """
        pass
