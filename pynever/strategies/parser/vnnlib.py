"""This module defines the VNN-LIB parser"""

import torch

from pynever.strategies.parser.tokenizer import Token, Tokenizer
from pynever.strategies.parser.tree import InfoNode, OperationNode, AssertionNode
from pynever.strategies.parser.util import Operation, Assertion, NodeType


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
                    a_node = self.parse_assertion()
                    if a_node is not None:
                        nodes.append(a_node)

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

    def parse_assertion(self) -> InfoNode | None:
        """
        Procedure to parse an assertion tree.

        Returns
        -------
        InfoNode | None
            The new InfoNode tree representing the assertion, if possible
        """
        token = self.read_par()
        operation = token.tag

        # Two possible cases: OR as the root of output property...
        # TODO self.parse_or
        if operation == Operation.OR:
            and_nodes = []  # Potential operations in and

            l = token.line
            token = self.read_par()
            if token is None:
                raise SyntaxError(f'Expected output constraints, but the file has ended at line {l}')

            if token.tag == Operation.AND:
                # If no AND statement is found we interpret it as a single output constraint
                temp = self.parse_op(output=True, root=True)
                and_nodes.append(temp)

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

            assertion_node = AssertionNode(Assertion.OUT_COMPLEX)
            assertion_node.add_child(or_node)

        # ...or a constraint
        else:
            self.safe_next()
            left_node, right_node, token, input_flag = self.parse_operands(token)

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
