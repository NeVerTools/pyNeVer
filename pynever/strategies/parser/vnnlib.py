"""This module defines the VNN-LIB parser"""

import collections

from strategies.parser.tokenizer import Token
from strategies.parser.tree import InfoNode


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
    """

    def __init__(self, input_name: str = 'X', output_name: str = 'Y'):
        self.input_name = input_name
        self.output_name = output_name
        self.input_len = 0
        self.output_len = 0

    def parse(self, tokens: list[Token]) -> list[InfoNode]:
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
        n_input = 0
        n_output = 0
        p_counter = 0  # Counter for balancing parentheses

        if len(tokens) == 0:
            return nodes

        # Build an iterator for the tokens to preserve the pointer
        token_itr = iter(tokens)
        stream_end = len(tokens)
        i = 0
        output_reached = False

        while i < stream_end:
            token, i = VnnlibParser.safe_next(token_itr)

        return nodes

    @staticmethod
    def safe_next(token_itr: collections.Iterator[Token]) -> tuple[int, Token]:
        pass
