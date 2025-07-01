"""This module defines the nodes representing the syntax tree"""

import uuid

from pynever.strategies.parser.util import Assertion
from pynever.strategies.parser.util import NodeType


class InfoNode:
    """
    This class represents a node in the syntax tree to be parsed

    Attributes
    ----------
    id: float
        The node identifier
    value: Any
        The value of the node if a number, the symbol otherwise
    children: list[InfoNode]
        The list of possible children
    type: NodeType
        The classification of this node
    """

    def __init__(self, value):
        self.id = uuid.uuid4()
        self.value = value
        self.children = []
        self.type = None

    def add_child(self, child: 'InfoNode'):
        self.children.append(child)

    def __str__(self):
        return f'InfoNode(type: {self.type if self.type else "-"}, # children: {len(self.children)})'

    def __repr__(self):
        return str(self)


class InputVariableNode(InfoNode):
    """
    A class to store an input variable node in the tree.

    Attributes
    ----------
    value: int
        The variable number
    """

    def __init__(self, value: int):
        super().__init__(value)
        self.type = NodeType.IN_VAR


class OutputVariableNode(InfoNode):
    """
    A class to store an output variable node in the tree.

    Attributes
    ----------
    value: int
        The variable number
    """

    def __init__(self, value: int):
        super().__init__(value)
        self.type = NodeType.OUT_VAR


class ConstantNode(InfoNode):
    """
    A class to store a constant number node in the tree.

    Attributes
    ----------
    value: float
        The constant value
    """

    def __init__(self, value: float):
        super().__init__(value)
        self.type = NodeType.CONST


class AssertionNode(InfoNode):
    """
    A class to store an assertion node in the tree.

    Attributes
    ----------
    value: Assertion
        The kind of assertion represented by the node
    """

    def __init__(self, value: Assertion):
        super().__init__(value)
        self.type = NodeType.ASSERTION


class OperationNode(InfoNode):
    """
    A class to store an operation node in the tree.

    Attributes
    ----------
    value: str
        The operation symbol
    """

    def __init__(self, value: str):
        super().__init__(value)
        self.type = NodeType.OPERATION

    def __str__(self):
        msg = f'Operation Node(Operation: {self.value}, # children: {len(self.children)})\n'
        for i, child in enumerate(self.children):
            msg += f'\n - Child {i} of {self.value}: {child}'
        return msg + '\n'
