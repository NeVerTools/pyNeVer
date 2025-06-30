"""
Contains ``Node`` classes and some constants to be used by parser and visitor.
"""

#NODE TYPES
from Constants import id2word


IN_VAR : int = 0 #Input Variable node type
OUT_VAR : int = 1 #Output variable node type
CONST : int = 2 #Constant node type
ASSERTION : int = 3 #Assertion node type
OPERATION : int = 4 #Operation node type (and, or, >=, +, -, *, /, change sign)
##########
type2word = ["Input", "Output", "Constant", "Assertion", "Operation"]


class Node:
    """
    ``Node`` class is the parent class of the classes that store the information of a node in the tree.

    Attributes
    ----------
    type: int
        type of the node (0 1 2 3 4 respectively: Input, Output, Constant, Assertion, Operation). None for the parent.
    value: int or float
        value of the node (if applicable, e.g. variable number for input/output nodes, constant value for constant nodes).
    children: list
        list of child nodes (if applicable, e.g. operation nodes have two children (except OR and AND), assertions have one).
    """
    def __init__(self, value=None):
       # self.name = name  # Name of the node (If applicable))
        self.value = value  # Value of the node (if applicable)
        self.children = []  # List to hold child nodes
        self.type = None  # Type of the node (0 1 2 3  4 respectively, Input, Output, Constant, Assertion, Operation)

    def add_child(self, child_node):
        """Append a child node to this node children field."""
        self.children.append(child_node)

    def __repr__(self):
        if self.type is not None:
            return f"Node (Type: {type2word[self.type]}, number of children: {len(self.children)})"
        else: return f"Node (Type: None, number of children: {len(self.children)})"

    def __str__(self):
        if self.type is not None:
            return f"Node (Type: {type2word[self.type]}, number of children: {len(self.children)})"
        else: return f"Node (Type: None, number of children: {len(self.children)})"

#Children is empty
class InputVariableNode(Node):
    """
    ``InputVariableNode`` class stores an input variable node in the tree.

    Attributes
    ----------
    type: int
        It is IN_VAR (0) for input variable nodes.
    value: int
        It stores the variable number (e.g. 0 for X_0, 1 for X_1, etc.).
    """
    def __init__(self, value):
        super().__init__(int(value))
        #self.value = value #Value stores the variable number
        self.type = IN_VAR

#Children is empty
class OutputVariableNode(Node):
    """
    ``OutputVariableNode`` class stores an output variable node in the tree.

    Attributes
    ----------
    type: int
        It is OUT_VAR (1) for input variable nodes.
    value: int
        It stores the variable number (e.g. 0 for X_0, 1 for X_1, etc.).
    """
    def __init__(self, value):
        super().__init__(int(value))
        #self.value = value #Value stores the variable number
        self.type = OUT_VAR

# Children is empty, only carries the contstant value
class ConstantNode(Node):
    """
    ``ConstantNode`` class stores a constant node in the tree.

    Attributes
    ----------
    type: int
        It is CONST (2) for input variable nodes.
    value: float
        It stores the value of the constant (e.g. 0.5, -1.0, etc.).
    """
    def __init__(self, value):
        super().__init__(float(value))
        self.type = CONST


######ASSERTION CODES########
INPUT : int = 0 #Input assertion code
OUTSIMPLE : int = 1
OUTCOMPL : int = 2 #Assertion with or and and operations inside

#Assertion has an operation node child and two (that can be operation nodes, variable nodes or constant nodes) as children
class AssertionNode(Node):
    """
    ``AssertionNode`` class stores an assertion node in the tree.

    Attributes
    ----------
    type: int
        It is ASSERTION (3).
    value: int
        It stores the assertion type (INPUT, OUTSIMPLE, OUTCOMPL). OUTSIMPLE is deprecated.
    """
    #Value stores the assertion type
    def __init__(self, value):
        super().__init__(value)
        self.type = ASSERTION




    #Operation node has two operand nodes as children (that can be operation nodes, variable nodes or constant nodes)
class OperationNode(Node):
    """
    ``OperationNode`` class stores an operation node in the tree.

    Attributes
    ----------
    type: int
        It is OPERATION (4).
    value: int
        It stores the operation type (from Constants.py).
    """
    def __init__(self, opType):
        super().__init__(int(opType))
        self.type = OPERATION
        

    def __str__(self):
            msg =  f"Node (Type: {type2word[self.type]}, Operation: {id2word[self.value]}, number of children: {len(self.children)})\n"
            for i, child in enumerate(self.children):
                if child.type is not None:
                    msg += f"\n - Child {i} of {id2word[self.value]}: {child}"
                else:
                    msg += f"\n - Child {i}: Node (Type: None, number of children: {len(child.children)})"
            return msg+'\n'

    #An operation can be a logical operation (and, or) or a mathematical operation (addition, subtraction, multiplication, division, change sign ex: -5 => do it in the parser?)

    #OR AND can have more than two children!!!!
