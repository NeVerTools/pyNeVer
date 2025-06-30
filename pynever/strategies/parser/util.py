from enum import Enum


class NodeType(Enum):
    """
    Defines the possible kinds of nodes to parse
    """

    IN_VAR = "Input"
    OUT_VAR = "Output"
    CONST = "Constant"
    ASSERTION = "Assertion"
    OPERATION = "Operation"


class Assertion(Enum):
    """
    Defines the assertion types

    * INPUT defines an input constraint
    * OUT_SIMPLE defines an output constraint with a single OR statement
    * OUT_COMPLEX defines an output constraint with nested AND/OR statements
    """

    INPUT = 0
    OUT_SIMPLE = 1
    OUT_COMPLEX = 2


class Operation(Enum):
    LP = '('
    RP = ')'
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    NUM = 'NUM'
    DEC = 'declare-const'
    REA = 'Real'
    ASS = 'assert'
    OR = 'OR'
    AND = 'AND'
    GE = '>='
    LE = '<='
    IN = 'in'
    OUT = 'out'
