from venus.utils.constraint import LinearConstraint, WeightedSum
from venus.utils.formula_interface import Formula
from operator import __le__, __ge__, __eq__

(LT, GT, NE) = ('<', '>', '!=')
(LE, GE, EQ) = ('<=', '>=', '==')

SENSE_MAP = {GE: __ge__, LE: __le__, EQ: __eq__}


class BinaryFormula(Formula):
    """
    the abstract class representing a binary formula
    """
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def accept(self, visitor):
        visitor.visitBinaryFormula(self)

    def acceptI(self, visitor):
        return visitor.visitBinaryFormula(self)


class ConjFormula(BinaryFormula):

    def __init__(self, left, right):
        super(ConjFormula, self).__init__(left, right)
        self.name = "AND"

    def __str__(self):
        return self.name + "(" + self.left.__str__() + ", " + self.right.__str__() + ")"

    def accept(self, visitor):
        visitor.visitConjFormula(self)

    def acceptI(self, visitor):
        return visitor.visitConjFormula(self)


class DisjFormula(BinaryFormula):

    def __init__(self, left, right):
        super(DisjFormula, self).__init__(left, right)
        self.name = "OR"

    def __str__(self):
        return self.name + "(" + self.left.__str__() + ", " + self.right.__str__() + ")"

    def accept(self, visitor):
        visitor.visitDisjFormula(self)

    def acceptI(self, visitor):
        return visitor.visitDisjFormula(self)


class AtomicDisjFormula(BinaryFormula):

    def __init__(self, left, right):
        super(AtomicDisjFormula, self).__init__(left, right)
        self.name = "OR"

    def __str__(self):
        return self.name + "(" + self.left.__str__() + ", " + self.right.__str__() + ")"

    def accept(self, visitor):
        visitor.visitAtomicDisjFormula(self)

    def acceptI(self, visitor):
        return visitor.visitAtomicDisjFormula(self)


class AtomicConjFormula(BinaryFormula):

    def __init__(self, left, right):
        super(AtomicConjFormula, self).__init__(left, right)
        self.name = "AND"

    def __str__(self):
        return self.name + "(" + self.left.__str__() + ", " + self.right.__str__() + ")"

    def accept(self, visitor):
        visitor.visitAtomicConjFormula(self)

    def acceptI(self, visitor):
        return visitor.visitAtomicConjFormula(self)


class NAryFormula(Formula):
    """
    the abstract class representing an n-ary formula
    """

    def __init__(self, clauses):
        self.clauses = clauses

    def accept(self, visitor):
        visitor.visitNAryFormula(self)

    def acceptI(self, visitor):
        return visitor.visitNAryFormula(self)


class NAryDisjFormula(NAryFormula):

    def __init__(self, clauses):
        super(NAryDisjFormula, self).__init__(clauses)
        self.name = "OR"

    def __str__(self):
        return self.name + "(" + ",".join([clause.__str__() for clause in self.clauses]) + ")"

    def accept(self, visitor):
        visitor.visitNAryDisjFormula(self)

    def acceptI(self, visitor):
        return visitor.visitNAryDisjFormula(self)


class NAryConjFormula(NAryFormula):

    def __init__(self, clauses):
        super(NAryConjFormula, self).__init__(clauses)
        self.name = "AND"

    def __str__(self):
        return self.name + "(" + ",".join([clause.__str__() for clause in self.clauses]) + ")"

    def accept(self, visitor):
        visitor.visitNAryConjFormula(self)

    def acceptI(self, visitor):
        return visitor.visitNAryConjFormula(self)


class EUntilFormula(BinaryFormula):
    """
    Represents formulas of the form  E phi U^k psi
    """
    def __init__(self, k, left, right):
        super(EUntilFormula, self).__init__(left, right)
        self.k = k
        self.name = "E U"

    def __str__(self):
        return self.name + "{}(".format(self.k) + self.left.__str__() + ", " + self.right.__str__() + ")"

    def accept(self, visitor):
        visitor.visitEUntilFormula(self)

    def acceptI(self, visitor):
        return visitor.visitEUntilFormula(self)


class AUntilFormula(BinaryFormula):
    """
    Represents formulas of the form  A phi U^k psi
    """
    def __init__(self, k, left, right):
        super(AUntilFormula, self).__init__(left, right)
        self.k = k
        self.name = "A U"

    def __str__(self):
        return self.name + "{}(".format(self.k) + self.left.__str__() + ", " + self.right.__str__() + ")"

    def accept(self, visitor):
        visitor.visitAUntilFormula(self)

    def acceptI(self, visitor):
        return visitor.visitAUntilFormula(self)


class UnaryFormula(Formula):
    """
    the abstract class representing a unary formula
    """
    def __init__(self, left):
        super(UnaryFormula, self).__init__()
        self.left = left

    def accept(self, visitor):
        visitor.visitUnaryFormula(self)

    def acceptI(self, visitor):
        return visitor.visitUnaryFormula(self)


class NegationFormula(UnaryFormula):

    def __init__(self, left):
        super(NegationFormula, self).__init__(left)
        self.name = "NOT"

    def __str__(self):
        return self.name + "(" + self.left.__str__() + ")"

    def acceptI(self, visitor):
        return visitor.visitNegationFormula(self)


class ENextFormula(UnaryFormula):

    """
    Represents formulas of the form  E X^k phi
    """
    def __init__(self, k, left):
        super(ENextFormula, self).__init__(left)
        self.k = k
        self.name = "E X"
    
    def __str__(self):
        return self.name + "{}(".format(self.k) + self.left.__str__() + ")"

    def accept(self, visitor):
        visitor.visitENextFormula(self)

    def acceptI(self, visitor):
        return visitor.visitENextFormula(self)


class ANextFormula(UnaryFormula):
    """
    Represents formulas of the form  A X^k phi
    """

    def __init__(self, k, left):
        super(ANextFormula, self).__init__(left)
        self.k = k
        self.name = "A X"

    def __str__(self):
        return self.name + "{}(".format(self.k) + self.left.__str__() + ")"

    def accept(self, visitor):
        visitor.visitANextFormula(self)

    def acceptI(self, visitor):
        return visitor.visitANextFormula(self)


class StateCoordinate(Formula):
    """
    the class representing a component of an n-ary tuple,
    to be used in VarVarConstraint and VarConstConstraint
    for describing conditions on layers of neural networks
    """

    def __init__(self, i):
        self.i = i

    def __str__(self):
        return "({})".format(self.i)

    def accept(self, visitor):
        visitor.visitStateCoordinateFormula(self)

    def acceptI(self, visitor):
        return visitor.visitStateCoordinateFormula(self)


class LinearExpression:
    """
    the class representing a linear combination of components of an n-ary tuple,
    to be used in LinExprConstraint for describing arbitrary linear constraints
    on layers of neural networks
    """
    def __init__(self, coord_coeff_map):
        """
        :param coord_coeff_map: a dictionary where
            the keys are integers referring to components of n-ary tuples and
            the values are corresponding coefficients.

            For example an expression of the form 2 x0 + 3 x1 is encoded in dict
            {0: 2, 1: 3}
        """
        self.coord_coeff_map = coord_coeff_map

    def __str__(self):
        return " + ".join(["{}*({})".format(self.coord_coeff_map[i], i) for i in self.coord_coeff_map])


class Constraint(Formula):
    """
    the abstract class for an atomic formula which is a comparison between two terms
    """

    def __init__(self, op1, sense, op2):
        """
        :param op1:
        :param sense: a comparison operator, one of LT, GT, NE, LE, GE, EQ
        :param op2:
        """
        assert sense in {LT, GT, NE, LE, GE, EQ}
        super(Constraint, self).__init__()
        self.op1 = op1
        self.op2 = op2
        self.sense = sense

    def accept(self, visitor):
        visitor.visitConstraintFormula(self)

    def acceptI(self, visitor):
        return visitor.visitConstraintFormula(self)


class VarVarConstraint(Constraint):
    """
    the class representing an inequality between two components,
    e.g., (0) > (1) read as the value of the first component is greater
    than the value of the second component.
    """

    def __init__(self, op1, sense, op2):
        """
        :param op1: an instance of StateCoordinate
        :param sense: a comparison operator (see Constraint)
        :param op2: an instance of StateCoordinate
        """
        assert isinstance(op1, StateCoordinate)
        assert isinstance(op2, StateCoordinate)
        super(VarVarConstraint, self).__init__(op1, sense, op2)

    def __str__(self):
        return self.op1.__str__() + self.sense + self.op2.__str__()

    def accept(self, visitor):
        visitor.visitVarVarConstraintFormula(self)

    def acceptI(self, visitor):
        return visitor.visitVarVarConstraintFormula(self)

    def get_atomic_constraint(self, state_vars):
        op1 = state_vars[self.op1.i]
        op2 = state_vars[self.op2.i]
        if self.sense in SENSE_MAP:
            return SENSE_MAP[self.sense](op1, op2)
        else:
            raise Exception("Unexpected sense", self.sense)

    def get_custom_atomic_constraint(self, state_vars):
        op1 = state_vars[self.op1.i]
        op2 = state_vars[self.op2.i]
        if self.sense in SENSE_MAP:
            lhs = WeightedSum([op1, op2], [1, -1])
            return LinearConstraint(lhs, SENSE_MAP[self.sense], 0)
        else:
            raise Exception("Unexpected sense", self.sense)


class VarConstConstraint(Constraint):
    """
    the class representing an inequality between a component and a constant,
    e.g., (0) > 1500 read as the value of the first component is greater than 1500.
    """

    def __init__(self, op1, sense, op2):
        """
        :param op1: an instance of StateCoordinate
        :param sense: a comparison operator (see Constraint)
        :param op2: a number
        """
        assert isinstance(op1, StateCoordinate)
        assert isinstance(op2, int) or isinstance(op2, float)
        super(VarConstConstraint, self).__init__(op1, sense, op2)

    def __str__(self):
        return self.op1.__str__() + self.sense + "{}".format(self.op2)

    def accept(self, visitor):
        visitor.visitVarConstConstraintFormula(self)

    def acceptI(self, visitor):
        return visitor.visitVarConstConstraintFormula(self)

    def get_atomic_constraint(self, state_vars):
        op1 = state_vars[self.op1.i]
        op2 = self.op2
        if self.sense in SENSE_MAP:
            return SENSE_MAP[self.sense](op1, op2)
        else:
            raise Exception("Unexpected sense", self.sense)

    def get_custom_atomic_constraint(self, state_vars):
        op1 = state_vars[self.op1.i]
        op2 = self.op2
        if self.sense in SENSE_MAP:
            lhs = WeightedSum([op1], [1])
            return LinearConstraint(lhs, SENSE_MAP[self.sense], op2)
        else:
            raise Exception("Unexpected sense", self.sense)


class LinExprConstraint(Constraint):
    """
    the class representing an arbitrary linear constraint on a layer
    of a neural network
    """

    def __init__(self, op1, sense, op2):
        """
        :param op1: an instance of LinearExpression
        :param sense: a comparison operator (see Constraint)
        :param op2: a number
        """
        assert isinstance(op1, LinearExpression)
        assert isinstance(op2, int) or isinstance(op2, float)
        super(LinExprConstraint, self).__init__(op1, sense, op2)

    def __str__(self):
        return self.op1.__str__() + self.sense + self.op2.__str__()

    def get_atomic_constraint(self, state_vars):
        op1 = 0
        for i, c in self.op1.coord_coeff_map.items():
            op1 += c * state_vars[i]
        op2 = self.op2
        if self.sense in SENSE_MAP:
            return SENSE_MAP[self.sense](op1, op2)
        else:
            raise Exception("Unexpected sense", self.sense)

    def get_custom_atomic_constraint(self, state_vars):
        vars = []
        coeffs = []
        for i, c in self.op1.coord_coeff_map.items():
            vars.append(state_vars[i])
            coeffs.append(c)

        lhs = WeightedSum(vars, coeffs)
        op2 = self.op2
        if self.sense in SENSE_MAP:
            return LinearConstraint(lhs, SENSE_MAP[self.sense], op2)
        else:
            raise Exception("Unexpected sense", self.sense)


class EmptyFormula(Formula):

    def accept(self, visitor):
        visitor.visitEmptyFormula(self)

    def acceptI(self, visitor):
        return visitor.visitEmptyFormula(self)


class FalseFormula(Formula):

    def __str__(self):
        return "FALSE"

    def accept(self, visitor):
        visitor.visitFalseFormula(self)

    def acceptI(self, visitor):
        return visitor.visitFalseFormula(self)


class TrueFormula(Formula):

    def __str__(self):
        return "TRUE"

    def accept(self, visitor):
        visitor.visitTrueFormula(self)

    def acceptI(self, visitor):
        return visitor.visitTrueFormula(self)
