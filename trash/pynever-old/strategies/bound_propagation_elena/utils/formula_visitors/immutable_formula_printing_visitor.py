from venus.utils.formula_visitors.immutable_formula_visitor_interface import FormulaVisitorI


class FormulaPrintVisitorI(FormulaVisitorI):
    """
    An immutable visitor implementation for printing formulae.
    """

    def visitConstraintFormula(self, element):
        return ""

    def visitENextFormula(self, element):
        return self.visitTemporalUnaryFormula(element)

    def visitANextFormula(self, element):
        return self.visitTemporalUnaryFormula(element)

    def visitAUntilFormula(self, element):
        return self.visitTemporalBinaryFormula(element)

    def visitConjFormula(self, element):
        return self.visitBinaryFormula(element)

    def visitDisjFormula(self, element):
        return self.visitBinaryFormula(element)

    def visitEUntilFormula(self, element):
        return self.visitTemporalBinaryFormula(element)

    def visitBinaryFormula(self, element):
        return element.name + "(" + element.left.acceptI(self) + ", " + element.right.acceptI(self) + ")"

    def visitUnaryFormula(self, element):
        return element.name + "(" + element.left.acceptI(self) + ")"

    def visitTrueFormula(self, element):
        return "True"

    def visitFalseFormula(self, element):
        return "False"

    def visitEmptyFormula(self, element):
        return ""

    def visitNegationFormula(self, element):
        return self.visitUnaryFormula(element)

    def visitNAryFormula(self, element):
        return element.name + "(" + ','.join(clause.acceptI(self) for clause in element.clauses) + ")"

    def visitStateCoordinateFormula(self, element):
        return "({})".format(element.i)

    def visitVarConstConstraintFormula(self, element):
        return element.op1.acceptI(self) + element.sense + "{}".format(element.op2)

    def visitVarVarConstraintFormula(self, element):
        return element.op1.acceptI(self) + element.sense + element.op2.acceptI(self)

    def visitTemporalUnaryFormula(self, element):
        return element.name + str(element.k) + "(" + element.left.acceptI(self) + ")"

    def visitTemporalBinaryFormula(self, element):
        return element.name + str(element.k) + "(" + element.left.acceptI(self) + ", " + element.right.acceptI(self) + ")"