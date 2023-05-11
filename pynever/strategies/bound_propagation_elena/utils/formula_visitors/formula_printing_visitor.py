from venus.utils.formula_visitors import FormulaVisitor


class FormulaPrintVisitor(FormulaVisitor):
    def __init__(self, string_builder):
        """
        A mutable visitor implementation for printing formulae.
        :param string_builder: A string to mutate, representing the formula.
        """
        self.sb = string_builder

    def visitConstraintFormula(self, element):
        pass

    def visitENextFormula(self, element):
        self.visitTemporalUnaryFormula(element)

    def visitANextFormula(self, element):
        self.visitTemporalUnaryFormula(element)

    def visitAUntilFormula(self, element):
        self.visitTemporalBinaryFormula(element)

    def visitConjFormula(self, element):
        self.visitBinaryFormula(element)

    def visitDisjFormula(self, element):
        self.visitBinaryFormula(element)

    def visitEUntilFormula(self, element):
        self.visitTemporalBinaryFormula(element)

    def visitBinaryFormula(self, element):
        self.sb.append(element.name + "(")
        element.left.accept(self)
        self.sb.append(", ")
        element.right.accept(self)
        self.sb.append(")")

    def visitUnaryFormula(self, element):
        self.sb.append(element.name)
        self.sb.append("(")
        element.left.accept(self)
        self.sb.append(")")

    def visitTrueFormula(self, element):
        self.sb.append("True")

    def visitFalseFormula(self, element):
        self.sb.append("False")

    def visitEmptyFormula(self, element):
        pass

    def visitNegationFormula(self, element):
        self.sb.append("Not")
        element.left.accept(self)

    def visitNAryFormula(self, element):
        self.sb.append(element.name + "(")
        for i, clause in enumerate(element.clauses):
            clause.accept(self)

            # Mimic str.join(..) by omitting comma from last element of sequence.
            if i < len(element.clauses):
                self.sb.append(",")
        self.sb.append(")")

    def visitStateCoordinateFormula(self, element):
        self.sb.append("({})".format(element.i))

    def visitVarConstConstraintFormula(self, element):
        element.op1.accept(self)
        self.sb.append(element.sense + "{}".format(element.op2))

    def visitVarVarConstraintFormula(self, element):
        element.op1.accept(self)
        self.sb.append(element.sense)
        element.op2.accept(self)

    def visitTemporalUnaryFormula(self, element):
        self.sb.append(element.name)
        self.sb.append(str(element.k))
        self.sb.append("(")
        element.left.accept(self)
        self.sb.append(")")

    def visitTemporalBinaryFormula(self, element):
        self.sb.append(element.name + str(element.k) + "(")
        element.left.accept(self)
        self.sb.append(", ")
        element.right.accept(self)
        self.sb.append(")")