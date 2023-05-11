from venus.utils.formula import *
from venus.utils.formula_visitors.immutable_formula_visitor_interface import FormulaVisitorI

INVERTED_SENSE = {LE: GT, GT: LE, GE: LT, LT: GE, EQ: NE, NE: EQ}


class FormulaVisitorNNF(FormulaVisitorI):
    """
    An immutable visitor implementation for converting a formula to NNF.
    """

    def visitNAryDisjFormula(self, element):
        return NAryDisjFormula([e.acceptI(self) for e in element.clauses])

    def visitNAryConjFormula(self, element):
        return NAryConjFormula([e.acceptI(self) for e in element.clauses])

    def visitUnaryFormula(self, element):
        return element.left.acceptI(self)

    def visitNegationFormula(self, element):
        subformula = element.left
        if isinstance(subformula, NegationFormula):
            return subformula.left.acceptI(self)

        if isinstance(subformula, VarVarConstraint):
            return VarVarConstraint(subformula.op1, INVERTED_SENSE[subformula.sense], subformula.op2)

        if isinstance(subformula, VarConstConstraint):
            return VarConstConstraint(subformula.op1, INVERTED_SENSE[subformula.sense], subformula.op2)

        if isinstance(subformula, LinExprConstraint):
            return LinExprConstraint(subformula.op1, INVERTED_SENSE[subformula.sense], subformula.op2)

        if isinstance(subformula, AtomicConjFormula):
            return AtomicDisjFormula(NegationFormula(subformula.left).acceptI(self), NegationFormula(subformula.right).acceptI(self))

        if isinstance(subformula, AtomicDisjFormula):
            return AtomicConjFormula(NegationFormula(subformula.left).acceptI(self), NegationFormula(subformula.right).acceptI(self))

        if isinstance(subformula, ConjFormula):
            return DisjFormula(NegationFormula(subformula.left).acceptI(self), NegationFormula(subformula.right).acceptI(self))

        if isinstance(subformula, DisjFormula):
            return ConjFormula(NegationFormula(subformula.left).acceptI(self), NegationFormula(subformula.right).acceptI(self))

        if isinstance(subformula, NAryDisjFormula):
            return NAryConjFormula([NegationFormula(clause).acceptI(self) for clause in subformula.clauses])

        if isinstance(subformula, NAryConjFormula):
            return NAryDisjFormula([NegationFormula(clause).acceptI(self) for clause in subformula.clauses])

        if isinstance(subformula, ENextFormula):
            return ANextFormula(subformula.k, NegationFormula(subformula.left).acceptI(self))

        if isinstance(subformula, ANextFormula):
            return ENextFormula(subformula.k, NegationFormula(subformula.left).acceptI(self))

        if isinstance(subformula, EUntilFormula):
            """
            NOT(E phi1 U^k phi2) =>
                    AG^k NOT(phi2) OR A NOT(phi2) U^k (NOT(phi2) AND NOT(phi1))
            """
            left_nnf = NegationFormula(subformula.left).acceptI(self)
            right_nnf = NegationFormula(subformula.right).acceptI(self)

            # left
            left_subformula = right_nnf
            for i in range(subformula.k):
                left_subformula = ConjFormula(right_nnf, ANextFormula(1, left_subformula))

            right_subformula = AUntilFormula(subformula.k, right_nnf, ConjFormula(right_nnf, left_nnf)).acceptI(self)

            return DisjFormula(left_subformula, right_subformula)

        if isinstance(subformula, AUntilFormula):
            """
            NOT(A phi1 U^k phi2) =>
                    AG^k NOT(phi2) OR E NOT(phi2) U^k (NOT(phi2) AND NOT(phi1))
            """
            left_nnf = NegationFormula(subformula.left).acceptI(self)
            right_nnf = NegationFormula(subformula.right).acceptI(self)

            left_subformula = right_nnf
            for i in range(subformula.k):
                left_subformula = ConjFormula(right_nnf, ENextFormula(1, left_subformula))

            right_subformula = EUntilFormula(subformula.k, right_nnf, ConjFormula(right_nnf, left_nnf)).acceptI(self)

            return DisjFormula(left_subformula, right_subformula)

        return NegationFormula(subformula.to_NNF())

    def visitVarConstConstraintFormula(self, element):
        return VarConstConstraint(element.op1, INVERTED_SENSE[element.sense], element.op2)

    def visitEUntilFormula(self, element):
        left_nnf = element.left.acceptI(self)
        right_nnf = element.right.acceptI(self)

        subformula = right_nnf
        for i in range(element.k):
           subformula = DisjFormula(right_nnf, ConjFormula(left_nnf, ENextFormula(1, subformula)))

        return subformula

    def visitDisjFormula(self, element):
        return DisjFormula(element.left.acceptI(self), element.right.acceptI(self))

    def visitConjFormula(self, element):
        return ConjFormula(element.left.acceptI(self), element.right.acceptI(self))

    def visitAUntilFormula(self, element):
        left_nnf = element.left.acceptI(self)
        right_nnf = element.right.acceptI(self)

        subformula = right_nnf
        for i in range(element.k):
            subformula = DisjFormula(right_nnf, ConjFormula(left_nnf, ANextFormula(1, subformula)))

        return subformula
