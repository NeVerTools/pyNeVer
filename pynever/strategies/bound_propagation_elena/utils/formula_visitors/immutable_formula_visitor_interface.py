from abc import ABCMeta, abstractmethod

from venus.utils.formula_interface import NOT_IMPLEMENTED


class FormulaVisitorI:
    """
    Immutable visitor interface for formulas. Assumes visit/accept methods have a return value; does not
    rely on mutability to construct a data structure while visiting a formula.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def visitBinaryFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitNAryFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitUnaryFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitTrueFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitFalseFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitEmptyFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitVarVarConstraintFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitNegationFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitStateCoordinateFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitVarConstConstraintFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitTemporalUnaryFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitTemporalBinaryFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitEUntilFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitENextFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitANextFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitDisjFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitConjFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitAUntilFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visitConstraintFormula(self, element):
        raise NotImplementedError(NOT_IMPLEMENTED)