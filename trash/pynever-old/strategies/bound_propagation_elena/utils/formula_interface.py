from abc import ABCMeta, abstractmethod

NOT_IMPLEMENTED = "Needs to be implemented."


class Formula:
    """
    the abstract class representing a temporal logic formula. Subsumes propositional logic.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def accept(self, visitor):
        """
        Mutable accept. See Docstring of FormulaVisitor (mutable formula visitor).
        :param visitor: Visitor object
        :return: None
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def acceptI(self, visitor):
        """
        Immutable accept. See Docstring of FormulaVisitorI (immutable formula visitor).
        :param visitor: Visitor object
        :return: Any concrete return type
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

