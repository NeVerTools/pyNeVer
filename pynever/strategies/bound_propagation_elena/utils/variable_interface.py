from abc import ABCMeta, abstractmethod

NOT_IMPLEMENTED = "Needs to be implemented."


class Variable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def accept(self, visitor):
        """
        Mutable accept. See Docstring of VariableVisitor (mutable variable visitor).
        :param visitor: Visitor object
        :return: None
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def acceptI(self, visitor):
        """
        Immutable accept. See Docstring of VariableVisitorI (immutable variable visitor).
        :param visitor: Visitor object
        :return: Any concrete return type
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def to_gurobi_var(self):
        """
        :return: kwargs to be passed into a call to gurobi.addVar().
        """
        raise NotImplementedError(NOT_IMPLEMENTED)
