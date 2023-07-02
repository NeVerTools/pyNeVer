from gurobipy import GRB
from venus.utils.variable_interface import Variable


class AbstractVariable(Variable):
    def __init__(self, name, lb, ub):
        self.ub = ub
        self.lb = lb
        self.varName = name

    def __str__(self):
        return self.varName

    def __repr__(self):
        return self.__str__()


class BinaryVariable(AbstractVariable):
    def __init__(self, name, lb=0, ub=1):
        super(BinaryVariable, self).__init__(name, lb, ub)

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_var(self):
        return {'name': self.varName, 'vtype': GRB.BINARY, 'lb': self.lb, 'ub': self.ub}


class IntegerVariable(AbstractVariable):
    def __init__(self, name, lb, ub):
        super(IntegerVariable, self).__init__(name, lb, ub)

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_var(self):
        return {'name': self.varName, 'vtype': GRB.INTEGER, 'lb': self.lb, 'ub': self.ub}


class ContinuousVariable(AbstractVariable):
    def __init__(self, name, lb, ub):
        super(ContinuousVariable, self).__init__(name, lb, ub)

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_var(self):
        return {'name': self.varName, 'lb': self.lb, 'ub': self.ub}
