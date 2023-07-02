import numpy as np
from venus.utils.constraint_interface import Constraint
from operator import __le__, __ge__, __eq__

from venus.utils.variable import BinaryVariable


class LinearConstraint(Constraint):

    def __init__(self, lhs, sense, rhs, name=""):
        """

        :param lhs: Left-hand side for the new constraint,
        a weighted sum of variables.
        :param sense: Sense for the new constraint.
        :param rhs: Right-hand side for the new constraint,
        a constant.
        :param name: Name for new constraint.
        """

        assert isinstance(lhs, WeightedSum)

        self.name = name
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs
        self.sense_repr = {__eq__: "==", __le__: "<=", __ge__: ">="}[sense]

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_constr(self, all_grb_vars, gmodel):
        vars = self.lhs.vars
        coeffs = np.array(self.lhs.coeffs)
        grb_vars = []
        for var in vars:
            grb_vars.append(all_grb_vars[var.varName])
        grb_constr = self.sense(coeffs.dot(grb_vars), self.rhs)
        return grb_constr

    def __str__(self):
        return "({} {} {})".format(self.lhs, self.sense_repr, self.rhs)

    def __repr__(self):
        return self.__str__()


class WeightedSum:
    def __init__(self, variables, coefficients):
        """

        :param variables: a list of variables
        :param coefficients: a list of matching length of coefficients
        """
        assert len(variables) == len(coefficients)

        self.vars = variables
        self.coeffs = coefficients
        self.var_coeff_map = {}
        for i in range(len(variables)):
            self.var_coeff_map[str(variables[i])] = coefficients[i]

    def __str__(self):
        return " + ".join(["{}*{}".format(coeff, varname) for varname, coeff in self.var_coeff_map.items()])

    def __repr__(self):
        return self.__str__()


class IndicatorConstraint(Constraint):

    def __init__(self, binvar, binval, lin_constr, name=""):
        """
        Add a new indicator constraint.
        :param binvar: A binary indicator variable.
        :param binval: The value for the binary indicator variable that would force the linear constraint to be satisfied.
        :param linear_constr: Linear constraint triggered by the indicator.
        :param name: Name for the new indicator constraint.
        """

        assert isinstance(lin_constr, LinearConstraint)
        assert isinstance(binvar, BinaryVariable)

        self.name = name
        self.lin_constr = lin_constr
        self.binval = binval
        self.binvar = binvar

    def accept(self, visitor):
        pass

    def acceptI(self, visitor):
        pass

    def to_gurobi_constr(self, all_grb_vars, gmodel):
        grb_bin_var = all_grb_vars[self.binvar.varName]
        grb_lin_constr = self.lin_constr.to_gurobi_constr(all_grb_vars, gmodel)
        grb_constr = (grb_bin_var == self.binval) >> grb_lin_constr
        return grb_constr

    def __str__(self):
        return "({} == {}) >> {}".format(self.binvar, self.binval, self.lin_constr)

    def __repr__(self):
        return self.__str__()
