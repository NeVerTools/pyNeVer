import numpy as np
from pysmt.smtlib.parser import SmtLibParser

import pynever.strategies.processing as proc
from pynever.tensor import Tensor
from pynever.strategies.verification import SMTLIBProperty, NeVerProperty


class SmtPropertyParser:
    """
    SMTLIB parser utility for building a Star expressed in normal form:
    in_coef_mat * x <= in_bias_mat
    out_coef_mat * y <= out_bias_mat

    Attributes
    ----------
    smtlib_property: SMTLIBProperty
        Container of the SMTLIB file in which the property is defined.
    x: list
        Input vector.
    y: list
        Output vector.
    x_name: str
        Name of the input vector.
    y_name: str
        Name of the output vector.
    in_coef_mat: Tensor
        Matrix of the coefficients for the input constraints.
    in_bias_mat: Tensor
        Matrix of the biases for the input constraints.
    out_coef_mat: list
        List of matrices of the coefficients for the output constraints.
    out_bias_mat: list
        List of matrices of the biases for the output constraints.

    Methods
    -------
    __as_script()
        Procedure to serialize the SMTLIB file as a list of commands via pysmt.
    __get_assert_commands_for(x)
        Procedure to extract the SMTLIB 'assert' commands regarding the literal 'x'.
    __get_coef_mat(list, str, list)
        Procedure to extract the input vector coefficients of the given Tensor
        from the SMTLIB file.
    __get_bias_mat(list)
        Procedure to extract the bias coefficients of the given Tensor from the
        SMTLIB file.
    get_components_of(str)
        Procedure to build a list containing the components of the 'str' vector,
        i.e., the declared variables beginning with 'str' in the SMTLIB file.
    parse_property()
        Exposed procedure to parse the SMTLIB file and build the NeVerProperty.

    """

    def __init__(self, smtlib_property: SMTLIBProperty, x_name: str, y_name: str):
        self.smtlib_property = smtlib_property
        self.x_name = x_name
        self.y_name = y_name
        self.x = self.get_components_of(self.x_name)
        self.y = self.get_components_of(self.y_name)
        self.in_coef_mat = Tensor([])
        self.in_bias_mat = Tensor([])
        self.out_coef_mat = []
        self.out_bias_mat = []

    def __as_script(self):
        """
        This method makes use of pysmt for extracting the commands.

        Returns
        ----------
        SmtLibScript
            The list of SMTLIB commands contained in the file.

        """

        parser = SmtLibParser()
        return parser.get_script_fname(self.smtlib_property.smtlib_path)

    def __get_assert_commands_for(self, x: str) -> list:
        """
        Utility method for retrieving the SMTLIB assertions regarding the variable 'x'.

        Parameters
        ----------
        x: str
            The name of the literal of interest.

        Returns
        ----------
        list
            The list of SMTLIB commands obtained via pysmt.

        """

        script = self.__as_script()
        assertions = script.filter_by_command_name('assert')
        assert_var = []

        # Filter assertions by vec_name
        for a in assertions:
            line = str(a.args[0]).replace('\'', '')
            if x in line:
                assert_var.append(line)

        return assert_var

    def __get_coef_mat(self, vector: list, vec_name: str, asserts: list) -> Tensor:
        """
        This method reads the assertions and extracts the coefficients
        for the given vector.

        Parameters
        ----------
        vector: list
            The vector of interest.
        vec_name: str
            The vector name.
        asserts: list
            The list of SMTLIB 'assert' commands obtained via pysmt.

        Returns
        ----------
        Tensor
            The coefficient matrix associated to 'vec_name' and the
            assertion list.

        """

        # Prepare output
        n_var = len(vector)
        coef_mat = np.zeros((len(asserts), n_var))

        # Row counter
        loop = 0

        for a in asserts:
            # Tokenize
            line = a.replace('(', ' ( ').replace(')', ' ) ').split()
            row = np.zeros(n_var)
            idx = []

            # Get variables index
            for elem in line:
                if vec_name in elem:
                    idx.append(line.index(elem))

            # Get coefficients for each index
            for i in idx:
                # Variable number --- corresponding to the vector index
                v_num = int(line[i].replace(vec_name + '_', ''))
                before = line[i - 1]

                # var on lhs
                if before == '(':
                    # Check if the coefficient is on the right side
                    if line[i + 1] == '*':
                        row[v_num] += proc.read_smt_num(line[i + 2])
                    elif line[i + 1] == '/':
                        row[v_num] += float(1 / proc.read_smt_num(line[i + 2]))
                    else:
                        row[v_num] += 1

                # var as addendum (c = 1)
                elif before == '+':
                    row[v_num] += 1

                # Same, but negative (c = -1)
                elif before == '-':
                    row[v_num] -= 1

                # var with coefficient
                elif before == '*':
                    row[v_num] += proc.read_smt_num(line[i - 2])

            # Row now is complete and can be put in the matrix
            coef_mat[loop] = row
            loop += 1

        return coef_mat

    def __get_bias_mat(self, asserts: list) -> Tensor:
        """
        This method reads the assertions and extracts the
        known terms in order to build the bias vector.

        Parameters
        ----------
        asserts: list
            The list of SMTLIB 'assert' commands obtained via pysmt.
            
        Returns
        ----------
        Tensor
            The bias matrix associated to the assertion list.

        """

        # Init
        bias_mat = np.zeros((len(asserts), 1))
        tree_converter = proc.ExpressionTreeConverter()

        # Row counter
        loop = 0

        for a in asserts:
            # Tokenize
            line = a.replace('(', ' ( ').replace(')', ' ) ')

            b_leaf = tree_converter.build_from_infix(line).node_right
            assert b_leaf.is_leaf(), 'The node' + b_leaf + 'was not leaf'

            bias_mat[loop] = proc.read_smt_num(b_leaf.data)
            loop += 1

        return bias_mat

    def get_components_of(self, vec_name: str) -> list:
        """
        This method reads the components of the given named vector and
        returns the corresponding list.

        Parameters
        ----------
        vec_name: str
            The vector name to find the components of.

        Returns
        ----------
        list
            A list containing the input vector components.

        """

        script = self.__as_script()
        declarations = script.filter_by_command_name(['declare-fun', 'declare-const'])
        vec_list = []

        # Loop declarations and match with name
        for d in declarations:
            v = str(d.args[0]).replace('\'', '')  # args[0] contains the variable name, e.g., 'X_0'
            if vec_name in v:  # If vec_name == 'X' the variable is part of it
                vec_list.append(v)

        return vec_list

    def parse_property(self) -> NeVerProperty:
        """
        This method exposes the propriety parsing, performing all the steps and
        filling the Tensors.

        Returns
        ----------
        NeVerProperty
            The parsed property wrapped in the corresponding class

        """

        x_assert = self.__get_assert_commands_for(self.x_name)
        y_assert = self.__get_assert_commands_for(self.y_name)
        disjunct_list = []
        tree_converter = proc.ExpressionTreeConverter()

        # If y_assert contains some 'OR' statement I should separate
        for a in y_assert:
            if '|' in a:
                tree = tree_converter.build_from_infix(a)
                disjunct_list.extend(tree.get_disjunctions_infix())

        # Refine assert statements
        for idx, a in enumerate(x_assert):
            x_assert[idx] = proc.refine_smt_statement(a, self.x_name)

        if len(disjunct_list) == 0:
            for idx, a in enumerate(y_assert):
                y_assert[idx] = proc.refine_smt_statement(a, self.y_name)
        else:
            for idx, a in enumerate(disjunct_list):
                disjunct_list[idx] = proc.refine_smt_statement(a, self.y_name)

        # Warning without reasons. self.x, self.y ARE lists but PyCharm is not convinced
        self.in_coef_mat = self.__get_coef_mat(self.x, self.x_name, x_assert)
        self.in_bias_mat = self.__get_bias_mat(x_assert)

        if len(disjunct_list) == 0:  # If no 'OR' statements are present, just proceed normally
            self.out_coef_mat.append(self.__get_coef_mat(self.y, self.y_name, y_assert))
            self.out_bias_mat.append(self.__get_bias_mat(y_assert))
        else:  # Otherwise, separate in different matrices
            disjunct_list.reverse()
            for d in disjunct_list:
                self.out_coef_mat.append(self.__get_coef_mat(self.y, self.y_name, [d]))
                self.out_bias_mat.append(self.__get_bias_mat([d]))

        return NeVerProperty(self.in_coef_mat, self.in_bias_mat, self.out_coef_mat, self.out_bias_mat)
