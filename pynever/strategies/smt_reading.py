import re

import numpy as np
from pysmt.smtlib.parser import SmtLibParser

from pynever.tensors import Tensor


class ExprNode:
    """
    Class representing a binary Expression Tree in form of a recursive node.

    Attributes
    ----------
    data: str
        This node content, may be a number or an algebraic operator.
    node_left: ExprNode
        The left sub-node, if empty this node is a leaf.
    node_right: ExprNode
        The right sub-node, if empty this node is a leaf.

    Methods
    ----------
    is_leaf()
        Procedure to check whether the node is a leaf or not.
    as_prefix()
        Procedure to create the prefix string describing the tree.
    as_infix()
        Procedure to create the infix string describing the tree.
    get_disjunctions_infix()
        Procedure to return the list of disjunct atoms, if present.

    """

    def __init__(self, char: str):
        self.data = char
        self.node_left = None
        self.node_right = None

    def is_leaf(self) -> bool:
        """
        This method checks whether the node is a leaf or not.

        Returns
        ----------
        bool
            True if node_left and node_right are None, False otherwise.

        """

        return self.node_left is None and self.node_right is None

    def as_prefix(self) -> str:
        """
        This method converts the Expression Tree in a prefix string

        Returns
        ----------
        str
            The tree representation as a prefix string

        """

        if self.data.startswith('-'):
            data = self.data.replace('-', '(- ')
            self.data = data + ')'

        if self.node_left is None:
            return self.data
        else:
            prefix = '(' + self.data
            prefix += ' ' + self.node_left.as_prefix()
            prefix += ' ' + self.node_right.as_prefix()
            return prefix + ')'

    def as_infix(self) -> str:
        """
        This method converts the Expression Tree in an infix string

        Returns
        ----------
        str
            The tree representation as an infix string

        """

        if self.node_left is None:
            return self.data
        else:
            infix = '(' + self.node_left.as_infix()
            infix += ' ' + self.data
            infix += ' ' + self.node_right.as_infix()
            return infix + ')'

    def get_disjunctions_infix(self) -> list:
        """
        This method is used in order to separate OR statements in
        the input file

        Returns
        ----------
        list
            The list of infix-notated disjunctions

        """

        if self.data != '|':
            raise Exception('No disjunctions detected')

        d_list = [self.node_right.as_infix()]  # Right depth is 1
        if self.node_left.data != '|':
            d_list.append(self.node_left.as_infix())
        else:
            d_list.extend(self.node_left.get_disjunctions_infix())

        return d_list


class ExpressionTreeConverter:
    """
    Class for converting infix expressions to expression trees.

    Courtesy of Nikhil Kumar Singh(nickzuck_007) and aashish1995.

    Attributes
    ----------
    precedence: dict
        A dictionary assigning precedences to operators.
    charStack: list
        Stack for operands, needed in the conversion routine.
    nodeStack: list
        Stack for operators, needed in the conversion routine.

    Methods
    ----------
    build_from_infix(str)
        Procedure to generate an expression tree from an infix string.

    """

    def __init__(self):
        self.precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '>=': 3, '<=': 3, '>': 3, '<': 3, '=': 3, '&': 4, '|': 4}
        self.charStack = []
        self.nodeStack = []

    def build_from_infix(self, infix: str) -> ExprNode:
        """
        This method builds an Expression Tree using the ExprNode class.

        Parameters
        ----------
        infix: str
            The infix-notated string to be converted.

        Returns
        ----------
        ExprNode
            The root node containing the Expression Tree.

        """

        infix = '(' + infix + ')'  # Redundancy for correctness
        # Expand parentheses in order to split correctly
        infix = infix.replace('(', '( ').replace(')', ' )')

        # Read tokens
        for char in infix.split():
            # Expression beginning
            if char == '(':
                self.charStack.append(char)

            # Expression end
            elif char == ')':
                while len(self.charStack) > 0 and self.charStack[-1] != '(':
                    t = ExprNode(self.charStack.pop())
                    t1 = self.nodeStack.pop()
                    t2 = self.nodeStack.pop()

                    t.node_left = t2
                    t.node_right = t1
                    self.nodeStack.append(t)

                self.charStack.pop()

            # Operand
            elif not is_operator(char):
                t = ExprNode(char)
                self.nodeStack.append(t)

            # Operator
            elif self.precedence[char] > 0:
                while len(self.charStack) > 0 and self.charStack[-1] != '(' and \
                        self.precedence[self.charStack[-1]] >= self.precedence[char]:
                    # Assign nodes
                    t = ExprNode(self.charStack.pop())
                    t1 = self.nodeStack.pop()
                    t2 = self.nodeStack.pop()

                    # Link left and right nodes
                    t.node_left = t2
                    t.node_right = t1
                    self.nodeStack.append(t)

                self.charStack.append(char)

        return self.nodeStack.pop()


class SmtPropertyParser:
    """
    SMTLIB parser utility for building a Star expressed in normal form:
    in_coef_mat * x <= in_bias_mat
    out_coef_mat * y <= out_bias_mat

    Attributes
    ----------
    smtlib_path: str
        Path to the SMTLIB file in which the property is defined.
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

    def __init__(self, smtlib_path: str):
        self.smtlib_path = smtlib_path

        self.x_name, self.y_name = self.__get_variable_names()
        self.x = self.get_components_of(self.x_name)
        self.y = self.get_components_of(self.y_name)

        self.in_coef_mat = None
        self.in_bias_mat = None
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
        return parser.get_script_fname(self.smtlib_path)

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

    def __get_variable_names(self) -> tuple[str, str]:
        """
        This method extracts the name of the declared variables in the file
        and returns them in the form of a tuple (input, output)

        """

        script = self.__as_script()
        declarations = script.filter_by_command_name(['declare-fun', 'declare-const'])

        input_name = ''
        output_name = ''
        pattern = '^(.*?)(?=_)'  # Regex to find '_i' after variable name

        for d in declarations:
            suffix = re.sub(pattern, '', str(d.args[0]))

            if input_name != '':
                temp_name = str(d.args[0]).replace(suffix, '')
                if temp_name != input_name:
                    output_name = temp_name
                    break
            else:
                input_name = str(d.args[0]).replace(suffix, '')

        return input_name, output_name

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
                        row[v_num] += read_smt_num(line[i + 2])
                    elif line[i + 1] == '/':
                        row[v_num] += float(1 / read_smt_num(line[i + 2]))
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
                    row[v_num] += read_smt_num(line[i - 2])

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
        tree_converter = ExpressionTreeConverter()

        # Row counter
        loop = 0

        for a in asserts:
            # Tokenize
            line = a.replace('(', ' ( ').replace(')', ' ) ')

            b_leaf = tree_converter.build_from_infix(line).node_right
            assert b_leaf.is_leaf(), 'The node' + b_leaf + 'was not leaf'

            bias_mat[loop] = read_smt_num(b_leaf.data)
            loop += 1

        return bias_mat

    @staticmethod
    def remove_parentheses(cmd: str) -> str:
        strip_idx = 0

        # Forward
        while cmd[strip_idx + 1] == '(':
            strip_idx += 1

        ret = cmd[strip_idx:]
        strip_idx = len(ret)

        # Backward
        while ret[strip_idx - 1] == ')':
            strip_idx -= 1

        ret = ret[:strip_idx + 1]

        return ret

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

    def parse_property(self) -> (Tensor, Tensor, list, list):
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
        tree_converter = ExpressionTreeConverter()

        # If y_assert contains some 'OR' statement I should separate
        for asrt in y_assert:
            if '|' in asrt:
                tree = tree_converter.build_from_infix(asrt)
                disjunct_list.extend(tree.get_disjunctions_infix())
            else:
                ands = []
                for idx, a in enumerate(y_assert):
                    if ' & ' in y_assert[idx]:
                        and_expr = y_assert[idx].split(' & ')

                        for out_asrt in and_expr:
                            ands.append(self.remove_parentheses(out_asrt))

                if len(ands) > 0:
                    y_assert = ands

        # Process 'and' in output properties disjunct
        for i in range(len(disjunct_list)):
            if ' & ' in disjunct_list[i]:

                # Here we "abuse" Python's freedom and make a list of lists out of a list of statements
                disjunct_list[i] = disjunct_list[i].split(' & ')

                count = 0
                for a in disjunct_list[i]:
                    disjunct_list[i][count] = self.remove_parentheses(a)
                    count += 1

        # Refine assert statements
        for idx, a in enumerate(x_assert):
            x_assert[idx] = refine_smt_statement(a, self.x_name)

        if len(disjunct_list) == 0:
            for idx, a in enumerate(y_assert):
                y_assert[idx] = refine_smt_statement(a, self.y_name)
        else:
            for d_idx, statement in enumerate(disjunct_list):
                if isinstance(statement, list):
                    for idx, a in enumerate(statement):
                        disjunct_list[d_idx][idx] = refine_smt_statement(a, self.y_name)
                else:
                    disjunct_list[d_idx] = refine_smt_statement(statement, self.y_name)

        # Warning without reasons. self.x, self.y ARE lists but PyCharm is not convinced
        self.in_coef_mat = self.__get_coef_mat(self.x, self.x_name, x_assert)
        self.in_bias_mat = self.__get_bias_mat(x_assert)

        if len(disjunct_list) == 0:  # If no 'OR' statements are present, just proceed normally
            self.out_coef_mat.append(self.__get_coef_mat(self.y, self.y_name, y_assert))
            self.out_bias_mat.append(self.__get_bias_mat(y_assert))
        else:  # Otherwise, separate in different matrices
            disjunct_list.reverse()
            for d in disjunct_list:
                if isinstance(d, list):
                    self.out_coef_mat.append(self.__get_coef_mat(self.y, self.y_name, d))
                    self.out_bias_mat.append(self.__get_bias_mat(d))
                else:
                    self.out_coef_mat.append(self.__get_coef_mat(self.y, self.y_name, [d]))
                    self.out_bias_mat.append(self.__get_bias_mat([d]))

        return self.in_coef_mat, self.in_bias_mat, self.out_coef_mat, self.out_bias_mat


def is_operator(c: str):
    """
    Utility for checking operators.

    Parameters
    ----------
    c: str
        The character or string to check.

    Returns
    ----------
    bool
        True if c is part of the operators set, False otherwise.

    """

    return c == '*' or c == '+' or c == '-' or c == '/' or c == '>' \
        or c == '>=' or c == '<' or c == '<=' or c == '=' \
        or c == '&' or c == '|'


def read_smt_num(val: str):
    """
    Procedure to convert a SMTLIB string to a number.

    Parameters
    ----------
    val: str
        A string containing a number from a SMT file.

    Returns
    ----------
    Any
        int if the given string represents an integer,
        float if it represents a float or
        None if it does not represent a number.

    """

    try:
        if '.' in val:
            return float(val)
        elif '/' in val:
            val = val.split('/')
            return float(int(val[0]) / int(val[1]))
        else:
            return int(val)
    except ValueError:
        return


def prefix2infix(prefix: str) -> str:
    """
    Procedure for converting a prefix string to an infix string.

    Parameters
    ----------
    prefix: str
        The prefix string that should be converted.

    Returns
    ----------
    str
        The infix-converted string.

    """

    # Preprocess
    prefix = prefix.replace('(', ' ').replace(')', ' ').replace('assert', '').split()
    stack = []

    # Read tokens in reverse order
    i = len(prefix) - 1
    while i >= 0:
        # Symbol is operand
        if not is_operator(prefix[i]):
            stack.append(prefix[i])
            i -= 1
        # Symbol is operator
        else:
            res = '(' + stack.pop() + ' ' + prefix[i] + ' ' + stack.pop() + ')'
            stack.append(res)
            i -= 1

    return stack.pop()


def refine_smt_statement(assertion: str, vec_name: str) -> str:
    """
    This method refines a SMTLIB statement by intercepting
    malformed Normal Form formulas and rewriting them
    correctly.

    Parameters
    ----------
    assertion
        The SMT formula to verify.
    vec_name
        The variable name in use

    Returns
    -------
    str
        The formula in input if it is already well-formed,
        its refinement otherwise.

    """

    tree_converter = ExpressionTreeConverter()
    tree = tree_converter.build_from_infix(assertion)

    # Case Xi REL_OP Xj -> Xi - Xj REL_OP 0
    # This is skipped if one of the children is a number
    if read_smt_num(tree.node_left.data) is None and read_smt_num(tree.node_right.data) is None:
        assertion = '((' + tree.node_left.as_infix() + ' - ' + \
                    tree.node_right.as_infix() + ') ' + tree.data + ' 0.0)'

    # Case beta REL_OP Xi -> (-1.0 * Xi) REL_OP eval(-beta)
    left_num = read_smt_num(tree.node_left.data)
    if left_num is not None:
        sign = '' if left_num <= 0 else '-'

        if tree.node_right.is_leaf():
            assertion = '((-1.0 * ' + tree.node_right.as_infix() + ') ' + \
                        tree.data + f" {sign}" + tree.node_left.as_infix().replace('-', '') + ')'
        else:
            # Case beta REL_OP (alpha * Xi) -> (-alpha * Xi) REL_OP eval(-beta)
            tokens = assertion.replace('(', '').replace(')', '').split(' ')
            alpha_idx = tokens.index('*') - 1  # Assume there is only a * character
            tokens[alpha_idx] = str(-read_smt_num(tokens[alpha_idx]))
            assertion = ('((' + tokens[alpha_idx] + ' * ' + tokens[alpha_idx + 2] + ') ' + tokens[1] +
                         f" {sign}" + tree.node_left.as_infix().replace('-', '') + ')')

    # Case (Xi + alpha) REL_OP beta -> Xi REL_OP eval(beta - alpha)
    # This is always performed
    line = assertion.replace('(', ' ( ').replace(')', ' ) ').split()
    idx = []
    acc = 0

    # Get variables index
    for elem in line:
        if vec_name in elem:
            idx.append(line.index(elem))

    for i in idx:
        before = line[i - 1]
        if before == '(':
            # Try to evaluate the token as a number
            post = read_smt_num(line[i + 2])
            if post is not None:
                if line[i + 1] == '+':
                    acc += post
                    line.pop(i + 1)
                    line.pop(i + 1)
                elif line[i + 1] == '-':
                    acc -= post
                    line.pop(i + 1)
                    line.pop(i + 1)
        elif before == '+':
            post = read_smt_num(line[i - 2])
            if post is not None:
                acc += post
                line.pop(i - 1)
                line.pop(i - 1)
        elif before == '-':
            post = read_smt_num(line[i - 2])
            if post is not None:
                acc -= post
                line.pop(i - 1)
                line.pop(i - 1)

    # If extra bias coefficients have been found, sum up the right hand side
    if acc > 0:
        relop_idx = line.index('<=')
        line[relop_idx + 1] = str(read_smt_num(line[relop_idx + 1]) - acc)

    # Save assertion
    assertion = ''
    for elem in line:
        assertion += elem + ' '

    return assertion.replace('( ', '(').replace(' )', ')')
