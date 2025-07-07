class ExpressionNode:
    """
    Class representing a binary Expression Tree in form of a recursive node.

    Attributes
    ----------
    data: str
        This node content, may be a number or an algebraic operator.
    node_left: ExpressionNode
        The left sub-node, if empty this node is a leaf.
    node_right: ExpressionNode
        The right sub-node, if empty this node is a leaf.
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
    """

    def __init__(self):
        self.precedence = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '>=': 3,
            '<=': 3,
            '>': 3,
            '<': 3,
            '=': 3,
            '&': 4,
            '|': 4
        }
        self.charStack = []
        self.nodeStack = []

    def build_from_infix(self, infix: str) -> ExpressionNode:
        """
        This method builds an Expression Tree using the ExprNode class.

        Parameters
        ----------
        infix: str
            The infix-notated string to be converted.

        Returns
        -------
        ExpressionNode
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
                    t = ExpressionNode(self.charStack.pop())
                    t1 = self.nodeStack.pop()
                    t2 = self.nodeStack.pop()

                    t.node_left = t2
                    t.node_right = t1
                    self.nodeStack.append(t)

                self.charStack.pop()

            # Operand
            elif not is_operator(char):
                t = ExpressionNode(char)
                self.nodeStack.append(t)

            # Operator
            elif self.precedence[char] > 0:
                while len(self.charStack) > 0 and self.charStack[-1] != '(' and \
                        self.precedence[self.charStack[-1]] >= self.precedence[char]:
                    # Assign nodes
                    t = ExpressionNode(self.charStack.pop())
                    t1 = self.nodeStack.pop()
                    t2 = self.nodeStack.pop()

                    # Link left and right nodes
                    t.node_left = t2
                    t.node_right = t1
                    self.nodeStack.append(t)

                self.charStack.append(char)

        return self.nodeStack.pop()


def is_operator(c: str):
    """
    Utility for checking operators.

    Parameters
    ----------
    c: str
        The character or string to check.

    Returns
    -------
    bool
        True if c is part of the operators set, False otherwise.
    """
    return c == '*' or c == '+' or c == '-' or c == '/' or c == '>' \
        or c == '>=' or c == '<' or c == '<=' or c == '=' \
        or c == '&' or c == '|'


def read_smt_num(val: str):
    """
    Procedure to convert an SMT-LIB string to a number.

    Parameters
    ----------
    val: str
        A string containing a number from an SMT file.

    Returns
    -------
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
        return None


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
    This method refines an SMT-LIB statement by intercepting
    malformed Normal Form formulas and rewriting them correctly.

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
        match line[i - 1]:
            case '(':
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
            case '+':
                post = read_smt_num(line[i - 2])
                if post is not None:
                    acc += post
                    line.pop(i - 1)
                    line.pop(i - 1)
            case '-':
                post = read_smt_num(line[i - 2])
                if post is not None:
                    acc -= post
                    line.pop(i - 1)
                    line.pop(i - 1)
            case _:
                continue

    # If extra bias coefficients have been found, sum up the right hand side
    if acc > 0:
        relop_idx = line.index('<=')
        line[relop_idx + 1] = str(read_smt_num(line[relop_idx + 1]) - acc)

    # Save assertion
    assertion = ''
    for elem in line:
        assertion += elem + ' '

    return assertion.replace('( ', '(').replace(' )', ')')
