"""
Contains the visitor functions to evaluate the nodes (assertions) and return the input and output matrices.
"""

import torch

import Constants
import Node


# Function called to visit an operation ( + - * / ) and return the coefficients vector and known member and if it contains variables (coefficients != 0)
def visitOperation(node, varNumber):
    """
    Visits an operation node and evaluates it, returning the coefficients vector,
    known member and if it contains variables (coefficients are not all zeros).

    Parameters
        ----------
        node: ``Node``
            operation node to be visited, expected to have two children (left and right member)
        varNumber: int
            how many variables are expected in the coefficients vector (length of the vector)

        Returns
        -------
        list of float 
        float
        bool
    """

    # These will be returned but temporarily they will store the left member information
    coefficients = [0.0] * varNumber
    knownMember = 0.0

    # Get the operation value (e.g. +, -, *, /)
    OperationTag = node.value
    if OperationTag not in (Constants.ADD, Constants.SUB, Constants.MUL, Constants.DIV):
        raise ValueError(f"Node {node} is not a valid operation node, cannot evaluate it (visitOperation).")

    # Chekc if the operation has two children
    if len(node.children) != 2:
        raise ValueError(
            f"Node {node} is not a valid operation node, it has {len(node.children)} children instead of 2.")

    LeftMember = node.children[0]
    RightMember = node.children[1]
    # Flag to check if the left member is a variable node (input or output)
    LeftIsVariable = False
    RightIsVariable = False

    """Left member of the operation"""
    if LeftMember.type in (Node.INVAR, Node.OUTVAR):  # Input variable node
        # Set the coefficient for the input variable
        coefficients[LeftMember.value] += 1.0
        LeftIsVariable = True

    elif LeftMember.type == Node.CONST:  # Constant node
        knownMember = LeftMember.value  # Set the known member to the constant value


    # Otherwise, the left member could be an operation node => evaluate it and update the coefficients vector and known member accordingly
    elif LeftMember.type == Node.OPERATION:
        coefficients, knownMember, LeftIsVariable = visitOperation(LeftMember, varNumber)

    else:
        raise ValueError(f"Left member {LeftMember} is not a valid input or output variable or constant node.")

    """Right member of the operation"""

    Rcoeff = [0.0] * varNumber  # Coefficients vector for the right member
    Rknown = 0.0  # Known member for the right member

    if RightMember.type in (Node.INVAR, Node.OUTVAR):  # Input variable node
        # Set the coefficient for the input variable
        Rcoeff[RightMember.value] += 1.0
        RightIsVariable = True

    elif RightMember.type == Node.CONST:  # Constant node
        Rknown = RightMember.value  # Set the known member to the constant value


    # Otherwise, the left member could be an operation node => evaluate it and update the coefficients vector and known member accordingly
    elif RightMember.type == Node.OPERATION:
        Rcoeff, Rknown, RightIsVariable = visitOperation(RightMember, varNumber)

    else:
        raise ValueError(f"Rigth member {RightMember} is not a valid input or output variable or constant node.")

    """Update the coefficients vector and known member according to the operation"""

    match OperationTag:
        case Constants.ADD:
            if LeftIsVariable and RightIsVariable:
                for i in range(varNumber):
                    coefficients[i] += Rcoeff[i]
            elif RightIsVariable:
                for i in range(varNumber):
                    coefficients[i] = Rcoeff[i]
            knownMember += Rknown

        case Constants.SUB:
            if LeftIsVariable and RightIsVariable:
                for i in range(varNumber):
                    coefficients[i] -= Rcoeff[i]
            elif RightIsVariable:
                for i in range(varNumber):
                    coefficients[i] = -Rcoeff[i]
            knownMember -= Rknown

        case Constants.MUL:
            if LeftIsVariable and RightIsVariable:
                raise ValueError("Multiplication of two variables is not supported in linear constraints.")
            elif LeftIsVariable:
                for i in range(varNumber):
                    coefficients[i] = coefficients[i] * Rknown
            elif RightIsVariable:
                for i in range(varNumber):
                    coefficients[i] = Rcoeff[i] * knownMember
            knownMember *= Rknown

        case Constants.DIV:
            if LeftIsVariable and RightIsVariable:
                raise ValueError("Division of two variables is not supported in linear constraints.")
            elif LeftIsVariable:
                for i in range(varNumber):
                    coefficients[i] = coefficients[i] / Rknown
            elif RightIsVariable:
                raise ValueError("Division by a variable is not supported in linear constraints.")
            knownMember /= Rknown

        case _:
            raise ValueError(f"Operation {OperationTag} is not a valid operation, cannot evaluate it (visitOperation).")

    # Return the coefficients vector, known member and if it contains variables
    return coefficients, knownMember, (LeftIsVariable or RightIsVariable)


# This function evaluates a simple assertion - form (assert (>= X_0 X_1) )
# It return a line of the matrix (number of columns is varNumber)
def visitSimple(operationNode, varNumber):
    """
    Visits a simple assertion node (input constraints), returning the coefficients vector and known member.

    Parameters
        ----------
        operationNode: ``Node``
            operation node to be visited (disequality expected), expected to have two children (left and right member)
        varNumber: int
            how many variables are expected in the coefficients vector (length of the vector)

        Returns
        -------
        list of float 
        float
        bool
    """

    # Get the operation node (the first child and only of the assertion node). If it is >= raise change sign flag
    # The matrices must be in canonical form
    changeSignFlag = False

    if operationNode.type != Node.OPERATION:
        raise ValueError(f"Node {operationNode} is not an operation node, cannot evaluate it.")

    if operationNode.value not in (Constants.LE, Constants.GE):
        raise ValueError(f"Node {operationNode} is not a >= or <= operation, cannot evaluate it.")

    if operationNode.value == Constants.GE:
        changeSignFlag = True

    # Get the two operands of the operation node
    LeftMember = operationNode.children[0]
    RightMember = operationNode.children[1]
    # Known member will have to be in the right member and variables in the left member (at the end)
    # Initialize the coefficients vector with zeros (everything must be float!)
    coefficients = [0.0] * varNumber  # In Python, questo � il modo corretto e idiomatico per creare una lista di float inizializzati a zero
    knownMember = 0.0

    # Check the left member and generate its coefficients vector
    if LeftMember.type in (Node.INVAR, Node.OUTVAR):  # Input variable node
        # Set the coefficient for the input variable

        coefficients[LeftMember.value] += 1.0


    elif LeftMember.type == Node.CONST:  # Constant node

        knownMember -= LeftMember.value


    # Otherwise, the left member could be an operation node => evaluate it and update the coefficients vector and known member accordingly
    elif LeftMember.type == Node.OPERATION:
        coeff, known, IsVar = visitOperation(LeftMember, varNumber)

        # coeff is a vector of coefficients long as varNumber, known  (all float)
        # Update the coefficients vector and known member

        if IsVar:
            for i in range(varNumber):
                coefficients[i] += coeff[i]

            knownMember -= known



    else:
        raise ValueError(f"Left member {LeftMember} is not a valid input or output variable or constant.")

    # Check the right member and generate its coefficients vector
    if RightMember.type in (Node.INVAR, Node.OUTVAR):  # Input variable node
        # Set the coefficient for the input variable

        coefficients[RightMember.value] -= 1.0


    elif RightMember.type == Node.CONST:  # Constant node

        knownMember += RightMember.value


    # Otherwise, the right member could be an operation node => evaluate it and update the coefficients vector and known member accordingly
    elif RightMember.type == Node.OPERATION:

        coeff, known, IsVar = visitOperation(RightMember, varNumber)
        # coeff is a vector of coefficients long as varNumber, known  (all float)
        # Update the coefficients vector and known member
        if IsVar:
            for i in range(varNumber):
                coefficients[i] -= coeff[i]

            knownMember += known


    else:
        raise ValueError(f"Right member {RightMember} is not a valid input or output variable or constant node.")

    # Return the coefficients vector and known member
    if changeSignFlag:
        coefficients = [-c for c in coefficients]
        knownMember = -knownMember

    return coefficients, knownMember


def visitAnd(node, varNumber):
    """
        This method visits an and. Expected to have multiple children (constraints)
       and return a tensor with the matrix of coefficients and a vector of known members

        Parameters
        ----------
        node: ``Node``
            And node to be visited, expected to have at least one child (simple constraint)
        varNumber: int
            Number of variables needed to make proper matrices

        Returns
        -------
        pytorch tensor
        pytorch tensor
        """

    # Inside an and there should be only simple constraints (>=, <=) => call visitSimple on each child and return the coefficients and known members as a tensor
    if len(node.children) == 0:
        raise ValueError(f"Node {node} is not a valid AND node, it has no children instead of at least 1.")

    rows = len(node.children)
    ATensor = torch.zeros((rows, varNumber), dtype=torch.float32)  # Initialize a tensor with zeros
    BTensor = torch.zeros((rows, 1), dtype=torch.float32)  # Initialize a tensor for known members

    for i in range(rows):
        child = node.children[i]
        if child.type != Node.OPERATION:
            raise ValueError(f"Node {child} is not an assertion node, cannot evaluate it (visitAnd).")

        # Write the coefficients and known member in the tensors
        # visitSimple returns a tuple (coefficients, knownMember)
        # coefficients: list of float, knownMember: float
        # ATensor[i, :] and BTensor[i, 0] are the i-th row of the tensors

        coeff, known = visitSimple(child, varNumber)
        # print(f"Coefficients: {coeff}, Known: {known}")
        ATensor[i, :] = torch.tensor(coeff, dtype=torch.float32)
        BTensor[i, 0] = known

    return ATensor, BTensor


def visitComplex(node, varNumber):
    """
       This method visits the output assertion. Expected OR operation with ANDs or a single constraint

       Parameters
       ----------
       node: ``Node``
           Assertion to be visited, expected to have one child (OR node)
       varNumber: int
           Number of variables needed to make proper matrices

       Returns
       -------
       List of pytorch tensors tuples (A and B for each AND node inside the OR node)
       """

    if len(node.children) != 1:
        raise ValueError(
            f"Output Assertion {node} was expected to have only one child (OR node), instead it has {len(node.children)} children.")

    orNode = node.children[0]
    # throw error if it is not an OR Node
    if orNode.value != Constants.OR:
        raise ValueError(f"Output Assertions is missing an OR node, instead found {orNode}")

    # Initialize a list to store the tensors in couples (coefficients, known member)
    OutputTensorList = []

    # Now the or could have a single constraint or AND node as child or multiple AND nodes
    orLength: int = len(orNode.children)

    if orLength == 0:
        raise ValueError(f"Output assertions has an empty or node inside => {orNode}")


    elif orLength == 1:

        singleConstraint = orNode.children[0]

        # Initialize the tensors to store the coefficients and known member
        TensorA = torch.zeros((1, varNumber), dtype=torch.float32)
        TensorB = torch.zeros((1, 1), dtype=torch.float32)
        # Check if the child is a constraint or an and and act accordingly (they should be operation nodes, so the operation is stored in value)
        if singleConstraint.value in (Constants.GE, Constants.LE):
            # Call visitOperation and return the matrix built on the coefficients and known member returned
            coeff, known = visitSimple(singleConstraint, varNumber)
            # Make a tensor out of the coefficients and known member and return it
            TensorA[0, :] = torch.tensor(coeff, dtype=torch.float32)
            TensorB[0, 0] = known

        elif singleConstraint.value == Constants.AND:
            # Call parseAnd and return the matrices it returns
            coeff, known = visitAnd(singleConstraint, varNumber)
            # Make a tensor out of the coefficients and known member and return it
            TensorA = coeff
            TensorB = known

        else:
            raise ValueError(
                f"Output OR assertion with a single child expected to have a >= , <= or only AND inside, found {orNode.children[0]}")

        OutputTensorList.append((TensorA, TensorB))
        return OutputTensorList

    for i in range(0, orLength):

        # Children of OR Node are expected to be AND nodes => Throw error if they are not and if they are correct call parseAnd and create a list of
        child = orNode.children[i]
        andLength = len(child.children)  # Number of children of the AND node
        if andLength == 0:
            raise ValueError(f"Output OR assertion has an AND node with no children inside, found {child}.")

        if (child.value != Constants.AND):
            raise ValueError(
                f"Output OR assertion with multiple children expected to have only ANDs inside, found {child}")
        # Call parseAnd and get the matrix (pytorch tensor )=> put it in a list
        ATensor, BTensor = visitAnd(child, varNumber)

        OutputTensorList.append((ATensor, BTensor))  # Aggiungi la coppia (TensorA, TensorB) alla lista

    # Now we have a list of tensors, we can return it
    return OutputTensorList


# This function has to iterate through the assertion nodes and evaluate them, returning the input and output matrices => tensors
def visit(nodes, inputVarNumber, outputVarNumber) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """
        Main method to visit the nodes and evaluate them, returning the input and output matrices.

        Parameters
        ----------
        nodes: list of ``Node``
            Assertions to be visited
        inputVarNumber: int
            Number of variables needed to make proper matrices
        outputVarNumber: int
            Number of output variables needed to make proper matrices

        Returns
        -------
        pytorch tensor
        pytorch tensor
        List of pytorch tensors tuples (A and B for each AND node inside the OR node)
        """

    outputVisited = False
    OutputList = None

    inputRows: int = len(nodes)
    if outputVarNumber != 0:
        # It means that there is an output assertion, not all nodes are input assertions
        inputRows -= 1.

    if inputRows <= 0:
        raise ValueError(
            f"Input assertion nodes are not enough to build a matrix, expected at least 1, got {inputRows}.")

    print(f"Input Rows: {int(inputRows)}, Input Variables: {inputVarNumber}, Output Variables: {outputVarNumber}")

    # Don't knwo why but inputRows is a float, so we have to convert it to int
    InputA = torch.zeros((int(inputRows), inputVarNumber),
                         dtype=torch.float32)  # Initialize a tensor for input coefficients
    InputB = torch.zeros((int(inputRows), 1), dtype=torch.float32)  # Initialize a tensor for input known members

    InputCounter = 0  # Counter for input assertions

    for node in nodes:
        if node.type != Node.ASSERTION:
            raise ValueError(f"Node {node} is not an assertion node, cannot evaluate it.")

        # Value of assertion stores which assertion it is (input, output, simple, complex)
        match node.value:
            # If it is an input assertion => simple assertion 
            case Node.INPUT:
                operationNode = node.children[0]
                if operationNode.type != Node.OPERATION:
                    raise ValueError(
                        f"Node {operationNode} is not an operation node, cannot evaluate it (input assertion).")

                InputCoeff, InputKnown = visitSimple(operationNode, inputVarNumber)
                # Add it to the Input tensors
                InputA[InputCounter, :] = torch.tensor(InputCoeff, dtype=torch.float32)
                InputB[InputCounter, 0] = InputKnown
                InputCounter += 1

                # Debugging
                # print(f"Input Coefficients: {InputCoeff}, Input Known: {InputKnown}")

            # Cambiata policy di progetto: l'output deve avere sempre l'OR nell'assert, e questo assert deve essere unico
            case Node.OUTSIMPLE:
                # Error! Output assertion must be complex

                raise ValueError(f"Output assertion {node} is not complex! It must have an OR node inside.")
                # Add it to the Output matrix

            case Node.OUTCOMPL:
                if outputVisited:
                    raise ValueError(
                        f"Output assertion {node} is already visited, there can be only one output assertion in the model.")
                OutputList = visitComplex(node, outputVarNumber)
                outputVisited = True

                # Debugging
                # print (f"Output List: {OutputList}")

            case _:
                raise ValueError(
                    f"Node {node} is not a valid assertion node, cannot evaluate it, value : {node.value}.")

    return InputA, InputB, OutputList if outputVisited else None
