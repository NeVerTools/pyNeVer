"""
Takes a token list and parses it creating nodes (syntax analysis) => Each node represents a declaration, an assertion or an expression
Nodes can be nested. 
The parse can also handle output assertions written as the input assertions creating nodes with value OUTSIMPLE,
but the visitor raises an error if it finds OUTSIMPLE. (The code was already done, so I didn't delete it)
"""

#Need token for constants and Token class
import Constants
import Token
import Node


#Get the next token or raise an error
def safe_next(iterator, i):
    """
    Advances the iterator and returns the next token and the updated index.
    Raises a SyntaxError if the iterator is exhausted.

    Parameters
        ----------
        iterator: iterator
            iterator over the tokens list
        i: int
            iteration counter

        Returns
        -------
        iterator
        int
    """
    try:
        #Return a tuple with two elements
        return next(iterator), i+1
    except StopIteration:
        raise SyntaxError("Unexpected end of input")




#Every possible operation is enclosed in parentheses (SMT-LIB format) => check for errors in that sense while parsing

def parseDeclaration(TokenItr, i, token, inputVarNum, outputVarNum):
    """
    Parse a declaration node and updata the input and output variable numbers.

     Parameters
        ----------
        TokenItr: iterator
            itator over the tokens list
        i: int
            iteration counter to keep track of the current position in the token list
        token: Token
            the current token being parsed
        inputVarNum: int
            Keeps track of how many input variables have been declared so far, used to check if the input variable number matches the expected one
        outputVarNum: int
            Keeps track of how many output variables have been declared so far, used to check if the output variable number matches the expected one

        """
    token, i = safe_next(TokenItr, i)

    match token.tag:

        case Constants.IN :
            varNumber = int(token.value)
            if varNumber == inputVarNum:
                inputVarNum += 1
            else: 
                raise SyntaxError(f"Input variable number {varNumber} does not match the expected input variable number {inputVarNum}")
        case Constants.OUT:
            varNumber = int(token.value)
            if varNumber == outputVarNum:
                outputVarNum += 1
            else:
                raise SyntaxError(f"Output variable number {varNumber} does not match the expected output variable number {outputVarNum}")

        case _: 
            raise SyntaxError(f"Expected input or output variable token during declaration, found {Token.id2word[token.tag]}, row {token.row}")

     # Check for optional real token
    token, i = safe_next(TokenItr, i)
    if token.tag == Constants.REA:
        #Do nothing, just skip it
        token, i = safe_next(TokenItr, i)
        # Check for right parenthesis
    if token.tag != Constants.RP:
        raise SyntaxError(f"Expected right parenthesis after declaration, found {Token.id2word[token.tag]}, row {token.row}")

    return i, inputVarNum, outputVarNum





#Check if the number is written in fraction form (e.g. 1/3) => NUM TOKEN
#Returns the token, the iteration counter and the value to put in the constant node
def parseConstant(token, TokenItr, i):
    """
    Parse a constant token and return the corresponding value to put in the node.

     Parameters
        ----------
        token: Token
            the current token being parsed
        i: int
            iteration counter to keep track of the current position in the token list
        TokenItr: iterator
            itator over the tokens list
        
    Returns
        -------
        Token
        int
        float
        """

    numerator = float(token.word)  # Convert the number to float

    token, i = safe_next(TokenItr, i)  # Get the next token
# DIV TOKEN NUM TOKEN if it is in fraction form
    if token.tag == Constants.DIV:
        #Get the next token, it should be a number
        token, i = safe_next(TokenItr, i)
        if token.tag != Constants.NUM:
            raise SyntaxError(f"Expected number after division token (expected fraction form number), found {Token.id2word[token.tag]}, row {token.row}")
        denominator = float(token.value)
        temp = float(numerator / denominator)

        token, i = safe_next(TokenItr, i)  # Get the next token after the number
    else:
        #Otherwise generate a constant node with the number value
        temp = float(numerator)
        #Token has already been advanced, so no need to advance it again here
    return token, i, temp




#Only called when IN or OUT tokens are found
def parseVariable(token, i , isInput, isOutput, inputVarNum, outputVarNum) :
    """
    Parse a variable token and return the corresponding node.

     Parameters
        ----------
        token: Token
            the current token being parsed
        i: int
            iteration counter to keep track of the current position in the token list
        isInput: bool
            flag indicating if the what is being parsed is expected to be an input variable
        isOutput: bool
            flag indicating if the what is being parsed is expected to be an output variable
        inputVarNum: int
            number of input variables declared so far, used to check if the variable being parsed is valid
        outputVarNum: int
            number of output variables declared so far, used to check if the variable being parsed is valid
        
    Returns
        -------
        Token
        int
        bool
        bool
        """
    if token.tag == Constants.IN:
        if isOutput:
            raise SyntaxError(f"Mixed input and output in assertion! Row {token.row}")

        if int(token.word) >= inputVarNum:
            raise SyntaxError(f"Input variable number {token.word} is not valid! Only {inputVarNum} input variables have been declared yet")

        isInput = True
        temp = Node.InputVariableNode(value=int(token.word))
    else :
        if isInput:
            raise SyntaxError(f"Mixed input and output in assertion! Row {token.row}")

        if int(token.word) >= outputVarNum:
            raise SyntaxError(f"Output variable number {token.word} is not valid! Only {outputVarNum} output variables have been declared yet")

        isOutput = True
        #Generate variable node (output)
        temp = Node.OutputVariableNode(value=int(token.word))
    return temp, i, isInput, isOutput




#This function parses an operand of an operation. Returns the node,the iteration counter, the flags and the current token (to be parsed)
def parseOperand(TokenItr, i, token, isInput, isOutput, inputVarNum, outputVarNum):
    """
    Parses an operand of an operation and returns the corresponding node, the updated iteration counter, the input/output flags, and the current token.

     Parameters
        ----------
        TokenItr: iterator
            itator over the tokens list
        i: int
            iteration counter to keep track of the current position in the token list
        token: Token
            the current token being parsed
        isInput: bool
            flag indicating if the what is being parsed is expected to be input related
        isOutput: bool
            flag indicating if the what is being parsed is expected to be output related
        inputVarNum: int
            number of input variables declared so far, used to check if the variable being parsed is valid
        outputVarNum: int
            number of output variables declared so far, used to check if the variable being parsed is valid
        
    Returns
        -------
        Node
        int
        bool
        bool
        Token
        """
    if token.tag == Constants.LP:
        #Recursive call to parse the operation inside the parentheses
        tempNode, i, isInput, isOutput = parseOperation(TokenItr, i, inputVarNum, outputVarNum, isInput, isOutput)
        #After this call, the iterator should be on the right parenthesis of the operation
        token, i = safe_next(TokenItr, i)


    elif token.tag == Constants.IN or token.tag == Constants.OUT:

        tempNode, i, isInput, isOutput = parseVariable(token, i ,isInput, isOutput, inputVarNum, outputVarNum)         
        token, i = safe_next(TokenItr, i)  # Get the next token after the variable


    elif token.tag == Constants.NUM:
        
        token, i, value = parseConstant(token, TokenItr, i)
        tempNode =Node.ConstantNode(float(value))
        #No need to iterate because parseConstant already returns with the successive token
        

    elif token.tag == Constants.SUB:
        #Expected a number (fraction form possible)
        token, i = safe_next(TokenItr, i)
        if token.tag == Constants.NUM:
            token, i, value = parseConstant(token, TokenItr, i)
        else:
            raise SyntaxError(f"Expected constant after subtraction token, found {Token.id2word[token.tag]}, row {token.row}")
        tempNode =Node.ConstantNode(float(-value))

    else:
        raise SyntaxError(f"Expected operand while parsing operation, found {Token.id2word[token.tag]}, row {token.row}")

    #The token should be the token after the operand
    return tempNode, i, isInput, isOutput, token









    #Function for * / + - operations. It must be able to manage the input or output flags. Returns with iterator on RP
    #Called with iterator on LP token, so it returns with iterator on RP token (of the operation)
def parseOperation(TokenItr, i: int, inputVarNum, outputVarNum, isInput : bool = False,  isOutput : bool = False, isAnd : bool = False) -> Node.Node:
    """
    Parse an operation and return the corresponding node, the updated iteration counter, and the input/output flags.

     Parameters
        ----------
        TokenItr: iterator
            itator over the tokens list
        i: int
            iteration counter to keep track of the current position in the token list
        inputVarNum: int
            number of input variables declared so far, used to check if the variable being parsed is valid
        outputVarNum: int
            number of output variables declared so far, used to check if the variable being parsed is valid
        isInput: bool
            flag indicating if the what is being parsed is expected to be input related
        isOutput: bool
            flag indicating if the what is being parsed is expected to be output related
        isAnd: bool
            flag indicating if the operation is an AND operation (used to check if the operation token is valid)
        
    Returns
        -------
        Node
        int
        bool
        bool
        """
    #Should return a node representing an operation, with two operands as children and the two flags indicating if the assertion is an input assertion or output assertion
    token, i = safe_next(TokenItr, i)
    
    if token.tag == Constants.RP:
        raise SyntaxError(f"Unexpected right parenthesis at row {token.row}, expected operation token or operand")
    #Expected an operation token (+ - * /), then two operands which can be variables or constants or other operations (wrapped in parentheses)

    if not isAnd:
       if token.tag not in (Constants.ADD, Constants.SUB, Constants.MUL, Constants.DIV):
           raise SyntaxError(f"Expected operation token in assertion, found {Token.id2word[token.tag]}, row {token.row}")

    #If it is and AND operation, it should be an >= or <= operation
    else: 
       if token.tag not in (Constants.GE, Constants.LE):
           raise SyntaxError(f"Expected >= or <= token as AND operand, found {Token.id2word[token.tag]}, row {token.row}")

    # Save the operation tag for later use and get the next token
    opTag = token.tag  
    token, i = safe_next(TokenItr, i) 

    # Initialize operand nodes
    leftNode = None  
    rightNode = None  

    #Expected operands (variables, constants or operations wrapped in parentheses)
    leftNode, i, isInput, isOutput, token = parseOperand(TokenItr, i, token, isInput, isOutput, inputVarNum, outputVarNum)

    rightNode, i, isInput, isOutput, token = parseOperand(TokenItr, i, token, isInput, isOutput, inputVarNum, outputVarNum)


    #After having parsed both operands, expect a right parenthesis
    if token.tag != Constants.RP:
        raise SyntaxError(f"Expected right parenthesis after operation, found {Token.id2word[token.tag]}, row {token.row}")
    

    #Now we can create the operation node with the two operands as children
    result = Node.OperationNode(opType = opTag)
    result.add_child(leftNode)
    result.add_child(rightNode)
    #Return the operation node, the iterator index and the input/output flags
    return result, i, isInput, isOutput








#This function parses an AND operation. An and operation can have 1 constraint inside or multiple.
def parseAnd(TokenItr, i:int, token, inputVarNum, outputVarNum):
    """
    Parse an and operation and return the corresponding node

     Parameters
        ----------
        TokenItr:
            iterator
        i: int
            iteration counter to keep track of the current position in the token list
        token: Token
            the current token being parsed
        inputVarNum: int
            number of input variables declared so far, used to check if the variables being parsed are valid
        outputVarNum: int
            number of output variables declared so far, used to check if the variables being parsed are valid
        
    Returns
        -------
        Token
        int
        """
    AndOperands = []  # List to store the AND operation nodes

    if token.tag != Constants.AND:
        raise SyntaxError(f"Expected at least an AND token in output assertion beginning with OR, found {Token.id2word[token.tag]}, row {token.row}")

    #I should find only output variables in the AND operations, so throw an error if an input variable is found
    isInput = False
    isOutput = True  # Output assertion, so this should be True
    
    #operationParse need the iterator to be on the LP token
    token, i = safe_next(TokenItr, i)  # Get the next token after the AND token
    #Cases are (or (and ())(and ())(and ()) ) or (or (and (...)(...)(...)) (and (...)(...)(...)) (and (...)(...)(...)) )

    while True :
        if token.tag != Constants.LP:
            raise SyntaxError(f"Expected left parenthesis after AND token in output assertion, found {Token.id2word[token.tag]}, row {token.row}")

        #Parse the operation inside the parentheses (isAnd is True because it is an AND operation)
        assertion, i, _, _ = parseOperation(TokenItr, i, inputVarNum, outputVarNum, isInput, isOutput, True)
        #Add the assertion node to the list of AND nodes
        AndOperands.append(assertion)
        #After operationParse, the iterator should be on the right parenthesis of the operation
        #print(f"Parsed assertion: {assertion}")  # Debugging output

        #Possible to find LP (statement with multiple constraints) or RP (statement with a single constraint) => another LP (for another AND) or another RP (end of OR)
        token, i = safe_next(TokenItr, i)  # Get the next token after the assertion
        if token.tag == Constants.LP:
            #LP means another and operand, so continue parsing
            continue
        elif token.tag == Constants.RP:
            #RP means the AND operation is finished, so break the loop
            break
        else:
            raise SyntaxError(f"Expected left or right parenthesis after AND operand, found {Token.id2word[token.tag]}, row {token.row}")

    #Create the AND node with the AND operands as children
    temp = Node.OperationNode(Constants.AND)  # Create an AND node
    for operand in AndOperands:

        temp.add_child(operand)

    #return the token iterator index and the AND node
    return temp, i






#Main function that ensures all tokens are read => recursive call to another parsing function?
def parse(tokens : list) -> list :
    """
    Parses a list of tokens and returns a list of nodes representing the syntax tree. Also returns the number of input and output variables declared.
    
    tokens:
        List of tokens to parse.

    Returns
        -------
        list
        int
        int
    """
    nodes = []

    #Declarations should be sequential, so the declaration should match the expected input/output variable number
    inputVarNum = 0  # Counter for input variables
    outputVarNum = 0  # Counter for output variables

    ParNumber = 0 # Counter for parentheses (don't know if it is needed)
    #See if the iterator is empty

    if not tokens:
        raise SyntaxError("Missing input")

    TokenItr = iter(tokens)

    #ParNumber += 1  # Increment the parentheses counter

    i = 0

    streamEnd = len(tokens)      

    OutputChecker = False  # Flag to check if an output assertion has been found


    # Iterate through the tokens until the end of the stream
    while i < streamEnd:
        #Anything starts with a left parenthesis, so check for it
        token, i = safe_next(TokenItr, i)
        
        #print(f"Current token: {token}, index: {i}, stream end: {streamEnd}, parentheses count: {ParNumber}")
        
        #token = tokens[i]
        #i +=1
        if token.tag != Constants.LP:  # Check for left curly brace
            raise SyntaxError(f"Missing left bracket at row {token.row}, found token ' {Token.id2word[token.tag]} '!")
        else:
            token, i = safe_next(TokenItr, i)  # Get the next token after the left parenthesis
            ParNumber += 1

        #print(f"Parsing token {token} at index {i}, parentheses count: {ParNumber}")
        
        #Declare-const token => expected input or output variable token and optionally real token => RP token
        if token.tag == Constants.DEC:  # Declaration
          
            i, inputVarNum, outputVarNum = parseDeclaration(TokenItr, i, token, inputVarNum, outputVarNum)

            

        #Assert token => expected left parenthesis, then an operation token, then two operands (which can be variables or constants or other operations wrapped in parentheses) and finally a right parenthesis
        elif token.tag == Constants.ASS:
            #First token expected is ( => error otherwise
            token, i = safe_next(TokenItr, i)
            if token.tag != Constants.LP:
                raise SyntaxError(f"Expected left parenthesis after assertion token, found {Token.id2word[token.tag]}, row {token.row}")
            #Expect at least an operation
            #First token expected is an operation token, followed by two operands which can be variables or constants or other operations (wrapped in parentheses)
            token, i = safe_next(TokenItr, i)

            #Expected GRE or LSE for generic assertions, and OR is possible for output assertions (two possible cases with OR token to manage later).
            opTag = token.tag

            #Check if the assertion is valid
            if opTag not in (Constants.LE, Constants.GE, Constants.OR):
                raise SyntaxError(f"Expected disequality or OR token at the beginning of assertion, found {Token.id2word[opTag]}, row {token.row}")



            """#Anche se sono teoricamente binari gli operatori and e or hanno un arbitrario numero di operandi >= 1. 
            #Ad esempio, (or (and A B) (and C D E) (and F)) significa (A & B) | (C & D & E) | (F) con A...F espressioni"""


            #Output assertion must have the OR token, but they could be only one AND, multiple ANDs or a single simple constraint
            if opTag == Constants.OR:
                #If the token is an OR token, it means that the assertion is an output assertion
                #Also there are two possible cases:
                # - Multiple statements with a single constraint (or (and...)(and...)...(and...))  => visitor will have to make an array of matrices
                # - Single statement with multiple constraints (or (and (...)(...)...(...)))  => visitor will have to make a single matrix
                OutputChecker = True  # Set the output assertion flag to True
                isOutput = True
                
                #AndNodes is a list of nodes representing the AND operations in the assertion. 
                AndNodes = []
                #Check if the next token is a left parenthesis
                token, i = safe_next(TokenItr, i)
                if token.tag != Constants.LP:
                    raise SyntaxError(f"Expected left parenthesis after OR token in assertion, found {Token.id2word[token.tag]}, row {token.row}")

                #Get the next token, it should be an AND token or a single constraint, if it is a single constraint parse it and close the OR operation
                row = token.row
                token = tokens[i]
                #print(f"Next token after OR: {token}, index: {i+1}, row: {row}")  # Debugging output)
                if not token:
                    raise SyntaxError(f"Expected AND token or a single constraint after OR token, found end of input, row {row}")
                
                #Expected AND token or a single constraint
                if token.tag != Constants.AND:
                    #If it is not an AND token, it should be a single constraint (get it and put it in the AND nodes list)
                        
                        temp, i, _ , _ = parseOperation(TokenItr, i, inputVarNum, outputVarNum, False, isOutput, isAnd=True)
                        AndNodes.append(temp)  # Add the single constraint node to the list of AND nodes
                        
                        #Next token should be the right parenthesis of the OR operation
                        token, i = safe_next(TokenItr, i)  
                        if token.tag != Constants.RP:
                            raise SyntaxError(f"Expected right parenthesis after single constraint in OR operation, found {Token.id2word[token.tag]}, row {token.row}")

                else:                        

                    while True:
                        token, i = safe_next(TokenItr, i)

                
                        #parseAnd will return an and node
                        temp, i= parseAnd(TokenItr, i, token, inputVarNum, outputVarNum)
                        AndNodes.append(temp)  # Add the AND node to the list of AND nodes

                        #This function return on the AND RP token, expect LP for another AND operation or RP for the end of the OR operation
                        token, i = safe_next(TokenItr, i)  # Get the next token after the AND operation
                        if (token.tag == Constants.LP):
                            #Expected another AND operation, so continue parsing
                            continue
                        elif (token.tag == Constants.RP):
                            #End of the OR operation, so break the loop
                            break
                    
                
                #Found the end of the OR operation, expect assert's right parenthesis
                token, i =safe_next(TokenItr, i)
                if token.tag != Constants.RP:
                    raise SyntaxError(f"Expected right parenthesis after OR operation, found {Token.id2word[token.tag]}, row {token.row}")
                #Create OR node with AND nodes as children
                orNode = Node.OperationNode(Constants.OR)  
                for andNode in AndNodes:
                    orNode.add_child(andNode)
                
                assertionNode = Node.AssertionNode(Node.OUTCOMPL)  # Create an output assertion node
                assertionNode.add_child(orNode)
                #Add the assertion node to the nodes list
                nodes.append(assertionNode)






            #If the assertion doesn't start with OR token, expect simple constraint with two operands (that may be variables or constants or operations)
            else:

                #To see if it is an input assertion or an output assertion
                isInput = False
                isOutput = False

                leftNode = None
                rightNode = None
                token, i = safe_next(TokenItr, i)                

                #Check if it is an operation or a constant or a variable
                """LEFT OPERAND"""


                leftNode, i, isInput, isOutput, token = parseOperand(TokenItr, i, token, isInput, isOutput, inputVarNum, outputVarNum)


                """RIGHT OPERAND"""
                
                rightNode, i, isInput, isOutput, token = parseOperand(TokenItr, i, token, isInput, isOutput, inputVarNum, outputVarNum,)



                #After having parsed both operands, there should be one of the two flags set to True
                if not isInput and not isOutput:
                   raise SyntaxError(f"Expected input or output variable as at least one of the operands in assertion, found constants, row {token.row}")
                if isInput and isOutput:
                   raise SyntaxError(f"Found mixed input and output variables in assertion! Row {token.row}")

                if isInput:
                    andType = Node.INPUT  # Input assertion
                else:
                    andType = Node.OUTSIMPLE  # Output assertion with a simple constraint


                #Check for right parenthesis (relative to the assert token) and generate the node
            
                if token.tag != Constants.RP:
                    raise SyntaxError(f"Expected right parenthesis after disequality was parsed, found {Token.id2word[token.tag]}, row {token.row}")
                ParNumber -= 1
                #Get the next token, it should be the right parenthesis of the assertion
                token, i = safe_next(TokenItr, i)
                if token.tag != Constants.RP:
                    raise SyntaxError(f"Expected right parenthesis after assertion, found {Token.id2word[token.tag]}, row {token.row}")
                ParNumber -= 1

                #Generate the assertion node with the <= or >= operation node as child (with the two operands as children)
                ChildNode = Node.OperationNode(opType=opTag)  # Create an operation node with the appropriate operation
                ChildNode.add_child(leftNode)
                ChildNode.add_child(rightNode)

                #Generate the assertion node with the right type
                assertionNode = Node.AssertionNode(andType)
                assertionNode.add_child(ChildNode)
                #Add the assertion node to the nodes list
                nodes.append(assertionNode)

    #If there is no output constraint, set the output variable number to 0 
    return nodes, inputVarNum, outputVarNum if OutputChecker else 0