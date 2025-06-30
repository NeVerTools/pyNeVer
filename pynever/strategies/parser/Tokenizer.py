"""
Here the Input file is read line by line and converted into a vector of tokens. This vector will be passed to the parser. 
"""

import Constants
import Token
# This is the tokenizer class. It will read the input file and convert it into a vector of tokens.
# It will also handle lexical errors and throw exceptions if necessary.
#Until the file is not empty, it will read the file character by character and convert valid keywords into tokens. There will also be a row counter to help with error messages.
def tokenize(fileName : str) -> list:
    """
        Method that reads the file passed and makes a list of corresponding ``Tokens`` for the parser:

        Parameters
        ----------
        filename: str
            Name of the file to be read

        Returns
        -------
        list
            List of ``Tokens`` read from the file
        """

    #List to store tokens
    tokens = []
    with open(fileName, 'r') as file:
        row = 1
        #Used to check if the input variable (the first one has been read, so that we can declare the output variable)
        inputRead = False
        inputName = None
        outputRead = False
        outputName = None
        #Used to check if the alphanumeric string that has been read is a variable id or not
        serviceBool = False
        while True:
            # Read here reads one character at a time
            ch = file.read(1)
            if not ch:
                break  # Fine file
            if ch.isspace():
                if ch == '\n':
                    row += 1
                continue

            
            match ch:
                case '(':
                    tokens.append(Token.Token(Constants.LP, row))
                case ')':
                    tokens.append(Token.Token(Constants.RP, row))
                case '+':
                    tokens.append(Token.Token(Constants.ADD, row))
                case '-':
                    tokens.append(Token.Token(Constants.SUB, row))
                case '*':
                    tokens.append(Token.Token(Constants.MUL, row))
                case '/':
                    tokens.append(Token.Token(Constants.DIV, row))

                case _ if ch.isdigit():
                    num_str = ch
                    while True:
                        next_ch = file.read(1)
                        if next_ch == '.':
                            num_str += next_ch
                            next_ch = file.read(1)
                            if not next_ch.isdigit():
                                raise ValueError(f"Invalid number format at row {row}: {num_str}")
                            num_str += next_ch
                            while True:
                                next_ch = file.read(1)
                                if not next_ch or not next_ch.isdigit():
                                    file.seek(file.tell() - 1)
                                    break
                                num_str += next_ch
                            break
                        elif not next_ch or not next_ch.isdigit():
                            file.seek(file.tell() - 1)
                            break
                        num_str += next_ch
                    tokens.append(Token.Token(Constants.NUM, row, num_str))
                case '>':
                    next_ch = file.read(1)
                    if next_ch == '=':
                        tokens.append(Token.Token(Constants.GRE, row))
                    else:
                        tokens.append(Token.Token(Constants.GRE, row))
                        file.seek(file.tell() - 1)
                case '<':
                    next_ch = file.read(1)
                    if next_ch == '=':
                        tokens.append(Token.Token(Constants.LSE, row))
                    else:
                        tokens.append(Token.Token(Constants.LSE, row))
                        file.seek(file.tell() - 1)

                case _ if ch.isalnum():
                    # (resto del codice per identificatori/keyword)
                    identifier = ch

                    #Read until there is a space
                    while True:
                        next_ch = file.read(1)

                        # underscore is allowed in identifiers, and it is used to separate the variable name from the number
                        if next_ch == '_':
                            varName = identifier
                            identifier = None
                            next_ch = file.read(1)
                            
                            #Read all the digits untile there is a space
                            if not next_ch.isdigit():
                                raise ValueError(f"Invalid identifier format: {varName + next_ch} at row {row}")
                            identifier = next_ch

                            while True:
                                next_ch = file.read(1)
                                if not next_ch or not next_ch.isdigit():
                                    file.seek(file.tell() - 1)
                                    break
                                identifier += next_ch

                            #If the input has not been read yet, we set the input variable name. Same on cascade for the output
                            if not inputRead:
                                inputName = varName
                                inputRead = True
                                tokens.append(Token.Token(Constants.IN, row, identifier))
                            elif varName == inputName:
                                tokens.append(Token.Token(Constants.IN, row, identifier))
                            elif not outputRead:
                                outputName = varName
                                outputRead = True
                                tokens.append(Token.Token(Constants.OUT, row, identifier))
                            elif varName == outputName:
                                tokens.append(Token.Token(Constants.OUT, row, identifier))

                            else:
                                raise ValueError(f"Input ({inputName}) and Output ({outputName}) variable names  differ from: {varName} at row {row}")

                            #The string read is a variable id
                            serviceBool = True
                            break

                        # If the next character is not alphanumeric or a dash, we stop reading the identifier
                        elif not next_ch or not (next_ch.isalnum() or next_ch == '-'):
                            file.seek(file.tell() - 1)
                            break

                        #Otherwise, we append the next character to the identifier
                        identifier += next_ch
                        
                    #Once we have the identifier, we check if it is a valid keyword or not
                    match identifier :
                        case "or":
                            tokens.append(Token.Token(Constants.OR, row))
                        case "and":
                            tokens.append(Token.Token(Constants.AND, row))
                        case "declare-const":
                            tokens.append(Token.Token(Constants.DEC, row))
                        case "Real":
                            tokens.append(Token.Token(Constants.REA, row))
                        case "assert":
                            tokens.append(Token.Token(Constants.ASS, row))
                        case _ if serviceBool:
                            #The token was already built, just set the flag off
                            serviceBool = False
                        case _ :
                            raise ValueError(f"Unknown token '{identifier}' found at row {row}")
                
                case _:
                    raise ValueError(f"Unknown token literal '{ch}' found at row {row}")

                # Handle identifiers and keywords
                
            
    return tokens
