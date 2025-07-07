"""
This module handles the tokenization of the file. The ``Tokenizer`` class reads it line-by-line
and creates a list of ``Token`` objects
"""
from io import TextIOWrapper

from pynever.strategies.parser import PARSER_LOGGER
from pynever.strategies.parser.util import Operation


class Token:
    """
    A class to represent a token to parse.

    Attributes
    ----------
    tag: Operation
        The token type
    line: int
        The line of the file where the token is read
    value: str
        The value of the token, if it is a variable name or a number
    """

    def __init__(self, tag: Operation, line: int = 0, value: str = ''):
        self.tag = tag
        self.line = line
        self.value = value

    def __str__(self):
        return f"Token(tag: '{self.tag.value}', value: '{self.value}', line: '{self.line}')"

    def __repr__(self):
        return str(self)


class Tokenizer:
    """
    A class to tokenize a file for the VNN-LIB parser.
    It reads the input file and converts it into a list of ``Token`` objects. It also handles lexical errors.
    Until the file is not empty, it reads the file character by character and converts valid keywords
    into tokens.

    Attributes
    ----------
    filepath: str
        Path to the input file.
    cur_line: int
        Current line of the input file.
    input_name: str
        Name of the input variables.
    output_name: str
        Name of the output variables.
    input_read: bool
        Flag to tell when the input constraints have been read.
    output_read: bool
        Flag to tell when the output constraints have been read.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

        self.cur_line = 1
        self.input_name = ''
        self.output_name = ''

        self.input_read = False
        self.output_read = False

        self.logger = PARSER_LOGGER

    def tokenize(self) -> list[Token]:
        """
        Procedure to create the list of :class:`~pynever.strategies.parser.tokenizer.Token` objects for the Parser to process.

        Returns
        -------
        list[Token]
            List of :class:`~pynever.strategies.parser.tokenizer.Token` objects
        """
        tokens = []

        with open(self.filepath, 'r') as file:
            # Flag to trace the execution
            is_variable_id = False

            # Main loop
            while True:
                ch = file.read(1)
                if not ch:
                    break  # EOF

                # Update line counter
                if ch.isspace():
                    if ch == '\n':
                        self.cur_line += 1
                    continue

                # Update tokens
                match ch:
                    case '(':
                        tokens.append(Token(Operation.LP, self.cur_line))
                    case ')':
                        tokens.append(Token(Operation.RP, self.cur_line))
                    case '+':
                        tokens.append(Token(Operation.ADD, self.cur_line))
                    case '-':
                        tokens.append(Token(Operation.SUB, self.cur_line))
                    case '*':
                        tokens.append(Token(Operation.MUL, self.cur_line))
                    case '/':
                        tokens.append(Token(Operation.DIV, self.cur_line))
                    case '>':
                        next_ch = file.read(1)
                        if next_ch == '=':
                            tokens.append(Token(Operation.GE, self.cur_line))
                        else:
                            self.logger.warning(
                                f'Warning: encountered ">" at line {self.cur_line}, expected ">=" instead'
                            )
                            tokens.append(Token(Operation.GE, self.cur_line))
                            file.seek(file.tell() - 1)
                    case '<':
                        next_ch = file.read(1)
                        if next_ch == '=':
                            tokens.append(Token(Operation.LE, self.cur_line))
                        else:
                            self.logger.warning(
                                f'Warning: encountered "<" at line {self.cur_line}, expected "<=" instead'
                            )
                            tokens.append(Token(Operation.LE, self.cur_line))
                            file.seek(file.tell() - 1)
                    case _ if ch.isdigit():
                        tokens.append(self.read_num(ch, file))
                    case _:
                        new_tokens, is_variable_id = self.read_alpha(ch, file, is_variable_id)
                        tokens.extend(new_tokens)

        return tokens

    def read_alpha(self, ch: str, file: TextIOWrapper, is_variable_id: bool) -> tuple[list[Token], bool]:
        """
        Procedure that reads a stream from a file that may represent several tokens

        Parameters
        ----------
        ch: str
            The first character of the stream previously read
        file: TextIOWrapper
            The file to keep reading from
        is_variable_id: bool
            Flag to check the variable identifier

        Returns
        -------
        tuple[list[Token], bool]
            A list of new Tokens and the updated flag ``is_variable_id``
        """
        identifier = ch
        new_tokens = []

        # Read until there is a space
        while True:
            next_ch = file.read(1)

            # underscore is allowed in identifiers, and it is used to separate the variable name from the number
            if next_ch == '_':
                var_name = identifier
                next_ch = file.read(1)

                # Read all the digits untile there is a space
                if not next_ch.isdigit():
                    raise ValueError(f'Invalid identifier format: {var_name + next_ch} at line {self.cur_line}')
                identifier = next_ch

                while True:
                    next_ch = file.read(1)
                    if not next_ch or not next_ch.isdigit():
                        file.seek(file.tell() - 1)
                        break
                    identifier += next_ch

                # If the input has not been read yet, we set the input variable name. Same on cascade for the output
                if not self.input_read:
                    self.input_name = var_name
                    self.input_read = True
                    new_tokens.append(Token(Operation.IN, self.cur_line, identifier))

                elif var_name == self.input_name:
                    new_tokens.append(Token(Operation.IN, self.cur_line, identifier))

                elif not self.output_read:
                    self.output_name = var_name
                    self.output_read = True
                    new_tokens.append(Token(Operation.OUT, self.cur_line, identifier))

                elif var_name == self.output_name:
                    new_tokens.append(Token(Operation.OUT, self.cur_line, identifier))

                else:
                    raise ValueError(
                        f'Input {self.input_name} and Output {self.output_name} variable names'
                        f'differ from: {var_name} at line {self.cur_line}'
                    )

                # The string read is a variable id
                is_variable_id = True
                break

            # If the next character is not alphanumeric or is a hyphen, we stop reading the identifier
            elif not next_ch or not (next_ch.isalnum() or next_ch == '-'):
                file.seek(file.tell() - 1)
                break

            # Otherwise, we append the next character to the identifier
            identifier += next_ch

        # Once we have the identifier, we check if it is a valid keyword or not
        match identifier:
            case 'or':
                new_tokens.append(Token(Operation.OR, self.cur_line))
            case 'and':
                new_tokens.append(Token(Operation.AND, self.cur_line))
            case 'declare-const':
                new_tokens.append(Token(Operation.DEC, self.cur_line))
            case 'Real':
                new_tokens.append(Token(Operation.REA, self.cur_line))
            case 'assert':
                new_tokens.append(Token(Operation.ASS, self.cur_line))
            case _ if is_variable_id:
                # The token was already built, just set off the flag
                is_variable_id = False
            case _:
                raise ValueError(f"Unknown token '{identifier}' found at line {self.cur_line}")

        return new_tokens, is_variable_id

    def read_num(self, ch: str, file: TextIOWrapper) -> Token:
        """
        Procedure that reads a stream from a file that represents a number,
        and returns a Token containing that number

        Parameters
        ----------
        ch: str
            The first character of the number previously read
        file: TextIOWrapper
            The file to keep reading the number from

        Returns
        -------
        Token
            The number Token
        """
        while True:
            next_ch = file.read(1)

            if next_ch == '.':
                ch += next_ch
                next_ch = file.read(1)

                if not next_ch.isdigit():
                    raise ValueError(f'Invalid number format at line {self.cur_line}: {ch}')

                ch += next_ch

                while True:
                    next_ch = file.read(1)
                    if not next_ch or not next_ch.isdigit():
                        file.seek(file.tell() - 1)
                        break
                    ch += next_ch
                break

            elif not next_ch or not next_ch.isdigit():
                file.seek(file.tell() - 1)
                break

            ch += next_ch

        return Token(Operation.NUM, self.cur_line, ch)
