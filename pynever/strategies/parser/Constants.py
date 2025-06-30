"""
Contains useful constants and a list to convert said constants to words.
"""

LP : int = 0 #Left Parenthesis token
RP : int = 1 #Right Parenthesis token
ADD : int = 2 #Addition token
SUB : int = 3 #Subtraction token
MUL : int = 4 #Multiplication token
DIV  : int = 5 #Division token
NUM  : int = 6 #Numerical token (0.5 has to be accepted as a number)
DEC : int = 7 #Declare-const token
REA : int = 8 #Real token
ASS : int = 9 #Assert token
OR : int = 10 #Or token
AND : int = 11 #And token
GRE : int = 12 #Greater equal token
#GRT : int = 13 #Greater than token Commented out because it is not used in the current implementation
LSE : int = 13 #Less equal token
#LST : int = 15 #Less than token
IN : int = 14 #Input variable token
OUT : int = 15 #Output variable token

id2word = [ "(", ")", "+", "-", "*", "/", "NUM", "DECLARE-CONST", "REAL", "ASSERT", "OR", "AND", ">=", "<=", "IN", "OUT"]