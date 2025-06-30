"""
File containing ``Token`` that is to be used during tokenizing and parsing.
"""

import Constants
id2word = Constants.id2word
#Oppure uso un dizionario cos� da poter usare le parole come chiavi e i numeri come valori? Per esempio:
# id2word = {
#     LP : '(' ...


# Token class to
class Token:
	"""
    This is the docstring of ``Token``. Defines a token with its type and value

    Attributes
    ----------
    tag: int
        Contains the token type, which is an integer representing the type of token (e.g., identifier, keyword, operator).
	word: str
		Contains the token value, which is a string representing the actual value of the token (e.g.,  constant value, variable number).
	row: int
		Contains the row number in the source code where the token was found, useful for error reporting.
    """

	def __init__(self, tag: int, row : int = 0, word: str = ""):
		self.tag = tag
		self.word = word
		self.row = row

	def __str__(self):
		return f"Token(tag= '{Constants.id2word[self.tag]}', word='{self.word}', row='{self.row}')"

	def __repr__(self):
		return self.__str__()