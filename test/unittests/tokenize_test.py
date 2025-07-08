from pynever.strategies.parser.tokenizer import Tokenizer
from pynever.strategies.parser.vnnlib import VnnlibParser

tokens = Tokenizer('2d_prop.vnnlib').tokenize()
nodes = VnnlibParser().parse_tokens(tokens)
