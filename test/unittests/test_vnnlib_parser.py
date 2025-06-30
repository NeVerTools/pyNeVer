"""
Main file that manages workflow and calls all functions
"""

#Main file, here the parser reads the .vnnlib file and launches its routines

 #Uso per leggere il file da riga di comando
import sys
#uso per assicurarmi che il file esista e sia leggibile
import pathlib
from pynever.strategies.parser import Tokenizer_OLD
from pynever.strategies.parser import Parser
from pynever.strategies.parser import Visitor

import time

# Uso per creare poi i tensori con le condizioni su input e output
#import torch

# Controllo se e' stato passato un argomento
if len(sys.argv) < 2:
    print("Errore: specificare il nome del file come argomento.")
    print(f"Uso: {sys.argv[0]} <nome_file>")
    sys.exit(1)
# Ottengo il nome del file dal primo argomento
nome_file = sys.argv[1]

# Verifica se il file esiste e se e' leggibile
tokens = None
file_path = pathlib.Path(nome_file)
if file_path.exists() and file_path.is_file():
    print(f"Il file '{nome_file}' esiste ed e' leggibile.")


    start = time.perf_counter()

    #Read the file and tokenize it (lexical analysis)
    tokens = Tokenizer.tokenize(nome_file)

    end_tokenize = time.perf_counter()

    print(f"Time of tokenization: ({end_tokenize - start}) seconds")

    #For tokenizer debugging
    #for token in tokens:
    #    print(token)


else:
    print(f"Errore: il file '{nome_file}' non esiste o non e' leggibile.")
    sys.exit(1)


#Parse the tokens (syntactic analysis)
nodes, inputVarNumber, outputVarNumber = Parser.parse(tokens)
end_parsing = time.perf_counter()

print(f"Time of parsing: ({end_parsing - end_tokenize}) seconds")

# For parser debugging
#for node in nodes:
 #   print(node)
  #  for child in node.children:
   #      print(f"  Child: {child}")
        # for grandchild in child.children:
        #     if (grandchild):
        #         print(f"    Grandchild: {grandchild}")

print(f"Input variables: {inputVarNumber}, Output variables: {outputVarNumber}")

# Visitor (Semantic analysis)

#Visit returns A and B of input and a list of output tensors in tuples (each A and B are in a single tuple)
InputA, InputB, OutputList = Visitor.visit(nodes, inputVarNumber, outputVarNumber)

end = time.perf_counter()

print(f"Input A: {InputA},\n Input B: {InputB}")
print(f"Output List: {OutputList}")

print(f"Time of execution : ({end - start}) seconds")

#OutputList can be None if no output conditions are found
#return InputA, InputB, OutputList

