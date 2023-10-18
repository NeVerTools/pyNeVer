import os
import sys

import csv

from pynever.networks import SequentialNetwork
from pynever.strategies import conversion
from pynever.strategies.conversion import ONNXNetwork, ONNXConverter
from pynever.strategies.smt_reading import SmtPropertyParser
from pynever.strategies.verification import NeVerProperty, NeverVerification


def show_help():
    print("usage: python never2.py [--verify] [model] [property] [strategy], [--batch] [CSV file] [strategy] ")
    print()
    print("Options and arguments:")
    print("--verify args ... : verify the VNN-LIB property in args[1] on the\n"
          "                    ONNX model in args[2] with the strategy in args[3]")
    print()
    print("--batch args ...  : verify the VNN-LIB property for all the ONNX models\n"
          "                    specified in the CSV file in args[1] with the strategy in args[3]")
    print()
    print("[strategy]        : one between 'complete', 'approx', 'mixed' ")
    print("args ...          : arguments passed to program in sys.argv[1:]")
    print()


def verify_single_model(model_file: str, property_file: str, strategy: str):
    """
    This method starts the verification procedure on the network model
    provided in the model_file path and prints the result

    Parameters
    ----------
    property_file : str
        Path to the .vnnlib or .smt2 file of the property
    model_file : str
        Path to the .onnx file of the network
    strategy : str
        Verification strategy (either complete, approximate, mixed)
    """
    nn_path = os.path.abspath(model_file)
    prop_path = os.path.abspath(property_file)
    if not os.path.isfile(nn_path):
        print('Invalid path for the network model.')
        return False
    elif not os.path.isfile(prop_path):
        print('Invalid path for the property.')
        return False
    else:
        # Read the network file
        alt_repr = conversion.load_network_path(nn_path)

        if alt_repr is not None:
            if isinstance(alt_repr, ONNXNetwork):
                network = ONNXConverter().to_neural_network(alt_repr)

                if isinstance(network, SequentialNetwork):
                    # Read the property file
                    # parser = SmtPropertyParser(prop_path, network.input_id,
                    #                            network.get_last_node().identifier)
                    # to_verify = NeVerProperty(*parser.parse_property())
                    to_verify = NeVerProperty()
                    to_verify.from_smt_file(prop_path)

                    ver_strategy = None
                    if strategy == 'complete':
                        ver_strategy = NeverVerification('best_n_neurons',
                                                         [[10000] for _ in range(network.count_relu_layers())])
                    elif strategy == 'approx':
                        ver_strategy = NeverVerification('best_n_neurons',
                                                         [[0] for _ in range(network.count_relu_layers())])
                    elif strategy == 'mixed':
                        ver_strategy = NeverVerification('best_n_neurons',
                                                         [[1] for _ in range(network.count_relu_layers())])

                    return ver_strategy.verify(network, to_verify)
            else:
                print('The model is not an ONNX model.')
                return False


def verify_CSV_model(csv_file: str, strategy: str):
    csv_file_path = os.path.abspath(csv_file)
    if not os.path.isfile(csv_file_path):
        print('Invalid path for the CSV file.')
        return False
    else:
        try:
            csv_file_iti = csv.reader(open(csv_file_path, newline=''))
        except Exception as e:
            print("Cannot open the file: ", csv_file, "\n")
            return False
        else:
            for row in csv_file_iti:
                if len(row) >= 2:
                    verify_single_model(row[0], row[1], strategy)
                else:
                    print("This row is not valid: ", row, "\n")
