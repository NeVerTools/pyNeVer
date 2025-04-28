"""
Module never2_launcher.py

Provides an entry point for the execution of NeVer2 on a single instance

Authors: Stefano Demarchi, Pedro Achete

"""

import os
import re
import time
import warnings
from argparse import ArgumentParser

import torch

from pynever.strategies.abstraction.bounds_propagation.bounds import HyperRectangleBounds
from pynever.strategies.abstraction.bounds_propagation.manager import BoundsManager
from pynever.strategies.abstraction.networks import networks
from pynever.strategies.conversion.converters.onnx import ONNXConverter
from pynever.strategies.conversion.converters.pytorch import PyTorchConverter
from pynever.strategies.conversion.representation import load_network_path, ONNXNetwork
from pynever.strategies.verification.properties import VnnLibProperty

warnings.simplefilter("error", RuntimeWarning)

# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

POST_CONDITIONS_TEMP_FILE = r'C:\Users\andr3\PycharmProjects\pyNeVer\pynever\strategies\bounds_propagation\test\test\intermediate.vnnlib'


def add_options(p: ArgumentParser):
    """
    Common options for the execution of NeVer2

    """

    # Options
    p.add_argument('-o', '--out', type=str,
                   default='output.csv', help='output file for execution log')
    p.add_argument('-t', '--timeout', type=int, default=300,
                   help='execution timeout in seconds')

    # Algorithm
    algorithm = p.add_subparsers(dest='algorithm', description='Verification algorithm to use')

    # SSBP
    ssbp = algorithm.add_parser('ssbp', description='Starset with bounds propagation')
    ssbp.add_argument('-p', '--params', nargs='?', default='', metavar='FILE',
                      help='JSON file with parameters')

    # SSLP
    sslp = algorithm.add_parser('sslp', description='Starset with linear programs')
    sslp.add_argument('-s', '--strategy', choices=['overapprox', 'mixed', 'complete'], metavar='STRATEGY',
                      default='complete', help='Verification strategy to use, complete by default')

    return p


def neg_post_condition(prop_path: str) -> None:
    """
    This method negates the property post-condition in order
    to represent both safety and unsafety properties

    Parameters
    ----------
    prop_path : str
        Path to the property file

    """

    def replace_with_negatives(match):
        number = match.group()
        if number not in ("0", "0.0") and match.string[match.start() - 1] != '_':
            number = float(number)
            negative_number = -number
            return str(negative_number)
        return number

    with open(prop_path, 'r', newline='') as cur_prop:
        with open(POST_CONDITIONS_TEMP_FILE, 'w', newline='') as new_prop:
            # List of post-condition constraints
            y_constraints = []

            # Read file
            for row in cur_prop:
                # Filter declarations
                if not (row.find('Y') and row.find('assert')):
                    new_prop.write(row)

                else:
                    if row.find('<') > 0 and row.find('Y') > 0:
                        if row.find('(* -1.0') > 0:
                            temp_row = row.replace('(assert (<= (* -1.0', '(<=')
                            temp_row = temp_row[:temp_row.find(')')] + temp_row[temp_row.find(')') + 1:]
                            pattern = r'(?<!_)-?\d+\.\d+|(?<!_)-?\d+'
                            temp_row = re.sub(pattern, replace_with_negatives, temp_row)
                        else:
                            temp_row = row.replace('(assert (<=', '(>=')

                    elif row.find('>') > 0 and row.find('Y') > 0:
                        if row.find('(* -1.0') > 0:
                            temp_row = row.replace('(assert (>= (* -1.0', '(>=')
                            temp_row = temp_row[:temp_row.find(')')] + temp_row[temp_row.find(')') + 1:]
                            pattern = r'(?<!_)-?\d+\.\d+|(?<!_)-?\d+'
                            temp_row = re.sub(pattern, replace_with_negatives, temp_row)
                        else:
                            temp_row = row.replace('(assert (>=', '(<=')

                    else:
                        new_prop.write(row)
                        continue

                    temp_row = temp_row[:temp_row.rfind(')')] + temp_row[temp_row.rfind(')') + 1:]
                    y_constraints.extend(temp_row)

            new_prop.write('(assert (or \n')

            for row in y_constraints:
                new_prop.write(row)
            new_prop.write('\n))')


if __name__ == '__main__':
    nn_path = r"C:\Users\andr3\PycharmProjects\pyNeVer\pynever\strategies\bounds_propagation\test\mnist_fcnn_double_conv.onnx"
    prop_path = r"C:\Users\andr3\PycharmProjects\pyNeVer\pynever\strategies\bounds_propagation\test\test\loc_rob_property_0.vnnlib"
    safety_prop = False

    if not os.path.isfile(nn_path):
        raise Exception(f'Error: file {nn_path} not found!')

    if not os.path.isfile(prop_path):
        raise Exception(f'Error: file {prop_path} not found!')

    alt_repr = load_network_path(nn_path)

    if not isinstance(alt_repr, ONNXNetwork):
        raise Exception('The network is not an ONNX network!')

    network = ONNXConverter().to_neural_network(alt_repr)

    if not isinstance(network, networks.SequentialNetwork):
        raise Exception('The network is not a sequential network!')

    prop = VnnLibProperty(prop_path)

    if safety_prop:
        neg_post_condition(prop_path)
        prop = VnnLibProperty(os.path.abspath(POST_CONDITIONS_TEMP_FILE))
        os.remove(POST_CONDITIONS_TEMP_FILE)

    lower = torch.rand(784)
    upper = lower + 10e-8
    input = HyperRectangleBounds(lower, upper)

    results_dict = BoundsManager(network, input_bounds=input)
    start_time = time.time()
    bounds_dict = results_dict.compute_bounds()
    num_bounds = bounds_dict.numeric_post_bounds[network.get_last_node().identifier]
    end_time = time.time()
    execution_time = end_time - start_time

    py_net = PyTorchConverter().from_neural_network(network)
    py_net.pytorch_network.eval()
    py_net.pytorch_network.float()
    output_nn = py_net.pytorch_network(lower.view(1, 1, 28, 28))

    lower_output = bounds_dict.numeric_post_bounds[bounds_dict.identifiers[-1]].lower
    upper_output = bounds_dict.numeric_post_bounds[bounds_dict.identifiers[-1]].upper

    assert ((output_nn >= lower_output) & (output_nn <= upper_output)).all(), "Bounds errati"

    # Check bounds: all lower bounds must be smaller than the upper bounds
    for index, id in enumerate(bounds_dict.identifiers):
        print(f'Bounds #{index}: {id}')
        lower_num_bounds = bounds_dict.numeric_post_bounds[id].lower
        upper_num_bounds = bounds_dict.numeric_post_bounds[id].upper
        print(f'Dims {lower_num_bounds.shape}')
        assert (lower_num_bounds <= upper_num_bounds).all(), "Lower bounds greater than upper bounds"

    print(execution_time)

    exit(0)
