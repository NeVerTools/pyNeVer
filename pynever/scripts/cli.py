import csv
import json
import logging
import os
import re
import sys
import time

import pynever.networks as nets
import pynever.strategies.conversion as conv
import pynever.strategies.verification as ver
from pynever import utilities
from pynever.tensors import Tensor
from pynever.utilities import execute_network

# Log to stdout
logger = logging.getLogger('pynever.strategies.verification')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def sslp_verify_single(safety_prop: bool, model_file: str, property_file: str, strategy: str, logfile: str) -> bool:
    """
    This method starts the verification procedure on the network model
    provided in the model_file path and prints the result with the SSLP algorithm

    Parameters
    ----------
    safety_prop : bool
        Specifies if the property is for safe or unsafe zone
    model_file : str
        Path to the .onnx file of the network
    property_file : str
        Path to the .vnnlib or .smt2 file of the property
    strategy : str
        Verification strategy (either complete, approximate, mixed)
    logfile : str
        Path to CSV file output

    Returns
    ----------
    bool
        True if the network is safe, False otherwise

    """

    nn_path = os.path.abspath(model_file)
    prop_path = os.path.abspath(property_file)

    if not os.path.isfile(nn_path):
        print(f'Error: file {nn_path} not found!')
        return False

    elif not os.path.isfile(prop_path):
        print(f'Error: file {prop_path} not found!')
        return False

    else:
        # Read the network file
        alt_repr = conv.load_network_path(nn_path)

        if alt_repr is not None:
            if isinstance(alt_repr, conv.ONNXNetwork):
                network = conv.ONNXConverter().to_neural_network(alt_repr)

                if isinstance(network, nets.SequentialNetwork):
                    # Read the property file
                    to_verify = ver.NeVerProperty()

                    if safety_prop:
                        neg_post_condition(prop_path)
                        to_verify.from_smt_file(os.path.abspath('pynever/scripts/intermediate.vnnlib'))
                        os.remove('pynever/scripts/intermediate.vnnlib')
                    else:
                        to_verify.from_smt_file(prop_path)

                    params = []
                    if strategy == 'overapprox':
                        params = [[0] for _ in range(network.count_relu_layers())]
                    elif strategy == 'mixed':
                        params = [[1] for _ in range(network.count_relu_layers())]
                    elif strategy == 'complete':
                        params = [[10000] for _ in range(network.count_relu_layers())]
                    ver_strategy = ver.NeverVerification(strategy, params)

                    model_name = os.path.basename(nn_path)
                    property_name = os.path.basename(property_file)

                    ver_start_time = time.perf_counter()
                    safe = ver_strategy.verify(network, to_verify)
                    fancy_cex = None
                    fancy_out = None

                    if not safe:
                        if strategy == 'complete':
                            answer = 'Falsified'
                            counter_stars = ver_strategy.counterexample_stars
                            if counter_stars is not None:
                                cexs = []

                                # Extract counterexamples (one per star is enough)
                                for cex_star in counter_stars:
                                    cexs.extend(cex_star.get_samples(num_samples=1))

                                if len(cexs) > 0:
                                    fancy_cex = reformat_counterexample(cexs[0])
                                    fancy_out = reformat_counterexample(execute_network(network, cexs[0]))
                        else:
                            answer = 'Unknown'
                    else:
                        answer = 'Verified'

                    ver_end_time = time.perf_counter() - ver_start_time
                    print(f'Benchmark: {model_name}, {property_name}')
                    print('----------------------------')
                    print(f'Result: {answer}')

                    if answer == 'Falsified':
                        print(f'Counterexample: {fancy_cex} -> {fancy_out}')

                    print(f'Time elapsed: {ver_end_time}\n\n')

                    with open(logfile, 'a+', newline='') as csv_out:
                        # Init writer with current file pointer
                        writer = csv.writer(csv_out)

                        # Set file pointer to the beginning to read the first line
                        csv_out.seek(0)
                        if csv_out.readline() == '':
                            writer.writerow(['Network', 'Property', 'Verification strategy', 'Verification time',
                                             'Answer', 'Counterexample', 'Unsafe output'])

                        # Write with the writer which still points at the end
                        writer.writerow([model_name, property_name, strategy, ver_end_time,
                                         answer, fancy_cex, fancy_out])
                    return True

            else:
                print('The model is not an ONNX model.')
                return False


def ssbp_verify_single(model_file: str, property_file: str, logfile: str, timeout: int, params_file: str) -> bool:
    """
    This method starts the verification procedure on the network model provided
    in the model_file path for the property specified in property_file, using
    the SSBP algorithm

    Parameters
    ----------
    model_file : str
        Path to the ONNX network file
    property_file : str
        Path to the VNNLIB property file
    logfile : str
        Path to the CSV output file
    timeout : int
        Execution timeout, optional
    params_file : str, optional
        Path to the JSON parameters file

    Returns
    ----------
    bool
        True if the network is safe w.r.t. the property, False otherwise

    """

    nn_path = os.path.abspath(model_file)
    prop_path = os.path.abspath(property_file)
    params_path = os.path.abspath(params_file)
    params = None

    if not os.path.isfile(nn_path):
        print(f'Error: file {nn_path} not found!')
        return False

    elif not os.path.isfile(prop_path):
        print(f'Error: file {prop_path} not found!')
        return False

    elif params_file != '' and not os.path.isfile(params_path):
        print(f'Error: file {params_path} not found!')
        return False

    else:
        # Read the network file
        alt_repr = conv.load_network_path(nn_path)

        if alt_repr is not None:
            if isinstance(alt_repr, conv.ONNXNetwork):
                network = conv.ONNXConverter().to_neural_network(alt_repr)

                if isinstance(network, nets.SequentialNetwork):
                    # Read the property file
                    to_verify = ver.NeVerProperty()
                    to_verify.from_smt_file(prop_path)

                    if os.path.isfile(params_path):
                        params = json.loads(params_path)

                    ver_strategy = ver.SearchVerification(params)
                    ver_strategy.search_params['timeout'] = timeout

                    start_time = time.perf_counter()
                    result = ver_strategy.verify(network, to_verify)
                    lap = time.perf_counter() - start_time

                    p_name = prop_path.split('/')[-1].split('\\')[-1]
                    net_name = network.identifier.split('/')[-1].split('\\')[-1]
                    instance_name = f'{net_name} - {p_name}'
                    dump_results(instance_name, network, result, lap, logfile)

                    return result[0]
            else:
                print('Not an ONNX model.')
                return False
        else:
            print('The model is not readable.')
            return False


def sslp_verify_batch(safety_prop: bool, csv_file: str, strategy: str, logfile: str) -> bool:
    """
    This method starts the verification procedure on the instances provided
    in the csv_file path and prints the result with the SSLP algorithm

    Parameters
    ----------
    safety_prop : bool
        Specifies if the property is for safe or unsafe zone
    csv_file : str
        Path to the .csv file of the instances
    strategy : str
        Verification strategy (either complete, approximate, mixed)
    logfile : str
        Path to CSV file output

    Returns
    ----------
    bool
        True if all the instances executed, False otherwise

    """

    csv_file_path = os.path.abspath(csv_file)
    folder = os.path.dirname(csv_file_path)
    exec_ok = True

    if not os.path.isfile(csv_file_path):
        print('Invalid path for the CSV file.')
        return False

    else:
        with open(csv_file_path, newline='') as f:
            try:
                reader = csv.reader(f)
            except OSError:
                print('Cannot open file ', csv_file)
                return False

            else:
                for row in reader:
                    if len(row) >= 2:
                        net_path = f'{folder}/{row[0]}'
                        prop_path = f'{folder}/{row[1]}'
                        sslp_verify_single(safety_prop, net_path, prop_path, strategy, logfile)
                    else:
                        print('Invalid row: ', row)
                        exec_ok = False
    return exec_ok


def ssbp_verify_batch(csv_file: str, logfile: str, timeout: int, params_file: str) -> bool:
    """
    This method starts the verification procedure on the batch of instances
    provided in the csv_file path, using the SSBP algorithm

    Parameters
    ----------
    csv_file : str
        Path to the instances file
    logfile : str
        Path to the CSV output file
    timeout : int
        Execution timeout, optional
    params_file : str, optional
        Path to the JSON parameters file

    Returns
    ----------
    bool
        True if all the instances executed, False otherwise

    """

    csv_file_path = os.path.abspath(csv_file)
    folder = os.path.dirname(csv_file_path)
    exec_ok = True

    if not os.path.isfile(csv_file_path):
        print('Invalid path for the CSV file.')
        return False

    else:
        with open(csv_file_path, newline='') as f:
            try:
                reader = csv.reader(f)
            except OSError:
                print('Cannot open file ', csv_file)
                return False

            else:
                for row in reader:
                    if len(row) >= 2:
                        net_path = f'{folder}/{row[0]}'
                        prop_path = f'{folder}/{row[1]}'
                        ssbp_verify_single(net_path, prop_path, logfile, timeout, params_file)
                    else:
                        print('Invalid row: ', row)
                        exec_ok = False
    return exec_ok


def dump_results(name, net, ans, t, out):
    """
    This method prints the result of the verification procedure to a CSV file and to a single instance file
    as per VNN-COMP directive

    """

    # CSV structure: name,time,result
    with open(out, 'a', encoding='utf-8') as csv_out, open(f'{name.replace(".vnnlib", "")}.txt', 'w') as inst_out:
        csv_out.write(f'{name},')

        # If answer is False with no counterexample -> timeout
        if len(ans) == 1:
            if ans[0]:
                csv_out.write(f'{t},Verified\n')
                inst_out.write('unsat')
            else:
                csv_out.write(f'-,Unknown\n')
                inst_out.write('unknown')
        else:
            unsafe_out = utilities.execute_network(net, ans[1])
            text = ''
            for i in range(len(ans[1])):
                text += f'(X_{i} {ans[1][i]})\n'

            for j in range(len(unsafe_out)):
                text += f'(Y_{j} {unsafe_out[j]})\n'
            text = text.replace('[', '').replace(']', '')[:-1]

            csv_out.write(f'{t},Unsafe\n')
            inst_out.write(f'sat\n({text})')


def reformat_counterexample(counterexample: Tensor) -> str:
    """
    This method writes the tensor data in a nice way for printing

    """

    formatted = '['
    for component in counterexample:
        formatted += str(float(component[0]))
        formatted += ' '
    formatted = formatted[:-1]
    formatted += ']'

    return formatted


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
        with open('pynever/scripts/intermediate.vnnlib', 'w', newline='') as new_prop:
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
