import csv
import json
import logging
import os
import re
import sys
import time

import pynever.networks as networks
import pynever.strategies.verification.algorithms as veralg
import pynever.strategies.verification.parameters as verparams
import pynever.strategies.verification.properties as verprop
import pynever.utilities as utilities
from pynever.strategies.conversion.converters.onnx import ONNXConverter
from pynever.strategies.conversion.representation import load_network_path, ONNXNetwork
from pynever.tensors import Tensor

# Log to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

POST_CONDITIONS_TEMP_FILE = 'pynever/scripts/intermediate.vnnlib'


def sslp_verify_single(safety_prop: bool, model_file: str, property_file: str, strategy: str, logfile: str | None) \
        -> None:
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
        raise Exception(f'Error: file {nn_path} not found!')

    if not os.path.isfile(prop_path):
        raise Exception(f'Error: file {prop_path} not found!')

        # Read the network file
    alt_repr = load_network_path(nn_path)

    if not isinstance(alt_repr, ONNXNetwork):
        raise Exception('The network is not an ONNX network!')

    network = ONNXConverter().to_neural_network(alt_repr)

    if not isinstance(network, networks.SequentialNetwork):
        raise Exception('The network is not a sequential network!')

    # Read the property file
    prop = verprop.VnnLibProperty(prop_path)

    if safety_prop:
        neg_post_condition(prop_path)
        prop = verprop.VnnLibProperty(os.path.abspath(POST_CONDITIONS_TEMP_FILE))
        os.remove(POST_CONDITIONS_TEMP_FILE)

    params = verparams.SSLPVerificationParameters(strategy,
                                                  [1 for _ in range(network.count_relu_layers())]
                                                  if strategy == 'mixed' else None)

    ver_strategy = veralg.SSLPVerification(params)

    model_name = os.path.basename(nn_path)
    property_name = os.path.basename(property_file)

    ver_start_time = time.perf_counter()
    safe, counterexample = ver_strategy.verify(network, prop)
    ver_end_time = time.perf_counter() - ver_start_time

    fancy_cex = None
    fancy_out = None

    answer = 'Unknown'
    if safe:
        answer = 'Verified'
    else:
        if counterexample is not None:
            answer = 'Falsified'
            fancy_cex = reformat_counterexample(counterexample)
            fancy_out = reformat_counterexample(utilities.execute_network(network, counterexample))

    print(f'Benchmark: {model_name}, {property_name}')
    print('----------------------------')
    print(f'Result: {answer}')

    if counterexample is not None:
        print(f'Counterexample: {fancy_cex} -> {fancy_out}')

    print(f'Time elapsed: {ver_end_time}\n\n')

    if logfile is not None:
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


def ssbp_verify_single(model_file: str, property_file: str, out_dir: str, logfile: str | None,
                       timeout: int, params_file: str) -> None:
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

    if not os.path.isfile(nn_path):
        raise Exception(f'Error: file {nn_path} not found!')

    if not os.path.isfile(prop_path):
        raise Exception(f'Error: file {prop_path} not found!')

    if params_file != '' and not os.path.isfile(params_path):
        raise Exception(f'Error: file {params_path} not found!')

    # Read the network file
    alt_repr = load_network_path(nn_path)

    if not isinstance(alt_repr, ONNXNetwork):
        raise Exception('The network is not an ONNX network!')

    network = ONNXConverter().to_neural_network(alt_repr)

    if not isinstance(network, networks.SequentialNetwork):
        raise Exception('The network is not a sequential network!')

    # Read the property file
    prop = verprop.VnnLibProperty(prop_path)
    ver_params = verparams.SSBPVerificationParameters(timeout=timeout)

    if os.path.isfile(params_path):
        params = json.loads(params_path)
        if not ({'heuristic', 'bounds', 'intersection'} <= set(params.keys())):
            raise Exception(f'Error: parameters file {params_path} does not contain valid parameters')
        ver_params = verparams.SSBPVerificationParameters(params['heuristic'],
                                                          params['bounds'],
                                                          params['intersection'],
                                                          timeout)

    ver_strategy = veralg.SSBPVerification(ver_params)

    start_time = time.perf_counter()
    result = ver_strategy.verify(network, prop)
    lap = time.perf_counter() - start_time

    p_name = prop_path.split('/')[-1].split('\\')[-1]
    net_name = network.identifier.split('/')[-1].split('\\')[-1]
    instance_name = f'{net_name} - {p_name}'
    dump_results(instance_name, network, result, lap, logfile, out_dir)


def sslp_verify_batch(safety_prop: bool, csv_file: str, strategy: str, logfile: str | None) -> bool:
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
        raise Exception('Invalid path for the CSV file.')

    with open(csv_file_path, newline='') as f:
        try:
            reader = csv.reader(f)
        except OSError:
            raise Exception('Cannot open file ', csv_file)

        for row in reader:
            try:
                net_path = f'{folder}/{row[0]}'
                prop_path = f'{folder}/{row[1]}'
                sslp_verify_single(safety_prop, net_path, prop_path, strategy, logfile)
            except Exception as e:
                exec_ok = False
                print(e)

    return exec_ok


def ssbp_verify_batch(csv_file: str, out_dir: str, logfile: str | None, timeout: int, params_file: str) -> bool:
    """
    This method starts the verification procedure on the batch of instances
    provided in the csv_file path, using the SSBP algorithm

    Parameters
    ----------
    csv_file : str
        Path to the instances file
    out_dir : str
        Output directory for the experiments
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
        raise Exception('Invalid path for the CSV file.')

    with open(csv_file_path, newline='') as f:
        try:
            reader = csv.reader(f)
        except OSError:
            raise Exception('Cannot open file ', csv_file)

        for row in reader:
            try:
                net_path = f'{folder}/{row[0]}'
                prop_path = f'{folder}/{row[1]}'
                ssbp_verify_single(net_path, prop_path, out_dir, logfile, timeout, params_file)
            except Exception as e:
                exec_ok = False
                print(e)

    return exec_ok


def dump_results(name, net, ans, t, out, out_dir):
    """
    This method prints the result of the verification procedure to a CSV file and to a single instance file
    as per VNN-COMP directive

    """

    if 'txt' in out:
        write_instance(net, ans, out)

    else:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # CSV structure: name,time,result
        with open(out, 'a', encoding='utf-8') as csv_out:
            csv_out.write(f'{name},')

            # If answer is False with no counterexample -> timeout
            if ans[1] is None:
                if ans[0]:
                    csv_out.write('{:9.4f},Verified\n'.format(t))
                else:
                    csv_out.write(f'-,Timeout\n')
            else:
                csv_out.write('{:9.4f},Unsafe\n'.format(t))


def write_instance(network, answer, out_file):
    with open(out_file, 'w', encoding='utf-8') as inst_out:
        if answer[1] is None:
            if answer[0]:
                inst_out.write('unsat')
            else:
                inst_out.write('timeout')
        else:
            unsafe_out = utilities.execute_network(network, answer[1])
            text = ''
            for i in range(len(answer[1])):
                text += f'(X_{i} {answer[1][i]})\n'

            for j in range(len(unsafe_out)):
                text += f'(Y_{j} {unsafe_out[j]})\n'
            text = text.replace('[', '').replace(']', '')[:-1]

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
