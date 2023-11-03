import csv
import os
import re

import time

from pynever import tensor
from pynever.networks import SequentialNetwork
from pynever.strategies import conversion
from pynever.strategies.conversion import ONNXNetwork, ONNXConverter
from pynever.strategies.verification import NeVerProperty, NeverVerification


def show_help():
    print("usage: python never2.py [--verify] [-s | -u] [model] [property] [strategy] | "
          "[--batch] [-s | -u] [CSV file] [strategy] ")
    print()
    print("Options and arguments:")
    print("--verify args ... : verify the VNN-LIB property in args[2],\n "
          "                    which specifies the safe or unsafe zone args[1],\n"
          "                    on the ONNX model in args[3] with the strategy in args[4]")
    print()
    print("--batch args ...  : verify the VNN-LIB property for all the ONNX models\n"
          "                    specified in the CSV file in args[1] with the strategy in args[3]")
    print()
    print("[strategy]        : one between 'complete', 'approx', 'mixed' ")
    print("[-s | -u]         : -s for safe and -u for the unsafe zone ")
    print("args ...          : arguments passed to program in sys.argv[1:]")
    print()


def verify_single_model(property_type: str, model_file: str, property_file: str, strategy: str, writer_file):
    """
    This method starts the verification procedure on the network model
    provided in the model_file path and prints the result

    Parameters
    ----------
    property_file : str
        Path to the .vnnlib or .smt2 file of the property
    model_file : str
        Path to the .onnx file of the network
    property_type : str
        Specifies if the property is for safe or unsafe zone
    strategy : str
        Verification strategy (either complete, approximate, mixed)
    writer_file
        Output File

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
                    if property_type == '-s':
                        invert_conditions(prop_path)
                        to_verify.from_smt_file(os.path.abspath('pynever/scripts/intermediate.vnnlib'))
                    elif property_type == '-u':
                        to_verify.from_smt_file(prop_path)
                    else:
                        show_help()
                        return False
                    ver_strategy = None
                    if strategy == 'complete':
                        ver_strategy = NeverVerification('complete',
                                                         [[10000] for _ in range(network.count_relu_layers())])
                    elif strategy == 'approx':
                        ver_strategy = NeverVerification('best_n_neurons',
                                                         [[0] for _ in range(network.count_relu_layers())])
                    elif strategy == 'mixed':
                        ver_strategy = NeverVerification('best_n_neurons',
                                                         [[1] for _ in range(network.count_relu_layers())])

                    model_name = os.path.basename(nn_path)
                    property_name = os.path.basename(property_file)
                    ver_start_time = time.perf_counter()
                    unsafe = ver_strategy.verify(network, to_verify)
                    printable_counterexample = None
                    output = None
                    if unsafe:
                        if strategy == 'complete':
                            answer = 'Falsified'
                            counter_stars = ver_strategy.counterexample_stars
                            if counter_stars is not None:
                                some_counterexamples = []
                                for cex_star in counter_stars:
                                    try:
                                        some_counterexamples.extend(cex_star.get_samples(num_samples=1))
                                    except:
                                        print("Error finding the counterexample")
                                if len(some_counterexamples) > 0:
                                    tensor_counterexample = some_counterexamples[0]
                                    output = network.execute(tensor_counterexample)
                                    printable_counterexample = reformat_counterexample(tensor_counterexample)
                                    output = reformat_counterexample(output)
                        else:
                            answer = 'Unknown'
                    else:
                        answer = 'Verified'

                    ver_end_time = time.perf_counter() - ver_start_time
                    print("Benchmark ", model_name, ", ", property_name, "\n",
                          "Answer: ", answer, "\n",
                          "Time elapsed: ", ver_end_time)
                    if answer == 'Falsified':
                        print("Counterexample input: ", printable_counterexample, "\n",
                              "Counterexample output: ", output, "\n")

                    writer = csv.writer(writer_file)
                    writer.writerow(
                        [model_name, property_name, strategy, answer, ver_end_time, printable_counterexample, output])
                    return True
            else:
                print('The model is not an ONNX model.')
                return False


def verify_CSV_model(property_type: str, csv_file: str, strategy: str):
    csv_file_path = os.path.abspath(csv_file)
    print(csv_file_path)
    folder = os.path.dirname(csv_file_path)
    writer_file = open(os.path.abspath('output.csv'), 'w', newline='')
    writer_file.close()
    response = True
    if not os.path.isfile(csv_file_path):
        print('Invalid path for the CSV file.')
        return False
    else:
        try:
            csv_file_iti = csv.reader(open(csv_file_path, newline=''))
        except Exception:
            print("Cannot open the file: ", csv_file, "\n")
            return False
        else:
            for row in csv_file_iti:
                if len(row) >= 2:
                    writer_file = open(os.path.abspath('output.csv'), 'a', newline='')
                    net_path = folder + chr(47) + row[0] + ".onnx"
                    prop_path = folder + chr(47) + row[1]
                    response = response and verify_single_model(property_type, net_path, prop_path, strategy,
                                                                writer_file)
                else:
                    print("This row is not valid: ", row, "\n")
                    return False
    return response


def reformat_counterexample(counterexample: tensor):
    response = "["
    for component in counterexample:
        response += str(float(component))
        response += " "
    response = response[:-1]
    response += "]"
    return response


def invert_conditions(prop_path):
    writer = open('pynever/scripts/intermediate.vnnlib', 'w', newline='')
    reader = open(prop_path, 'r', newline='')
    y_constraints = []
    for row in reader:
        if row[0:7] != '(assert':
            writer.write(row)
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
                writer.write(row)
                continue

            temp_row = temp_row[:temp_row.rfind(')')] + temp_row[temp_row.rfind(')') + 1:]
            y_constraints.extend(temp_row)
    writer.write('(assert (or \n')
    for row in y_constraints:
        writer.write(row)
    writer.write('\n))')


def replace_with_negatives(match):
    number = match.group()
    if number not in ("0", "0.0") and match.string[match.start() - 1] != '_':
        number = float(number)
        negative_number = -number
        return str(negative_number)
    return number
