import csv
import logging
import time

from pynever.networks import SequentialNetwork
from pynever.strategies import verification, conversion

BENCHMARKS_PATH = '../../examples/benchmarks'

logger_stream = logging.getLogger("pynever.strategies.verification")
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.INFO)

logger_file = logging.getLogger("log_file")
logger_file.addHandler(logging.FileHandler('logs/experiments.csv'))
logger_file.setLevel(logging.INFO)


def format_csv(answer: list) -> str:
    # If answer is False without counterexample -> timeout
    if len(answer) == 1:
        return 'Verified' if answer[0] else 'Unknown'

    else:
        fancy_cex = '['
        for component in answer[1]:
            fancy_cex += str(float(component[0]))
            fancy_cex += ' '
        fancy_cex = fancy_cex[:-1]
        fancy_cex += ']'

        return f'Falsified,{fancy_cex}'


def launch_instances(instances_file: str):
    logger_file.info('Instance,Time,Result,Counterexample')
    folder = instances_file.replace(instances_file.split('/')[-1], '')
    with open(instances_file, 'r') as f:
        csv_reader = csv.reader(f)

        for instance in csv_reader:

            network_path = f'{folder}/{instance[0]}'
            ver_strategy = verification.SearchVerification()
            onnx_net = conversion.load_network_path(network_path)
            if isinstance(onnx_net, conversion.ONNXNetwork):
                net = conversion.ONNXConverter().to_neural_network(onnx_net)

                if isinstance(net, SequentialNetwork):
                    prop = verification.NeVerProperty()
                    prop.from_smt_file(f'{folder}/{instance[1]}')

                    inst_name = f'{instance[0].split("/")[-1]} - {instance[1].split("/")[-1]}'

                    logger_stream.info(f'Instance: {inst_name}')

                    start_time = time.perf_counter()
                    result = ver_strategy.verify(net, prop)
                    lap = time.perf_counter() - start_time

                    logger_stream.info(f'{result} - {lap}')
                    logger_file.info(f'{inst_name},{lap},{format_csv(result)}')


if __name__ == '__main__':
    launch_instances(f'{BENCHMARKS_PATH}/ACAS XU/instances.csv')
