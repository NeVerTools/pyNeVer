import logging
import os
import time

from pynever.networks import SequentialNetwork
from pynever.strategies import verification, conversion

NETS_PATH = '../../examples/submissions/2023_AppliedIntelligence/vnncomp_benchmarks/clean_onnx'
PROPS_PATH = '../../examples/submissions/2023_AppliedIntelligence/vnncomp_benchmarks/vnnlib_specs'

logger_stream = logging.getLogger("pynever.strategies.verification")
logger_stream.addHandler(logging.StreamHandler())
logger_stream.setLevel(logging.INFO)

logger_file = logging.getLogger("log_file")
logger_file.addHandler(logging.FileHandler('logs/experiments.csv'))
logger_file.setLevel(logging.INFO)


def format_csv(answer: list) -> str:
    if len(answer) == 1:
        return answer[0]

    else:
        fancy_cex = '['
        for component in answer[1]:
            fancy_cex += str(float(component))
            fancy_cex += ' '
        fancy_cex = fancy_cex[:-1]
        fancy_cex += ']'

        return f'{answer[0]},{fancy_cex}'


if __name__ == '__main__':
    networks = [f'{NETS_PATH}/{nn_name}' for nn_name in os.listdir(NETS_PATH)]

    # This is amazing
    benchmarks = {
        # Key = name of the network
        nn.split('/')[-1].replace('.onnx', ''):

        # Value = list of properties
            [f'{PROPS_PATH}/{p_name}' for p_name in os.listdir(PROPS_PATH)
             if p_name.split('_')[0] in nn]
        for nn in networks
    }

    logger_file.info('Instance,Time,Result,Counterexample')

    for network_name, properties in benchmarks.items():

        network_path = f'{NETS_PATH}/{network_name}.onnx'

        for prop_path in properties:
            ver_strategy = verification.SearchVerification()
            onnx_net = conversion.load_network_path(network_path)
            if isinstance(onnx_net, conversion.ONNXNetwork):
                net = conversion.ONNXConverter().to_neural_network(onnx_net)

                if isinstance(net, SequentialNetwork):
                    prop = verification.NeVerProperty()
                    prop.from_smt_file(prop_path)

                    logger_stream.info(f'Instance: {network_name} - {prop_path.split("/")[-1]}')

                    start_time = time.perf_counter()
                    result = ver_strategy.verify(net, prop)
                    lap = time.perf_counter() - start_time

                    logger_stream.info(f'{result} - {lap}')
                    logger_file.info(f'{network_name} - {prop_path.split("/")[-1]},{lap},{format_csv(result)}')
