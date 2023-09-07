# Script for launching the experiments of 'Soft Computing' journal paper
import csv
import logging
import time

import pynever.strategies.conversion as conv
from pynever.strategies.conversion import ONNXConverter, ONNXNetwork
from pynever.strategies.verification import NeVerProperty, NeverVerification

pynever_setting = [['Over-approx.', 'overapprox', [0]],
                   ['Mixed1', 'mixed', [1]],
                   ['Complete', 'complete', [10000]]]

logger_stream = logging.getLogger("pynever.strategies.verification")
logger_file = logging.getLogger("log_file")

logger_stream.addHandler(logging.StreamHandler())
logger_file.addHandler(logging.FileHandler('logs/experiments.csv'))

logger_stream.setLevel(logging.INFO)
logger_file.setLevel(logging.INFO)

if __name__ == '__main__':

    logger_file.info('Benchmark,Over-approx.,,Mixed1,,Complete,,')
    logger_file.info(',Result,Time,Result,Time,Result,Time')

    # ACAS XU launcher
    with open('data/ACAS XU/instances.csv') as instances:
        folder = 'data/ACAS XU'
        csv_reader = csv.reader(instances)

        for instance in csv_reader:
            network_instance = conv.load_network_path(f"{folder}/Networks/{instance[0]}")
            if isinstance(network_instance, ONNXNetwork):
                onnx_net = ONNXConverter().to_neural_network(network_instance)

            property_instance = NeVerProperty()
            property_instance.from_smt_file(f"{folder}/Properties/{instance[1]}")

            for setting in pynever_setting:
                logger_stream.info(f"Benchmark: {instance}")
                logger_stream.info(f"PyNeVer setting: {setting[0]}")

                strategy = NeverVerification(setting[1], setting[2])
                time_start = time.perf_counter()
                safe = not strategy.verify(onnx_net, property_instance)
                time_end = time.perf_counter()
                logger_file.info(f"{str(instance).replace(',', ' -')},{safe},{time_end - time_start}")

    # acas_dir = ('data/ACAS XU')
    # for property_file in os.listdir(f"{acas_dir}/Properties"):
    #     f = os.path.join(acas_dir, property_file)
    #
    #     network_files = os.listdir(f"{acas_dir}/Networks")
    #
    #     if os.path.isfile(f):
