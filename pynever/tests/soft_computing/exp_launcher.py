# Script for launching the experiments of 'Soft Computing' journal paper
import csv
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager

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

TIMEOUT = 300  # 5 minutes


class TimeoutException(Exception):
    """
    Exception class for timeout

    """

    pass


@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException('Timeout')

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


def exec_instance(network_path: str, property_path: str, property_id: str):
    network_instance = conv.load_network_path(network_path)
    onnx_net = None
    if isinstance(network_instance, ONNXNetwork):
        onnx_net = ONNXConverter().to_neural_network(network_instance)

    property_instance = NeVerProperty()
    property_instance.from_smt_file(property_path)

    for setting in pynever_setting:
        inst_name = f"[{network_instance.identifier} - {property_id}]"
        logger_stream.info(f"Benchmark: {inst_name}")
        logger_stream.info(f"PyNeVer setting: {setting[0]}")

        try:
            with time_limit(TIMEOUT):
                strategy = NeverVerification(setting[1], setting[2])
                time_start = time.perf_counter()
                safe = strategy.verify(onnx_net, property_instance)
                time_end = time.perf_counter()
                logger_file.info(f"{inst_name},{safe},{time_end - time_start}")
        except TimeoutException:
            logger_file.info(f"{inst_name},---,---")
            break


if __name__ == '__main__':

    TEST_ACAS = True if sys.argv[1] == '1' else False
    TEST_ACC = True if sys.argv[2] == '1' else False
    TEST_RL = True if sys.argv[3] == '1' else False
    TEST_DRONES = True if sys.argv[4] == '1' else False

    logger_file.info('Benchmark,Over-approx.,,Mixed1,,Complete,,')
    logger_file.info(',Result,Time,Result,Time,Result,Time')

    # ACAS XU launcher
    if TEST_ACAS:
        with open('data/ACAS XU/instances.csv') as instances:
            folder = 'data/ACAS XU'
            csv_reader = csv.reader(instances)

            for instance in csv_reader:
                exec_instance(f"{folder}/Networks/{instance[0]}",
                              f"{folder}/Properties/{instance[1]}",
                              instance[1])

    # ACC and RL launcher
    dirs = []
    if TEST_ACC:
        dirs.append('data/ACC')

    if TEST_RL:
        dirs.extend(['data/RL/Cartpole', 'data/RL/Lunar Lander', 'data/RL/Dubins Rejoin'])

    for dir_name in dirs:
        for property_file in os.listdir(f"{dir_name}/Properties"):
            p_f = os.path.join(f"{dir_name}/Properties", property_file)

            if os.path.isfile(p_f):
                for network_file in os.listdir(f"{dir_name}/Networks"):
                    n_f = os.path.join(f"{dir_name}/Networks", network_file)

                    if os.path.isfile(n_f):
                        exec_instance(n_f, p_f, property_file)

    # Drones launcher
    if TEST_DRONES:
        with open('data/Drones/instances.csv') as instances:
            folder = 'data/Drones'
            csv_reader = csv.reader(instances)

            for instance in csv_reader:
                exec_instance(f"{folder}/Networks/{instance[0]}",
                              f"{folder}/Properties/{instance[1]}",
                              instance[1])
