import csv
import glob
import sys
import time

import numpy as np
import sampling

import logging

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def test_sampling(network_name: str, n_samples: list[int]):
    networks = glob.glob(f"networks/{network_name}.onnx")
    properties = sorted(glob.glob(f"properties/{network_name}/*.vnnlib"))

    ground_truth_file = f"test/{network_name}_ground_truth.csv"

    ground_truth = []
    with open(ground_truth_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            typed_row = {"index": int(row["index"]), "safe": row["safe"] == "True"}
            ground_truth.append(typed_row)

    print(len(ground_truth))
    print(len(properties))
    assert len(ground_truth) == len(properties)

    total = len(networks) * len(properties)
    results = []
    for n in n_samples:
        correct = 0
        times = []
        for network in networks:
            for prop, truth in zip(properties, ground_truth):
                logger.log(logging.INFO, f"Testing {prop} for {network}")
                start = time.time()
                result = sampling.sampling(sampling.load_network(network), prop, n_points=n)
                end = time.time()
                times.append(end - start)
                if (result is None) != truth["safe"]:
                    correct += 1
        results.append({"sample_points": n, "accuracy": correct / total * 100, "average_time": np.mean(times)})

    result_file = f"{network_name}_results.csv"

    field_names = ["sample_points", "accuracy", "average_time"]

    with open(result_file, 'w') as file:
        csv_writer = csv.DictWriter(file, fieldnames=field_names)
        csv_writer.writeheader()
        csv_writer.writerows(results)

    print(f"Result file for {network_name} has been created.")


if __name__ == "__main__":
    np.random.seed(0)
    samples = [100, 1000]
    test_sampling("cartpole", samples)
    test_sampling("lunarlander", samples)
    test_sampling("dubinsrejoin", samples)
