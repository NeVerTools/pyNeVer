import os
import sys

# Get the current working directory
current_directory = os.getcwd()

# Get the directory two levels above
parent_directory = os.path.dirname(os.path.dirname(current_directory))

# Add the parent directory to the Python path
sys.path.insert(0, parent_directory)


from examples.pruning_experiments.networks_generation.generate_network_number_filter_growth_wr_rsloss import generate_no_batch_networks, \
    load_yaml_config

from datetime import datetime
import csv


if __name__ == '__main__':
    yaml_file = 'configs/config_MNIST_filter_growth.yaml'
    config = load_yaml_config(yaml_file)

    mul_factors = [24]

    old_weights = None

    for mul_factor in mul_factors:
        print(f"Number of filters {mul_factor} * 8: {mul_factor * 2}")
        old_weights = generate_no_batch_networks(config, mul_factor, old_weights=old_weights, rs_factor=20,
                                                 hidden_layer_dim=)




