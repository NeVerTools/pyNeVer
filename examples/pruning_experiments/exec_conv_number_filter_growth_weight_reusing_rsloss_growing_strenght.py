import os
import sys

from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics

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

    filters_numbers = [4,6,8,10,12,24,48,96,124]
    hidden_layer_dims = [30, 33, 36, 39, 42, 56, 64, 80]
    #increment_list = [0.5, 0.5, 0.5, 0.5, 0.5, 1 , 2]
    #increment_list = [0.5, 0, 0.5, 0.5, 1, 2, 3, 5]


    old_weights = None
    old_model = None
    HIDDEN_LAYER_DIM = 30
    rs_factor = 3

    for idx, filters in enumerate(filters_numbers):
        print(f"Number of filters {filters}")
        metrics, old_model = generate_no_batch_networks(config, filters_number=filters, old_weights = old_weights, RS_FACTOR=rs_factor, HIDDEN_LAYER_DIM=HIDDEN_LAYER_DIM)

        if old_model is not None:
            old_weights = (old_model.fcLikeConv.weight, old_model.fcLikeConv.bias,  old_model.fc1.weight, old_model.fc1.bias, old_model.fc2.weight, old_model.fc2.bias)
        HIDDEN_LAYER_DIM += 3




