import os
import sys

# Get the current working directory
current_directory = os.getcwd()

# Get the directory two levels above
parent_directory = os.path.dirname(os.path.dirname(current_directory))

# Add the parent directory to the Python path
sys.path.insert(0, parent_directory)


from examples.pruning_experiments.networks_generation.generate_deep_network import generate_no_batch_networks, \
    load_yaml_config

from datetime import datetime
import csv


def generate_csv():
    # List of CSV files to update
    csv_files = [
        "accuracies_neuron_pruning.csv",
        "accuracies_no_batch.csv",
        "accuracies_no_batch_dropout.csv",
        "accuracies_no_batch_leaky.csv",
        "accuracies_no_batch_sparse.csv",
        "accuracies_no_batch_weight_decay.csv",
        "accuracies_weight_pruning.csv",
        "accuracies_with_batch.csv",
        "accuracies_with_batch_sparse.csv",
        "accuracies_with_batch_weight_decay.csv"
    ]

    # Header to write
    header = ['h_dim', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']

    # Iterate over each file and update the header
    for file_name in csv_files:
        try:
            file_path = f'results/{file_name}'
            # Open the file in write mode to overwrite existing content
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(header)
            print(f"Updated header for file: {file_name}")
        except Exception as e:
            print(f"Failed to update file {file_name}: {e}")


if __name__ == '__main__':
    yaml_file = 'config_deep_MNIST.yaml'
    config = load_yaml_config(yaml_file)

    hdims = config['hidden_layer_dims']
    n_hidden_layers = config['n_of_layers']

    for hdim in hdims:
        for n_hidden_layer in n_hidden_layers:
            print(f"Hidden layer dimsension: {hdim}")
            generate_no_batch_networks(config, hdim, n_hidden_layer)


