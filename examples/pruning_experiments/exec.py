from examples.pruning_experiments.networks_generation.generate_network import generate_no_batch_networks, \
    load_yaml_config
from datetime import datetime
import csv


def generate_csv():
    # List of CSV files to update
    csv_files = [
        'accuracies_no_batch.csv',
        'accuracies_neuron_pruning.csv',
        'accuracies_no_batch_sparse.csv',
        'accuracies_weight_pruning.csv',
        'accuracies_with_batch.csv',
        'accuracies_with_batch_sparse.csv'
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
    yaml_file = 'config1.yaml'  # Modifica con il percorso corretto del tuo file YAML'
    config = load_yaml_config(yaml_file)

    hdims = config['hidden_layer_dims']

    for hdim in hdims:
        generate_no_batch_networks(config, hdim)

