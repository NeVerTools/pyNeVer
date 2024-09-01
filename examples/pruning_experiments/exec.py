from examples.pruning_experiments.networks_generation.generate_network import generate_no_batch_networks
from networks_generation.generate_config import *
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
    generate_csv()
    yaml_file = 'config1.yaml'  # Modifica con il percorso corretto del tuo file YAML

    timestamp = datetime.now().strftime("%Y%m%d")
    csv_file_path = f'temp/config_{timestamp}.csv'
    config = load_yaml_config(yaml_file)

    if config:
        csv_header = write_csv_from_config(config, csv_file_path)
    else:
        raise FileNotFoundError

    while True:
        # Opening the file without 'with' statement
        csv_file = open(csv_file_path, mode='r', newline='')
        try:
            reader = list(csv.DictReader(csv_file))
            if not reader:
                print("No more rows to process.")
                break
            else:
                data_dict = reader[0]  # Get the first row
                generate_no_batch_networks(data_dict)

                # Remove the first row and rewrite the CSV file
                csv_file.close()  # Close the file before reopening in writing mode

                # Open the file in writing mode to update it
                csv_file = open(csv_file_path, mode='w', newline='')
                writer = csv.DictWriter(csv_file, fieldnames=data_dict.keys())
                writer.writeheader()
                writer.writerows(reader[1:])  # Write all rows except the first one
        finally:
            csv_file.close()  # Ensure the file is closed even if an error occurs
