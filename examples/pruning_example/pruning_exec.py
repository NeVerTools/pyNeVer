import sys
import os
import numpy as np
import csv

# Add the path to the pynever module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pruning_example import train_and_prune_mnist_network


ARCHITECTURES = numbers = [
    [30],
    [40],
    [45],
    [47],
    [49],
    [50],
    [51],
    [53],
    [55],
    [60],
    [70],
    [80],
    [90],
    [100],
    [110],
    [130],
    [150],
    [170],
    [200],
    [250],
    [500],
    [900],
    [1100]
]

upper_weight_pruning_rate = 0.7
upper_neuron_pruning_rate = 0.7
lower_pruning_rate = 0.4

# Define the step size for the pruning rates
step_size = 0.1

# Create the folder path for saving networks
base_folder = os.path.join(os.getcwd(), "networks")
os.makedirs(base_folder, exist_ok=True)

# open a file to save results
csv_file = "accuracies.csv"

weight_pruning_rate = 0.4
neuron_pruning_rate = 0.6
# Iterate through each architecture
for architecture in ARCHITECTURES:
    # Iterate through all combinations of pruning rates

    arch_layers = architecture

    # Convert each element to string and join them with underscores for folder name
    folder_name = '_'.join(map(str, arch_layers))

    # Create the folder for the specific architecture and pruning rates
    specific_folder_path = os.path.join(base_folder, f"{folder_name}_wp{weight_pruning_rate}_np{neuron_pruning_rate}")
    os.makedirs(specific_folder_path, exist_ok=True)

    # train data and get accuracies

    data_dict = train_and_prune_mnist_network(weight_pruning_rate, neuron_pruning_rate, arch_layers.copy(),
                                              specific_folder_path)

    header = list(data_dict.keys())

    # Open the CSV file in append mode and write the data
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)

        # Check if file is empty and write header if necessary
        file.seek(0, 2)  # Go to the end of file to check if empty
        if file.tell() == 0:
            writer.writeheader()  # Write header if file is empty

        # Write the data from data_dict to the CSV file
        writer.writerow(data_dict)
