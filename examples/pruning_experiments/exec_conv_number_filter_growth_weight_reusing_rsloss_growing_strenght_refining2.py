import math
import os
import sys
import torch
import csv
from datetime import datetime

# Check CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

# Add parent directory to the system path
current_directory = os.getcwd()
parent_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, parent_directory)

from examples.pruning_experiments.networks_generation.generate_network_convolution_improved import generate_no_batch_networks, \
    load_yaml_config

def save_model(model, filters_number, folder, device):
    # Export the models to ONNX format
    dummy_input = torch.rand(1, 1, 28, 28).to(device)  # Ensure input is on the same device

    # Save the model in ONNX and PyTorch formats
    torch.onnx.export(
        model,
        dummy_input,
        f"{folder}/{filters_number}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )

    torch.save(model, f"{folder}/{filters_number}.pth")


def save_metrics_to_csv(metrics, csv_file):
    headers = list(metrics.keys())
    data = list(metrics.values())

    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)  # Write headers if file is new
        writer.writerow(data)  # Write metrics data


if __name__ == '__main__':
    yaml_file = 'configs/config_MNIST_filter_growth.yaml'
    config = load_yaml_config(yaml_file)

    filters_numbers = [4, 12, 24, 48, 64, 96, 124, 148]
    hidden_layer_dims = [33, 45, 48, 51, 53, 54, 57, 60]



    OUTPUT_FOLDER = r"results/best_selection"
    CSV_FILE = os.path.join(OUTPUT_FOLDER, "results.csv")



    old_weights = None
    old_model = None

    min_increment = 0
    max_increment = 5
    steps_limit = 10

    # Rs lambda for the smallest network, this values has to increase
    rs_factor = 3

    # Best accuracy for the smallest netowork. That value is gonna be the value to improve with over-parametrization
    metrics, model = generate_no_batch_networks(config, filters_number=filters_numbers.pop(0),
                                                rs_factor=rs_factor, hidden_layer_dim=hidden_layer_dims.pop(0))
    save_model(model, 4, OUTPUT_FOLDER, device)
    save_metrics_to_csv(metrics, CSV_FILE)

    # Baseline accuracy
    previous_accuracy = metrics['test_accuracy']




    for idx, filters in enumerate(filters_numbers):
        #print(f"Number of filters {filters}")

        #print(f"Baseline accuracy: {previous_accuracy}")

        target_model = None
        target_metrics = None
        target_rs_loss = None

        # Increasing the lambda parameter of the RSLoss until it stops decreasing the accuracy
        min_increment = 0.1
        max_increment = 7
        increment = (max_increment - min_increment)/2

        steps_counter = 0
        failure_bool = True

        while steps_counter <= steps_limit:
            metrics, model = generate_no_batch_networks(config, filters_number=filters, old_weights=old_weights,
                                                        rs_factor=rs_factor + increment,
                                                        hidden_layer_dim=hidden_layer_dims[idx])


            if metrics['test_accuracy'] + 0.001 >= previous_accuracy:
                target_rs_loss = rs_factor + increment
                target_model = model
                target_metrics = metrics
                min_increment = increment
                increment = min_increment + (max_increment - min_increment)/2
                failure_bool = False

            elif metrics['test_accuracy'] < previous_accuracy:
                max_increment = increment
                increment = min_increment +(max_increment - min_increment)/2

            steps_counter += 1

            if min_increment > max_increment:
                raise ValueError("min_increment > max_increment. Error in binary research implementation")


        if not failure_bool:
            print(f"Accuracy of network with {filters} has set the accuracy minimum to {target_metrics['test_accuracy']}")
            previous_accuracy = target_metrics['test_accuracy']
            save_model(target_model, filters, OUTPUT_FOLDER, device)
            save_metrics_to_csv(target_metrics, CSV_FILE)
            rs_factor = target_rs_loss
        else:
            print(f"Network with {filters} filters has failed.")
