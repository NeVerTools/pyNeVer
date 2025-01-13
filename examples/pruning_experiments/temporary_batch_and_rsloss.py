import math
import os
import sys
import torch
import csv
from datetime import datetime
import cProfile
import pstats
import io


# Check CUDA availability and set device
device = torch.device('cuda' )#'cuda' if torch.cuda.is_available() else 'cpu')
print(device)
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
    filters_numbers = [4]

    OUTPUT_FOLDER = r"results/best_selection"
    CSV_FILE = os.path.join(OUTPUT_FOLDER, "results.csv")

    old_weights = None
    old_model = None

    min_increment = 0
    max_increment = 5
    steps_limit = 10

    # Rs lambda for the smallest network, this values has to increase
    rs_factor = 3

    def profile_generate_no_batch_networks(config, filters_number, rs_factor, hidden_layer_dims):
        profiler = cProfile.Profile()
        profiler.enable()
        metrics, model = generate_no_batch_networks(
            config,
            filters_number=filters_number,
            rs_factor=rs_factor,
            hidden_layer_dim=hidden_layer_dims
        )
        profiler.disable()

        # Analizza e stampa i risultati
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        return metrics, model

    # Best accuracy for the smallest netowork. That value is gonna be the value to improve with over-parametrization
    metrics, model = profile_generate_no_batch_networks(config, filters_number=filters_numbers.pop(0),
                                                rs_factor=rs_factor, hidden_layer_dims=hidden_layer_dims.pop(0))
    save_model(model, 4, OUTPUT_FOLDER, device)
    save_metrics_to_csv(metrics, CSV_FILE)

    # Baseline accuracy
    previous_accuracy = metrics['test_accuracy']
    #
    # for idx, filters in enumerate(filters_numbers):
    #
    #     increment = 0
    #     metrics, model = generate_no_batch_networks(config, filters_number=filters,
    #                                                 rs_factor=rs_factor + increment,
    #                                                 hidden_layer_dim=hidden_layer_dims[idx])

