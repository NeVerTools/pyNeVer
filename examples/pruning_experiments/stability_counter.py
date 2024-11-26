import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx
import torch.nn.functional as F
import csv
import pandas as pd


# Funzione per estrarre il numero dal nome del file
def extract_number(filename):
    return int(filename.split('.')[0])  # Restituisce il numero prima di '.onnx'


# Funzione per ottenere i parametri del modello ONNX
def get_model_parameters(model_path):
    model = onnx.load(model_path)

    params = {}
    for initializer in model.graph.initializer:
        weight_or_bias = onnx.numpy_helper.to_array(initializer)
        params[initializer.name] = weight_or_bias
        print(f"Name: {initializer.name}")
        print(f"Shape: {weight_or_bias.shape}")
        print(f"Data:\n{weight_or_bias}\n")

    return params


# Funzione per l'aritmetica degli intervalli nei fully connected layer
def interval_arithmetic_fc(lb, ub, W, b):
    W = W.T
    W_max = torch.maximum(W, torch.tensor(0.0))
    W_min = torch.minimum(W, torch.tensor(0.0))

    new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
    new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b

    return new_lb, new_ub


# Funzione per contare i neuroni instabili in un singolo layer
def count_unstable_neurons(lb, ub):
    unstable_neurons = (lb < 0) & (ub > 0)
    return unstable_neurons.sum().item()  # Restituisce il conteggio come intero


if __name__ == '__main__':
    output_file_path = 'instability_csv.csv'

    # Imposta il dataset MNIST e il DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # Dizionario per memorizzare i risultati
    instability_data = {}

    # Ottieni i file ONNX dalla cartella
    directory_path = r"D:\11-12-24\pyNeVer\examples\pruning_experiments\results\no_batch"
    onnx_files = [f for f in os.listdir(directory_path) if f.endswith('.onnx')]

    # Ordina i file ONNX per numero nel nome
    onnx_files.sort(key=extract_number)

    # Ciclo per ogni file ONNX
    for onnx_file in onnx_files:
        model_path = os.path.join(directory_path, onnx_file)
        print(f"Elaborando il modello: {onnx_file}")

        # Imposta FILTER_NUMBER come il numero estratto dal nome del file
        FILTER_NUMBER = extract_number(onnx_file)

        # Ottieni i parametri del modello
        params = get_model_parameters(model_path)

        # Converte i pesi e i bias in tensori PyTorch
        W1 = torch.tensor(params.get('fcLikeConv.weight'), dtype=torch.float32)
        b1 = torch.tensor(params.get('fcLikeConv.bias'), dtype=torch.float32)
        W2 = torch.tensor(params.get('fc1.weight'), dtype=torch.float32)
        b2 = torch.tensor(params.get('fc1.bias'), dtype=torch.float32)

        counter = 0
        for i, (input_data, target) in enumerate(test_loader):
            if counter == 150:  # Usa il valore di NUMBER_OF_SAMPLES per limitare i campioni
                break

            # Propagazione con aritmetica degli intervalli
            noise = 0.1
            ubs = input_data + noise
            lbs = input_data - noise


            def convert_input(bounds, kernel_param_size=3, stride=1, padding=0):
                patches_lb = F.unfold(bounds, kernel_size=kernel_param_size, stride=stride, padding=padding)
                patches_lb = patches_lb.squeeze(0)
                bounds = patches_lb.transpose(0, 1)
                return bounds


            lb_1 = convert_input(lbs)
            ub_1 = convert_input(ubs)

            lb_1, ub_1 = interval_arithmetic_fc(lb_1, ub_1, W1, b1)


            def reshape(bounds):
                batch_size, _, height, width = input_data.size(0), input_data.size(1), input_data.size(
                    2), input_data.size(3)
                out_height = (height - 3) // 1 + 1  # Usa il kernel_size=3 e stride=1
                out_width = (width - 3) // 1 + 1
                bounds = bounds.view(batch_size, out_height, out_width, FILTER_NUMBER).permute(0, 3, 1, 2)
                return bounds


            lb_1 = reshape(lb_1).flatten(start_dim=1)
            ub_1 = reshape(ub_1).flatten(start_dim=1)

            lbh_1 = torch.relu(lb_1)
            ubh_1 = torch.relu(ub_1)

            lb_2, ub_2 = interval_arithmetic_fc(lbh_1, ubh_1, W2, b2)

            conv_unstable_neurons = count_unstable_neurons(lb_1, ub_1)
            fc1_unstable_neurons = count_unstable_neurons(lb_2, ub_2)

            # Memorizza i dati separatamente per ogni modello
            if f"conv_{FILTER_NUMBER}" not in instability_data:
                instability_data[f"conv_{FILTER_NUMBER}"] = []
            if f"fc_{FILTER_NUMBER}" not in instability_data:
                instability_data[f"fc_{FILTER_NUMBER}"] = []

            instability_data[f"conv_{FILTER_NUMBER}"].append(conv_unstable_neurons)
            instability_data[f"fc_{FILTER_NUMBER}"].append(fc1_unstable_neurons)

            counter += 1

    # Crea un DataFrame per i risultati
    df = pd.DataFrame(instability_data)

    # Salva i risultati su un file CSV
    df.to_csv(output_file_path, index=False)

    # Salva anche in Excel
    df.to_excel('instability_analysis.xlsx', index=False)
