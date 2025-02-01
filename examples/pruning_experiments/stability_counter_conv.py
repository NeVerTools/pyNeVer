import os
import torch
from torchvision import datasets, transforms
from examples.pruning_experiments.utils.convolution_bp import propagate_conv_bp_sparse
from torch.utils.data import DataLoader
import onnx
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
    return unstable_neurons.sum().item()


# Funzione principale per analizzare l'instabilità nei modelli ONNX
def analyze_onnx_instability(directory_path, output_file_path, noise=0.1, num_samples=5):
    # Imposta il dataset MNIST e il DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # Dizionario per memorizzare i risultati
    instability_data = {}

    # Ottieni i file ONNX dalla cartella
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
        W1 = torch.tensor(params.get('onnx::Conv_20'), dtype=torch.float32)
        b1 = torch.tensor(params.get('onnx::Conv_21'), dtype=torch.float32)
        W2 = torch.tensor(params.get('fc1.weight'), dtype=torch.float32)
        b2 = torch.tensor(params.get('fc1.bias'), dtype=torch.float32)

        counter = 0
        for i, (input_data, target) in enumerate(test_loader):
            if counter == num_samples:  # Usa il valore di num_samples per limitare i campioni
                break

            # Propagazione con aritmetica degli intervalli
            ubs = input_data + noise
            lbs = input_data - noise

            lb_1, ub_1, _ = propagate_conv_bp_sparse(kernel_size=3, padding=(0, 0, 0, 0), stride=1, lb=lbs, ub=ubs,
                                                     device=torch.device('cpu'), filter_weights=W1, filter_biases=b1)
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
    df.to_excel(output_file_path.replace('.csv', '.xlsx'), index=False)


# Esempio di utilizzo della funzione
if __name__ == '__main__':
    analyze_onnx_instability(
        directory_path=r"C:\Users\andr3\PycharmProjects\pyNeVer\examples\pruning_experiments\results\no_batch",
        output_file_path='instability_csv.csv',
        noise=0.1,
        num_samples=100
    )
