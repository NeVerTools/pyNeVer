import yaml
import csv
import os
from datetime import datetime


def load_yaml_config(yaml_file_path):
    """
    Carica la configurazione YAML da un file specificato.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Errore: Il file {yaml_file_path} non è stato trovato.")
        return None
    except yaml.YAMLError as exc:
        print(f"Errore nella lettura del file YAML: {exc}")
        return None


def write_csv_from_config(config, csv_file_path):
    """
    Scrive un file CSV utilizzando i parametri dalla configurazione YAML.
    """
    csv_header = [
        'optimizer',
        'hdim',
        'learning_rate',
        'train_batch_size',
        'test_batch_size',
        'validation_batch_size',
        'scheduler_lr',
        'scheduler_patience',
        'validation_percentage',
        'loss_type',
        'dataset',
        'train_dim',
        'test_dim',
        'input_dim',
        'output_dim',
        'wp_strength',
        'np_strength',
        'batch_norm_decay',
        'l1_decay',
        'num_epochs'
    ]

    try:
        optimizer = config['training']['optimizer']
        learning_rate = config['training']['learning_rate']
        train_batch_size = config['training']['train_batch_size']
        test_batch_size = config['training']['test_batch_size']
        validation_batch_size = config['training']['validation_batch_size']
        scheduler_lr = config['training']['scheduler_lr']
        scheduler_patience = config['training']['scheduler_patience']
        validation_percentage = config['training']['validation_percentage']
        loss_type = config['training']['loss_type']
        dataset = config['data']['dataset']
        train_dim = config['data']['train_dim']
        test_dim = config['data']['test_dim']
        input_dim = config['data']['input_dim']
        output_dim = config['data']['output_dim']
        wp_strength = config['weight_pruning']['wp_strength']
        np_strength = config['neuron_pruning']['np_strength']
        batch_norm_decay = config['neuron_pruning']['batch_norm_decay']
        l1_decay = config['l1_sparse']['l1_decay']
        num_epochs = config['training']['epochs']

    except KeyError as e:
        print(f"Errore: Mancanza della chiave di configurazione richiesta: {e}")
        return

    hidden_layer_dims = config['network'].get('hidden_layer_dims', [])

    try:
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_header)
            writer.writeheader()

            for hdim in hidden_layer_dims:
                row = {
                    'optimizer': optimizer,
                    'hdim': hdim,
                    'learning_rate': learning_rate,
                    'train_batch_size': train_batch_size,
                    'test_batch_size': test_batch_size,
                    'validation_batch_size': validation_batch_size,
                    'scheduler_lr': scheduler_lr,
                    'scheduler_patience': scheduler_patience,
                    'validation_percentage': validation_percentage,
                    'loss_type': loss_type,
                    'dataset': dataset,
                    'train_dim': train_dim,
                    'test_dim': test_dim,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'wp_strength': wp_strength,
                    'np_strength': np_strength,
                    'batch_norm_decay': batch_norm_decay,
                    'l1_decay': l1_decay,
                    'num_epochs': num_epochs
                }
                writer.writerow(row)
        print(f"File CSV scritto con successo in: {csv_file_path}")
    except Exception as e:
        print(f"Errore durante la scrittura del file CSV: {e}")
    return csv_header
