from pynever import datasets, nodes, networks
from pynever.strategies import training, conversion
import torch
import torch.optim as opt


INPUT_SIZE = 8
OUTPUT_SIZE = 6

layer_sizes = [16, 32, 64, 128]
max_layers = 5

nn_trainer = training.PytorchTraining(optimizer_con=opt.Adam,
                                      opt_params={'lr': 0.1},
                                      loss_function=torch.nn.MSELoss(),
                                      n_epochs=5,
                                      validation_percentage=0.3,
                                      train_batch_size=128,
                                      validation_batch_size=64,
                                      scheduler_con=opt.lr_scheduler.ReduceLROnPlateau,
                                      sch_params={'patience': 5},
                                      precision_metric=torch.nn.functional.mse_loss,
                                      network_transform=None)
james_dataset = datasets.GenericFileDataset("james_pos_train.txt", 8)

converter = conversion.ONNXConverter()

for layer_size in layer_sizes:
    for n_layers in range(1, max_layers + 1):
        # Create the network
        nn = networks.SequentialNetwork(f"james_{layer_size}x{n_layers}", "X")
        # Adding input layer
        layer = 0
        nn.add_node(nodes.FullyConnectedNode(f"FC_{layer}", (INPUT_SIZE, ), layer_size))
        nn.add_node(nodes.ReLUNode(f"ReLU_{layer}", (layer_size, )))
        layer += 1
        # Adding hidden layers
        while layer != n_layers:
            nn.add_node(nodes.FullyConnectedNode(f"FC_{layer}", (layer_size,), layer_size))
            nn.add_node(nodes.ReLUNode(f"ReLU_{layer}", (layer_size,)))
            layer += 1
        # Adding output layer
        nn.add_node(nodes.FullyConnectedNode(f"FC_{layer}", (layer_size,), OUTPUT_SIZE))
        # Training
        trained_net = nn_trainer.train(nn, james_dataset)

        # Conversion to ONNX
        onnx_net = converter.from_neural_network(trained_net)
        conversion.save_network_path(onnx_net, f"james_{layer_size}x{n_layers}.onnx")

