from pynever import datasets, nodes, networks
from pynever.strategies import training, conversion
from torch.nn import CrossEntropyLoss
import torch.optim as opt


INPUT_SIZE = 8
OUTPUT_SIZE = 6

layer_sizes = [16, 32]
max_layers = 2

nn_trainer = training.PytorchTraining(optimizer_con=opt.Adam,
                                      opt_params={'lr': 0.1},
                                      loss_function=CrossEntropyLoss(),
                                      n_epochs=5,
                                      validation_percentage=0.3,
                                      train_batch_size=128,
                                      validation_batch_size=64,
                                      scheduler_con=opt.lr_scheduler.ReduceLROnPlateau,
                                      sch_params={'patience': 5},
                                      precision_metric=training.PytorchMetrics.inaccuracy,
                                      network_transform=None)
james_dataset = datasets.GenericFileDataset("james_pos_train.txt", 8)

converter = conversion.ONNXConverter()

for layer_size in layer_sizes:
    print(layer_size)
    for n_layers in range(1, max_layers + 1):
        print(n_layers)
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
        print(nn)
        # Training
        trained_net = nn_trainer.train(nn, james_dataset)

        # Conversion to ONNX
        onnx_net = converter.from_neural_network(trained_net)
        conversion.save_network_path(onnx_net, f"james_{layer_size}x{n_layers}.onnx")

