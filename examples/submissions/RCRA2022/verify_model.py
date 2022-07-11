import argparse
import time
import pynever.strategies.conversion as conv
import pynever.strategies.verification as ver
import pynever.nodes as nodes
from gym import spaces
import numpy as np
import onnx
import logging
import torch
from datetime import datetime
import shared_constants

logger_exp_stream = logging.getLogger("pynever.strategies.verification")
logger_exp_file = logging.getLogger("exp_file")

exp_file_handler = logging.FileHandler(f"logs/{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}-ExperimentLog.txt")
exp_stream_handler = logging.StreamHandler()

exp_file_handler.setLevel(logging.INFO)
exp_stream_handler.setLevel(logging.INFO)

logger_exp_file.addHandler(exp_file_handler)
logger_exp_stream.addHandler(exp_stream_handler)

logger_exp_file.setLevel(logging.INFO)
logger_exp_stream.setLevel(logging.INFO)

acceptable_models = shared_constants.id_arch_dict.keys()

in_low_b = np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
in_high_b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
input_space = spaces.Box(low=in_low_b, high=in_high_b, dtype=np.float32)

num_neuron_mixed = 1

logger_exp_file.info("NETWORK_ID, HEURISTIC, EPSILON, MIN_LB, MAX_UB, DELTA, TIME")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Verification Experiment with Hover Actors')
    parser.add_argument('--model', default='AC1', type=str, choices=acceptable_models, help='ONNX network to verify')
    parser.add_argument('--algo',  default='overapprox', type=str, choices=["overapprox", "mixed", "complete"],
                        help='Verification Algorithm')
    parser.add_argument('--eps', default=0.1, type=float, help='Infinity norm bound')
    parser.add_argument('--seed', default=shared_constants.SEED, type=int,
                        help='Seed to use for the sample generation.')
    args = vars(parser.parse_args())

    net_path = f"onnx_nets/{args['model']}.onnx"
    heuristic = args['algo']
    epsilon = args['eps']

    input_space.seed(args['seed'])
    random_input = input_space.sample()
    logger_exp_stream.info(f"Input Sample:{random_input}")
    tensor_input = torch.from_numpy(random_input)
    logger_exp_file.info("NETWORK_ID, HEURISTIC, EPSILON, MIN_LB, MAX_UB, DELTA, TIME")

    net_id = net_path.replace("onnx_nets/", "").replace(".onnx", "")
    logger_exp_stream.info(f"EVALUATING NET {net_id} ({net_path})")

    onnx_net = conv.ONNXNetwork(net_id, onnx.load(net_path))
    net = conv.ONNXConverter().to_neural_network(onnx_net)
    torch_net = conv.PyTorchConverter().from_neural_network(net).pytorch_network

    numpy_output = torch_net(tensor_input).detach().numpy()

    ver_param = []

    for node in net.nodes.values():
        if isinstance(node, nodes.ReLUNode):
            if heuristic == "overapprox":
                ver_param.append([0])
            elif heuristic == "mixed":
                ver_param.append([num_neuron_mixed])
            elif heuristic == "complete":
                ver_param.append([node.in_dim[0]])

    # INPUT CONSTRAINTS DEFINITION

    in_pred_mat = []
    in_pred_bias = []
    data_size = len(random_input)
    for i in range(len(random_input)):

        lb_constraint = np.zeros(data_size)
        ub_constraint = np.zeros(data_size)
        lb_constraint[i] = -1
        ub_constraint[i] = 1
        in_pred_mat.append(lb_constraint)
        in_pred_mat.append(ub_constraint)
        if random_input[i] - epsilon < -1:
            in_pred_bias.append([1])
        else:
            in_pred_bias.append([-(random_input[i] - epsilon)])

        if random_input[i] + epsilon > 1:
            in_pred_bias.append([1])
        else:
            in_pred_bias.append([random_input[i] + epsilon])

    in_pred_bias = np.array(in_pred_bias)
    in_pred_mat = np.array(in_pred_mat)

    in_prop = ver.NeVerProperty(in_pred_mat, in_pred_bias, [], [])

    verifier = ver.NeverVerification(heuristic="best_n_neurons", params=ver_param)

    start = time.perf_counter()
    output_starset, computing_time = verifier.get_output_starset(net, in_prop)
    lbs = []
    ubs = []
    for star in output_starset.stars:
        lb, ub = star.get_bounds(0)
        lbs.append(lb)
        ubs.append(ub)

    min_lb = np.min(np.array(lbs))
    max_ub = np.max(np.array(ubs))

    delta = max(numpy_output[0] - min_lb, max_ub - numpy_output[0])

    stop = time.perf_counter()

    logger_exp_file.info(f"{net_id}, {heuristic}, {epsilon}, {min_lb}, {max_ub}, {delta}, {stop - start}")
    logger_exp_stream.info(f"{net_id}, {heuristic}, {epsilon}, {min_lb}, {max_ub}, {delta}, {stop - start}")
