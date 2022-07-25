import logging
import time

import numpy as np
import onnx

from pynever.strategies import verification, conversion

if __name__ == '__main__':
    logger = logging.getLogger('pynever.strategies.verification')
    log_handler = logging.StreamHandler()
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)

    # Global parameters
    input_size = 12
    output_size = 1

    input_lbs = [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    input_ubs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # One unsafe delta (threshold) for each network
    drone_nets = ['AC1', 'AC2', 'AC3', 'AC4', 'AC5', 'AC6', 'AC7', 'AC8']
    unsafe_deltas = [4.0, 2.2, 12.5, 6.0, 17.0, 1.8, 2.5, 28.0]

    # Verification parameters
    ver_params = {
        'Complete': ['best_n_neurons', [[10000] for _ in range(3)]],
        'Over-approximate': ['best_n_neurons', [[0] for _ in range(3)]],
        'Mixed': ['best_n_neurons', [[1] for _ in range(3)]]
    }
    net_path = 'RCRA22_nets/'

    # Build input matrices
    in_pred_mat = []
    in_pred_bias = []

    for i in range(input_size):
        lb_constraint = np.zeros(input_size)
        ub_constraint = np.zeros(input_size)
        lb_constraint[i] = -1
        ub_constraint[i] = 1
        in_pred_mat.append(lb_constraint)
        in_pred_mat.append(ub_constraint)

        in_pred_bias.append([-float(input_lbs[i])])
        in_pred_bias.append([float(input_ubs[i])])

    in_pred_bias = np.array(in_pred_bias)
    in_pred_mat = np.array(in_pred_mat)

    out_pred_mats = [np.ones((1, 1)) for _ in range(len(drone_nets))]
    out_pred_bias = [np.ones((1, 1)) for _ in range(len(drone_nets))]
    for i in range(len(drone_nets)):
        out_pred_bias[i][0, 0] = unsafe_deltas[i]

    # Begin tests
    for i in range(len(drone_nets)):
        print(f"Verify robustness property on network {drone_nets[i]}...")
        net_full_path = f"{net_path}{drone_nets[i]}.onnx"
        p = verification.NeVerProperty(in_pred_mat, in_pred_bias, [out_pred_mats[i]], [out_pred_bias[i]])
        p.to_smt_file('X', 'Y', f"smt_prop_{i}")

        for v_type, params in ver_params.items():
            print(f"Verification methodology: {v_type}")
            verifier = verification.NeverVerification(params[0], params[1])

            onnx_net = conversion.ONNXNetwork(drone_nets[i], onnx.load(net_full_path))
            net = conversion.ONNXConverter().to_neural_network(onnx_net)

            time_start = time.perf_counter()
            safe = verifier.verify(net, p)
            time_end = time.perf_counter()
