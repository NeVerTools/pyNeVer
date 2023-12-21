from pynever.networks import SequentialNetwork
from pynever.nodes import FlattenNode, FullyConnectedNode, ReLUNode
from pynever.strategies import smt_reading, verification, conversion

import cpp.build.bin.cpp_verify as cpp_verify

import func_timeout

def verify_cpp(network_path, property_path, get_output_bounds=False):
    # Load the network
    # TODO better error handling
    nn = conversion.load_network_path(network_path)
    if not isinstance(nn, conversion.ONNXNetwork):
        raise Exception

    nn = conversion.ONNXConverter().to_neural_network(nn)
    if not isinstance(nn, SequentialNetwork):
        raise Exception

    # Checking that the first node is a fully connected node
    current_node = nn.get_first_node()
    if not isinstance(current_node, FullyConnectedNode):
        # If the first node is a flatten node, it is skipped and the fully connected node after it is considered
        if isinstance(current_node, FlattenNode):
            current_node = nn.get_next_node(current_node)
            if not isinstance(current_node, FullyConnectedNode):
                raise Exception
        else:
            raise Exception

    in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias = smt_reading. \
        SmtPropertyParser(property_path, "X", "Y").parse_property()

    bounds = verification.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)

    bounds = cpp_verify.PropertyBounds(bounds.in_coef_mat,
                                       bounds.in_bias_mat.transpose()[0],
                                       bounds.out_coef_mat[0],
                                       bounds.out_bias_mat[0].transpose()[0])
    layers = []
    while current_node is not None:
        if isinstance(current_node, FullyConnectedNode):
            weights = current_node.weight
            bias = current_node.bias
            followed_by_relu = False
            if isinstance(nn.get_next_node(current_node), ReLUNode):
                followed_by_relu = True
            layers.append(cpp_verify.FullyConnectedLayer(weights, bias, followed_by_relu))

            current_node = nn.get_next_node(current_node)
            if current_node is not None:
                current_node = nn.get_next_node(current_node)
        else:
            raise Exception

    network = cpp_verify.FullyConnectedNetwork(layers)
    solver = cpp_verify.CPVerificationSolver(network, bounds)

    try:
        res = func_timeout.func_timeout(600, solver.solve)
    except func_timeout.FunctionTimedOut:
        res = "Timeout"
    return res


if __name__ == "__main__":
    # verify_cpp("networks/ACAS_XU_1_1.onnx", "properties/ACAS.smt2")
    import glob
    import time

    import sys

    networks = [sys.argv[1]]
    # networks = glob.glob("networks/james/*.onnx")
    properties = [sys.argv[2]]
    # properties = glob.glob("properties/james/*.vnnlib")

    # networks = glob.glob("networks/ex/ex2.onnx")
    # properties = glob.glob("properties/ex/ex5.smt2")

    results = {}
    times = {}

    for prop in properties:
        for network in networks:
            print(network.split('/')[2])
            print(prop.split('/')[2])
            start = time.time()
            result = verify_cpp(network, prop)
            end = time.time()
            results[(network, prop)] = result
            times[(network, prop)] = end - start

    with open("cpp_times_no_prop.txt", "a") as f:
        for prop in properties:
            for network in networks:
                f.write(f"{network.split('/')[2]}\t{prop.split('/')[2]}\t{results[(network, prop)]}\t{times[network, prop]}\n")

    print(networks)
    print(properties)
