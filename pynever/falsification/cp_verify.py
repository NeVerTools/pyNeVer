from docplex.cp.model import CpoModel
import docplex.cp.expression as expr
import docplex.cp.modeler as mdlr

from pynever.networks import SequentialNetwork
from pynever.nodes import FlattenNode, FullyConnectedNode, ReLUNode
from pynever.strategies import conversion, smt_reading, verification

# The following script takes as input an ONNX network, an SMT property with bounds on the input and the output of the
# network, and checks whether the property on the output is satisfied given that on the input using constraint
# programming


def verify_cp(network_path, property_path, get_output_bounds=False):
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

    # Create the model
    mdl = CpoModel()

    # Create variables corresponding to the input
    input_size = current_node.in_features
    x = expr.float_var_list(size=input_size, name="X")

    # Defining the bounds on the input

    # The following variables are matrices which define a polytope which limits the research area in such a way that the
    # considered points are all those that if multiplied by X_C are less or equal to X_d, i.e. all x such that
    # X_C @ x <= X_d
    # FIXME X and Y should be changed when they will be the same as the input and output of the network
    in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias = smt_reading.\
        SmtPropertyParser(property_path, "X", "Y").parse_property()

    bounds = verification.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)

    X_C = [list(x) for x in bounds.in_coef_mat]
    X_d = [x[0] for x in bounds.in_bias_mat]

    for i in range(len(X_C)):
        products = []
        for idx, c in enumerate(X_C[i]):
            products.append(c * x[idx])
        mdl.add(mdlr.sum(products) <= X_d[i])

    # This is an array of expressions containing the output of the last layer
    last_output = x
    while current_node is not None:
        if isinstance(current_node, FullyConnectedNode):
            weights = list(current_node.weight)
            # If the bias is not set it is set to 0
            bias = list(current_node.bias)
            if bias is None:
                bias = [0] * len(weights)

            # The following conditional statement concerns the case in which a Fully Connected layer is not followed by
            # a ReLU layer, and generates constraint which set the output to the product of the input and the weights
            if nn.get_next_node(current_node) is None:
                current_output = []
                for i in range(len(weights)):
                    products = []
                    for idx, c in enumerate(weights[i]):
                        products.append(c * last_output[idx])
                    current_output.append(sum(products) + bias[i])
                last_output = current_output

        elif isinstance(current_node, ReLUNode):
            current_output = []
            for i in range(len(weights)):
                products = []
                for idx, c in enumerate(weights[i]):
                    products.append(c * last_output[idx])
                current_output.append(mdlr.max(sum(products) + bias[i], 0))
            last_output = current_output

        else:
            raise Exception

        current_node = nn.get_next_node(current_node)

    n_alternatives = len(bounds.out_coef_mat)
    alts = []
    for alternative in range(n_alternatives):
        Y_C = [list(y) for y in bounds.out_coef_mat[alternative]]
        Y_d = [y[0] for y in bounds.out_bias_mat[alternative]]

        exprs = []
        for i in range(len(Y_C)):
            products = []
            for idx, c in enumerate(Y_C[i]):
                products.append(c * last_output[idx])
            exprs.append(mdlr.sum(products) <= Y_d[i])
        alts.append(exprs)
    alts = [mdlr.logical_and(exprs) for exprs in alts]
    mdl.add(mdlr.logical_or(alts))

    output_bounds = None
    if get_output_bounds:
        y = expr.float_var_list(size=len(last_output), name="Y")
        for idx, o in enumerate(last_output):
            mdl.add(y[idx] == o)
        var_bounds = mdl.propagate().solution.var_solutions_dict
        output_bounds = []
        for i in range(len(y)):
            output_bounds.append(var_bounds[f'Y_{i}'].value)

    # mdl.solve()
    #return mdl.solve().solve_status, output_bounds
    return mdl.solve().solve_status


if __name__ == "__main__":
    # import glob
    # import time
    #
    # networks = glob.glob("networks/james/*.onnx")
    # properties = ["properties/james_vnnlib.smt2"] * len(networks)
    #
    # results = []
    # times = []
    #
    # for network, prop in zip(networks, properties):
    #     print(network.split('/')[1].split('\\')[1])
    #     start = time.time()
    #     result = verify_cp(network, prop)
    #     end = time.time()
    #     results.append(result)
    #     times.append(end - start)
    #
    # with open("cp_times.txt", "w") as f:
    #     for network, prop, result, time in zip(networks, properties, results, times):
    #         f.write(f"{network}\t{prop}\t{result}\t{time}\n")

    # import glob
    # import time
    #
    # networks = glob.glob("networks/james/*.onnx")
    # properties = ["properties/james_vnnlib.smt2"] * 5
    #
    # results = []
    # times = []
    #
    # for network, prop in zip(networks, properties):
    #     print(network.split('/')[1].split('\\')[1])
    #     start = time.time()
    #     _, result = verify_cp(network, prop, True)
    #     end = time.time()
    #     results.append(result)
    #     times.append(end - start)
    #
    # with open("cp_bounds.txt", "w") as f:
    #     for network, prop, result, time in zip(networks, properties, results, times):
    #         f.write(f"{network}\t{prop}\t{result}\t{time}\n")
    import glob
    import time

    networks = glob.glob("networks/james/*.onnx")
    properties = glob.glob("properties/james/*.vnnlib")

    # networks = glob.glob("networks/ex/ex*.onnx")
    # properties = glob.glob("properties/ex/ex*.smt2")

    results = {}
    times = {}

    for prop in properties:
        for network in networks:
            print(network.split('/')[2])
            print(prop.split('/')[2])
            start = time.time()
            result = verify_cp(network, prop)
            end = time.time()
            results[(network, prop)] = result
            times[(network, prop)] = end - start

    with open("cp_times.txt", "w") as f:
        for prop in properties:
            for network in networks:
                f.write(
                    f"{network.split('/')[2]}\t{prop.split('/')[2]}\t{results[(network, prop)]}\t{times[network, prop]}\n")

