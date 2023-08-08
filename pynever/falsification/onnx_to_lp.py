import cplex

from pynever.networks import SequentialNetwork
from pynever.nodes import FlattenNode, FullyConnectedNode, ReLUNode
from pynever.strategies import conversion, smt_reading, verification

# The following script takes as input an ONNX network, an SMT property with bounds on the input and the output of the
# network, and generates a MILP problem which checks if the property on the output is satisfied given that on the input,
# optionally writing it


def verify_milp(network_path, property_path, file_path=""):
    # Big M definition
    M = 1e2

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

    # Definition of the problem
    m = current_node.in_dim[0]  # m is the input dimension
    # Creating the x variables and adding their bounds, their types and their coefficients in the cost function
    var_names = [f"x{i}" for i in range(1, m+1)]
    upper_bounds = [cplex.infinity] * m
    lower_bounds = [-cplex.infinity] * m
    var_types = "C" * m
    obj = [0.0] * m

    # Creating the variables which contain the constraint information
    constraint_names = []  # Contains the name of each constraint
    constraint_sense = ""  # Contains the sense of each constraint, which can be G, L or E
    rows = []  # Contains the index of every constraint
    variables = []  # Contains the index of every variable
    coefficients = []  # Contains the coefficients of the variables
    rhs = []  # Contains the known term of each constraint
    # For example, if the variables look like this:
    # constraint_sense = "GL"
    # rows = [1, 1, 2]
    # variables = [1, 2, 1]
    # coefficients = [5, 8, 3]
    # rhs = [2, 7]
    #
    # the constraints will look like this:
    # 5 x1 + 8 x2 >= 2
    # 3 x1 <= 7

    n_constraints = 0
    n_variables = 0

    # Adding constraints on the input interval limitation X
    rhs += X_d
    for idx, cs in enumerate(X_C):
        rows += [idx+n_constraints] * m
        variables += list(range(m))
        coefficients += cs
    n_constraints += len(X_C)

    # FIXME expects both upper and lower bounds on each input, is this a correct assumption?
    n_variables += len(X_C) // 2

    constraint_names += [f"c_x{i}" for i in range(1, len(X_C)+1)]
    constraint_sense += "L" * len(X_C)

    # Constraints on ReLU
    if not isinstance(current_node, FullyConnectedNode):
        raise Exception

    last_output = list(range(m))  # Indices of the variables of the last output, at the beginning it's the input
    # The following cycle goes through ReLU and Fully connected layers to set
    while current_node is not None:
        if isinstance(current_node, FullyConnectedNode):
            # The weights are opposite sign because of the structure of the ReLU constraints
            o_weights = [list(map(lambda x: -x, row)) for row in current_node.weight]
            # If the bias is not set it is set to 0
            bias = current_node.bias
            if bias is None:
                bias = [0] * len(o_weights)

            # The following conditional statement concerns the case in which a Fully Connected layer is not followed by a
            # ReLU layer, and generates constraint which set the output to the product of the input and the weights
            if nn.get_next_node(current_node) is None:
                new_last_output = []
                for idx, row in enumerate(o_weights):
                    var_names += [f"id{current_node.identifier}y{idx}"]
                    upper_bounds += [cplex.infinity]
                    lower_bounds += [-cplex.infinity]
                    var_types += "C"
                    obj += [0.0]
                    n_variables += 1
                    new_last_output.append(n_variables - 1)

                    rhs += [bias[idx]]

                    rows += [n_constraints] * len(last_output) + [n_constraints]
                    variables += last_output + [n_variables - 1]
                    coefficients += row + [1.0]

                    n_constraints += 1
                    constraint_names += [f"c{current_node.identifier}{idx}"]
                    constraint_sense += "E"
                last_output = new_last_output

        # If the current layer is a ReLU layer, the constraints for each ReLU are created
        elif isinstance(current_node, ReLUNode):
            new_last_output = []  # Temporary variable which is populated by the following cycle
            for idx, row in enumerate(o_weights):
                var_names += [f"id{current_node.identifier}y{idx}", f"id{current_node.identifier}a{idx}"]
                upper_bounds += [cplex.infinity, 1.0]
                lower_bounds += [0.0, 0.0]
                var_types += "CI"
                obj += [0.0, 0.0]
                n_variables += 2
                new_last_output.append(n_variables-2)

                rhs += [M + bias[idx], bias[idx], 0]

                rows += [n_constraints] * len(last_output) + [n_constraints, n_constraints]
                variables += last_output + [n_variables-2, n_variables-1]
                coefficients += row + [1.0, M]

                rows += [n_constraints+1] * len(last_output) + [n_constraints+1]
                variables += last_output + [n_variables-2]
                coefficients += row + [1.0]

                rows += [n_constraints+2, n_constraints+2]
                variables += [n_variables-2, n_variables-1]
                coefficients += [1.0, -M]

                n_constraints += 3
                constraint_names += [f"c{current_node.identifier}{idx}{i}" for i in range(1, 4)]
                constraint_sense += "LGL"
            last_output = new_last_output
        else:
            raise Exception

        current_node = nn.get_next_node(current_node)

    # Setting constraints on output properties
    n_delta = len(bounds.out_coef_mat)
    var_names += [f"delta{i}" for i in range(n_delta)]
    upper_bounds += [1.0] * n_delta
    lower_bounds += [0.0] * n_delta
    var_types += "I" * n_delta
    obj += [0.0] * n_delta
    delta_indices = list(range(n_variables, n_variables + n_delta))
    n_variables += n_delta

    for i in range(n_delta):
        Y_C = [list(x) for x in bounds.out_coef_mat[i]]
        Y_d = [x[0] for x in bounds.out_bias_mat[i]]

        rhs += [x + M for x in Y_d]
        for idx, cs in enumerate(Y_C):
            rows += [idx+n_constraints] * (len(last_output) + 1)
            variables += last_output + [delta_indices[i]]
            coefficients += cs + [M]
        n_constraints += len(Y_C)

        constraint_names += [f"c_out{i}{j}" for j in range(1, len(Y_C)+1)]
        constraint_sense += "L" * len(Y_C)

    rhs += [1.0]
    constraint_names += ["deltas"]
    constraint_sense += "G"

    rows += [n_constraints] * n_delta
    variables += delta_indices
    coefficients += [1.0] * n_delta

    # Recasting to float to avoid cplex error
    coefficients = [float(x) for x in coefficients]
    rhs = [float(x) for x in rhs]

    # Creating, writing and solving the problem
    prob = cplex.Cplex()

    prob.objective.set_sense(prob.objective.sense.minimize)

    prob.linear_constraints.add(rhs=rhs, senses=constraint_sense,
                                names=constraint_names)
    prob.variables.add(obj=obj, lb=lower_bounds, ub=upper_bounds, types=var_types,
                       names=var_names)

    prob.linear_constraints.set_coefficients(zip(rows, variables, coefficients))

    if file_path != "":
        prob.write(file_path)

    prob.solve()
    try:
        print(prob.solution.get_objective_value())
        # Print the values of the x variables and the delta variables
        print(prob.variables.get_names(list(range(m)) + delta_indices))
        print(prob.solution.get_values(list(range(m)) + delta_indices))
    except cplex.exceptions.errors.CplexSolverError:
        pass

    return prob.solution.get_status()


if __name__ == "__main__":
    verify_milp("networks/ex.onnx", "properties/ex.smt2", "problems/ex.lp")
    # import glob
    # import time
    #
    # networks = glob.glob("networks/ACAS_XU*.onnx")
    # properties = glob.glob("properties/ACAS\*.vnnlib")
    #
    # results = {}
    # times = {}
    #
    # for network in networks:
    #     for prop in properties:
    #         print(network.split('\\')[1])#.split('\\')[1])
    #         print(prop.split('/')[1].split('\\')[1])
    #         start = time.time()
    #         result = verify_milp(network, prop)
    #         end = time.time()
    #         results[f"{network}\t{prop}"] = result
    #         times[f"{network}\t{prop}"] = end - start
    #         print(end-start)
    #
    # with open("ACAS_milp_times.txt", "w") as f:
    #     for result in results.keys():
    #         f.write(f"{result}\t{results[result]}\t{times[result]}")
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
    #     result = verify_milp(network, prop)
    #     end = time.time()
    #     results.append(result)
    #     times.append(end - start)
    #
    # with open("milp_times.txt", "w") as f:
    #     for network, prop, result, time in zip(networks, properties, results, times):
    #         f.write(f"{network}\t{prop}\t{result}\t{time}\n")
    # with open("times.txt", "w") as f:
    #     networks = ["ACAS_XU_2_3.onnx",
    #                 "ACAS_XU_1_3.onnx",
    #                 "ACAS_XU_1_1.onnx",
    #                 "ACAS_XU_4_3.onnx"]
    #     prop = "ACAS.smt2"
    #     for n in networks:
    #         f.write(f"{n} \t {prop} \t {verify_milp(f'networks/{n}', f'properties/{prop}')}\n")

    # with open("times.txt", "a") as f:
    #     network = "mnist_clean.onnx"
    #     properties = ["mnist_prop_1_0.03.vnnlib",
    #                   "mnist_prop_1_0.05.vnnlib",
    #                   "mnist_prop_2_0.03.vnnlib",
    #                   "mnist_prop_2_0.05.vnnlib",
    #                   "mnist_prop_3_0.03.vnnlib",
    #                   "mnist_prop_3_0.05.vnnlib"]
    #     for p in properties:
    #         f.write(f"{network} \t {p} \t {verify_milp(f'networks/{network}', f'properties/{p}')}\n")
