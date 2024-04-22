
import numpy as np
import networkx
import matplotlib.pyplot as plt

import pynever.nodes as pyn_nodes
import pynever.strategies.abstraction as pyn_abst


def print_star_data(p_star: pyn_abst.Star):

    print("PREDICATE CONSTRAINTS:")
    for row in range(p_star.predicate_matrix.shape[0]):
        constraint = ""
        for col in range(p_star.predicate_matrix.shape[1]):
            if p_star.predicate_matrix[row, col] < 0:
                sign = "-"
            else:
                sign = "+"
            constraint = constraint + f"{sign} {abs(p_star.predicate_matrix[row, col])} * x_{col} "

        constraint = constraint + f"<= {p_star.predicate_bias[row, 0]}"
        print(constraint)

    print("VARIABLES EQUATIONS:")
    for row in range(p_star.basis_matrix.shape[0]):
        equation = f"z_{row} = "
        for col in range(p_star.basis_matrix.shape[1]):
            if p_star.basis_matrix[row, col] < 0:
                sign = "-"
            else:
                sign = "+"
            equation = equation + f"{sign} {abs(p_star.basis_matrix[row, col])} * x_{col} "

        if p_star.center[row, 0] < 0:
            c_sign = "-"
        else:
            c_sign = "+"
        equation = equation + f"{c_sign} {abs(p_star.center[row, 0])}"
        print(equation)


def test_single_concat():

    predicate_matrix = np.array([[-1], [1]])
    predicate_bias = np.array([[0], [1]])

    first_star = pyn_abst.Star(predicate_matrix, predicate_bias)
    print("FIRST STAR: ")
    print_star_data(first_star)
    second_star = pyn_abst.Star(predicate_matrix, predicate_bias)
    print("\nSECOND STAR: ")
    print_star_data(second_star)

    star_set = pyn_abst.single_concat_forward(first_star, second_star)
    print("\nCONCAT STAR: ")
    print_star_data(list(star_set)[0])


def test_abst_concat_node():

    first_predicate_matrix = np.array([[-1], [1]])
    first_predicate_bias = np.array([[0], [1]])

    second_predicate_matrix = np.array([[-1], [1]])
    second_predicate_bias = np.array([[-2], [3]])

    print("FIRST STAR: ")
    print_star_data(pyn_abst.Star(first_predicate_matrix, first_predicate_bias))
    print("\nSECOND STAR: ")
    print_star_data(pyn_abst.Star(second_predicate_matrix, second_predicate_bias))

    first_starset = pyn_abst.StarSet({pyn_abst.Star(first_predicate_matrix, first_predicate_bias),
                                      pyn_abst.Star(second_predicate_matrix, second_predicate_bias)})

    second_starset = pyn_abst.StarSet({pyn_abst.Star(first_predicate_matrix, first_predicate_bias),
                                       pyn_abst.Star(second_predicate_matrix, second_predicate_bias)})

    abst_conc_node = pyn_abst.AbsConcatNode("temp", pyn_nodes.ConcatNode("temp", [(1,), (1,)]))

    output_starset = abst_conc_node.forward([first_starset, second_starset])

    print("\nOUTPUT STARSET:")

    for i, star in enumerate(output_starset.stars):
        print(f"\nSTAR {i}:")
        print_star_data(star)


def test_single_sum():

    predicate_matrix = np.array([[-1], [1]])
    predicate_bias = np.array([[0], [1]])

    first_star = pyn_abst.Star(predicate_matrix, predicate_bias)
    print("FIRST STAR: ")
    print_star_data(first_star)
    second_star = pyn_abst.Star(predicate_matrix, predicate_bias)
    print("\nSECOND STAR: ")
    print_star_data(second_star)

    star_set = pyn_abst.single_sum_forward(first_star, second_star)
    print("\nSUM STAR: ")
    print_star_data(list(star_set)[0])


def test_abst_sum_node():

    first_predicate_matrix = np.array([[-1], [1]])
    first_predicate_bias = np.array([[0], [1]])

    second_predicate_matrix = np.array([[-1], [1]])
    second_predicate_bias = np.array([[-2], [3]])

    print("FIRST STAR: ")
    print_star_data(pyn_abst.Star(first_predicate_matrix, first_predicate_bias))
    print("\nSECOND STAR: ")
    print_star_data(pyn_abst.Star(second_predicate_matrix, second_predicate_bias))

    first_starset = pyn_abst.StarSet({pyn_abst.Star(first_predicate_matrix, first_predicate_bias),
                                      pyn_abst.Star(second_predicate_matrix, second_predicate_bias)})

    second_starset = pyn_abst.StarSet({pyn_abst.Star(first_predicate_matrix, first_predicate_bias),
                                       pyn_abst.Star(second_predicate_matrix, second_predicate_bias)})

    abst_conc_node = pyn_abst.AbsSumNode("temp", pyn_nodes.SumNode("temp", [(1,), (1,)]))

    output_starset = abst_conc_node.forward([first_starset, second_starset])

    print("\nOUTPUT STARSET:")

    for i, star in enumerate(output_starset.stars):
        print(f"\nSTAR {i}:")
        print_star_data(star)


def test_relu():

    first_predicate_matrix = np.array([[-1.0], [1.0]])
    first_predicate_bias = np.array([[0.0], [1.0]])

    print("FIRST STAR: ")
    print_star_data(pyn_abst.Star(first_predicate_matrix, first_predicate_bias))

    first_starset = pyn_abst.StarSet({pyn_abst.Star(first_predicate_matrix, first_predicate_bias)})
    abst_relu_node = pyn_abst.AbsReLUNode("TEST", pyn_nodes.ReLUNode("TEST", (2,)), heuristic="best_n_neurons", params=[2])

    output_starset = abst_relu_node.forward(first_starset)

    print("\nOUTPUT STARSET:")

    for i, star in enumerate(output_starset.stars):
        print(f"\nSTAR {i}:")
        print_star_data(star)


def test_abst_acy_net():

    first_predicate_matrix = np.array([[-1.0], [1.0]])
    first_predicate_bias = np.array([[0.0], [1.0]])

    second_predicate_matrix = np.array([[-1.0], [1.0]])
    second_predicate_bias = np.array([[-2.0], [3.0]])

    print("FIRST STAR: ")
    print_star_data(pyn_abst.Star(first_predicate_matrix, first_predicate_bias))
    print("\nSECOND STAR: ")
    print_star_data(pyn_abst.Star(second_predicate_matrix, second_predicate_bias))

    first_starset = pyn_abst.StarSet({pyn_abst.Star(first_predicate_matrix, first_predicate_bias)},
                                     identifier='input1')

    second_starset = pyn_abst.StarSet({pyn_abst.Star(second_predicate_matrix, second_predicate_bias)},
                                      identifier='input2')

    abs_inputs = [first_starset, second_starset]

    abs_network = pyn_abst.AbsAcyclicNetwork("Test Neural Network", ['input1', 'input2'],
                                             input_edges={'input1': ['root1'], 'input2': ['root2']})

    root_node_1 = pyn_nodes.FullyConnectedNode("root1", (1,), 16)
    abs_root_node_1 = pyn_abst.AbsFullyConnectedNode("root1", root_node_1)
    root_node_2 = pyn_nodes.FullyConnectedNode("root2", (1,), 32)
    abs_root_node_2 = pyn_abst.AbsFullyConnectedNode("root2", root_node_2)

    child_node_1 = pyn_nodes.ReLUNode("inter1", (16,))
    abs_child_node_1 = pyn_abst.AbsReLUNode("inter1", child_node_1, heuristic="best_n_neurons", params=[0])
    child_node_2 = pyn_nodes.ReLUNode("inter2", (32,))
    abs_child_node_2 = pyn_abst.AbsReLUNode("inter2", child_node_2, heuristic="best_n_neurons", params=[0])

    multi_input_node = pyn_nodes.ConcatNode("union", [(16,), (32,)])
    abs_multi_input_node = pyn_abst.AbsConcatNode("union", multi_input_node)

    leaf_node_1 = pyn_nodes.FullyConnectedNode("leaf1", (48,), 8)
    abs_leaf_node_1 = pyn_abst.AbsFullyConnectedNode("leaf1", leaf_node_1)
    leaf_node_2 = pyn_nodes.FullyConnectedNode("leaf2", (48,), 4)
    abs_leaf_node_2 = pyn_abst.AbsFullyConnectedNode("leaf2", leaf_node_2)

    abs_network.add_node(abs_root_node_1)
    abs_network.add_node(abs_root_node_2)

    abs_network.add_node(abs_child_node_1, [abs_root_node_1])
    abs_network.add_node(abs_child_node_2, [abs_root_node_2])

    abs_network.add_node(abs_leaf_node_1)

    abs_network.add_node(abs_multi_input_node, [abs_child_node_1, abs_child_node_2], [abs_leaf_node_1])

    abs_network.add_node(abs_leaf_node_2, [abs_multi_input_node])

    temp_g = networkx.DiGraph(abs_network.edges)
    networkx.draw_networkx(temp_g)
    plt.show()

    output_starsets = abs_network.forward(abs_inputs)

    print("\nOUTPUT STARSETS:")

    for k, output_starset in enumerate(output_starsets):
        print(f"\nSTARSET {k}:")

        for i, star in enumerate(output_starset.stars):
            print(f"\nSTAR {i}:")
            print_star_data(star)


if __name__ == "__main__":
    # test_single_concat()
    # test_abst_concat_node()
    # test_single_sum()
    # test_abst_sum_node()
    test_abst_acy_net()

