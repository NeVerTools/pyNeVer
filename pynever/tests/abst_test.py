
import numpy as np

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



if __name__ == "__main__":
    test_single_concat()
    test_abst_concat_node()
    print("\n\n\n")
    test_single_sum()
    test_abst_sum_node()
    # test_relu()

