
import pytest


@pytest.mark.parametrize("num_nodes,cluster_dim", generate_optimized_test_params())
def test_one_way_gtsp_algorithm(num_nodes, cluster_dim):
    """
    Test the One-Way GTSP algorithm using both exact and heuristic methods.

    This test compares the performance of the exact approach (using Integer Programming) and the heuristic
    approach (using Nearest Neighbor) for solving the One-Way GTSP. The objective is to verify that the optimal
    solution (from `GraphPathSolver`) has a cost that is less than or equal to the suboptimal solution (from
    `NearestNeighbour`).

    Parameters:
    - num_nodes (int): The number of nodes in the graph.
    - cluster_dim (int): The number of nodes per cluster.

    The function performs the following steps:
    1. Generates a random graph with the specified number of nodes and cluster dimensions.
    2. Solves the GTSP using the exact `GraphPathSolver` method to obtain the optimal path and cost.
    3. Solves the GTSP using the heuristic `NearestNeighbour` method to obtain the suboptimal path and cost.
    4. Validates that both the optimal and suboptimal paths respect the cluster constraints.
    5. Asserts that the optimal cost is less than or equal to the suboptimal cost.
    6. Ensures that both the optimal and suboptimal costs are positive values.

    This test is parameterized to run with multiple combinations of node and cluster sizes.
    """
    matrix, colors_dict = generate_random_graphs(num_nodes, cluster_dim)
    solver = GraphPathSolver(matrix, colors_dict)
    optimal_path, optimal_cost = solver.launch_ip_problem()
    suboptimal_solver = NearestNeighbour(matrix, colors_dict)
    suboptimal_path, suboptimal_cost = suboptimal_solver.short_path_heuristic()

    print(f" {num_nodes=} {cluster_dim=} {optimal_cost=} {suboptimal_cost=}")

    # Check that paths respect the constraint that exactly one node from each cluster must be in the path
    validate_clusters(optimal_path, colors_dict)
    validate_clusters(suboptimal_path, colors_dict)

    assert optimal_cost <= suboptimal_cost, "Optimal cost should be less than or equal to suboptimal cost"
    assert optimal_cost > 0, "The optimal cost must be positive"
    assert suboptimal_cost > 0, "The suboptimal cost must be positive"