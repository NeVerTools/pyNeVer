import matplotlib.pyplot as plt
import numpy as np
import pypoman
import scipy.spatial

from pynever.strategies.abstraction.star import Star, StarSet


def plot_star(p_star: Star, show=False, title=None, color=np.array([0.5, 0.5, 1])):
    # color = np.random.rand(3)
    alpha_vertexes = pypoman.compute_polytope_vertices(p_star.predicate_matrix, p_star.predicate_bias)
    x_vertexes = []
    for v in alpha_vertexes:
        v_ver = np.zeros((v.shape[0], 1))
        for k in range(v.shape[0]):
            v_ver[k, 0] = v[k]
        new_x_v = p_star.center + np.matmul(p_star.basis_matrix, v_ver)
        x_vertexes.append(new_x_v)

    x1_plot = []
    x2_plot = []
    for v in x_vertexes:
        x1_plot.append(v[0, 0])
        x2_plot.append(v[1, 0])

    for k in range(len(x1_plot)):
        for j in range(len(x1_plot)):
            plt.plot([x1_plot[k], x1_plot[j]], [x2_plot[k], x2_plot[j]], color=color)

    plt.scatter(x1_plot, x2_plot, color=color)
    if show:
        if title is not None:
            plt.title(title)
        plt.grid()


def plot_starset(p_starset: StarSet, title, color=np.array([0.5, 0.5, 1])):
    for p_star in p_starset.stars:
        plot_star(p_star, color=color)

    if title is not None:
        plt.title(title)
    plt.grid()


def get_star_extreme_points(p_star: Star):
    x_vertexes = []
    alpha_vertexes = pypoman.compute_polytope_vertices(p_star.predicate_matrix, p_star.predicate_bias)
    for v in alpha_vertexes:
        v_ver = np.zeros((v.shape[0], 1))
        for k in range(v.shape[0]):
            v_ver[k, 0] = v[k]
        new_x_v = p_star.center + np.matmul(p_star.basis_matrix, v_ver)
        new_x_v = np.array([new_x_v[0, 0], new_x_v[1, 0]])
        x_vertexes.append(new_x_v)

    x_vertexes = np.array(x_vertexes)

    return x_vertexes


def get_starset_extreme_points(p_starset: StarSet):
    x_vertexes = []
    for p_star in p_starset.stars:
        alpha_vertexes = pypoman.compute_polytope_vertices(p_star.predicate_matrix, p_star.predicate_bias)
        for v in alpha_vertexes:
            v_ver = np.zeros((v.shape[0], 1))
            for k in range(v.shape[0]):
                v_ver[k, 0] = v[k]
            new_x_v = p_star.center + np.matmul(p_star.basis_matrix, v_ver)
            new_x_v = np.array([new_x_v[0, 0], new_x_v[1, 0]])
            x_vertexes.append(new_x_v)

    x_vertexes = np.array(x_vertexes)
    return x_vertexes


def plot_convex_hull(points: np.ndarray, title=None, color=None, alpha=1, label=None):
    hull = scipy.spatial.ConvexHull(points)
    vertexes = points[hull.vertices, :]
    vertexes = np.vstack((vertexes, vertexes[0, :]))
    if color is None:
        plt.fill(vertexes[:, 0], vertexes[:, 1], alpha=0.5, label=label)
    else:
        plt.fill(vertexes[:, 0], vertexes[:, 1], color=color, alpha=alpha, label=label)

    if title is not None:
        plt.title(title)
    plt.grid()


def print_star_data(p_star: Star):
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
