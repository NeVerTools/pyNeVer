import numpy as np

from pynever.strategies.abstraction.star import Star
from pynever.strategies.bp.bounds import AbstractBounds
from pynever.tensors import Tensor


def check_stable(var_index: int, bounds: AbstractBounds) -> int:
    """

    Parameters
    ----------
    var_index
    bounds

    Returns
    -------
    0 if unstable, 1 if positive stable, -1 if negative stable

    """

    precision_guard = 10e-15

    lb = bounds.get_lower()[var_index]
    ub = bounds.get_upper()[var_index]

    # Positive stable
    if lb >= precision_guard:
        return 1

    # Negative stable
    elif ub <= -precision_guard:
        return -1

    # Unstable
    else:
        return 0


def intersect_with_halfspace(star: Star, coef_mat: Tensor, bias_mat: Tensor) -> Star:
    """
    Function which takes as input a Star and a halfspace defined by its coefficient matrix and bias vector
    and returns the Star resulting from the intersection of the input Star with the halfspace.

    """

    new_center = star.center
    new_basis_matrix = star.basis_matrix
    hs_pred_matrix = np.matmul(coef_mat, star.basis_matrix)
    hs_pred_bias = bias_mat - np.matmul(coef_mat, star.center)
    new_pred_matrix = np.vstack((star.predicate_matrix, hs_pred_matrix))
    new_pred_bias = np.vstack((star.predicate_bias, hs_pred_bias))

    new_star = Star(new_pred_matrix, new_pred_bias, new_center, new_basis_matrix)

    return new_star


def approx_relu_forward(star: Star, bounds: AbstractBounds, dim: int, start_idx: int = 0) -> Star:
    """
    Approximate abstract propagation for a ReLU layer starting from a
    specific index

    Parameters
    ----------
    star : Star
        The star to propagate in this layer
    bounds : AbstractBounds
        The bounds of this layer
    dim : int
        The number of neurons in this layer
    start_idx : int
        The neuron to start the propagation from

    Returns
    ----------
    Star
        The abstract star result from the propagation

    """

    out_star = star

    for i in range(start_idx, dim):
        # i is the number of neurons to process
        stable = check_stable(i, bounds)
        mask = np.identity(out_star.center.shape[0])
        mask[i, i] = 0
        lb = bounds.get_lower()[i]
        ub = bounds.get_upper()[i]

        # Positive stable
        if stable == 1:
            continue

        # Negative stable
        elif stable == -1:
            new_c = np.matmul(mask, out_star.center)
            new_b = np.matmul(mask, out_star.basis_matrix)
            new_pred = out_star.predicate_matrix
            new_bias = out_star.predicate_bias

            out_star = Star(new_pred, new_bias, new_c, new_b)

        # Unstable
        else:
            col_c_mat = out_star.predicate_matrix.shape[1]
            row_c_mat = out_star.predicate_matrix.shape[0]

            c_mat_1 = np.zeros((1, col_c_mat + 1))
            c_mat_1[0, col_c_mat] = -1
            c_mat_2 = np.hstack((np.array([out_star.basis_matrix[i, :]]), -np.ones((1, 1))))
            coef_3 = - ub / (ub - lb)
            c_mat_3 = np.hstack((np.array([coef_3 * out_star.basis_matrix[i, :]]), np.ones((1, 1))))
            c_mat_0 = np.hstack((out_star.predicate_matrix, np.zeros((row_c_mat, 1))))

            d_0 = out_star.predicate_bias
            d_1 = np.zeros((1, 1))
            d_2 = -out_star.center[i] * np.ones((1, 1))
            d_3 = np.array([(ub / (ub - lb)) * (out_star.center[i] - lb)])

            new_pred_mat = np.vstack((c_mat_0, c_mat_1, c_mat_2, c_mat_3))
            new_pred_bias = np.vstack((d_0, d_1, d_2, d_3))

            new_center = np.matmul(mask, out_star.center)
            temp_basis_mat = np.matmul(mask, out_star.basis_matrix)
            temp_vec = np.zeros((out_star.basis_matrix.shape[0], 1))
            temp_vec[i, 0] = 1
            new_basis_mat = np.hstack((temp_basis_mat, temp_vec))

            out_star = Star(new_pred_mat, new_pred_bias, new_center, new_basis_mat)

    return out_star

