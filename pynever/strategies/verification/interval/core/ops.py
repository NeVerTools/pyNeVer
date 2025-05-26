"""
This module defines common operations involving intervals
"""

from interval import interval


def interval_from_value(value: float, epsilon: float):
    """Procedure to create an interval from a single value"""
    return interval[value - epsilon, value + epsilon]


def get_positive(a: list):
    """Procedure to extract the positive part of a matrix"""
    if not isinstance(a[0], list):
        aa = [a]
    else:
        aa = a

    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]

    for i in range(len(aa)):
        for j in range(len(aa[i])):
            if aa[i][j][0].sup > 0 > aa[i][j][0].inf:
                result[i][j] = interval[0, aa[i][j][0].sup]
            else:
                result[i][j] = aa[i][j] if aa[i][j][0].inf > 0 else interval[0.0, 0.0]

    return result


def get_negative(a: list):
    """Procedure to extract the negative part of a matrix"""
    if not isinstance(a[0], list):
        aa = [a]
    else:
        aa = a

    result = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]

    for i in range(len(aa)):
        for j in range(len(aa[i])):
            if aa[i][j][0].sup > 0 > aa[i][j][0].inf:
                result[i][j] = interval[aa[i][j][0].inf, 0]
            else:
                result[i][j] = aa[i][j] if aa[i][j][0].sup < 0 else interval[0.0, 0.0]

    return result


def add(a: list, b: list, *args):
    """Procedure to sum lists of intervals"""
    result = []

    if len(a) > 1:
        for i in range(len(a)):
            result.append(a[i][0] + b[i][0])
            for x in args:
                result[i] += x[i][0]

    else:
        for i in range(len(a[0])):
            result.append(a[0][i] + b[0][i])
            for x in args:
                result[i] += x[0][i]

    return result


def matmul_left(a: list, b: list):
    """Procedure to multiply a matrix and a vector"""

    assert len(a[0]) == len(b)

    n = len(a)
    m = len(b)
    q = len(b[0])

    result = [[0 for _ in range(q)] for _ in range(n)]

    for i in range(n):
        for j in range(q):
            s = 0
            for k in range(m):
                s += a[i][k] * b[k]
            result[i][j] = s

    return result


def matmul(a: list, b: list):
    """Procedure to multiply two matrices"""

    assert len(a[0]) == len(b)

    n = len(a)
    m = len(b)
    q = len(b[0])

    result = [[0 for _ in range(q)] for _ in range(n)]

    for i in range(n):
        for j in range(q):
            s = 0
            for k in range(m):
                s += a[i][k] * b[k][j]
            result[i][j] = s

    return result


def create_disjunction_matrix(n_outs: int, label: int):
    """Procedure to create the matrix of the output property"""
    matrix = []
    c = 0

    if n_outs == 1:
        return [[interval[1.0]] if label == 1 else [interval[-1.0]]]

    for i in range(n_outs):
        if i != label:
            matrix.append([interval[0.0] for _ in range(n_outs)])
            matrix[c][label] = interval[1.0]
            matrix[c][i] = interval[-1.0]
            c += 1

    return matrix


def compute_max(weights: list, bounds: dict):
    """Procedure to compute the max value of the bounds through a linear transformation"""
    # The result is a list with one element and I return the element directly
    return add(matmul_left(get_positive(weights), bounds['lower']), matmul_left(get_negative(weights), bounds['upper']))[0]


def compute_min(weights: list, bounds: dict):
    """Procedure to compute the min value of the bounds through a linear transformation"""
    # The result is a list with one element and I return the element directly
    return add(matmul_left(get_positive(weights), bounds['upper']), matmul_left(get_negative(weights), bounds['lower']))[0]


def check_unsafe(bounds: dict, matrix: list, epsilon: float) -> bool:
    """Procedure to check whether the output bounds are unsafe"""

    return compute_min(matrix, bounds)[0].inf <= epsilon


def check_unsafe_symbolic(bounds: dict, matrix: list, lbs: list, ubs: list, epsilon: float) -> bool:
    """Procedure to check whether the output bounds are unsafe"""

    out_m = add(matmul(get_positive(matrix), bounds['matrix']), matmul(get_negative(matrix), bounds['matrix']))
    out_b = add(matmul(get_positive(matrix), bounds['offset']), matmul(get_negative(matrix), bounds['offset']))

    min_value = add(matmul_left(get_positive([out_m]), lbs), matmul_left(get_negative([out_m]), ubs), [out_b])[0]

    return min_value[0].inf <= epsilon
