import time

import numpy as np
from pysmt.shortcuts import Symbol, GE, Real, Equals, Plus, Times, Minus, LE, Solver
from pysmt.typing import REAL

import pynever.strategies.abstraction as abst


def smt_variables(gamma_rows, gamma_cols, beta_rows, lambda_rows, lambda_cols) -> str:
    # Variables --- Gamma
    smt_string = ''
    for i in range(gamma_rows):
        for j in range(gamma_cols):
            smt_string += '(declare-fun '
            smt_string += f"gamma_{i}_{j}"
            smt_string += ' () Real)\n'
    smt_string += '\n'

    # Variables --- beta
    for i in range(beta_rows):
        smt_string += '(declare-fun '
        smt_string += f"beta_{i}"
        smt_string += ' () Real)\n'
    smt_string += '\n'

    # Variables --- Lambda
    for i in range(lambda_rows):
        for j in range(lambda_cols):
            smt_string += '(declare-fun '
            smt_string += f"lambda_{i}_{j}"
            smt_string += ' () Real)\n'
    smt_string += '\n'

    return smt_string


def to_smt_containment(inbody: abst.Star, circumbody: abst.Star) -> tuple:
    # Fast access
    Vx, Vy = inbody.basis_matrix, circumbody.basis_matrix
    cx, cy = inbody.center, circumbody.center
    Cx, Cy = inbody.predicate_matrix, circumbody.predicate_matrix
    dx, dy = inbody.predicate_bias, circumbody.predicate_bias
    m = Cx.shape[1]
    qx, qy = Cx.shape[0], Cy.shape[0]

    # --- VARIABLES ---
    variable_list = []
    gamma_vars = [[Symbol(f"gamma_{j}_{i}", REAL) for i in range(m)]
                  for j in range(m)]
    variable_list.append(item for sublist in gamma_vars for item in sublist)

    beta_vars = [Symbol(f"beta_{j}", REAL) for j in range(m)]
    variable_list.append(beta_vars)

    lambda_vars = [[Symbol(f"lambda_{j}_{i}", REAL) for i in range(qy)]
                   for j in range(qx)]
    variable_list.append(item for sublist in lambda_vars for item in sublist)

    # --- CONSTRAINTS ---
    constraint_list = []

    lambda_domain = [[GE(var, Real(0)) for var in row]
                     for row in lambda_vars]
    constraint_list.append(item for sublist in lambda_domain for item in sublist)

    gamma_basis = [[Equals(Real(float(Vx[j, i])),
                           Plus(*[Times(Real(float(Vy[j, k])), gamma_vars[k][i]) for k in range(m)]))
                    for i in range(m)]
                   for j in range(m)]
    constraint_list.append(item for sublist in gamma_basis for item in sublist)

    beta_center = [Equals(Minus(Real(float(cy[i])), Real(float(cx[i]))),
                          Plus(*[Times(Real(float(Vy[i, k])), beta_vars[k]) for k in range(m)]))
                   for i in range(m)]
    constraint_list.append(beta_center)

    lambda_gamma_predicates = [[Equals(Plus(*[Times(lambda_vars[j][k], Real(float(Cx[k, i]))) for k in range(qx)]),
                                       Plus(*[Times(Real(float(Cy[j, k])), gamma_vars[k][i]) for k in range(m)]))
                                for i in range(m)]
                               for j in range(qy)]
    constraint_list.append(item for sublist in lambda_gamma_predicates for item in sublist)

    lambda_beta_bias = [LE(Plus(*[Times(lambda_vars[i][k], Real(float(dx[k]))) for k in range(qx)]),
                           Plus(Real(float(dy[i])),
                                Plus(*[Times(Real(float(Cy[i, k])), beta_vars[k]) for k in range(m)])))
                        for i in range(qy)]
    constraint_list.append(lambda_beta_bias)

    return [item for sublist in variable_list for item in sublist], \
           [item for sublist in constraint_list for item in sublist]


# INPUT STARSET DEFINITION
C_1 = np.zeros((5, 2))
C_1[0, 0] = -1
C_1[1, 1] = -1
C_1[2, 0] = 1
C_1[2, 1] = 1
C_1[3, 0] = -1
C_1[4, 1] = -1

d_1 = np.ones((5, 1))
d_1[3, 0] = 0
d_1[4, 0] = 0

star_1 = abst.Star(C_1, d_1)

C_2 = np.zeros((5, 2))
C_2[0, 0] = -1
C_2[1, 1] = -1
C_2[2, 0] = 1
C_2[2, 1] = 1
C_2[3, 0] = 1
C_2[4, 1] = -1

d_2 = np.ones((5, 1))
d_2[3, 0] = 0
d_2[4, 0] = 0

star_2 = abst.Star(C_2, d_2)
star_2.basis_matrix[0, 0] = 0
star_2.basis_matrix[0, 1] = 0
star_2.basis_matrix[1, 0] = 0
star_2.basis_matrix[1, 1] = 0

start = time.time()
variables, constraints = to_smt_containment(star_2, star_1)
print(f"Total variables: {len(variables)}, total constrainst = {len(constraints)}")

with Solver(name='z3') as s:
    for c in constraints:
        s.add_assertion(c)

    res = s.solve()
    print("Containment check:", res, "in", time.time() - start)
