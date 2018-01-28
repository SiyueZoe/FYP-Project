# IL_LP - the LP solution for Interruptable type load(without y)
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices from pr_b, ..., pr_e [array with shape (e-b+1,)]
#       P - the power rate [scalar]
#       E - the energy [scalar]
# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape (e-b+1,)]

import numpy as np
from scipy import optimize


def IL_LP(dt, pr, P, E):
    N = len(pr)
#    c = N * dt * pr
    c = dt * pr
    A_eq = (1, N)
    A_eq = np.ones(A_eq)
    b_eq = E / dt
    #ub: upper bound; lb: lower bound
    ub = np.eye(N, dtype=int)
    lb = -1 * ub
    A_ub = np.concatenate((ub, lb))
    b_ub = (2 * N, 1)
    b_ub = np.zeros(b_ub)
    for i in range(N):
        b_ub[i, 0] = P
    for i in range(N, 2 * N):
        b_ub[i, 0] = 0
    solution = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, method='simplex')
    F = solution.fun
    x = solution.x
    return F, x
