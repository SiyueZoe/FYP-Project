# IL_LP - the LP solution for Interruptable type load(with y)
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices from pr_b, ..., pr_e [array with shape (e-b+1,)]
#       P - the power rate [scalar]
#       E - the energy [scalar]
#       T_off - the minimum off time[scalar]
# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape 2 * (e-b+1,)]

import numpy as np
from scipy import optimize


def IL_LP(dt, pr, P, E, T_off):
    N = len(pr)
    c1 = dt * pr
    c2 = N
    c2 = np.zeros(c2)
    c = np.concatenate((c1, c2))
    A_eq = (1, 2 * N)
    A_eq = np.ones(A_eq)
    b_eq = E / dt
    # ub: upper bound; lb: lower bound
    ub = np.eye(N, dtype=int)
    lb = -1 * ub
    combine_b = np.concatenate((ub, lb))
    A_ub = (4 * N, 2 * N)
    A_ub = np.zeros(A_ub)
    for i in range(2 * N):
        for j in range(N):
            A_ub[i, j] = combine_b[i, j]
    for i in range(2 * N, 4 * N):
        for j in range(N, 2 * N):
            A_ub[i, j] = combine_b[i - 2 * N, j - N]
    # Constraints related to y
    y = (N, 2 * N)
    y = np.zeros(y)
    for i in range(N):
        y[i, i + N] = 1
    for i in range(2, N - T_off + 2):
        for j in range(i, i + T_off):
            y_tmp = y[i - 1 - 1] - y[i - 1] + y[j - 1]
            y_tmp = np.matrix(y_tmp)
            if i is 2 and j is 2:
                y_cb = np.copy(y_tmp)
            else:
                y_cb = np.concatenate((y_cb, y_tmp))
    A_ub = np.concatenate((A_ub, y_cb))
    xy_tmp = (N, 2 * N)
    xy_tmp = np.zeros(xy_tmp)
    for i in range(0, N):
        xy_tmp[i, i] = 1
        xy_tmp[i, i + N] = -1 * P
    A_ub = np.concatenate((A_ub, xy_tmp))
    b_ub = (4 * N, 1)
    b_ub = np.zeros(b_ub)
    for i in range(N):
        b_ub[i, 0] = P
    for i in range(N, 2 * N):
        b_ub[i, 0] = 0
    for i in range(2 * N, 3 * N):
        b_ub[i, 0] = 1
    for i in range(3 * N, 4 * N):
        b_ub[i, 0] = 0
    b_tmp = ((N - T_off) * T_off, 1)
    b_tmp = np.ones(b_tmp)
    b_ub = np.concatenate((b_ub, b_tmp))
    b_tmp = (N, 1)
    b_tmp = np.zeros(b_tmp)
    b_ub = np.concatenate((b_ub, b_tmp))
    solution = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, method='simplex')
    F = solution.fun
    x = solution.x
    return F, x
