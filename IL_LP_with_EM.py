# IL_LP - the LP solution for the Interruptable type load with Electrical Machinery
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices from pr_b, ..., pr_e [array with shape (e-b+1,)]
#       P - the power rate [scalar]
#       E - the energy [scalar]
#       T_off - the minimum off time [scalar]
#       small_const - small constant used in connecting three z variables 
# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape (e-b+1,)]
#       y - the optimal scheduling ancillary binary [array with shape (e-b+1,)]
#       z - the optimal scheduling variables to help in achieving constraints [[array with shape 3 * (e-b+1,)]]

import numpy as np
from scipy.linalg import toeplitz
from scipy import optimize


def IL_LP(dt, pr, P, E, T_off, small_const):
    # N - the number of variables
    N = len(pr)

    ### Objective function
    # c_x - the sub-vector of c corresponding to power statuses(represented by x)
    c_x = dt * pr
    # c_y - the sub-vector of c corresponding to ancillary binary variables(represented by y)
    c_y = np.zeros(N)
    # c_z - the sub-vector of c corresponding to fractional variables to connect x and y(represented by z)
    c_z = np.zeros(3 * N)
    c = np.concatenate((c_x, c_y, c_z))

    ### Equality constraints
    # A_eq - the matrix for equality constraints
    # A_eq - the array of ones corresponding to the total load consumption throughout the scheduling horizon = E
    # supplemented by the array of zeros corresponding to ancillary binary variables
    A_eq = np.concatenate((np.ones((1, N), dtype=int), np.zeros((1, 4 * N), dtype=int)), axis=1)
    # y_tmp - sub-vector of A_eq to obtain y value at each time step
    y_tmp = np.zeros((N, N), dtype=int)
    np.fill_diagonal(y_tmp, 1)
    # z_tmp1 - sub-vector of A_eq to obtain -(z2 + z3) at each time step
    z_tmp1 = np.zeros((N, 3 * N), dtype=int)
    for i in range(N):
        z_tmp1[i, (i * 3 + 1): ((i + 1) * 3)] = -1
    # tmp1 - coefficient to satisfy equality constraint: y - z2 - z3 = 0
    tmp1 = np.concatenate((np.zeros((N, N), dtype=int), y_tmp, z_tmp1), axis=1)
    # x_tmp - sub-vector of A_eq to obtain x value at each time step
    x_tmp = np.zeros((N, N), dtype=int)
    np.fill_diagonal(x_tmp, 1)
    # z_tmp2 - sub-vector of A_eq to obtain (-z2 * small_const - z3 * P) at each time step
    z_tmp2 = np.zeros((N, 3 * N), dtype=int)
    for i in range(N):
        z_tmp2[i, i * 3 + 1] = -small_const
        z_tmp2[i, i * 3 + 2] = -P
    # tmp2 - coefficient to satisfy equality constraint: x - z2 * small_const - z3 * P = 0
    tmp2 = np.concatenate((x_tmp, np.zeros((N, N), dtype=int), z_tmp2), axis=1)
    # z_tmp3 - sub-vector of A_eq to obtain (z1 + z2 + z3) at each time step
    z_tmp3 = np.zeros((N, 3 * N), dtype=int)
    for i in range(N):
        z_tmp3[i, i * 3: (i + 1) * 3] = 1
    # tmp3 - coefficient to satisfy equality constraint: z1 + z2 + z3 = 1
    tmp3 = np.concatenate((np.zeros((N, 2 * N), dtype=int), z_tmp3), axis=1)
    A_eq = np.concatenate((A_eq, tmp1, tmp2, tmp3))
    # b_eq - the vector for equality constraints
    b_eq = np.zeros((1, 1), dtype=int)
    b_eq[0, 0] = E / dt
    # b_eq_tmp1 - sub-vector of b_eq to satisfy equality constraints: y - z2 - z3 = 0 and x - z2 * small_const - z3 * P = 0
    # b_eq_tmp2 - sub-vector of b_eq to satisfy equality constraints: z1 + z2 + z3 = 1
    b_eq_tmp1 = np.zeros((2 * N, 1), dtype=int)
    b_eq_tmp2 = np.ones((N, 1), dtype=int)
    b_eq = np.concatenate((b_eq, b_eq_tmp1, b_eq_tmp2))

    ### Inequality constraints
    ## Upper and lower bound for each variable
    # A_bound_up - the matrix of upper bounds; A_bound_low - the matrix of lower bounds
    A_bound_up = np.eye(N, dtype=int)
    A_bound_low = -1 * A_bound_up
    # A_ub - the matrix for inequality constraints
    # construct the sub-matrix of A_ub corresponding to upper and lower bounds
    A_ub = np.concatenate((A_bound_up, A_bound_low))
    A_ub = np.concatenate((A_ub, np.zeros(A_ub.shape, dtype=int)), axis=1)
    # b_ub - the vector for inequality constraints
    # b_bound_up - the vector correspond to upper bounds; b_bound_low - the vector correspond to lower bounds;
    b_bound_up = P * np.ones((N, 1), dtype=int)
    b_bound_low = np.zeros((N, 1), dtype=int)
    # construct the sub-vector of b_ub corresponding to upper and lower bounds
    b_ub = np.concatenate((b_bound_up, b_bound_low))

    ## Ancillary minimum off-time constraints
    # construct the matrix for inequalities
    for i in range(T_off - 1):
        # construct the band matrix with elements 1; -1; (corresponding to the changing a switching-off) and
        # 1 on the i-th position (corresponding to a switching-on after i time steps)
        # first_column = np.concatenate(([[1]], np.zeros((N - T_off - i, 1), dtype=int)))
        first_column = np.concatenate(([[1]], np.zeros((N - T_off - 1, 1), dtype=int)))
        first_row = np.concatenate(([1], [-1], np.zeros(i, dtype=int), [1], np.zeros(N - 3 - i, dtype=int)))
        # A_ub_mt - the band matrix
        A_ub_mt = toeplitz(first_column, first_row)
        # supplement A_ub_mt with additional zeros corresponding to power status variables
        A_ub_mt = np.concatenate((np.zeros(A_ub_mt.shape, dtype=int), A_ub_mt), axis=1)
        # join to the A_ub
        A_ub = np.concatenate((A_ub, A_ub_mt))
    # construct the vector for inequalities
    b_row_size = int((N - T_off) * (T_off - 1))
    b_ub = np.concatenate((b_ub, np.ones((b_row_size, 1), dtype=int)))
    # A_ub_row - row number of A_ub yet without considering z elements
    A_ub_row = np.size(A_ub, 0)
    A_ub = np.concatenate((A_ub, np.zeros((A_ub_row, 3 * N), dtype=int)), axis=1)
    ### LP solution
    solution = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, method='simplex')
    # the value of objective function
    F = solution.fun
    # the optimal schedule
    x = solution.x[0:N]
    y = solution.x[N:2 * N]
    z = solution.x[2 * N: 5 * N]

    return [F, x, y, z]
