# TCL_LP - the LP solution for Thermostatically Controlled Loads
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices from the first to the last [array]
#       P - the power rate [scalar]
#       c_water - specific heat of water(W min/gallon°C) [scalar]
#       m - mass of water in full storage [scalar]
#       temp_up - upper limit of water temperature in storage [scalar]
#       temp_o - initial water temperature in storage [scalar]
#       temp_req - desired / required water temperature [scalar]
#       temp_en - environmental temperature at the i-th time step [array]
#       di - demand of hot water drawn during i-th time step [array]

# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape 2 * (e-b+1,)]
import numpy as np
from scipy import optimize


def TCL_LP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di):
    N = len(pr)
    c = dt * pr
    x_tmp = (N, N)
    x_tmp = np.zeros(x_tmp)
    for i in range(N):
        for j in range(i + 1):
            x_tmp[i, j] = 1
    x_tmp *= dt
    x_tmp2 = np.copy(x_tmp)
    x_tmp2 *= -1
    A_ub = np.concatenate((x_tmp, x_tmp2))
    ub = np.eye(N, dtype=int)
    lb = -1 * ub
    combine_b = np.concatenate((ub, lb))
    A_ub = np.concatenate((A_ub, combine_b))
    C = N
    C = np.zeros(C)
    for i in range(N):
        C[i] = di[i] * c_water * (temp_req - temp_en[i])
    b_ub = (2 * N, 1)
    b_ub = np.zeros(b_ub)
    for i in range(N):
        b_ub[i, 0] = C[i] + m * c_water * (temp_up - temp_o)
        b_ub[i + N, 0] = -1 * C[i]
    # range for each variable
    b_tmp = (2 * N, 1)
    b_tmp = np.zeros(b_tmp)
    for i in range(N):
        b_tmp[i, 0] = P
    b_ub = np.concatenate((b_ub, b_tmp))
    solution = optimize.linprog(c, A_ub, b_ub, method='simplex')
    F = solution.fun
    x = solution.x
    return F, x
