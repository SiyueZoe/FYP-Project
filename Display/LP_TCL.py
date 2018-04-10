# LP_TCL - the LP solution for Thermostatically Controlled Loads
# Input:
#       dt - the time step [scalar]
#       pr - the real time price with unit of $/kWmin [vector]
#       P - the power rate with unit of kWmin [scalar]
#       c_water - specific heat of water with unit of W min/gallon°C [scalar]
#       m - mass of water in full storage with unit of gallon [scalar]
#       temp_up - upper limit of water temperature in storage with unit of °C [scalar]
#       temp_o - initial water temperature in storage with unit of °C [scalar]
#       temp_req - desired / required water temperature with unit of °C [scalar]
#       temp_en - environmental temperature at the i-th time step with unit of °C [vector]
#       di - demand of hot water drawn during i-th time step with unit of gallon [vector]

# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [vector]
import numpy as np
from scipy import optimize


def LP_TCL(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di):
    # N - the number of variables
    N = len(pr)

    ## objective function
    # c - array of costs
    c = dt * pr

    ## Inequality Constraints
    # x_tmp - to get the first i-s accumulated energy
    x_tmp = dt * np.ones((N, N))
    x_tmp = np.tril(x_tmp, 0)
    # x_tmp2 - opposite sign of x_tmp
    x_tmp2 = -1 * x_tmp
    A_ub = np.concatenate((x_tmp, x_tmp2))
    # ub - to obtain the power value
    ub = np.eye(N)
    # lb - opposite sign of ub
    lb = -1 * ub
    # A_ub - coefficient in inequality A_ub * x <= b_ub
    A_ub = np.concatenate((A_ub, ub, lb))
    # C - heat consumption at each time step
    C = np.zeros(N)
    for i in range(N):
        C[i] = di[i] * c_water * (temp_req - temp_en[i])
    b_ub = np.zeros((2 * N, 1))
    for i in range(N):
        # b_ub[i, 0] - lower bound of energy accumulated at each time step
        b_ub[i, 0] = sum(C[0:i + 1]) + m * c_water * (temp_up - temp_o)
        # b_ub[i + N, 0] - upper bound of energy accumulated at each time step
        b_ub[i + N, 0] = -1 * sum(C[0:i + 1])
    # b_tmp - upper bound and then lower bound for power value at each time step
    b_tmp = np.zeros((2 * N, 1))
    b_tmp[0:N] = P
    # b_ub - value in inequality A_ub * x <= b_ub
    b_ub = np.concatenate((b_ub, b_tmp))

    ## LP solution
    solution = optimize.linprog(c, A_ub, b_ub, method='simplex')
    # the value of objective function
    F = solution.fun
    # the optimal schedule
    x = solution.x

    ## Display
    lb_total = b_ub[0:N]
    ub_total = -b_ub[N:2 * N]
    tmp = np.dot(A_ub[0:N], x)
    print('************************Result************************')
    print('Power Status: ', x)
    print('Total Cost: ', F)
    return F, x, lb_total, ub_total, tmp