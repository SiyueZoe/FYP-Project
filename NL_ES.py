# NL_ES - Non-interruptable load solved with exhaustive search
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices [array]
#       L - the duration of the task [array]
#       P_NL - the power rate[array]
#       NL_b - the beginning time [array]
#       NL_e - the ending time [array]
# Output:
#       total_cost - the total cost of all appliances throughout day
#       result_NL - the power distribution of all appliances
#       total_NL - the total power consumption from all appliances

import numpy as np
from real_time_price import price


def NL_ES(dt, pr, L, P_NL, NL_b, NL_e):
    # N_NL - the number of appliances
    N_NL = len(L)
    # N - the number of time steps
    N = len(pr)
    total_cost = np.repeat(1000000.0, N_NL)
    # start - store the starting point of each appliance
    start = np.zeros(N_NL)
    result_NL = [None] * N_NL
    for m in range(N_NL):
        for i in range(NL_b[m], NL_e[m] - L[m] + 2):
            # cost - temporary value to record the cost of each possibility
            cost = 0
            for j in range(i, i + L[m]):
                cost += pr[j]
            if cost < total_cost[m]:
                total_cost[m] = cost 
                start[m] = i
        total_cost[m] *= P_NL[m] * dt
        result_NL[m] = np.zeros(N)
        for i in range(int(start[m]), int(start[m] + L[m])):
            result_NL[m][i] = P_NL[m]
    total_NL = np.sum(result_NL, axis=0)
    return np.sum(total_cost), result_NL, total_NL
