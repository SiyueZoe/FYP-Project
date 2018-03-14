# Real_Case_Pymprog - the MILP solution for scheduling 1 apartment with 10 residences in 24 hours using Pymprog
# Input:
#   For All:
#       dt - the time step [scalar]
#       pr - the array of prices [array]
#       DR - demand response with unit of kW*min [scalar]
#       DR_b - the beginning time of demand response [scalar]
#       DR_e - the ending time of demand response [scalar]
#   For NL: (1. Clothes Washer; 2. Clothes Dryer; 3. Dishwasher)
#       L - the duration of the task [array]
#       P_NL - the power rate[array]
#       NL_b - the beginning time [array]
#       NL_e - the ending time [array]
#   For IL(with EM): (1. AC)
#       P_IL - the power rate [array]
#       E_IL - the energy [array]
#       T_off - the minimum length of off time-step [array]
#       Pmin - minimum power to turn on the appliance [array]
#       IL_b - the beginning time [array]
#       IL_e - the ending time [array]
#   For TCL: (1. Water Heater)
#       P_TCL - the power rate [array]
#       c_water - specific heat of water(kW min/gallon°C) [scalar]
#       m - mass of water in full storage [array]
#       temp_up - upper limit of water temperature in storage [array]
#       temp_o - initial water temperature in storage [array]
#       temp_req - desired / required water temperature [array]
#       temp_en - environmental temperature at the i-th time step [array]
#       di - demand of hot water drawn during i-th time step [array]

# Output:


from pymprog import *
import numpy as np
from real_time_price import price

def Real_Case(pr, N, dt, DR, DR_b, DR_e, L, P_NL, NL_b, NL_e, P_IL, E_IL, T_off, Pmin, IL_b, IL_e, P_TCL, c_water,
              mass, temp_up, temp_o, temp_req, temp_en, di):

    ## Parameter Setting
    # N_NL - the number of NL
    N_NL = len(P_NL)
    # N_IL - the number of IL
    N_IL = len(P_IL)
    # N_TCL - the number of TCL
    N_TCL = len(P_TCL)
    # N_all - the number of all variables
    N_all = N_NL + 5 * N_IL + N_TCL
    # C - heat consumption at each time step
    C = np.zeros((N_TCL, N))
    for i in range(N_TCL):
        for j in range(N):
            C[i][j] = di[i][j] * c_water * (temp_req[i] - temp_en[i][j])
    # C_limit - constant value of heat limit
    C_limit = np.zeros(N_TCL)
    for i in range(N_TCL):
        C_limit[i] = mass[i] * c_water * (temp_up[i] - temp_o[i])

    ## Useful arrays prepared
    # pr2 - expanded price array to fit array with variables to obtain objective function
    # 1) pr2 Expanded for NL
    pr2 = np.tile(pr, N_NL)
    tmp_NL = np.repeat(P_NL, N)
    pr2 = pr2 * tmp_NL
    # 2) pr2 Expanded for last two parts of IL
    tmp_IL = np.concatenate((pr, np.zeros(4 * N)))
    tmp_IL = np.tile(tmp_IL, N_IL)
    # 3) pr2 Expanded for TCL
    tmp_TCL = np.tile(pr, N_TCL)
    pr2 = np.concatenate((pr2, tmp_IL, tmp_TCL))
    # DR_vector - array to help obtain total energy of all appliances within demand response time
    DR_vector = np.zeros(N_all * N)
    # NL part in DR_vector
    for i in range(N_NL):
        for j in range(DR_b + i * N, DR_e + i * N + 1):
            DR_vector[j] = P_NL[i]
    # IL part in DR_vector
    for i in range(N_IL):
        for j in range(DR_b + N * (N_NL + i * 5), DR_e + N * (N_NL + i * 5) + 1):
            DR_vector[j] = 1
    # TCL part in DR_vector:
    for i in range(N_TCL):
        for j in range(DR_b + N * (N_NL + 5 * N_IL + i), DR_e + N * (N_NL + 5 * N_IL + i) + 1):
            DR_vector[j] = 1

    ## Create a model
    begin('MILP_pymprog')

    ## Create variables
    # x_NL: NL part of variables. x is binary, (P * x) is the actual variable value
    x_NL = var('x_NL', N * N_NL, bool)
    # x_IL: IL part of variables.
    x_IL_tmp = [None] * 3 * N_IL
    x_IL = []
    for i in range(3 * N_IL):
        # Power values in IL
        if i % 3 is 0:
            x_IL_tmp[i] = var('x_IL', N, bounds=(0, P_IL[int(i / 3)]))
        # Ancillary binary in IL
        elif i % 3 is 1:
            x_IL_tmp[i] = var('x_IL', N, bool)
        # Helpful z values
        else:
            x_IL_tmp[i] = var('x_IL', 3 * N, bounds=(0, 1))
        x_IL += x_IL_tmp[i]
    # x_TCL: TCL part of variables.
    x_TCL_tmp = [None] * N_TCL
    x_TCL = []
    for i in range(N_TCL):
        x_TCL_tmp[i] = var('x_TCL', N, bounds=(0, P_TCL[i]))
        x_TCL += x_TCL_tmp[i]
    # Combine all variables into one
    x = x_NL + x_IL + x_TCL

    ## Objective Function
    minimize(sum(dt * x[i] * pr2[i] for i in range(N * N_all)), 'cost')

    ## Constraints
    # All:
    # Demand Response
    sum(x[i] * DR_vector[i] for i in range(N * N_all)) <= DR
    # NL:
    for i in range(N_NL):
        # Constraint(1): Guarantee the non-interruptibility
        for j in range(1, N - L[i] + 1):
            sum(x[k] for k in range(j + i * N, j + L[i] + i * N)) >= (x[j + i * N] - x[j - 1 + i * N]) * L[i]
        sum(x[j + i * N] for j in range(L[i])) >= (x[i * N] * L[i])
        # Constraint(1): Meet the required duration
        sum(x[j + i * N] for j in range(N)) == L[i]
        # Constraint(2): Restrict the power within the allowable time
        for j in range(i * N, (i + 1) * N):
            if j < NL_b[i] + i * N or j > NL_e[i] + i * N:
                x[j] == 0
    # IL:
    for i in range(N_IL):
        # Constraint(1):
        sum(x[j] for j in range(5 * N * i + N * N_NL, N * (5 * i + 1) + N * N_NL)) == E_IL[i] / dt
        # Constraint(2): limit frequent switch
        for j in range(N - T_off[i]):
            for k in range(j + 2, j + T_off[i] + 2):
                x[j + N * (5 * i + 1) + N * N_NL] - x[j + 1 + N * (5 * i + 1) + N * N_NL] + x[
                    k - 1 + N * (5 * i + 1) + N * N_NL] <= 1
        # Constraint(3): connection between x and y
        for j in range(N):
            x[j + N * (5 * i + 1) + N * N_NL] == x[3 * j + 1 + N * (5 * i + 2) + N * N_NL] + x[
                3 * j + 2 + N * (5 * i + 2) + N * N_NL]
            x[j + 5 * N * i + N * N_NL] - Pmin[i] * x[3 * j + 1 + N * (5 * i + 2) + N * N_NL] - P_IL[i] * x[
                3 * j + 2 + N * (5 * i + 2) + N * N_NL] == 0
            x[3 * j + N * (5 * i + 2) + N * N_NL] + x[3 * j + 1 + N * (5 * i + 2) + N * N_NL] + x[
                3 * j + 2 + N * (5 * i + 2) + N * N_NL] == 1
        # Constraint(4): Restrict the power within the allowable time
        for j in range((i * 5 + N_NL) * N, (i * 5 + 1 + N_NL) * N):
            if j < IL_b[i] + (i * 5 + N_NL) * N or j > IL_e[i] + (i * 5 + N_NL) * N:
                x[j] == 0
    # TCL:
    for i in range(N_TCL):
        for j in range(1, N + 1):
            dt * sum(x[N * (N_NL + 5 * N_IL + i) + k] for k in range(j)) - sum(C[i][k] for k in range(j)) >= 0
            dt * sum(x[N * (N_NL + 5 * N_IL + i) + k] for k in range(j)) - C_limit[i] - sum(C[i][k] for k in range(j)) <= 0

    ## Optimize
    solve()

    ## Display
    result = []
    result_NL = np.zeros(N)
    result_IL = np.zeros(N)
    result_TCL = np.zeros(N)
    result_ALL = np.zeros(N)
    for i in range(N * N_all):
        result.append(x[i].primal)
    result = np.asarray(result)
    print('*****************Result*****************')
    print('Total Cost: ', vobj())
    for i in range(N_NL):
        result_NL += P_NL[i] * result[i * N: (i + 1) * N]
        print('NL (%d): ' % (i + 1), P_NL[i] * result[i * N: (i + 1) * N])
    for i in range(N_IL):
        result_IL += result[N * N_NL + i * 5 * N:N * N_NL + (i * 5 + 1) * N]
        print('IL (%d): ' % (i + 1), result[N * N_NL + i * 5 * N:N * N_NL + (i * 5 + 1) * N])
    for i in range(N_TCL):
        result_TCL += result[N * (N_NL + 5 * N_IL) + i * N:N * (N_NL + 5 * N_IL) + (i + 1) * N]
        print('TCL (%d): ' % (i + 1), result[N * (N_NL + 5 * N_IL) + i * N:N * (N_NL + 5 * N_IL) + (i + 1) * N])

    result_ALL += result_NL + result_IL + result_TCL

    return result_NL, result_IL, result_TCL, result_ALL
