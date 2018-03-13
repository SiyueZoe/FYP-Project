# Real_Case - the MILP solution for scheduling 1 apartment with 10 residences in 24 hours
# Input:
#   For All:
#       dt - the time step [scalar]
#       pr - the array of prices [array]
#       DR - demand response with unit of kW*min [scalar]
#       DR_b - the beginning time of demand response [scalar]
#       DR_e - the ending time of demand response [scalar]
#   For NL:
#       L - the duration of the task [array]
#       P_NL - the power rate[array]
#       NL_b - the beginning time [array]
#       NL_e - the ending time [array]
#   For IL(with EM):
#       P_IL - the power rate [array]
#       E_IL - the energy [array]
#       T_off - the minimum length of off time-step [array]
#       Pmin - minimum power to turn on the appliance [array]
#       IL_b - the beginning time [array]
#       IL_e - the ending time [array]
#   For TCL:
#       P_TCL - the power rate [array]
#       c_water - specific heat of water(kW min/gallon°C) [scalar]
#       m - mass of water in full storage [array]
#       temp_up - upper limit of water temperature in storage [array]
#       temp_o - initial water temperature in storage [array]
#       temp_req - desired / required water temperature [array]
#       temp_en - environmental temperature at the i-th time step [array]
#       di - demand of hot water drawn during i-th time step [array]

# Output:
#       m.objVal - the value of objective function [scalar]
#       P_NL * result[0:N] - the optimal schedule for NL [array]
#       result[N:2 * N] - the optimal schedule for IL [array]
#       result[6 * N: 7 * N] - the optimal schedule for TCL [array]

from gurobipy import *
import numpy as np


def Real_Case(pr, N, dt, DR, DR_b, DR_e, L, P_NL, NL_b, NL_e, P_IL, E_IL, T_off, Pmin, IL_b, IL_e, P_TCL, c_water,
              mass, temp_up, temp_o, temp_req, temp_en, di):
    ## Parameter
    # N_NL - the number of NL
    N_NL = len(P_NL)
    # N_IL - the number of IL
    N_IL = len(P_IL)
    # N_TCL - the number of TCL
    N_TCL = len(P_TCL)
    # N_all - the number of all variables
    N_all = N_NL + 5 * N_IL + N_TCL
    # x - the list of the variables
    # NL[0:N_NL * N]; IL(with EM)[N_NL * N:N_NL * N + N_IL * 5N]; TCL[N_NL * N + N_IL * 5N:N_NL * N + N_IL * 5N + N_TCL * N]
    x = [None] * N * N_all
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
    # DR_vector - array to help obtain total power of all appliances within demand response time
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
    m = Model("ALL_MILP")
    # Create variables
    # NL: x is binary, (P * x) is the actual variable value
    for i in range(N_NL * N):
        x[i] = m.addVar(vtype=GRB.BINARY)
    # IL:
    for i in range(N_IL):
        for j in range(N_NL * N + i * 5 * N, N_NL * N + (i + 1) * 5 * N):
            # add in x values(power at each time step)
            if N_NL * N + i * 5 * N <= j < N_NL * N + i * 5 * N + N:
                x[j] = m.addVar(lb=0, ub=P_IL[i], vtype=GRB.CONTINUOUS)
            # add in y values(ancillary binary variables)
            elif N_NL * N + i * 5 * N + N <= j < N_NL * N + i * 5 * N + 2 * N:
                x[j] = m.addVar(vtype=GRB.BINARY)
            # add in z values(variables that help to achieve step function)
            else:
                x[j] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS)
    # TCL: x is continuous value between 0 and P
    for i in range(N_TCL):
        for j in range((N_NL + N_IL * 5 + i) * N, (N_NL + N_IL * 5 + i + 1) * N):
            x[j] = m.addVar(lb=0, ub=P_TCL[i], vtype=GRB.CONTINUOUS)

    ## Integrate new variables
    m.update()

    ## Set objective
    m.setObjective(sum(dt * x[i] * pr2[i] for i in range(N * N_all)), GRB.MINIMIZE)

    ## Add constraints
    # All:
    # Demand Response
    m.addConstr(sum(dt * x[i] * DR_vector[i] for i in range(N * N_all)) <= DR)
    # NL:
    for i in range(N_NL):
        # Constraint(1): Guarantee the non-interruptibility
        for j in range(1, N - L[i] + 1):
            m.addConstr(
                sum(x[k] for k in range(j + i * N, j + L[i] + i * N)) >= (x[j + i * N] - x[j - 1 + i * N]) * L[i])
        m.addConstr(sum(x[j + i * N] for j in range(L[i])) >= (x[i * N] * L[i]))
        # Constraint(1): Meet the required duration
        m.addConstr(sum(x[j + i * N] for j in range(N)) == L[i])
        # Constraint(1): Restrict the power within the allowable time
        for j in range(i * N, (i + 1) * N):
            if j < NL_b[i] + i * N or j > NL_e[i] + i * N:
                m.addConstr(x[j] == 0)
    # IL:
    for i in range(N_IL):
        # Constraint(1):
        m.addConstr(sum(x[j] for j in range(5 * N * i + N * N_NL, N * (5 * i + 1) + N * N_NL)) == E_IL[i] / dt)
        # Constraint(2): limit frequent switch
        for j in range(N - T_off[i]):
            for k in range(j + 2, j + T_off[i] + 2):
                m.addConstr(x[j + N * (5 * i + 1) + N * N_NL] - x[j + 1 + N * (5 * i + 1) + N * N_NL] + x[
                    k - 1 + N * (5 * i + 1) + N * N_NL] <= 1)
        # Constraint(3): connection between x and y
        for j in range(N):
            m.addConstr(x[j + N * (5 * i + 1) + N * N_NL] == x[3 * j + 1 + N * (5 * i + 2) + N * N_NL] + x[
                3 * j + 2 + N * (5 * i + 2) + N * N_NL])
            m.addConstr(x[j + 5 * N * i + N * N_NL] - Pmin[i] * x[3 * j + 1 + N * (5 * i + 2) + N * N_NL] - P_IL[i] * x[
                3 * j + 2 + N * (5 * i + 2) + N * N_NL] == 0)
            m.addConstr(x[3 * j + N * (5 * i + 2) + N * N_NL] + x[3 * j + 1 + N * (5 * i + 2) + N * N_NL] + x[
                3 * j + 2 + N * (5 * i + 2) + N * N_NL] == 1)
        # Constraint(4): Restrict the power within the allowable time
        for j in range((i * 5 + N_NL) * N, (i * 5 + 1 + N_NL) * N):
            if j < IL_b[i] + (i * 5 + N_NL) * N or j > IL_e[i] + (i * 5 + N_NL) * N:
                m.addConstr(x[j] == 0)
    # TCL:
    for i in range(N_TCL):
        for j in range(1, N + 1):
            m.addConstr(
                dt * sum(x[N * (N_NL + 5 * N_IL + i) + k] for k in range(j)) - sum(C[i][k] for k in range(j)) >= 0)
            m.addConstr(
                dt * sum(x[N * (N_NL + 5 * N_IL + i) + k] for k in range(j)) - C_limit[i] - sum(
                    C[i][k] for k in range(j)) <= 0)

    ## Optimize
    m.optimize()

    ## Display
    result = []
    for v in m.getVars():
        result.append(v.x)
    result = np.asarray(result)
    print('Total Cost: ', m.objVal)
    print('*****************Result*****************')
    for i in range(N_NL):
        print('NL (%d): ' % (i + 1), P_NL[i] * result[i * N: (i + 1) * N])
    for i in range(N_IL):
        print('IL (%d): ' % (i + 1), result[N * N_NL + i * 5 * N:N * N_NL + (i * 5 + 1) * N])
    for i in range(N_TCL):
        print('TCL (%d): ' % (i + 1), result[N * (N_NL + 5 * N_IL) + i * N:N * (N_NL + 5 * N_IL) + (i + 1) * N])
