# All_without_Energy_limit_MILP - the MILP solution for Three types of loads without adding energy limit
# Input:
#   For All:
#       dt - the time step [scalar]
#       pr - the array of prices from the first to the last [array]
#   For NL:
#       L - the duration of the task [scalar]
#       P_NL - the power rate[scalar]
#   For IL(with EM)
#       P_IL - the power rate [scalar]
#       E_IL - the energy [scalar]
#       T_off - the minimum off time [scalar]
#       Pmin - minimum power to turn on the appliance [scalar]
#   For TCL:
#       P_TCL - the power rate [scalar]
#       c_water - specific heat of water(W min/gallon°C) [scalar]
#       m - mass of water in full storage [scalar]
#       temp_up - upper limit of water temperature in storage [scalar]
#       temp_o - initial water temperature in storage [scalar]
#       temp_req - desired / required water temperature [scalar]
#       temp_en - environmental temperature at the i-th time step [array]
#       di - demand of hot water drawn during i-th time step [array]

# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape (e-b+1,)]

from gurobipy import *
import numpy as np

def ALL_MILP_without_Energy_Limit(dt, pr, L, P_NL, P_IL, E_IL, T_off, Pmin, P_TCL, c_water, m, temp_up, temp_o, temp_req, temp_en, di):
    
    # N - the number of time step
    N = len(pr)
    # x - the list of the variables
    # NL[0:N]; IL(with EM)[N:6N]; TCL[6N:7N]
    x = [None] * N * 7
    # C - heat consumption at each time step
    C = [None] * N
    for i in range(N):
        C[i] = di[i] * c_water * (temp_req - temp_en[i])
    # C_limit - constant value of heat limit
    C_limit = m * c_water * (temp_up - temp_o)

    ## Create a model
    m = Model("ALL_MILP")
    # Create variables
    # NL: x is binary, (P * x) is the actual variable value
    for i in range(N):
        x[i] = m.addVar(vtype=GRB.BINARY)
    # IL:
    for i in range(N, 6 * N):
        # add in x values(power at each time step)
        if N <= i < 2 * N:
            x[i] = m.addVar(lb=0, ub=P_IL, vtype=GRB.CONTINUOUS)
        # add in y values(ancillary binary variables)
        elif 2 * N <= i < 3 * N:
            x[i] = m.addVar(vtype=GRB.BINARY)
        # add in z values(variables that help to achieve step function)
        else:
            x[i] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS)
    # TCL: x is continuous value between 0 and P
    for i in range(6 * N, 7 * N):
        x[i] = m.addVar(lb=0, ub=P_TCL, vtype=GRB.CONTINUOUS)

    ## Integrate new variables
    m.update()

    ## Set objective
    m.setObjective(dt * sum(pr[i] * (P_NL * x[i] + x[i + N] + x[i + 6 * N]) for i in range(N)), GRB.MINIMIZE)

    ## Add constraints
    # NL:
    # Guarantee the non-interruptibility
    for j in range(1, N - L + 1):
        m.addConstr(sum(x[i] for i in range(j, j + L)) >= (x[j] - x[j - 1]) * L)
    m.addConstr(sum(x[i] for i in range(L)) >= (x[0] * L))
    # Meet the required duration
    m.addConstr(sum(x[i] for i in range(N)) == L)
    # IL:
    # Constraint(1): energy constraints
    m.addConstr(sum(x[i] for i in range(N, 2 * N)) == E_IL / dt)
    # Constraint(2): limit frequent switch
    for i in range(N - T_off):
        for j in range(i + 2, i + T_off + 2):
            m.addConstr(x[i + 2 * N] - x[i + 1 + 2 * N] + x[j - 1 + 2 * N] <= 1)
    # Constraint(3): connection between x and y
    for i in range(N):
        m.addConstr(x[i + 2 * N] == x[3 * N + 3 * i + 1] + x[3 * N + 3 * i + 2])
        m.addConstr(x[i + N] == Pmin * x[3 * N + 3 * i + 1] + P_IL * x[3 * N + 3 * i + 2])
        m.addConstr(x[3 * N + 3 * i] + x[3 * N + 3 * i + 1] + x[3 * N + 3 * i + 2] == 1)
    # TCL:
    for i in range(1, N + 1):
        m.addConstr(dt * sum(x[6 * N + j] for j in range(i)) >= sum(C[j] for j in range(i)))
        m.addConstr(dt * sum(x[6 * N + j] for j in range(i)) <= C_limit + sum(C[j] for j in range(i)))

    ## Optimize
    m.optimize()
    
    ## Display
    result = []
    for v in m.getVars():
        result.append(v.x)
    result.append(m.objVal)
    result = np.asarray(result)
    print('\n')
    print('**************Result**************')
    print('NL Power Status: ', P_NL * result[0:N])
    print('IL Power Status: ', result[N:2 * N])
    print('IL Ancillary Binary: ', result[2 * N:3 * N])
    print('TCL Power Status: ', result[6 * N:7 * N])
    print('Total Cost: ', result[7 * N])
    
    return result