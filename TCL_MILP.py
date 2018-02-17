
# TCL_MILP - the MILP solution for Thermostatically Controlled Loads
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
#       x - the optimal schedule [array with shape (e-b+1,)]

from gurobipy import *


def TCL_MILP(dt, pr, P, c_water, m, temp_up, temp_o, temp_req, temp_en, di):
    # N - the number of variables
    N = len(pr)
    # x - the list of the variables
    x = [None] * N
    # C - heat consumption at each time step
    C = [None] * N
    for i in range(N):
        C[i] = di[i] * c_water * (temp_req - temp_en[i])
    # C_limit - constant value of heat limit
    C_limit = m * c_water * (temp_up - temp_o)

    # Create a model
    m = Model("TCL_MILP")
    # Create variables - x is continuous value between 0 and P
    for i in range(N):
        x[i] = m.addVar(lb=0, ub=P, vtype=GRB.CONTINUOUS)

    # Integrate new variables
    m.update()

    # Set objective
    m.setObjective(sum(dt * pr[i] * x[i] for i in range(N)), GRB.MINIMIZE)

    # Add constraints
    for i in range(1, N + 1):
        m.addConstr(dt * sum(x[j] for j in range(i)) >= sum(C[j] for j in range(i)))
        m.addConstr(dt * sum(x[j] for j in range(i)) <= C_limit + sum(C[j] for j in range(i)))

    # Optimize
    m.optimize()
    for v in m.getVars():
        print(v.varName, v.x)
    print('Obj:', m.objVal)
