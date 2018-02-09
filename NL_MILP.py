# NL_MILP - the MILP solution for Non-Interruptable type load
# Input:
#      dt - the time step [scalar]
#      pr - the array of prices from p_b, ..., p_e [array with shape (e-b+1,)]
#      L - the duration of the task [scalar]
#      P - the power rate [scalar]
# Output:
#      F - the value of objective function [scalar]
#      x - the optimal schedule [array with shape (e-b+1,)]

from gurobipy import *


def NL_MILP(dt, pr, L, P):
    # N - the number of variables
    N = len(pr)
    # x - the list of the variables
    x = [None] * N

    ## Create a model
    m = Model("NL_MILP")
    # Create variables - x is binary, (P * x) is the actual variable value
    for i in range(N):
        x[i] = m.addVar(vtype=GRB.BINARY)
    # Integrate new variables
    m.update()
    # Set objective
    m.setObjective(sum(pr[i] * P * x[i] for i in range(N)), GRB.MINIMIZE)
    # Add constraints
    for j in range(1, N - L + 1):
        m.addConstr(sum(x[i] for i in range(j, j + L)) >= (x[j] - x[j - 1]) * L)
    m.addConstr(sum(x) == L)
    m.addConstr(sum(x[i] for i in range(L)) >= (x[0] * L))

    ## Optimize
    m.optimize()
    for v in m.getVars():
        print(v.varName, (v.x) * P)
    print('Obj:', m.objVal)
