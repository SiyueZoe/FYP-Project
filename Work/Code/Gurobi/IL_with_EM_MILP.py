# IL_MILP - the MILP solution for the Interruptable type load with Electrical Machinery
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices from pr_b, ..., pr_e [array with shape (e-b+1,)]
#       P - the power rate [scalar]
#       E - the energy [scalar]
#       T_off - the minimum off time [scalar]
#       Pmin - minimum power to turn on the appliance [scalar]
# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape 5 * (e-b+1,)]
from gurobipy import *


def IL_MILP(dt, pr, P, E, T_off, Pmin):
    # N - the number of time step
    N = len(pr)
    # x - list of variables
    x = [None] * N * 5

    # Create a new model
    m = Model("MILP_IL")

    # Create variables
    for i in range(5 * N):
        # add in x values(power at each time step)
        if i < N:
            x[i] = m.addVar(lb=0, ub=P, vtype=GRB.CONTINUOUS)
        # add in y values(ancillary binary variables)
        elif N <= i < 2 * N:
            x[i] = m.addVar(vtype=GRB.BINARY)
        # add in z values(variables that help to achieve step function)
        else:
            x[i] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS)

    # Integrate new variables
    m.update()

    # Set objective
    m.setObjective(dt * sum(pr[i] * x[i] for i in range(N)), GRB.MINIMIZE)

    # Add constraints
    # Constraint(1): energy constraints
    m.addConstr(sum(x[i] for i in range(N)) == E / dt)
    # Constraint(2): limit frequent swich
    for i in range(N - T_off):
        for j in range(i + 2, i + T_off + 2):
            m.addConstr(x[i + N] - x[i + 1 + N] + x[j - 1 + N] <= 1)
    # Constraint(3): connection between x and y
    for i in range(N):
        m.addConstr(x[i + N] == x[2 * N + 3 * i + 1] + x[2 * N + 3 * i + 2])
        m.addConstr(x[i] == Pmin * x[2 * N + 3 * i + 1] + P * x[2 * N + 3 * i + 2])
        m.addConstr(x[2 * N + 3 * i] + x[2 * N + 3 * i + 1] + x[2 * N + 3 * i + 2] == 1)

    # Optimize
    m.optimize()
    for v in m.getVars():
        print(v.varName, v.x)
    print('Obj:', m.objVal)
