# IL_MILP - the MILP solution for the Interruptable type load with Electrical Machinery
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices from pr_b, ..., pr_e [array with shape (e-b+1,)]
#       P - the power rate [scalar]
#       E - the energy [scalar]
#       T_off - the minimum off time [scalar]
# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape 5 * (e-b+1,)]

from pymprog import *
import numpy as np

def IL_Pymprog(dt, pr, P, E, T_off, P_min):

    # N - the number of time step
    N = len(pr)

    # Create a new model
    begin('Pymprog_IL')

    # Create variables
    IL_tmp = [None] * 3
    x_IL = []
    for i in range(3):
        # Power values in IL
        if(i % 3 is 0):
            IL_tmp[i] = var('IL', N, bounds=(0, P))
        elif(i % 3 is 1):
            IL_tmp[i] = var('IL', N, bool)
        else:
            IL_tmp[i] = var('IL', 3 * N, bounds=(0, 1))
        x_IL += IL_tmp[i]

    # Set objective
    minimize(sum(dt * x_IL[i] * pr[i] for i in range(N)), 'cost')

    # Add constraints
    # Constraint(1): energy constraints
    sum(x_IL[i] for i in range(N)) == E / dt
    # Constraint(2): limit frequent swich
    for i in range(N - T_off):
        for j in range(i + 2, i + T_off + 2):
            x_IL[i + N] - x_IL[i + 1 + N] + x_IL[j - 1 + N] <= 1
    # Constraint(3): connection between x and y
    for i in range(N):
        x_IL[i + N] == x_IL[2 * N + 3 * i + 1] + x_IL[2 * N + 3 * i + 2]
        x_IL[i] == P_min * x_IL[2 * N + 3 * i + 1] + P * x_IL[2 * N + 3 * i + 2]
        x_IL[2 * N + 3 * i] + x_IL[2 * N + 3 * i + 1] + x_IL[2 * N + 3 * i + 2] == 1

    # Optimize
    solve()

    # Display
    result = []
    for i in range(2 * N):
        result.append(x_IL[i].primal)
    result = np.asarray(result)
    print('*****************Result*****************')
    print('Total Cost: ', vobj())
    print('Power Status: ', result[0:N])
    print('Ancillary Binary: ', result[N:2 * N])
    
    return result