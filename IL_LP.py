# IL_LP - the LP solution for Interruptable type load(without y)
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices from pr_b, ..., pr_e [array with shape (e-b+1,)]
#       P - the power rate [scalar]
#       E - the energy [scalar]
# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape (e-b+1,)]

import numpy as np
from scipy import optimize


def IL_LP(dt, pr, P, E):
    # N - the number of variables
    N = len(pr)
    
    ### Objective function
    # c - the vector of costs for a LP problem
    c = N * dt * pr
#    c = dt * pr

    ### Equality constraints   
    #  A_eq - the matrix for equality constraints
    # A_eq - the array of ones corresponding to the total load consumption throughout the scheduling horizon = E
    A_eq = np.ones((1,N), dtype=int)
    # b_eq - the vector for equality constraints
    b_eq = E / (dt*N)
    
    ### Inequality constraints
    ## Upper and lower bound for each variable
    # bound_up - the upper bound; bound_low - the lower bound
    bound_up = np.eye(N, dtype=int)
    bound_low = -1 * bound_up
    # A_ub - the matrix for inequality constraints
    # construct the sub-matrix of A_ub corresponding to upper and lower bounds
    A_ub = np.concatenate((bound_up, bound_low))
    # b_ub - the vector for inequality constraints
    # b_bound_up - the vector correspond to upper bounds; b_bound_low - the vector correspond to lower bounds;
    b_bound_up = P*np.ones((N,1), dtype=int)
    b_bound_low = np.zeros((N,1), dtype=int)
    # construct the sub-vector of b_ub corresponding to upper and lower bounds
    b_ub = np.concatenate((b_bound_up, b_bound_low))
    
    ## LP solution
    solution = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, method='simplex')
    # the value of objective function
    F = solution.fun
    # the optimal schedule
    x = solution.x
    return F, x
