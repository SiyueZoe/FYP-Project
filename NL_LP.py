# NL_LP - the LP solution for Non-Interruptable type load
# Input:
#      dt - the time step [scalar]
#      pr - the array of prices from p_b, ..., p_e [array with shape (e-b+1,)]
#      L - the duration of the task [scalar]
#      P - the power rate [scalar]
# Output:
#      F - the value of objective function [scalar]
#      x - the optimal schedule [array with shape (e-b+1,)]

import numpy as np
from scipy.linalg import toeplitz
from scipy import optimize

def NL_LP(dt, pr, L, P):
    # N - the number of variables
    N = len(pr)
    
    ### Objective function
    # c - array of costs for LP
    c = dt*pr
    
    ### Equality constraints
    # A_eq, b_eq - the matrix and the vector for equality constraints  
    # A_eq - the array of ones corresponding to the total duration of 'on' load status throughout the scheduling horizon = L
    A_eq = np.ones((1,N), dtype=int)
    # b_eq - the scalar corresponding to the total load consumption
    b_eq = L*P
    
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
    ## Non-Interrubptability constraints
    # construct a band matrix corresponding to noninterruptability constraints (the continuous duration of 'on' load status >= L)
    first_column = np.concatenate(([[L-1]], [[-L]], np.zeros((N-L-1,1), dtype=int)))
    first_row = np.concatenate(([L-1], -np.ones(L-1, dtype=int), np.zeros(N-L, dtype=int)))
    A_ub_ni = toeplitz(first_column, first_row)
    # add identity matricies corresponding to variable bounds (0 <= x <= P)
    A_ub = np.concatenate((A_ub, A_ub_ni))
    # construct the inequlity constraints vector (the continuous duration >= L; 0 <= x <= P) 
    b_ub = np.concatenate((b_ub, np.zeros((N-L+1,1), dtype=int))
    # find the LP solution
    solution = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, method='simplex')
    # the value of objective function
    F = solution.fun
    # the optimal schedule
    x = solution.x
    return F, x
