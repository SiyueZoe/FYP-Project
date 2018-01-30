# IL_LP - the LP solution for the Interruptable type load with Electrical Machinery 
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices from pr_b, ..., pr_e [array with shape (e-b+1,)]
#       P - the power rate [scalar]
#       E - the energy [scalar]
#       T_off - the minimum off time [scalar]
# Output:
#       F - the value of objective function [scalar]
#       x - the optimal schedule [array with shape 2 * (e-b+1,)]

import numpy as np
from scipy.linalg import toeplitz
from scipy import optimize


def IL_LP(dt, pr, P, E, T_off):
    # N - the number of variables
    N = len(pr)
    
    ### Objective function
    # c1 - the sub-vector of c corresponding to power statuses 
    c1 = dt * pr
    # c2 - the sub-vector of c corresponding to ancillary binary variables
    c2 = np.zeros(N)
    c = np.concatenate((c1, c2))
    
    ### Equality constraints   
    # A_eq - the matrix for equality constraints
    # A_eq - the array of ones corresponding to the total load consumption throughout the scheduling horizon = E
    # supplemented by the array of zeros corresponding to ancillary binary variables
    A_eq = np.concatenate((np.ones((1,N), dtype=int), np.zeros((1,N), dtype=int)), axis=1)
    # b_eq - the vector for equality constraints
    b_eq = E / dt
    
    ### Inequality constraints
    ## Upper and lower bound for each variable
    # A_bound_up - the matrix of upper bounds; A_bound_low - the matrix of lower bounds
    A_bound_up = np.eye(N, dtype=int)
    A_bound_low = -1 * A_bound_up
    # A_ub - the matrix for inequality constraints
    # construct the sub-matrix of A_ub corresponding to upper and lower bounds
    A_ub = np.concatenate((A_bound_up, A_bound_low))
    A_ub = np.concatenate((A_ub, np.zeros(A_ub.shape, dtype=int)), axis=1)
    # b_ub - the vector for inequality constraints
    # b_bound_up - the vector correspond to upper bounds; b_bound_low - the vector correspond to lower bounds;
    b_bound_up = P*np.ones((N,1), dtype=int)
    b_bound_low = np.zeros((N,1), dtype=int)
    # construct the sub-vector of b_ub corresponding to upper and lower bounds
    b_ub = np.concatenate((b_bound_up, b_bound_low))
    print(A_ub.shape)
    print(b_ub.shape)
    ## Ancillary minimum off-time constaints\
    # construct the matrix for inequalities
    for i in range(T_off):
        # construct the band matrix with elements 1; -1; (corresponding to the changing a switching-off) and
        # 1 on the i-th possition (corresponding to a switching-on after i time steps)
        first_column = np.concatenate(([[1]], np.zeros((N-T_off-i,1), dtype=int)))
        first_row = np.concatenate(([1], [-1], np.zeros(i, dtype=int), [1], np.zeros(N-3-i, dtype=int)))
        # A_ub_mt - the band matrix 
        A_ub_mt = toeplitz(first_column, first_row)
        # supplement A_ub_mt with additional zeros corresponding to power status variables
        A_ub_mt = np.concatenate((np.zeros(A_ub_mt.shape, dtype=int), A_ub_mt), axis=1)
        # join to the A_ub
        A_ub = np.concatenate((A_ub, A_ub_mt))
    # construct the vector for inequalities
    b_row_size = int((N-T_off)*T_off + (3-T_off)*T_off/2)
    b_ub = np.concatenate((b_ub, np.ones((row_size,1), dtype=int)))
    
    ### LP solution
    print(A_ub.shape)
    print(b_ub.shape)
    solution = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, method='simplex')
    # the value of objective function
    F = solution.fun
    # the optimal schedule
    x = solution.x
    return F, x
