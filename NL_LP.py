# NL_LP - the LP solution for Non-Interruptable type load
# Input:
#      dt - the time step [scalar]
#      p - the array of prices from p_b, ..., p_e [array with shape (e-b+1,)]
#      L - the duration of the task [scalar]
#      P - the power rate [scalar]
# Output:
#      F - the value of objective function [scalar]
#      x - the optimal schedule [array with shape (e-b+1,)]

import numpy as np
from scipy.linalg import toeplitz
from scipy import optimize

def NL_LP(dt, p, L, P):
    # N - the number of variables
    N = len(p)
    # c - array of costs for LP
    c = N*dt*p
    # A_eq, b_eq - the matrix and the vector for equality constraints  
    # A_eq - the array of ones corresponding to the total duration of 'on' load status throughout the scheduling horizon = L
    A_eq = np.ones((1,N), dtype=int)
    # b_eq - the scalar
    b_eq = L*P
    
    # A_ub - the matrix for inequality constraints 
    # construct a band matrix corresponding to noninterruptability constraints (the continuous duration of 'on' load status >= L)
    first_column = np.concatenate(([[L-1]], [[-L]], np.zeros((N-L-1,1), dtype=int)))
    first_row = np.concatenate(([L-1], -np.ones((L-1,), dtype=int), np.zeros((N-L,), dtype=int)))
    A_ub = toeplitz(first_column, first_row)
    # add identity matricies corresponding to variable bounds (0 <= x <= P)
    A_ub = np.concatenate((A_ub, -np.eye(N), np.eye(N)))
    # construct the inequlity constraints vector (the continuous duration >= L; 0 <= x <= P) 
    b_ub = np.concatenate((np.zeros((2*N-L+1,1), dtype=int), P*np.ones((N,1), dtype=int)))
    # find the LP solution
    solution = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, method='simplex')
    # the value of objective function
    F = solution.fun
    # the optimal schedule
    x = solution.x
    return F, x
