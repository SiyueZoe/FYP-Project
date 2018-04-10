import numpy as np
import math
import random


class PSO():
    # PSO Parameter Setting
    def __init__(self, pN, dim, max_iter, max_V, lb, ub):
        self.w = 2
        self.c1 = 2.8
        self.c2 = 1.3
        self.r1 = random.uniform(0, 1)
        self.r2 = random.uniform(0, 1)
        # self.pN - number of particles
        self.pN = pN
        # self.dim - searching dimensions
        self.dim = dim
        # self.max_iter - number of iteration
        self.max_iter = max_iter
        # self.max_V - max velocity
        self.max_V = max_V
        # self.X, self.Y - velocities and positions for all particles
        self.X = np.zeros((self.pN, self.dim))
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))
        self.gbest = np.zeros((1, self.dim))
        # self.p_fit - personal best fit
        self.p_fit = np.zeros(self.pN)
        # self.fit - global best fit
        self.fit = 1e10
        # self.lb - lower bound of x
        self.lb = lb
        # self.ub - upper bound of x
        self.ub = ub

    # Objective Function
    def function(self, x):
        result = x * (math.sin(x))
        return result

    # Initialization
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                # Uniformly distributed
                self.X[i][j] = random.uniform(self.lb, self.ub)
                self.V[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                # Update the global best value at the beginning
                self.gbest = self.X[i]

    # Update Particle Position
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            # Update gbest / pbest
            for i in range(self.pN):
                if self.X[i] > self.ub:
                    # Set the range of x values
                    self.X[i] = self.ub
                elif self.X[i] < self.lb:
                    # Set the range of x values
                    self.X[i] = self.lb
                temp = self.function(self.X[i])
                # Update pbest
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    # Update gbest
                    if self.p_fit[i] < self.fit:
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            # Apply the formula
            for i in range(self.pN):
                if self.V[i] > self.max_V:
                    self.V[i] = self.max_V
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) \
                            + self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            # output the optimal fitness data
            print(self.fit)
        return fitness


# Execution
my_pso = PSO(pN=30, dim=1, max_iter=500, max_V=3, lb=0, ub=15)
my_pso.init_Population()
fitness = my_pso.iterator()
