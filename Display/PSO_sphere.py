import numpy as np
import random


class PSO():
    # PSO Parameter Setting
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        # self.pN - number of particles
        self.pN = pN
        # self.dim - searching dimensions
        self.dim = dim
        # self.max_iter - number of iteration
        self.max_iter = max_iter
        # self.X, self.Y - velocities and positions for all particles
        self.X = np.zeros((self.pN, self.dim))
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))
        self.gbest = np.zeros((1, self.dim))
        # self.p_fit - personal best fit
        self.p_fit = np.zeros(self.pN)
        # self.fit - global best fit
        self.fit = 1e10

    # Objective Function
    def function(self, x):
        sum = 0
        length = len(x)
        x = x ** 2
        for i in range(length):
            sum += x[i]
        return sum

    # Initialization
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 1)
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
            for i in range(self.pN):  # Update gbest / pbest
                temp = self.function(self.X[i])
                if temp < self.p_fit[i]:  # Update pbest
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:  # Update gbest
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):  # Apply the formula
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) \
                            + self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            # output the optimal fitness data
            print(self.fit)
        return fitness


# Execution
sphere = PSO(pN=30, dim=5, max_iter=100)
sphere.init_Population()
fitness = sphere.iterator()
