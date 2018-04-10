# PSO_IL - the PSO solution for the interruptable load with Electrical Machinery and small Pmin
# Input:
#       dt - the time step [scalar]
#       pr - the array of prices [vector]
#       power_ILmax - the maximum power [scalar]
#       E_IL - the total energy [scalar]
#       T_offmin_IL - the minimum off time [scalar]
#       begin_IL - the beginning of possible time duration [scalar]
#       end_IL - the ending of possible time duration [scalar]
# Output:
#       Total_cost - The total cost [scalar]
#       P_status - the optimal schedule [vector]
import numpy as np
import random
from real_time_price import price


class PSO():
    # PSO Parameter Setting
    def __init__(self, pN, dim, max_iter, max_V, power_ILmax, E_IL, T_offmin_IL, begin_IL, end_IL, dt):
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
        # self.fit - global best fit(initialized with a large value)
        self.fit = 1e10
        # self.max_pIL - max power for Appliance IL
        self.max_pIL = power_ILmax
        # self.E - total energy needed for IL task
        self.E = E_IL
        # self.T_offmin - num of times steps of minimum off-time
        self.T_offmin = T_offmin_IL
        # self.b_IL - load can work after this time
        self.b_IL = begin_IL
        # self.e_IL - load can work before this time
        self.e_IL = end_IL
        self.dt = dt

    # Objective Function
    def function(self, x):
        cost = 0
        for i in range(self.dim):
            cost += price()[i] * x[i] * self.dt
        return cost

    # Initialization
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.b_IL, self.e_IL + 1):
                # Uniformly distributed
                self.X[i][j] = random.uniform(0, self.max_pIL)
                self.V[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.X[i]  # Including all dimensions
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
            # Update pbest for each particle and gbest for all
            for i in range(self.pN):
                energy = 0
                for j in range(self.b_IL, self.e_IL + 1):
                    # Set the upper bound of x
                    if self.X[i][j] > self.max_pIL:
                        self.X[i][j] = self.max_pIL
                    # Set the lower bound of x
                    elif self.X[i][j] < 0:
                        self.X[i][j] = 0
                    energy += self.dt * self.X[i][j]
                # To satisfy the min-offtime constraints
                # [begin_IL, end_IL - 2)
                for j in range(self.b_IL, self.e_IL - 2):
                    if self.X[i][j] is not 0 and self.X[i][j + 1] is 0:
                        # [j+1,j+T] or [j+1,self.e_IL-1]
                        for k in range(j + 2, min(j + self.T_offmin + 1, self.e_IL)):
                            if self.X[i][k] is not 0:
                                energy -= (self.dt * self.X[i][k])
                                self.X[i][k] = 0
                # When energy is over the limit
                tmp_price = price()
                for m in range(self.dim):
                    if m < self.b_IL or m > self.e_IL:
                        tmp_price[m] = 0
                while energy > self.E:
                    diff = energy - self.E
                    while self.X[i][np.argmax(tmp_price)] is 0:
                        # Modify the max value to find the next max
                        tmp_price[np.argmax(tmp_price)] = 0
                    if self.X[i][np.argmax(tmp_price)] > (diff / self.dt):
                        self.X[i][np.argmax(tmp_price)] -= (diff / self.dt)
                        # Energy satisfied
                        energy = self.E
                    # Equal condition within
                    else:
                        tmp_rand = random.uniform(0, 1)
                        # 0 is not acceptable
                        while tmp_rand is 0:
                            tmp_rand = random.uniform(0, 1)
                        energy -= (self.dt * (self.X[i][np.argmax(tmp_price)] - tmp_rand))
                        # tmp_rand left
                        self.X[i][np.argmax(tmp_price)] = tmp_rand
                        # Modify the max value to find the next max
                        tmp_price[np.argmax(tmp_price)] = 0
                # When energy is below the limit
                tmp_price = price()
                for m in range(self.dim):
                    if m < self.b_IL or m > self.e_IL:
                        tmp_price[m] = 1e10
                while energy < self.E:
                    diff = self.E - energy
                    while self.X[i][np.argmin(tmp_price)] is 0:
                        # Modify the min value to find the next min
                        tmp_price[np.argmin(tmp_price)] = np.amax(tmp_price) + 1
                    if self.X[i][np.argmin(tmp_price)] * self.dt + diff <= self.max_pIL * self.dt:
                        self.X[i][np.argmin(tmp_price)] += (diff / self.dt)
                        # Energy satisfied
                        energy += diff
                    else:
                        energy += (self.dt * (self.max_pIL - self.X[i][np.argmin(tmp_price)]))
                        self.X[i][np.argmin(tmp_price)] = self.max_pIL
                        # Modify the min value to find the next min
                        tmp_price[np.argmin(tmp_price)] = np.amax(tmp_price) + 1
                # Output the power distribution
                if t is (self.max_iter - 1) and i is 7:
                    P_status = self.X[7]
                    print(P_status)
                tmp_cost = self.function(self.X[i])
                # Update pbest
                if tmp_cost < self.p_fit[i]:
                    self.p_fit[i] = tmp_cost
                    self.pbest[i] = self.X[i]
                    # Update gbest
                    if self.p_fit[i] < self.fit:
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            # Apply the formula
            for i in range(self.pN):
                for p in range(self.b_IL, self.e_IL + 1):
                    if self.V[i][p] > self.max_V:
                        self.V[i][p] = self.max_V
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + self.c2 * self.r2 * (
                        self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            # output the optimal fitness data
            Total_cost = self.fit / 1000
            print(Total_cost)
        return fitness


# Execution
my_pso = PSO(pN=30, dim=144, max_iter=50, max_V=10, power_ILmax=2000, E_IL=440000, T_offmin_IL=2, begin_IL=48,
             end_IL=90, dt=10)  # time: [8:00, 15:00]
my_pso.init_Population()
fitness = my_pso.iterator()