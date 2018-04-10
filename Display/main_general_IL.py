import numpy as np
from real_time_price import price
from IL_LP_with_EM import IL_LP_with_EM
from IL_LP_without_EM import IL_LP_without_EM
import matplotlib.pyplot as plt

# Paramers
pr = price()
dt = 10
P = np.array([1.47, 1.4])
E = np.array([264, 168])
T_off = np.array([0, 2])
b = np.array([0, 65])
e = np.array([144, 144])
N_IL = len(P)
result_IL = np.zeros(len(pr))
total_cost = 0
for i in range(N_IL):
    # IL without EM
    if T_off[i] == 0:
        solution = IL_LP_without_EM(dt, pr[(b[i]):(e[i])], P[i], E[i])
        result_IL[(b[i]):(e[i])] += solution[1]
    # IL with EM
    else:
        solution = IL_LP_with_EM(dt, pr[(b[i]):(e[i])], P[i], E[i], T_off[i], 0 * P[i])
        result_IL[(b[i]):(e[i])] += solution[1][:]
    total_cost += solution[0]

# Display
print('Total Cost: ', total_cost)
print('Power Status: ', result_IL)

# Plot
fig, ax = plt.subplots()
ax.plot(result_IL, 'm-', color='black')
ax.plot()
ax.set(xlabel='Time, min', ylabel='Consumption, kW');