import numpy as np
from real_time_price import price
from ES_NL import ES_NL
import matplotlib.pyplot as plt

# Parameter Setting
pr = price()
dt = 10
L = np.array([5, 6, 12])
P_NL = np.array([1.3, 3, 1.2])
NL_b = np.array([107, 107, 107])
NL_e = np.array([137, 143, 143])

# Exhaustive Search
solution = ES_NL(dt, pr, L, P_NL, NL_b, NL_e)

# Display
print('Total cost: ', solution[0])
print('Power distribution: ', solution[1])

# Plot
fig, ax = plt.subplots() # Price
fig, ax1 = plt.subplots() # Total Power consumed by NL
ax.plot(pr, 'o-')
ax1.plot(solution[2], 'm-', color='black')
ax.set(xlabel='Time', ylabel='Price, $', title='Prices Vs Time')
ax1.set(xlabel='Time', ylabel='Consumption, kW)', title='Load Scheduling Vs Time')
ax.plot()
ax1.plot()