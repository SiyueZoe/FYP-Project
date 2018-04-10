import numpy as np
from real_time_price import price
from water_consumption import consumption
from LP_TCL import LP_TCL

import matplotlib.pyplot as plt

# Parameters
pr = price()
N = len(pr)
dt = 10
P_TCL = np.array([4.5])
c_water = 0.000073
mass = np.array([50])
temp_up = np.array([80])
temp_o = np.array([25])
temp_req = np.array([37])
temp_en = np.repeat(26, N)
N_TCL = len(P_TCL)
di = consumption()
# Solution
for i in range(N_TCL):
    solution = LP_TCL(dt, pr, P_TCL[i], c_water, mass[i], temp_up[i], temp_o[i], temp_req[i], temp_en, di)

# Plot
fig, ax = plt.subplots()  # Price
ax.plot(solution[1], 'm-', color='black')
ax.set(xlabel='Time, min', ylabel='Consumption, kW')
