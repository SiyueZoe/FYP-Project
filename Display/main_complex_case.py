import numpy as np
import matplotlib.pyplot as plt
from real_time_price import price
from water_consumption import consumption
from Pymprog_all import Real_Case

## Parameter Setting
# For All
pr = price()
N = len(pr)
dt = 10
# the number of residence
Residence = 4
DR_free = 5.48
DR = Residence * DR_free * 60 * 0.8
# beginning time: 19:00
DR_b = 19
DR_b = DR_b * 6
# ending time: 21:00
DR_e = 21
DR_e = DR_e * 6 - 1
# For NL
L = np.array([5, 6, 12])
P_NL = np.array([1.3, 3, 1.2])
NL_b = np.array([107, 107, 107])
NL_e = np.array([137, 143, 143])
L = np.tile(L, Residence)
P_NL = np.tile(P_NL, Residence)
NL_b = np.tile(NL_b, Residence)
NL_e = np.tile(NL_e, Residence)
# For IL
P_IL = np.array([1.47, 1.4])
E_IL = np.array([264, 168])
T_off = np.array([0, 2])
IL_b = np.array([0, 65])
IL_e = np.array([143, 143])
P_IL = np.tile(P_IL, Residence)
E_IL = np.tile(E_IL, Residence)
T_off = np.tile(T_off, Residence)
IL_b = np.tile(IL_b, Residence)
IL_e = np.tile(IL_e, Residence)
Pmin = 0.5 * P_IL
# For TCL
P_TCL = np.array([4.5])
c_water = 0.000073
mass = np.array([50])
temp_up = np.array([80])
temp_o = np.array([25])
temp_req = np.array([37])
temp_en = np.array([[26]])
di = consumption()
P_TCL = np.tile(P_TCL, Residence)
mass = np.tile(mass, Residence)
temp_up = np.tile(temp_up, Residence)
temp_o = np.tile(temp_o, Residence)
temp_req = np.tile(temp_req, Residence)
temp_en = np.tile(temp_en, (Residence, N))
di = np.tile(di, (Residence, 1))

## MILP Solution
solution = Real_Case(pr, N, dt, DR, DR_b, DR_e, L, P_NL, NL_b, NL_e, P_IL, E_IL, T_off, Pmin, IL_b, IL_e, P_TCL,
                     c_water, mass, temp_up, temp_o, temp_req, temp_en, di)

## Plot
fig, ax_pr = plt.subplots()  # Price
fig, ax_ALL = plt.subplots()  # Total Power consumed by ALL
fig, ax_NL = plt.subplots()  # Total Power consumed by NL
fig, ax_IL = plt.subplots()  # Total Power consumed by IL
fig, ax_TCL = plt.subplots()  # Total Power consumed by TCL
ax_pr.plot(pr, 'o-')
ax_ALL.plot(solution[3], 'm-', color='black')
ax_NL.plot(solution[0], 'm-', color='black')
ax_IL.plot(solution[1], 'm-', color='black')
ax_TCL.plot(solution[2], 'm-', color='black')
ax_pr.set(xlabel='Time', ylabel='Price, $', title='Prices Vs Time')
ax_ALL.set(xlabel='Time, min', ylabel='Consumption, kW')
ax_NL.set(xlabel='Time, min', ylabel='Consumption, kW')
ax_IL.set(xlabel='Time, min', ylabel='Consumption, kW')
ax_TCL.set(xlabel='Time, min', ylabel='Consumption, kW')
