# BB_V0.3 (24/Jan/2018)
import numpy as np
from scipy import optimize

# -------------------------------------------- Parameter Setting -------------------------------------------------------
N = 144  # number of time intervals
t = 10  # time interval(in min)
c_water = 0.073  # specific heat of water(W min/gallon°C)
S = int(input("The number of appliances is: "))  # number of appliances
num_of_NL = int(input("The number of NL is: "))
num_of_IL = int(input("The number of IL is: "))
num_of_TCL = int(input("The number of TCL is: "))
begin = (1, num_of_NL + num_of_IL)  # begin time for NL & IL
begin = np.zeros(begin)
end = np.copy(begin)  # end time for NL & IL
T_off = np.copy(begin)  # minimum off time for IL
for i in range(num_of_NL + num_of_IL):
    begin[0, i] = int(input("Beginning time for app %s is: " % (i + 1)))
    end[0, i] = int(input("Ending time for app %s is: " % (i + 1)))
    if i >= num_of_NL:
        T_off[0, i] = int(input("Minimum off time for app %s is: " % (i + 1)))
c = [0.035] * 6 + [0.033] * 6 + [0.0317] * 6 + [0.0283] * 12 + [0.0317] * 6 + [0.04] * 6 + [0.0483] * 18 + \
    [0.0467] * 12 + [0.045] * 12 + [0.043] * 12 + [0.045] * 6 + [0.0583] * 6 + [0.0567] * 6 + [0.0533] * 6 + \
    [0.05] * 6 + [0.045] * 6 + [0.04] * 6 + [0.0367] * 6  # price
c = np.tile(c, S)
c_tmp = (N * S)  # for y parts
c_tmp = np.zeros(c_tmp)
c = np.concatenate((c, c_tmp))
c = c.transpose()
di = (num_of_TCL, N)
di = np.zeros(di)
tmp_step = np.copy(di)
tmp_required = (num_of_TCL, 1)
tmp_required = np.zeros(tmp_required)
tmp_up = np.copy(tmp_required)
tmp_o = np.copy(tmp_required)
mass = np.copy(tmp_required)
for i in range(num_of_TCL):
    tmp_required[i, 0] = input("Desired water temperature for TCL %s is: " % (i + 1))
    tmp_up[i, 0] = input("Upper limit of water temperature in storage for TCL %s is: " % (i + 1))
    tmp_o[i, 0] = input("Initial water temperature in storage for TCL %s is: " % (i + 1))
    mass[i, 0] = input("Mass of water in full storage(in gallon) for TCL %s is: " % (i + 1))
    for j in range(24):
        tmp_demand = input("Demand of hot water drawn for TCL %s at time %s hour is: " % ((i + 1), (j + 1)))
        tmp_environment = input("Environmental temperature for TCL %s at time %s hour is: " % ((i + 1), (j + 1)))
        for t in range(6):  # presented by one hour
            di[i, j * 6 + t] = tmp_demand
            tmp_step[i, j * 6 + t] = tmp_environment

# ---------------------------------------------- Implementing ---------------------------------------------------------
a = (S, 2 * N * S)
a = np.zeros(a)
for i in range(S):
    for j in range(S * N):
        if i is 0 or i is 2:  # App1&3
            if (i * N + begin[i] - 1) <= j <= (i * N + end[i] - 1):
                a[i, j] = 1
a1 = (S, 2 * N * S)
a1 = np.zeros(a1)
for i in range(S):
    for j in range(S * N):
        if i is 1 or i is 4:  # App2&5
            if (i * N + begin[i] - 1) <= j <= (i * N + end[i] - 1):
                a1[i, j] = 1
a = np.concatenate((a, a1))

b = np.array([[40, 0, 45, 0, 0], [0, 100, 0, 0, 80]])

d = (S, 2 * N * S)
d = np.zeros(d)
d[4 - 1, (4 - 1) * N] = 1
for m in range(1, 2 * N):  # m=[1,9]
    tmp = (S, 2 * N * S)
    tmp = np.zeros(tmp)
    if m < N:  # ax<=b
        for n in range(m + 1):
            tmp[4 - 1, (4 - 1) * N + n] = 1
    else:  # ax>=b
        for n in range(m - N + 1):
            tmp[4 - 1, (4 - 1) * N + n] = -1
    d = np.concatenate((d, tmp))
# For appliance 2
y = (1, 2 * N * S)
y = np.zeros(y)
y[0, N * S + 1 * S] = 1
for m in range(1, N):
    y_tmp = (1, 2 * N * S)
    y_tmp = np.zeros(y_tmp)
    y_tmp[0, N * S + 1 * S + m] = 1
    y = np.concatenate((y, y_tmp))
# For appliance 5
for m in range(N):
    y_tmp = (1, 2 * N * S)
    y_tmp = np.zeros(y_tmp)
    y_tmp[0, N * S + 4 * S + m] = 1
    y = np.concatenate((y, y_tmp))
count = 0
for t in range(S):
    if T_off[t] is not 0:  # Look for IL
        count += 1
        for p in range((count - 1) * N, N - T_off[t] + (count - 1) * N):
            for q in range(p + 1, T_off[t] + p + 1):
                y_tmp = y[p] - y[p + 1] + y[q]
                d = np.concatenate((d, y_tmp))

e = (2 * N, S)
e = np.zeros(e)
for i in range(N):  # ax<=b
    e[i, 3] = mass * c_water * (tmp_up - tmp_o)
    for j in range(i + 1):
        e[i, 3] += di[j] * c_water * (tmp_required - tmp_step[j])
for i in range(N, 2 * N):  # ax>=b
    for j in range(i - N + 1):
        e[i, 3] -= di[j] * c_water * (tmp_required - tmp_step[j])
e /= t

# ---------------------------------------------- Execution -------------------------------------------------------------
res = optimize.linprog(c, A_eq=a, b_eq=b, A_ub=d, b_ub=e,
                       bounds=((0, 20), (0, 20), (0, 20), (0, 20), (0, 20), (0, 40), (0, 40), (0, 40),
                               (0, 40), (0, 40), (0, 15), (0, 15), (0, 15), (0, 15), (0, 15), (0, 240), (0, 240),
                               (0, 240), (0, 240), (0, 240), (0, 25), (0, 25), (0, 25), (0, 25), (0, 25), (0, 1),
                               (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
                               (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
                               (0, 1), (0, 1)))
print(res.x)
print(res.fun)
