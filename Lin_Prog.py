# BB_V0.1 (19/Jan/2018)
import numpy as np
from scipy import optimize

# NL:Appliance1&3; IL:Appliance2&5; TCL:Appliance4
N = 5  # number of time intervals
S = 5  # number of appliances
t = 10  # time interval
begin = np.array([1, 2, 1, 0, 1])  # beginning time for NL & IL appliance
end = np.array([4, 5, 5, 0, 5])  # ending time for NL & IL appliance
c = np.array([10, 20, 30, 50, 40])
c = np.tile(c, S)
c = c.transpose()
di = np.array([25, 25, 25, 25, 25])  # demand of hot water drawn during i-th time step
c_water = 0.073  # specific heat of water(W min/gallon°C)
tmp_required = 37  # desired water temperature
tmp_up = 80  # upper limit of water temperature in storage
mass = 50  # mass of water in full storage(in gallon)
tmp_o = 25  # initial water temperature in storage
tmp_step = np.array([25, 25, 25, 25, 25]) # environmental temp at the i-th time step

a = (S, N * S)
a = np.zeros(a)
for i in range(S):
    for j in range(S * N):
        if i is 0 or i is 2:  # App1&3
            if (i * N + begin[i] - 1) <= j <= (i * N + end[i] - 1):
                a[i, j] = 1
a1 = (S, N * S)
a1 = np.zeros(a1)
for i in range(S):
    for j in range(S * N):
        if i is 1 or i is 4:  # App2&5
            if (i * N + begin[i] - 1) <= j <= (i * N + end[i] - 1):
                a1[i, j] = 1
a = np.concatenate((a, a1))

b = np.array([[40, 0, 45, 0, 0], [0, 100, 0, 0, 80]])

d = (N, N * S)
d = np.zeros(d)
d[4 - 1, (4 - 1) * N] = 1
for m in range(1, 2 * N):  # m=[1,9]
    tmp = (S, N * S)
    tmp = np.zeros(tmp)
    if m < N:  # ax<=b
        for n in range(m + 1):
            tmp[4 - 1, (4 - 1) * N + n] = 1
    else:  # ax>=b
        for n in range(m - N + 1):
            tmp[4 - 1, (4 - 1) * N + n] = -1
    d = np.concatenate((d, tmp))

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

res = optimize.linprog(c, A_eq=a, b_eq=b, A_ub=d, b_ub=e,
                       bounds=((0, 20), (0, 20), (0, 20), (0, 20), (0, 20), (0, 40), (0, 40), (0, 40),
                               (0, 40), (0, 40), (0, 15), (0, 15), (0, 15), (0, 15), (0, 15), (0, 240), (0, 240),
                               (0, 240),
                               (0, 240), (0, 240), (0, 25), (0, 25), (0, 25), (0, 25), (0, 25)))
print(res.x)
print(res.fun)
