# with unit of gallon
import numpy as np


def di():
    di = np.zeros(144)
    di[0:6] = 3
    di[6:12] = 4
    di[12:18] = 3.5
    di[18:24] = 2
    di[24:30] = 2.5
    di[30:36] = 26
    di[36:42] = 32.5
    di[42:48] = 28
    di[48:54] = 36
    di[54:60] = 23
    di[60:66] = 23
    di[66:72] = 20
    di[72:78] = 12.5
    di[78:84] = 20
    di[84:90] = 23.5
    di[90:96] = 14.5
    di[96:102] = 12
    di[102:108] = 10
    di[108:114] = 21
    di[114:120] = 14
    di[120:126] = 15
    di[126:132] = 12
    di[132:138] = 6
    di[138:144] = 1

    return di