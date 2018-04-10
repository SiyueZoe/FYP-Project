# with unit of $/kWmin
import numpy as np


def price():
    price = np.zeros(144)
    price[0:6] = 19
    price[6:12] = 1
    price[12:18] = 0.8
    price[18:24] = 3.8
    price[24:30] = 9
    price[30:36] = 17.5
    price[36:42] = 17
    price[42:48] = 28
    price[48:54] = 14.2
    price[54:60] = 13
    price[60:66] = 20.5
    price[66:72] = 12.5
    price[72:78] = 25.5
    price[78:84] = 17
    price[84:90] = 12.5
    price[90:96] = 17
    price[96:102] = 9.8
    price[102:108] = 17
    price[108:114] = 17
    price[114:120] = 4.8
    price[120:126] = 14.2
    price[126:132] = 14.8
    price[132:138] = 5.8
    price[138:144] = 7

    # Unit conversion from $/MWh to $/kWmin
    price /= 60000
    return price
