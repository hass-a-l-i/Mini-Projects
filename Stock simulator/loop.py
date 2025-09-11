import random
from vars import *

for i in range(1, days + 1):
    x_a.append(i)

for i in range(1, days + 1):
    x = random.uniform(0, 1)
    if x > trend:
        z = 1 + volatility
    else:
        z = 1 - volatility
    b = y_a[i-1] * z
    if b < 5:
        b = 5
    y_a.append(b)