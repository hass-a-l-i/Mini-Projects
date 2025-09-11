from vars import *
import numpy as np
import matplotlib.pyplot as plt

import print_outputs

plt.xlim(0, max(x_a))
plt.ylim((0,max(y_a)))
x_axis = np.array(x_a)
y_axis = np.array(y_a)
plt.plot(x_axis, y_axis)
plt.xlabel("Time (days)")
plt.ylabel("Price (£)")
plt.grid()
plt.title("Chachu Limited")
plt.show()
