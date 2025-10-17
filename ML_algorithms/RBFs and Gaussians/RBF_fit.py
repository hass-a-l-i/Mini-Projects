import numpy as np
from common_funcs import RBF_1d
import matplotlib.pyplot as plt

# cos function
grid_size = 0.2
xx = np.arange(10, 30, grid_size)
yy = np.cos(xx)

# note the c shift makes graph go weird around 0 therefore shifted to this x range and fits now
X_generated = np.arange(10, 30, 0.1)
f = RBF_1d(xx, yy, X_generated)
plt.plot(X_generated, f, linewidth=2)
plt.scatter(xx, yy, c='r', marker=".", label="test set")
plt.show()
