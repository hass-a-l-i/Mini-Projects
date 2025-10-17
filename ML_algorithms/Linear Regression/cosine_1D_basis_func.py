import numpy as np
import matplotlib.pyplot as plt

# plot cosine func - grid size shows how accurately we plot it, decrease = more points
# could use this as another basis function
grid_size = 0.1
x_grid = np.arange(-10, 10, grid_size)
f_vals = np.cos(x_grid)
plt.clf()
plt.plot(x_grid, f_vals, 'b-')
plt.plot(x_grid, f_vals, 'r.')
plt.show()