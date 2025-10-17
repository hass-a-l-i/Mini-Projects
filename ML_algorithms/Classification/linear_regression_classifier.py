import numpy as np
import matplotlib.pyplot as plt

# Train model on synthetic dataset
N = 200
X = np.random.rand(N, 1)*10
yy = (X > 1) & (X < 3)
def phi_fn(X):
    return np.concatenate([np.ones((X.shape[0],1)), X, X**2], axis=1)
ww = np.linalg.lstsq(phi_fn(X), yy, rcond=None)[0]

# Predictions
x_grid = np.arange(0, 10, 0.05)[:,None]
f_grid = np.dot(phi_fn(x_grid), ww)

# Predictions with alternative weights:
w2 = [-1, 2, -0.5] # Values set by hand - allows for 0.5 threshold for f(x)
f2_grid = np.dot(phi_fn(x_grid), w2)

# Show demo
plt.clf()
plt.title("binary dataset")
plt.plot(X[yy==1], yy[yy==1], 'r+', label="1's")
plt.plot(X[yy==0], yy[yy==0], 'bo', label="0's")
plt.plot(x_grid, f_grid, 'm-', label="quadratic fit - no threshold")
plt.plot(x_grid, f2_grid, 'k-', label="quadratic fit - threshold = 0.5")
plt.ylim([-0.1, 1.1])
plt.legend(loc="upper right")
plt.show()