import numpy as np
import math

# define mean and std dev
mu = 2.0
sigma = 3.0

# create data with fixed upper and lower bounds, step = infinitessimal dx
upper = mu + 10*sigma
lower = mu - 10*sigma
dx = sigma / 100.0
xx = np.arange(lower, upper, dx)

# integrate (sum)
integrand = np.exp(-0.5 * (xx - mu)**2 / sigma**2)
Z_approx = np.sum(integrand * dx)
Z_true = sigma * math.sqrt(2*math.pi)
print("Approx = %g" % Z_approx)
print("True = %g" % Z_true)
