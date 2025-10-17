import numpy as np
from common_funcs import plot_contours

N, D = 1000, 2
X = np.random.randn(N, D)
plot_contours(2, X, None)