import numpy as np
from common_funcs import plot_transformed_scatter_range_a, transform_A_scatter_plot

N = 10000
X = np.random.randn(N, 2)
#plot_transformed_scatter_range_a(0, 10, X, 100)
transform_A_scatter_plot(20, X, N) # check variances same for large N