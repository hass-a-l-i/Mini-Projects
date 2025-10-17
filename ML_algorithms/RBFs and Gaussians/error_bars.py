from common_funcs import bernoulli_std_error, std_error, generate_random_data2, fit_polynomial
import numpy as np
import matplotlib.pyplot as plt
# mean + or - std error show the error bars we would plot
#bernoulli_std_error(100)

np.random.seed(10)
x, y = generate_random_data2(-10,10, 1, 2, 6, 1)
x_grid, f = fit_polynomial(1, x, y, x)
std_error = std_error(y)
y_error = [std_error for i in range(len(y))]
print("Std Error = ", std_error)


plt.errorbar(x, y, yerr=y_error, fmt='.', c='r', ecolor='black', capsize=5)
plt.plot(x_grid, f)
plt.xlim(-10, 10)
plt.show()