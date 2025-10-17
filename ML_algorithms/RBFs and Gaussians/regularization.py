import numpy as np
import matplotlib.pyplot as plt
from common_funcs import generate_random_data, polynomial_regression, regularized_fit, RBF_1d

def choose_plot(poly, xx, yy, x_new, ff, ff_reg, lambd):
    if poly:
        polynomial_regression(len(x) + 20, xx, yy, 1, 0, 0, False, False)
    if not poly:
        plt.xlim(min(xx) - 1, max(xx) + 1)
        plt.ylim(min(yy) - 1, max(yy) + 1)
        plt.scatter(xx, yy, c="red", label="data")
        plt.plot(x_new, ff, c="blue", label="overfitted standard RBF")
        plt.plot(x_new, ff_reg, c="red", label="regularized RBF, lambda = %f" % lambd)
        plt.legend(loc="upper left")
        plt.show()

# make dataset
x, y = generate_random_data(0, 2, 0.4, 3, 4)
X_new = np.arange(-3, 3, 0.01)  # new data to fit
lamb = 0.0001  # lambda
f = RBF_1d(x, y, X_new)  # overfit RBF
f_reg = regularized_fit(x, y, X_new, lamb) # regularized RBF

choose_plot(False, x, y, X_new, f, f_reg, lamb)
