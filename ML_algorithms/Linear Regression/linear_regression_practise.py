import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.extras import average


# code to fit many features with same labels

def fit_many_features(N, D):
    # gen data
    mu = np.random.rand(N)
    X = np.tile(mu[:, None], (1, D)) + 0.01 * np.random.randn(N, D)
    yy = 0.1 * np.random.randn(N) + mu

    # average x's for each y label + create design matrix (extra col 0's for intercept b)
    X_av = [[average(item)] for item in X]
    X_av = np.array(X_av)
    X_design = np.concatenate([X_av, np.ones((X_av.shape[0], 1))], axis=1)

    # fit design and yy to find weights
    w_fit = np.linalg.lstsq(X_design, yy, rcond=None)[0]

    # create x values to plot linear regression line, X_grid1 for plotting, X_grid2 for finding f values
    X_grid1 = np.arange(0, 1, .1)[:, None]  # N,1
    X_grid2 = np.concatenate([X_grid1, np.ones((X_grid1.shape[0], 1))], axis=1)

    # find y values using f(x) = Xw (tilde)
    f = np.dot(X_grid2, w_fit)
    # intercept b and gradient
    a = (max(f) - min(f)) / (max(X_grid1) - min(X_grid1))
    b = w_fit[-1]
    print(b)

    # compare weights for different N and D
    print("Average weight = ", w_fit[0] / D)
    print("Proposed weight = ", 1 / D)

    # plot all data (black) avs and line
    plt.clf()
    X_T = np.transpose(X)
    for i in range(0, len(X_T)):
        plt.scatter(X_T[i], yy, c='black')
    plt.scatter(X_av, yy, c='g', label="weighted scatter")
    plt.scatter(X_grid1, f, c='r', label="weighted scatter")
    plt.plot(X_grid1, f, linewidth=2, label="linear regression line")
    plt.title("y = %.2f x + %.2f" % (a[0], b))
    plt.legend(loc="upper left")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

fit_many_features(10, 10)

# if N= 101, D = 100, we get average weights higher than 1/D
