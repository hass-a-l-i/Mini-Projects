import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.ma.extras import average

# train:val:test split
def train_val_test_split(train, val, test, x, y, seed):
    random.seed(seed)
    rand_list = random.sample(range(len(x)), len(x))
    train_index = int((train * len(x)))
    val_index = int(train_index + (val * len(x)))
    test_index = int(val_index + (test * len(x)))
    train_set = rand_list[:train_index]
    val_set = rand_list[train_index:val_index]
    test_set = rand_list[val_index:test_index]
    train_y, train_x = [], []
    for i in train_set:
        train_y.append(y[i])
        train_x.append(x[i])
    val_y, val_x = [], []
    for i in val_set:
        val_y.append(y[i])
        val_x.append(x[i])
    test_y, test_x = [], []
    for i in test_set:
        test_y.append(y[i])
        test_x.append(x[i])
    return train_x, val_x, test_x, train_y, val_y, test_y

# plot scatter train test val
def plot_train_test_val(train_x, val_x, test_x, train_y, val_y, test_y, x, y):
    plt.scatter(train_x, train_y, c='b', label="train set")
    plt.scatter(val_x, val_y, c='g', marker="^", label="val set")
    plt.scatter(test_x, test_y, c='r', marker=",", label="test set")
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)

# for creating matrix for polynomial fit
def polynomial_design_matrix(deg, x_in):
    X_des = np.ones(len(x_in), )
    for i in range(1, deg + 1):
        r = np.power(x_in, i)
        X_des = np.vstack((X_des, r))
    X_des = np.round(X_des, 5)
    X_des = np.transpose(X_des)
    return X_des

# fit weights to a set degree for polynomials
def fit_weights(deg, x, y):
    y_arr = np.array(y)
    x_arr = np.array(x)
    X_design = polynomial_design_matrix(deg, x_arr)
    w_fit = np.linalg.lstsq(X_design, y_arr, rcond=None)[0]
    return w_fit

# fit polynomial to data with set degree, returns plotting data
def fit_polynomial(deg, x_train, y_train, x_main):
    w_fit = fit_weights(deg, x_train, y_train)
    X_grid = np.arange(2 * min(x_main), 2 * max(x_main), .1)
    X_grid_matrix = polynomial_design_matrix(deg, X_grid)
    f = np.dot(X_grid_matrix, w_fit)
    return X_grid, f

# find loss for quadratic and cubics
def find_loss(deg, x, y, x_compare, y_compare):
    if len(x_compare) == 0:
        return
    w_fit = fit_weights(deg, x, y)
    y_pred = []
    if deg == 3:
        for i in x_compare:
            new_y = w_fit[0] + (w_fit[1] * i) + w_fit[2] * (i ** 2) + w_fit[3] * (i ** 3)
            y_pred.append(new_y)
    if deg == 2:
        for i in x_compare:
            new_y = w_fit[0] + (w_fit[1] * i) + w_fit[2] * (i ** 2)
            y_pred.append(new_y)
    all_se = []
    for i in range(len(y_compare)):
        diff = y_compare[i] - y_pred[i]
        se = diff ** 2
        all_se.append(se)
    return average(all_se)

# numpy linear regression checker
def numpy_polyfit(deg, x, y):
    model = np.poly1d(np.polyfit(x, y, deg))
    polyline = np.arange(min(x)-1, max(x)+1, .1)
    plt.plot(polyline, model(polyline), linewidth=2, label="polyfit", linestyle='--')

# polynomial regression function
def polynomial_regression(deg, x_in, y_in, train, val, test, polyfit_overlay, print_loss):
    train_xx, val_xx, test_xx, train_yy, val_yy, test_yy = (
        train_val_test_split(train, val, test, x_in, y_in, 10))
    plot_train_test_val(train_xx, val_xx, test_xx, train_yy, val_yy, test_yy, x_in, y_in)
    X_plot, f_plot = fit_polynomial(deg, train_xx, train_yy, x_in)

    if print_loss:
        val_loss = find_loss(deg, train_xx, train_yy, val_xx, val_yy)
        test_loss = find_loss(deg, train_xx, train_yy, test_xx, test_yy)
        print("Validation MSE = ", val_loss)
        print("Test MSE = ", test_loss)

    plt.plot(X_plot, f_plot, linewidth=2, label="degree-%d linear regr." % deg)

    if polyfit_overlay:
        numpy_polyfit(deg, x_in, y_in)  # polyfit comparison#

    plt.legend()
    plt.show()

# generates random data in some bound and deviates y values to make random
def generate_random_data(lower, upper, step, grd, c):
    xx = np.arange(lower, upper, step)
    yy = (grd * xx) + c
    dev = np.random.randn(len(xx))
    yy_dev = yy + dev
    return xx, yy_dev

def generate_random_data2(lower, upper, step, grd, c, dev_factor):
    xx = np.arange(lower, upper, step)
    yy = (grd * xx) + c
    dev = np.random.randn(len(xx)) * dev_factor
    yy_dev = yy + dev
    return xx, yy_dev

# uniform distribution of length k sampled n times
def gaussian_overlay_uniform(n, k, bins):
    xx = np.random.uniform(-1, 1, (n, k))
    ones = np.ones((k, 1))
    sums = np.dot(xx, ones)
    # find sums
    sums_f = sums.flatten()
    mu = sums_f.mean()
    sigma = sums_f.var()
    # plot histogram and fit gaussian curve
    histogram = plt.hist(sums_f, bins)
    _discard_, bin_centres = histogram[0], histogram[1]
    pdf = np.exp(-0.5 * (bin_centres - mu) ** 2 / sigma) / np.sqrt(2 * np.pi * sigma)  # remember sigma is squared
    bin_width = bin_centres[2] - bin_centres[1]
    predicted_bin_heights = pdf * n * bin_width
    plt.plot(bin_centres, predicted_bin_heights, '-r', linewidth=2)

# as above but now -log transform before plotting
def gaussian_overlay_log(n, k, bins):
    xx = np.random.rand(n, k)
    xx2 = -np.log(xx)
    ones = np.ones((k, 1))
    sums = np.dot(xx2, ones)
    # find sums
    sums_f = sums.flatten()
    mu = sums_f.mean()
    sigma = sums_f.var()
    # plot histogram and fit gaussian curve
    histogram = plt.hist(sums_f, bins)
    _discard_, bin_centres = histogram[0], histogram[1]
    pdf = np.exp(-0.5 * (bin_centres - mu) ** 2 / sigma) / np.sqrt(2 * np.pi * sigma)  # remember sigma is squared
    bin_width = bin_centres[2] - bin_centres[1]
    predicted_bin_heights = pdf * n * bin_width
    plt.plot(bin_centres, predicted_bin_heights, '-r', linewidth=2)

# returns joint pdf for bivariate distribution
def pdf_contour(x_1, x_2):
    mu1, mu2 = x_1.mean(), x_2.mean()
    var1, var2 = x_1.var(), x_2.var()
    pdf1 = np.exp(-0.5 * ((x_1 - mu1) ** 2) / var1) / np.sqrt(2 * np.pi * var1)
    pdf2 = np.exp(-0.5 * ((x_2 - mu2) ** 2) / var2) / np.sqrt(2 * np.pi * var2)
    pdf = pdf1 * pdf2 # ofc combine distributions as independent joint
    return pdf

# apply 2x2 transform to bivariate data
def transform_A_contour(a, X):
    A = np.ones((2, 2))
    A[0][1] = 0
    A[1][0] = a
    A[1][1] = 1 - a
    Z = np.dot(X, A.T)
    z1 = Z[:, 0]
    z2 = Z[:, 1]
    return z1, z2

# plot bivariate for both gaussian and transformed data
def plot_contours(aa, XX, gaussian):
    x1 = XX[:, 0]
    x2 = XX[:, 1]
    y1, y2 = transform_A_contour(aa, XX)
    pdf_x = pdf_contour(x1, x2)  # standard normal
    pdf_y = pdf_contour(y1, y2)  # shifted
    if gaussian:
        plt.tricontour(x1, x2, pdf_x, cmap="seismic")
    if not gaussian:
        plt.tricontour(y1, y2, pdf_y, cmap="Wistia")
    if gaussian is None:
        plt.tricontour(x1, x2, pdf_x, cmap="seismic")
        plt.tricontour(y1, y2, pdf_y, cmap="Wistia")
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.show()

# bernoulli distribution (1s and 0s)
def bernoulli_std_error(N):
    xx = 1 * (np.random.rand(N) < 0.3)
    mu = np.average(xx)
    sigma_sq = np.var(xx)
    std_dev = np.sqrt(sigma_sq)
    std_error = std_dev / N
    print("Mean = ", mu)
    print("Variance = ", sigma_sq)
    print("Std Dev = ", std_dev)
    print("Std Error = ", std_error)
    print("Mean + Std Error = ", mu + std_error)
    print("Mean - Std Error = ", mu - std_error)

def std_error(xx):
    mu = xx.mean()
    squares = (xx - mu) ** 2
    var_std_err = sum(squares) / (len(xx) - 1)
    std_err = np.sqrt(var_std_err) / np.sqrt(len(xx))
    return std_err


# plot gaussian histogram (standard normal)
def gauss_histogram(n, bins):
    N = int(n)  # 1e6 is a float, numpy wants int arguments
    xx = np.random.randn(N)
    histogram = plt.hist(xx, bins=bins)
    print('empirical_mean = %g' % np.mean(xx))  # or xx.mean()
    print('empirical_var = %g' % np.var(xx))  # or xx.var()
    plt.show()


# plot gaussian pdf line from hist
def gauss_distr(n, bins):
    N = int(n)
    xx = np.random.randn(N)

    # plot hist
    hist_stuff = plt.hist(xx, bins=bins)

    # find centre of bins
    bin_centres = 0.5 * (hist_stuff[1][1:] + hist_stuff[1][:-1])
    bin_width = bin_centres[1] - bin_centres[0]

    # create pdf using bin centres as x values
    # normal distr and randn used so 0 mean, var 1
    pdf = np.exp(-0.5 * bin_centres ** 2) / np.sqrt(2 * np.pi)

    # y values of bins will be pdf scaled by N values then * width of each bin
    predicted_bin_heights = pdf * N * bin_width

    # plot distr line over hist
    plt.plot(bin_centres, predicted_bin_heights, '-r')
    plt.show()

# plot gaussian or laplacian scatter distribution
def gaussian_laplace_2d(n, gaussian):
    N = int(n)  # no elements in each feature
    D = 2  # no features (x1 and x2)
    if gaussian:
        X = np.random.randn(N, D)
        plt.plot(X[:, 0], X[:, 1], '.')
        plt.axis('square')
        plt.ylabel("x2")
        plt.xlabel("x1")
        plt.title("Gaussian scatter")
        plt.show()
    if not gaussian:
        X = np.random.laplace(size=(N, D))
        plt.plot(X[:, 0], X[:, 1], '.')
        plt.axis('square')
        plt.ylabel("x2")
        plt.xlabel("x1")
        plt.title("Laplacian scatter")
        plt.show()

# define rbf basis function
def rbf(xx, cc, hh):
    return np.exp(-(xx-cc)**2 / hh**2)

# find rbf design matrix
def rbf_design_matrix(x, h):
    X_des = np.ones(len(x), )
    for i in range(len(x)):
        r = rbf(x, i, h)  # had to scale c to ensure it didn't tank => do regularization here
        X_des = np.vstack((X_des, r))
    X_des = np.delete(X_des, [0], axis=0)
    X_des = np.transpose(X_des)
    print(X_des)
    return X_des

# find plotting matrix from above
def plotting_matrix(w_in, x_values, h):
    X_des = np.ones(len(x_values), )
    for i in range(len(w_in)):
        r = np.array(rbf(x_values, i, h))
        X_des = np.vstack((X_des, r))
    X_des = np.delete(X_des, [0], axis=0)
    X_des = np.transpose(X_des)

    return X_des

# fit and plot rbf
def RBF_1d(x_data, y_data, x_new):
    X_design = rbf_design_matrix(x_data, 1)
    w_fit = np.linalg.lstsq(X_design, y_data, rcond=None)[0]
    X_grid = plotting_matrix(w_fit, x_new, 1)
    f = np.dot(X_grid, w_fit)
    return f

def regularized_fit(xx, yy, x_new, lamb):
    X_design = rbf_design_matrix(xx, 1)
    # this is rearranging L2 for w to find new weights
    # assumed inverse equal to prev - see notes
    w_reg = np.linalg.lstsq(X_design.T.dot(X_design) + lamb * np.identity(len(X_design)), X_design.T.dot(yy), rcond=None)[0]
    X_grid = plotting_matrix(w_reg, x_new, 1)
    f2 = np.dot(X_grid, w_reg)
    return f2

# single scatter plot transformed with A and hyperparameter a
def transform_A_scatter_plot(a, X, N):
    A = np.ones((2, 2))
    A[0][1] = 0
    A[1][0] = a
    A[1][1] = 1 - a
    Z = np.dot(X, A.T)

    # check variances match for large N
    # i.e. covar of Z should be equal to covar matrix = A*A.T as x is standard normal distr
    if N > 1000:
        print("Variance of Z: ")
        print(np.cov(Z.T))
        print("Variance of A: ")
        print((A @ A.T))

    plt.plot(X[:, 0], X[:, 1], '.', label="x points")
    plt.plot(Z[:, 0], Z[:, 1], '.', label="z points")
    plt.title("a = %d"%a)
    plt.legend(loc="upper left")
    plt.ylabel("x2 or z2")
    plt.xlabel("x1 or z1")
    plt.show()

# plot multiple scatters transformed different a's
def plot_transformed_scatter_range_a(l, u, XX, NN):
    for i in range(l, u):
        transform_A_scatter_plot(i, XX, NN)