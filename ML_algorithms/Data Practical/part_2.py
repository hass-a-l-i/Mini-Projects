from part_1 import *

"""PART A:"""
def polynomial_design_matrix(deg, x_in):
    X_des = np.ones(len(x_in), )
    for i in range(1, deg + 1):
        r = np.power(x_in, i)
        X_des = np.vstack((X_des, r))
    X_des = np.round(X_des, 5)
    X_des = np.transpose(X_des)
    return X_des

def fit_weights(deg, x, y):
    X_design = polynomial_design_matrix(deg, x)
    w_fit = np.linalg.lstsq(X_design, y, rcond=None)[0]
    return w_fit

def fit_polynomial(deg, x_train, y_train, xx_new):
    w_fit = fit_weights(deg, x_train, y_train)
    X_grid_matrix = polynomial_design_matrix(deg, xx_new)
    f = np.dot(X_grid_matrix, w_fit)
    return f

def linear_fit(xx, yy, xx_new):
    X_des = np.ones(len(xx), )
    X_des = np.vstack((X_des, xx))
    X_des = np.transpose(X_des)
    w_fit = np.linalg.lstsq(X_des, yy, rcond=None)[0]
    X_des_2 = np.ones(len(xx_new), )
    X_des_2 = np.vstack((X_des_2, xx_new))
    X_des_2 = np.transpose(X_des_2)
    f = np.dot(X_des_2, w_fit)
    return f

def _2_a(sample):
    x = X_shuf_train[sample, :]
    t = np.arange(0, 20, 1) / 20
    x_new = np.arange(0, 1.01, 0.01)
    f_quartic = fit_polynomial(4, t, x, x_new)
    f_linear = linear_fit(t, x, x_new)
    plt.scatter(t, x, label="input data", c='g')
    plt.plot(x_new, f_quartic, label="quartic fit")
    plt.plot(x_new, f_linear, label="linear fit")
    plt.scatter(1, y_shuf_train[sample], label="actual y")
    plt.legend(loc="upper left")
    print("Linear prediction = %f"%f_linear[-1])
    print("Quartic prediction = %f" %f_quartic[-1])
    print("Actual value = %f"%y_shuf_train[sample])
    #plt.show()

#_2_a(0)

"""
PART B:
as this captures the relationship between each two points and so is a better indicator for the predicted third
including 20 points means the linear regression line will predict the average trend across the 20
this will lead to an inaccurate prediction as the data has no clear linear pattern on this scale

PART C:
a context length of 5 data points fitted to a polynomial of degree 2
because locally (in areas of roughly 5 data points) the pattern follows a quadratic polynomial
"""

