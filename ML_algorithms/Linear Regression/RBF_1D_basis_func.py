import numpy as np
import matplotlib.pyplot as plt

# define RBF, note we use double letters to know that these belong to this function
def rbf_1d(xx, cc, hh):
    return np.exp(-(xx-cc)**2 / hh**2)

# plot 3 different RBFs using different parameters
def plot_test_rbf():
    plt.clf()
    grid_size = 0.01
    x_grid = np.arange(-40, 40, grid_size)
    plt.plot(x_grid, rbf_1d(x_grid, cc=10, hh=1), '-b', label="cc = 10, hh = 1")
    plt.plot(x_grid, rbf_1d(x_grid, cc=0, hh=2), '-r', label="cc = 0, hh = 2")
    plt.plot(x_grid, rbf_1d(x_grid, cc=-10, hh=5), '-g', label="cc = 10, hh = 5")
    plt.ylabel("x")
    plt.xlabel("c (centre) value")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # notice as hh increases, RBF is wider, and the cc is just where they are centred

# define log sigmoid function
def rbf_log_sigmoid_1d(xx, vv, bb):
    alpha = (-vv * xx) - bb
    return 1/(1 + np.exp(alpha ** 2))

# plot 3 different log sigmoid funcs using different v's
def plot_log_sigmoid_rbf_vv(vv1, vv2, vv3):
    plt.clf()
    grid_size = 0.01
    x_grid = np.arange(-100, 100, grid_size)
    plt.plot(x_grid, rbf_log_sigmoid_1d(x_grid, vv = vv1, bb = 0), '-b', label="vv = %.2f, bb = 0"%vv1)
    plt.plot(x_grid, rbf_log_sigmoid_1d(x_grid, vv = vv2, bb = 0), '-r', label="vv = %.2f, bb = 0"%vv2)
    plt.plot(x_grid, rbf_log_sigmoid_1d(x_grid, vv = vv3 , bb = 0), '-g', label="vv = %.2f, bb = 0"%vv3)
    plt.ylabel("σ(x)")
    plt.xlabel("x")
    plt.legend(loc="upper left")
    plt.grid()
    plt.title("σ(x) = 1/(1 + exp[- vx - b])")
    plt.show()
    # b moves the RBF along the x-axis
    # v affects width - smaller means wider

# plot 3 different log sigmoid funcs using different b's
def plot_log_sigmoid_rbf_bb(bb1, bb2, bb3):
    plt.clf()
    grid_size = 0.01
    x_grid = np.arange(-150, 150, grid_size)
    plt.plot(x_grid, rbf_log_sigmoid_1d(x_grid, vv = 0.05, bb = bb1), '-b', label="vv = 0.05, bb = %.2f"%bb1)
    plt.plot(x_grid, rbf_log_sigmoid_1d(x_grid, vv = 0.05, bb = bb2), '-r', label="vv = 0.05, bb = %.2f"%bb2)
    plt.plot(x_grid, rbf_log_sigmoid_1d(x_grid, vv = 0.05 , bb = bb3), '-g', label="vv = 0.05, bb = %.2f"%bb3)
    plt.ylabel("σ(x)")
    plt.xlabel("x")
    plt.legend(loc="upper left")
    plt.grid()
    plt.title("σ(x) = 1/(1 + exp[- vx - b])")
    plt.show()
    # b moves the RBF along the x-axis
    # v affects width - smaller means wider
    # shift for b is quicker if v small

# finds intersection points of curve with y-axis for given f, bb, vv
def find_curve_cross(f, bb, vv):
    plt.clf()
    grid_size = 0.01
    x_grid = np.arange(-50, 250, grid_size)
    plt.plot(x_grid, rbf_log_sigmoid_1d(x_grid, vv=vv, bb=bb), '-b')
    xx_neg = (- bb - np.sqrt(np.log((1 / f) - 1))) / vv
    xx_pos = (- bb + np.sqrt(np.log((1 / f) - 1))) / vv
    print("First intersection point = (", xx_neg, ", ", f, ")")
    print("Second intersection point = (", xx_pos, ", ", f, ")")
    plt.axhline(y=f, color='r', linestyle='--')
    plt.axvline(x=xx_neg, color='r', linestyle='--')
    plt.axvline(x=xx_pos, color='r', linestyle='--')
    plt.ylabel("σ(x)")
    plt.xlabel("x")
    plt.grid()
    plt.title("σ(x) = %.2f solutions"%f)
    plt.show()

# create rbf list  and quadratic list from input we can plot for linear combo
def create_rbf_plot_list(cc, hh):
    grid_size = 0.01
    x_grid = np.arange(-40, 40, grid_size)
    rbf = []
    for i in x_grid:
        rbf.append(rbf_1d(i, cc=cc, hh=hh))
    return rbf

def quadratic_basis_func(xx, cc):
    return -(xx - cc) ** 2

def create_quadratic_plot_list(cc):
    grid_size = 0.01
    x_grid = np.arange(-40, 40, grid_size)
    rbf = []
    for i in x_grid:
        rbf.append(quadratic_basis_func(i, cc=cc))
    return rbf


# define weights then f(x) = w^T * phi plotted, one for RBFs and other quadratics
def linear_combo_RBF(w1, w2, w3, c1, c2, c3, hh1, hh2, hh3):
    w = np.array([w1, w2, w3])
    rbf1 = create_rbf_plot_list(c1, hh1)
    rbf2 = create_rbf_plot_list(c2, hh2)
    rbf3 = create_rbf_plot_list(c3, hh3)
    f = np.multiply(w[0], rbf1) + np.multiply(w[1], rbf2) + np.multiply(w[2], rbf3)
    plt.clf()
    grid_size = 0.01
    x_grid = np.arange(-40, 40, grid_size)
    plt.plot(x_grid, f, '-b')
    plt.ylabel("x")
    plt.xlabel("c (centre) value")
    # plt.legend(loc="upper left")
    plt.grid()
    plt.show()

def linear_combo_quadratic(w1, w2, w3):
    w = np.array([w1, w2, w3])
    q1 = create_quadratic_plot_list(10)
    q2 = create_quadratic_plot_list(0)
    q3 = create_quadratic_plot_list(-10)
    f = np.multiply(w[0], q1) + np.multiply(w[1], q2) + np.multiply(w[2], q3)
    plt.clf()
    grid_size = 0.01
    x_grid = np.arange(-40, 40, grid_size)
    plt.plot(x_grid, f, '-b')
    plt.ylabel("x")
    plt.xlabel("c (centre) value")
    # plt.legend(loc="upper left")
    plt.grid()
    plt.show()



plot_test_rbf()
# plot_log_sigmoid_rbf_vv(0.03, 0.05, 0.1)
# plot_log_sigmoid_rbf_bb(-5, 0, 5)
# find_curve_cross(0.4, -5, 0.03)
# linear_combo_RBF(10, 20, 10, 10, 0, -10, 2, 3, 5)
# linear_combo_quadratic(1, 5, 10)