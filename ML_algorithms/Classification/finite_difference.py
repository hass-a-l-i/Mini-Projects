import numpy as np
import matplotlib.pyplot as plt


def log_sigmoid(a):
    sigma = 1 / (1 + np.exp(-a))
    return sigma

def finite_diff_check(a):
    epsilon = 10 ** -5
    f_plus = log_sigmoid(a + (epsilon/2))
    f_minus = log_sigmoid(a - (epsilon/2))
    diff_1 =(f_plus - f_minus)/epsilon
    diff_2 = log_sigmoid(a) * (1 - log_sigmoid(a))
    print("differentiated - finite diff", diff_2 - diff_1)


finite_diff_check(2)


