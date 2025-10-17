import numpy as np
import matplotlib.pyplot as plt

"""Set noisy data along linear function y = w1*x + w0 + std_normal_noise"""
def generate_random_data(lower, upper, step, grd, c, std_dev_noise, fix_seed):
    if fix_seed:
        np.random.seed(upper)
    xx = np.arange(lower, upper, step)
    noise = np.random.normal(0, std_dev_noise ** 2, len(xx))
    yy = (grd * xx) + c + noise
    return xx, yy


w0 = 4
w1 = 3
x, y = generate_random_data(-10, 10, 0.05, w1, w0, 2, True)
plt.scatter(x, y)
plt.show()


