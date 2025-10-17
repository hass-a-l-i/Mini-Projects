import numpy as np
from matplotlib import pyplot as plt

def sigmoid(a): return 1. / (1. + np.exp(-a))
def relu(x): return np.maximum(x, 0)
def linear(a): return a

def neural_net(X, layer_sizes, gg=None, sigma_w=1):
    for i in layer_sizes:
        Wt = sigma_w * np.random.randn(X.shape[1], i) # random weights
        X = gg(X @ Wt)
    return X

N = 100
layer_1 = 100
layer_2 = 50
X = np.linspace(-2, 2, num=N)[:, None]  # N,1
plt.clf()

for i in range(5):
    ff = neural_net(X, gg=sigmoid, layer_sizes=(layer_1, layer_2, 1))
    plt.plot(X, ff)
plt.show()

