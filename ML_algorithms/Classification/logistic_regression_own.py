import numpy as np
import matplotlib.pyplot as plt

def log_sigmoid(XX, ww):
    ff = np.dot(XX, ww)
    sigma = 1 / (1 + np.exp(-ff))
    return sigma

def gradient(XX, yy, ww):
    M = len(yy)
    ff = log_sigmoid(XX, ww)
    diff = ff - yy
    grd = np.dot(XX.T, diff) * 1/M  # change in x values * 1/M
    return grd

def log_likelihood(X, yy, ww):
    M = len(yy)
    sigma_y_1 = log_sigmoid(X, ww)
    sigma_y_0 = 1 - sigma_y_1
    summand = yy * sigma_y_1 + (1 - yy) * sigma_y_0
    L_mw = - np.sum(np.log(summand)) * 1/M  # monte carlo approx
    return L_mw

def SGD(XX, yy, lr, epochs):
    ww = np.zeros(XX.shape[1])  # start with 0s as weights
    for i in range(epochs):
        grad = gradient(XX, yy, ww)
        ww = ww - (lr * grad) # update each epoch
        log_like = log_likelihood(XX, yy, ww)
        loss = log_like * 100
        print("Epoch {0} : loss = {1:.2f}%".format(i+1, loss))
    return ww

y = np.concatenate((np.zeros(10,), np.ones(10,)))
x = np.arange(0, 10, 0.5)
X_des = np.vstack((np.ones(len(x), ), x))
X_des = np.transpose(X_des)
w = SGD(X_des, y, 0.1, 1000)

f = log_sigmoid(X_des, w)
plt.plot(x, f, c="r", label="logistic regression")
plt.scatter(x, y, label="data")
plt.legend(loc="upper left")
plt.show()

