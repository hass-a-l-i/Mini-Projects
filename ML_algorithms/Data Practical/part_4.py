import pylab as pl

from part_3 import *

def find_best_C(XX, yy, K):
    for i in range(2, XX.shape[1]+1):
        predictions(i, K, XX, yy, XX.shape[0])

def find_best_K(XX, yy, C):
    for i in range(2, 10):
        predictions(C, i, XX, yy, XX.shape[0])


#find_best_K(X_shuf_val, y_shuf_val, 20)

def histogram2(C, K, XX, yy, n):
    CC = len(XX[0, :][0:C])
    diffs = []
    for i in range(0, n):
        xx = XX[i, :][0:C]
        vv = make_vv(len(xx), K)
        pred = np.dot(vv, xx)
        if CC == C:
            YY = yy[i]
        else:
            YY = XX[i, :][C + 1]
        diff = pred - YY
        diffs.append(diff)
    diffs = np.array(diffs)
    plt.subplot(1, 2, 1)
    plt.hist(amp_data, density=True, bins=100)
    plt.ylabel('Frequency')
    plt.xlabel('Data')
    plt.title("amp data")
    plt.subplot(1, 2, 2)
    plt.hist(diffs, density=True, bins=100)  # density=False would make counts
    plt.title("Validation data")
    plt.ylabel('Frequency')
    plt.xlabel('Data')
    plt.tight_layout()
    plt.show()

histogram2(20,6,X_shuf_val, y_shuf_val, X_shuf_val.shape[0])
