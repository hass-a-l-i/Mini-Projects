import numpy as np
import matplotlib.pyplot as plt


amp_data = np.load('amp_data.npz')['amp_data']
x = np.arange(0, len(amp_data), 1)

"""PART A:"""
def histogram():
    plt.hist(amp_data, density=False, bins=100)
    plt.ylabel('Frequency')
    plt.xlabel('Data')
    plt.title("amp data")
    plt.show()

def plot(xx, yy):
    plt.plot(xx, yy)
    plt.show()

#plot(x, amp_data)
#histogram()

"""
IMPORTANT SENTENCES FOR A)
majority of data amplitude is around zero - mean likely close to zero too
spread of data around mean is quite high, larger variance compared to mean is likely
"""


def R_x_21():
    np.random.seed(10)
    R = int(len(amp_data) / 21)
    remainder = len(amp_data) - (R * 21)
    amp_data_21 = amp_data[0:-remainder]
    M = amp_data_21.reshape(R, 21)
    np.random.shuffle(M)
    return M

"""PART B:"""
def train_test_val_yy(MM, train, test , val, data_check):
    rows = len(MM[0:,:])
    train_idx = int(round((rows * train),0))
    val_idx = int(round(((rows * val) + train_idx), 0))
    test_idx = int(round(((rows  * test) + val_idx), 0))
    M_shuf_train, M_shuf_val, M_shuf_test = MM[0:train_idx, :], MM[train_idx:val_idx, :], MM[val_idx:test_idx, :]
    if data_check:
        print("Data check: MM (before yy taken)")
        print(MM.shape, M_shuf_train.shape, M_shuf_val.shape, M_shuf_test.shape)
        print(MM == np.vstack((M_shuf_train, M_shuf_val, M_shuf_test)))
    yy_shuf_train, yy_shuf_val, yy_shuf_test = M_shuf_train[:,-1], M_shuf_val[:,-1], M_shuf_test[:,-1]
    if data_check:
        print("Data check: yy")
        print(yy_shuf_train.shape, yy_shuf_val.shape, yy_shuf_test.shape)
        print(MM[:, -1] == np.concatenate((yy_shuf_train, yy_shuf_val, yy_shuf_test), axis=0))
    M_shuf_train, M_shuf_val, M_shuf_test = (
        np.delete(M_shuf_train, -1, axis=1), np.delete(M_shuf_val, -1, axis=1), np.delete(M_shuf_test, -1, axis=1))
    if data_check:
        print("Data check: MM (after yy taken)")
        print(MM.shape, M_shuf_train.shape, M_shuf_val.shape, M_shuf_test.shape)
        print(MM[:,0:-1] == np.vstack((M_shuf_train, M_shuf_val, M_shuf_test)))
    return M_shuf_train, yy_shuf_train, M_shuf_val, yy_shuf_val, M_shuf_test, yy_shuf_test


X = R_x_21()

X_shuf_train, y_shuf_train, X_shuf_val, y_shuf_val, X_shuf_test, y_shuf_test\
    = train_test_val_yy(X, 0.7, 0.15, 0.15, False)

