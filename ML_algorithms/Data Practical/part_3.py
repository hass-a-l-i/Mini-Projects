from part_1 import *
from part_2 import _2_a
"""PART A: see board"""

"""PART B)i):"""
def Phi(C, K):
    t = np.arange(0, C, 1) / C
    X_des = np.ones(len(t), )
    for i in range(1, K):
        r = np.power(t, i)
        X_des = np.vstack((X_des, r))
    X_des = np.round(X_des, 5)
    X_des = np.transpose(X_des)
    return X_des

phi_test = Phi(20, 5)

"""PART B)ii):"""
def make_vv(C, K):
    phi = Phi(C, K)
    inv_phi = np.linalg.inv(np.dot(phi.T, phi))
    phi_1t = np.ones(K,)
    inv_phi_phi = np.dot(inv_phi, phi.T)
    vv = np.dot(phi_1t.T, inv_phi_phi)
    return vv

"""PART B)iii):"""
def v_pred(xx, sample_no):
    v1 = make_vv(len(xx), 2)
    print("Linear prediction = %f"% np.dot(v1, xx))
    v2 = make_vv(len(xx), 5)
    print("Quartic prediction = %f"%np.dot(v2, xx))
    yy = y_shuf_train[sample_no]
    print("Actual value = %f" %yy)


"""sample = 15
x = X_shuf_train[sample, :]
_2_a(sample)
v_pred(x, sample)"""

"""PART C)i):"""
def predictions(C, K, XX, yy, n):
    CC = len(XX[0, :][0:C])
    #print("Polynomial order: ", K - 1)
    #print("No. predictions: ", n)
    #print("Context length: ", CC)
    diffs = []
    for i in range(0, n):
        xx = XX[i, :][0:C]
        vv = make_vv(len(xx), K)
        pred = np.dot(vv, xx)
        if CC == C:
            YY = yy[i]
        else:
            YY = XX[i, :][C+1]
        diff = pred - YY
        diffs.append(diff)
        #print("Sample %d : Prediction = %f \t Actual = %f" % (i, pred, YY))
    diffs = np.array(diffs)
    sq_error = np.sqrt(np.sum(diffs * diffs))  * 1/len(diffs)
    #print("Context length: ", CC, "\tSquare error = ", sq_error)
    print("Polynomial order: ", K - 1, "\tSquare error = ", sq_error)


def find_min_sq_err(XX, yy):
    for j in range(5, 21):
        print("C = ", j)
        for i in range(2, 8):
            predictions(j, i, XX, yy, XX.shape[0])

#find_min_sq_err()
"""
C = 20, K = 6 gives min Square error =  14.905942233659195
"""

"""PART C)ii):"""
"""print("Train set:")
predictions(20, 6, X_shuf_train, y_shuf_train, X_shuf_train.shape[0])
print("Val set:")
predictions(20, 6, X_shuf_val, y_shuf_val, X_shuf_val.shape[0])
print("Test set:")
predictions(20, 6, X_shuf_test, y_shuf_test, X_shuf_test.shape[0])"""

"""
Train set:
Polynomial order:  5 	Square error =  14.905942233659195
Val set:
Polynomial order:  5 	Square error =  6.901774870709162
Test set:
Polynomial order:  5 	Square error =  6.911827495627355
"""

