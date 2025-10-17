import numpy as np

# np cov computes expecations based on gaussian normal distr
# here we take x = vector with N random vars
# stack same x next to it then find covar between them = 2 x 2 matrix for each combo of x1, x2
x = np.random.randn(10**6);
x1 = x[:,None]
x2 = x1
X = np.hstack((x1, x2))
# need to transpose as input (D,N) required for cov
Sigma = np.cov(X.T)
print(Sigma)

# use cholesky decomp to find transform A (N > D)
N = 5
D = 2
Y = np.random.randn(D, N)
Sigma2 = np.cov(Y)
A = np.linalg.cholesky(Sigma2)
Sigma_from_A = A @ A.T # @ FOR MULTIPLYING NP MATRICES
print(Sigma2)
print(Sigma_from_A)

