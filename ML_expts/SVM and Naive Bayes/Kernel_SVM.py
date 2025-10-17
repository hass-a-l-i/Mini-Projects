# used when linear boundary clearly not good classifier
# e.g. if you have a 2d plot of circular data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_circles
from sklearn.svm import SVC
X, y = make_circles(100, factor=.1, noise=.1, random_state=0)
def view_data(data):
    plt.scatter(data[:, 0], data[:, 1], c=y, s=50, cmap='autumn')  # given array of array coords, so slice to get inside first array so can access coords then slice within coords array to get x's and y's for scatter
    plt.show()
# view_data(X)

# instead we project data into higher dimensions to make it linearly separable, use a radial basis function centred on central cluster
# first we move data into a higher dimensional representation
r = np.exp(-(X ** 2).sum(1))  # exp^sum as an array => we map the X into higher dimension by putting it in the power of exponent, sum here sums the each row (each coord) after squaring and then takes negative exp of result
def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()

# plot_3D()
# much easier to see separating hyperplane now as can see
# note we centred radial function on the middle cluster which worked out but if we did not then would not be as easy to separate cleanly
# note if wanted to centre elsewhere then use Euclidean dist squared (X-centre point)**2

# we want to create a basis function centred at EVERY point in dataset (kernel transformation) then use SVM to find optimal line by maximising margin in higher dimensional space which we can represent in our dimension (maximise dual problem, minimise primal problem)
# higher dimensional kernels basically do same thing as linear SVM but instead of measuring similarity between points with dot product, we replace this with another measure
# radial basis func is gaussian so allows for infinitely extrapolated a finite dimensional dataset as we can expand gaussian out using taylor expansion
# instead of doing computationally expensive projection of N points in N dimensions, we use kernel trick (map from 2D to 3D space using modified inner product of 2D vectors which is our kernel)
# apply kernel SVM using kernel hyperparameter
from sklearn.model_selection import train_test_split
SVM_model = SVC(kernel='linear', C=1E10)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   test_size=0.3, stratify=y,
                                   random_state=32)


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')

clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(clf)

