# supervised learning
# unlike naive bayes as is not generative classification (where focus on probability distribution of dataset to generate new labels)
# here we instead use discriminative classification => find line to divide classes in 2D (manifold in 3D)

# import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# let's import random dataset and view it first, so we can see what we want to classify with SVM
# use make_blobs => gives random sample with set random seed and std dev
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=500, centers=2,
                           random_state=0, cluster_std=0.60)
def view_data():
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')  # given array of array coords, so slice to get inside first array so can access coords then slice within coords array to get x's and y's for scatter
    plt.show()
#  view_data(X)

# can see many possible lines here can divide as linear classifier
def three_possible_divs():
    xfit = np.linspace(-1, 3.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)  # if we add new data point (marked X as e.g.) to plot then will be assigned diff label depending on which line decided so need SVmM instead
    for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
        plt.plot(xfit, m * xfit + b, '-k')
    plt.xlim(-1, 3.5)
    plt.show()
# three_possible_divs()


# this is why introduce margin from SVM => width of our dividing line
# want to max the margin to find optimal classifier
# use sklearn import for this, also remember to split into train and test data
from sklearn.svm import SVC  # "support vector classifier"
from sklearn.model_selection import train_test_split
SVM_model = SVC(kernel='linear', C=1E10)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   test_size=0.3, stratify=y,
                                   random_state=32)
SVM_model.fit(X_train, y_train)

# can make function to show the decision boundaries of classifier too
def plot_2D_svc_decision_function(model):
    plt.figure(figsize=(10, 8))
    # Plotting our two-features-space
    sns.scatterplot(x=X_train[:, 0],
                    y=X_train[:, 1],
                    hue=y_train,
                    s=8)
    # Constructing a hyperplane using a formula.
    w = model.coef_[0]  # w consists of 2 elements
    b = model.intercept_[0]  # b consists of 1 element
    x_points = np.linspace(-1, 4)  # generating x-points from -1 to 1
    y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
    # Plotting a red hyperplane (our main decision boundary)
    plt.plot(x_points, y_points, c='r')
    # Encircle support vectors (i.e. data points which our margin shifted lines cross (which we found as maximum distance from boundary))
    plt.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=50,
                facecolors='none',
                edgecolors='k',
                alpha=.5)
    # Step 2 (unit-vector for our margin):
    magnitude_w = np.sqrt(np.sum(model.coef_[0] ** 2))  # = sqrt(w_1 squared + w_2 squared)
    w_hat = w / magnitude_w
    # Step 3 (margin):
    margin = 1 / magnitude_w
    # Step 4 (calculate points of the margin lines):
    decision_boundary_points = np.array(list(zip(x_points, y_points)))
    points_of_line_above = decision_boundary_points + w_hat * margin
    points_of_line_below = decision_boundary_points - w_hat * margin
    # Plot margin lines
    # margin line above
    plt.plot(points_of_line_above[:, 0],
             points_of_line_above[:, 1],
             'b--',
             linewidth=2)
    # margin line below
    plt.plot(points_of_line_below[:, 0],
             points_of_line_below[:, 1],
             'b--',
             linewidth=2)
    plt.show()
#  plot_2D_svc_decision_function(SVM_model)

# note to access the support vectors (i.e. data points which our margin shifted lines cross (which we found as maximum distance from boundary)
# note these support vectors are all that matter to our model => loss function is dependent on maxing the margin
print(SVM_model.support_vectors_)
