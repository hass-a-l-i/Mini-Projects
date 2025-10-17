# if data has some overlap - this is what we call soft margin instead of hard like before

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.svm import SVC

X, y = datasets.make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=1.2)
def view_data():
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')  # given array of array coords, so slice to get inside first array so can access coords then slice within coords array to get x's and y's for scatter
    plt.show()
# view_data()
x = X[:, 0]
max_x = max(x)
min_x = min(x)
# softening margin means allow some data within the margin to allow for best fit => this hyperparameter is something e can tune, called C - larger C is harder margin
# lets plot for different C values

C = input("Please choose a float C value \n")
C = float(C)
soft_model = SVC(kernel='linear', C=C).fit(X, y)
def soft_margin_visualiser(model):
    sns.set()
    plt.figure(figsize=(10, 8))
    # Plotting our two-features-space
    sns.scatterplot(x=X[:, 0],
                    y=X[:, 1],
                    hue=y,
                    s=8)
    # Constructing a hyperplane using a formula.
    w = model.coef_[0]  # w consists of 2 elements
    b = model.intercept_[0]  # b consists of 1 element
    x_points = np.linspace(min_x, max_x)  # generating x-points from -1 to 1
    y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
    # Plotting a red hyperplane (our main decision boundary)
    plt.plot(x_points, y_points)
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
             'r--',
             linewidth=2)
    # margin line below
    plt.plot(points_of_line_below[:, 0],
             points_of_line_below[:, 1],
             'r--',
             linewidth=2)
    plt.show()


# noinspection PyTypeChecker
soft_margin_visualiser(soft_model)