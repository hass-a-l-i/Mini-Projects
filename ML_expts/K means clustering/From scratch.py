# unsupervised ML algo => all indep variables
# given dataset which form clusters when plotted
# classify a cluster by finding centre points of centroids and nearest centroid is the group for that data point
# we have k centroids => initialize k random points and find dist from these to centre of closest centroid and min each iteration

# generate dataset using sklearn
import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

def make_initial_clusters():
    centers = 5
    X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)  # normalises to unit variance bewtween datapoints
    sns.scatterplot(x=[X[0] for X in X_train],
                    y=[X[1] for X in X_train],
                    hue=true_labels,
                    palette="deep",
                    legend=None
                    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# calculate distances between points using euclidean distance
def euclidean(point,data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))

# now create K means class - allows us to create object with clusters and iterations attributes
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    # initialize centroids with function
    def fit(self, X_train):
        # first randomly generate our centroid points which we iteratively map toward each centroid centre
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [random.uniform(min_, max_) for _ in range(self.n_clusters)]  # centre of our centroid

        # iterate and min dist until centroid points stop moving or iteration number finished
        # adjust location of centroid points to mean of the points belonging to that centroid (aka where we want to be for each cluster)
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each data point, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)] # make list of lists
            for x in X_train:
                dists = euclidean(x, self.centroids)  # dist between points from each centroid and our x points we want to centre
                centroid_idx = np.argmin(dists)  # minimum distance i.e. we have assigned our centroid points we want to centre to their closes centroids
                sorted_points[centroid_idx].append(x)

            # reassign centroids from previous to current for current iteration using mean of the points belonging to each centroid
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]  # find mean of each cluster we want to approach with centroid points
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    # evaluation function to see if can correctly classify set of random points to correct centroids
    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs

# now train and test on original data
# same method of plotting with colours of diff labels but now our markers different style for predicted labels
def first_model():
    centers = 5
    X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)  # normalises to unit variance bewteen datapoints
    kmeans = KMeans(n_clusters=centers)
    kmeans.fit(X_train)
    # View results
    class_centers, classification = kmeans.evaluate(X_train)
    sns.scatterplot(x=[X[0] for X in X_train],
                    y=[X[1] for X in X_train],
                    hue=true_labels,
                    style=classification,
                    palette="deep",
                    legend=None
                    )
    plt.plot([x for x, _ in kmeans.centroids],
             [y for _, y in kmeans.centroids],
             '+',
             markersize=10,
             )
    plt.show()


# first_model()
# above model does not do well - if centroid initialised far from group, it's unlikely to move and conversely if too close to centroid unlikely to move away
# solve by using kmeans ++ to initialize centroid
# first initialize centroid as random selection of one of the data points
# second calc sum of dists between centroid and each data point in all centroids
# third select next centroid with prob proportional to dists to each cluster
# repeat for all centroids

class KMeans_plus_plus:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter


def fit(self, X_train):
    # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
    # then the rest are initialized w/ probabilities proportional to their distances to the first
    # Pick a random point from train data for first centroid
    self.centroids = [random.choice(X_train)]
    for _ in range(self.n_clusters - 1):
        # Calculate distances from points to the centroids
        dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
        # Normalize the distances
        dists /= np.sum(dists)
        # Choose remaining points based on their distances (prob distr)
        new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
        self.centroids += [X_train[new_centroid_idx]]
    # This initial method of randomly selecting centroid starts is less effective
    # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
    # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
    # Iterate, adjusting centroids until converged or until passed max_iter
    iteration = 0
    prev_centroids = None
    while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
        # Sort each datapoint, assigning to nearest centroid
        sorted_points = [[] for _ in range(self.n_clusters)]
        for x in X_train:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            sorted_points[centroid_idx].append(x)
        # Push current centroids to previous, reassign centroids as mean of the points belonging to them
        prev_centroids = self.centroids
        self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
        for i, centroid in enumerate(self.centroids):
            if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                self.centroids[i] = prev_centroids[i]
        iteration += 1

        def evaluate(self, X):
            centroids = []
            centroid_idxs = []
            for x in X:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                centroids.append(self.centroids[centroid_idx])
                centroid_idxs.append(centroid_idx)
            return centroids, centroid_idxs


# Create a dataset of 2D distributions
centers = 5
X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
X_train = StandardScaler().fit_transform(X_train)
# Fit centroids to dataset
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)
# View results
class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         'k+',
         markersize=10,
         )
plt.show()

