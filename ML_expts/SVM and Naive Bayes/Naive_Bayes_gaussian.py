# v. fast classifier, v. simple too for classification, good for high dimensional large datasets
# based on bayes theorem, we want to find label given we have features = P(L|features) = posterior prob
# if deciding between 2 labels then just take ratio of posteriors of L1 and L2
# naive bayes is a generative classifier = opposite of discriminative, it assigns probability to a feature given a label instead of a label given to a feature (discriminative)
# ends up generating new outputs e.g. ChatGPT generating next word in sentence using previous words = using probabilities and assigning feature to label
# naive because of assumptions made, one common one is all data points are independent of each other =, other assumptions limit this model, but it is fast classifier albeit very simplistic
# naive assumptions rarely match data which is why this is not used as much, but it does apply to high dimensional data where is it more likely that points are closer to each other so categories likely more well defined as dimensions increase
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# first we look at gaussian naive bayes=> naive assumption here is data from each label drawn is from simple gaussian distr
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
def view_data():
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
    plt.show()
# view_data()

# can also assume no covariance between dimensions
# fit model by finding mean and std dev of points under each label => the fit these as gaussians for each label
# we then compute likelihood P(features|L) for all data points and find posterior for each label with using bayes eq then using this we find the label most probable for each point (i.e. whichever probability is bigger from P(L1|features) or P(L2|features))
# make our model and fit data
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
# generate new data to start making predictions
rng = np.random.RandomState(0)
X_new = [-6, -14] + [14, 18] * rng.rand(2000, 2)  # generate array between same limits as before => multiple values so that we can see rough idea of decision boundary between clusters
y_new = model.predict(X_new)

# now we can "find" our decision boundary - for gaussian naive bayes decision boundary is quadratic, can see slight curve in below
def simple_gaussian_fit_plot():
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
    lim = plt.axis()
    plt.scatter(X_new[:, 0], X_new[:, 1], c=y_new, s=20, cmap='RdBu', alpha=0.1)
    plt.axis(lim)
    plt.show()
# simple_gaussian_fit_plot()

# we can also use library to find probabilistic classification using predict_proba
y_prob = model.predict_proba(X_new)
print(y_prob[-8:].round(2))  # returns rounded posteriors for first and second labels, only for last 8 features here as large dataset



