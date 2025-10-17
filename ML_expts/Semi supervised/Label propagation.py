# supervised learning = train algo using labeled data
# unsupervised = train using unlabled data and allow algo to identify own patterns and groupings
# semi supervised = mix of labelled and unlabelled data used to train

# label propagation algo is classification algo assigning labels to unlabelled data
# uses a network structure where connections denote similarity between data points for the classes we want to classify them in
# results in communities being formed where common classes cluster together (community = densely connected network as share same class)
# labels then propagate through network - works iteratively to connect like labels and form clusters
# reaches convergence when each node has majority label of neighbours
# semi supervised as to start clustering we form network with some labels being known then apply algo to network
# notion of strength of a label is apparent here given the neighbouring nodes of a given node

# relevant libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation
import numpy as np

# first we define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
# split into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
# split training set into portion we will label and other portion unlabelled (50:50)
X_train_labelled, X_train_unlabelled, y_train_labelled, y_train_unlabelled = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
# summarise training set size (remember y is labels to 1D array, X is data points so 2D)
print('Labeled Train Set:', X_train_labelled.shape, y_train_labelled.shape)
print('Unlabeled Train Set:', X_train_unlabelled.shape, y_train_unlabelled.shape)
# summarize test set size
print('Test Set:', X_test.shape, y_test.shape)


# now use supervised learning on labelled training set to see if we outperform it with semi supervised
# using logistic regression on labelled data
model = LogisticRegression()
model.fit(X_train_labelled, y_train_labelled)
# now make predictions on test set and compare to test labels to find accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print('Accuracy for supervised learning: %.3f' % (acc*100))


# now we compare this to label propagation - use sklearn to access algo
model2 = LabelPropagation()
# again fit model using training data and make predictions
# but we fit using mixed training set now with labeled examples and unlabelled examples with label -1
# model then assigns label to unlabeled examples by fitting model
# then use transduction function to find estimated labels from training
# first prepare label + unlabeled data
X_train_mixed = np.concatenate((X_train_labelled, X_train_unlabelled))
# now make -1 label for those with no label in unlabelled part of training set
no_label = [-1 for i in range(len(y_train_unlabelled))]
# now make mixed labels with -1 labels as unlabelled above
y_train_mixed = np.concatenate((y_train_labelled, no_label))

# now able to train label propagation model
model2.fit(X_train_mixed, y_train_mixed)
# now test accuracy of model with predictions on test set
y_pred2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
print('Accuracy for semi-supervised learning: %.3f' % (acc2*100))

# one more approach = estimate labels for training set using semi supervised above and fit supervised learning model
# we get labels from model2 above using transduction function
train_labels = model2.transduction_
# now we define supervised learning model, fit and predict then compare to test
model3 = LogisticRegression()
model3.fit(X_train_mixed, train_labels)
y_pred2 = model3.predict(X_test)
acc3 = accuracy_score(y_test, y_pred2)
print('Accuracy for semi supervised + supervised learning: %.3f' % (acc3*100))





