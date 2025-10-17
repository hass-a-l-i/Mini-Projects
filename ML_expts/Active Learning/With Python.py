# supervised learning - we now select and label examples which enhance the models quality => some subset of total unlabelled set
# improves performance when labeling is costly
# this is in contrast to traditional method which randomly selects examples
# steps:
# 1. random subset of unlabeled data is labelled
# 2. train model on labelled
# 3. query strategy - selects most informative examples from remaining unlabeled data (chooses complex or uncertain examples as these give most info)
# multiple types of query strategy - revolve around which metric to maximise, i.e. entropy, uncertainty, representational coverage, change in predictions
# 4. labelling
# 5. update model by adding labelled examples to training set then retraining model from scratch using new training set (which we added our most "important" data points to from last step)
# 6. repeat 3 to 5 iteratively with model increasing accuracy and needing fewer labelled examples each time trained
# used when there is limited data - need strategy to increase accuracy given limited dataset which is active learning
# can also apply to deep learning - same idea as we feed NN the examples which are most informative to train off of

# python example
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load your dataset and split into initial labeled and unlabeled sets
np.random.seed(10)
X = np.random.rand(100, 2)  # 100 data points i.e. coords of each data point so each feature is x, y here
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # binary classification target
dataset = list(zip(X, y))  # total dataset to assign label for each data point
X_labeled, X_validation, y_labeled, y_validation = train_test_split(X, y, test_size=0.8, stratify=y)
X_unlabeled = X_validation
# Initialize model and train on initial labeled data
model = LogisticRegression()
model.fit(X_labeled, y_labeled)

# Define the active learning loop
num_iterations = 10
batch_size = 10

# function for finding instances to update our learning algorithm
def get_labels_for_instances(data, input_instances):
    labels = []
    for i in range(len(input_instances)):
        lists_only = [i[0] for i in data]
        find = input_instances[i]
        index_found = np.where((lists_only == find).all(axis=1))[0][0]
        label_out = data[index_found][1]
        labels.append(label_out)
    return labels

# training loop
for iteration in range(num_iterations):
    # error handling
    if np.shape(X_unlabeled) == (0, 2):
        break

    # Implement a query strategy (e.g., uncertainty sampling)
    uncertainty = model.predict_proba(X_unlabeled)  # find log probabilities for each sample (2 columns as we have 2 classes, and we have this for each data point = prob data point belonging to each class (here 1 or 0))
    uncertainty_scores = np.max(uncertainty, axis=1)  # find max out of the two probs above i.e. the most likely class for each data point
    query_indices = np.argsort(uncertainty_scores)[-batch_size:]  # gives indices of largest probs we have from above, here its indices of top 10 (batch size is 10)

    # Label the selected instances
    labeled_instances = X_unlabeled[query_indices]  # returns the top 10 features which have largest probs so are easiest to classify as our instances
    labeled_labels = get_labels_for_instances(dataset, labeled_instances)  # finds the corresponding labels of the above

    # Update the labeled and unlabeled datasets - add on the ones we identified to the training set and remove from the test set
    X_labeled = np.concatenate((X_labeled, labeled_instances), axis=0)
    y_labeled = np.concatenate((y_labeled, labeled_labels), axis=0)
    X_unlabeled = np.delete(X_unlabeled, query_indices, axis=0)

    # Retrain the model on the updated labeled dataset
    model.fit(X_labeled, y_labeled)

    # Evaluate the model on a validation set
    validation_accuracy = accuracy_score(y_validation, model.predict(X_validation))
    print(f"Iteration {iteration+1}, Validation Accuracy: {validation_accuracy:.4f}")


