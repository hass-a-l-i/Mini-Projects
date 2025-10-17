# predict if customer likely to repay loan
# features include initial payment amount, last payment amount, credit score, house number,
# whether they managed to repay the loan is target variable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

data = pd.read_csv("Decision 2.csv")
print("Dataset length = ", len(data))
print("Dataset shape = ", data.shape)
result = {"yes": 1, "No": 0}
data['Result'] = data['Result'].map(result)
print(data.head())  # print first 5 rows to save computation power and to check formatting

# now we train and test, first split set
X = data.values[:, 0:4]
y = data.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)  # here test size = % of data used to test, remainder to train and random state fixed just fixes random number to get same result each run

# now use entropy (shannon) to train model instead of gini impurity used in prev tree
entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)  # can fix depth fo tree and samples per leaf (= min samples required to split a node i.e. below this and terminate and node is a leaf
entropy.fit(X_train, y_train)

# can now make predictions
y_prediction = entropy.predict(X_test)  # using our test set as we have trained with x and y train sets
print(y_prediction)

# check accuracy
# sees how close our predicted y's were to actual y's
print("Accuracy is", accuracy_score(y_test, y_prediction) * 100, "%")

# so can predict with this accuracy for a new customer who approaches (remember is a list in a list)
new_customer = entropy.predict([[200, 100112, 312, 3009]])
print("Will new customer be accepted for loan: ", new_customer)
