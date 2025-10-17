import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# first read file into dataframe
# if no header in date then header=None
# the data set will be used to see if person should go to comedy show or not based on info about each show and the comedian
# e.g. age of comedian, experience, rank of comedian, their nationality and if they went to that show
df = pd.read_csv("Decision 1.csv")

# all data numerical for decision tree to work
# so need nationality and go columns to also be in numerical form
# encode using own system, here encode list index for [UK, USA, N] and go / no go binary => use map method pandas
countries = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(countries)
went = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(went)
print(df)

# next need to ready data => split into feature X and target y columns
# X = data used to predict, y is prediction output
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]  # can do this in Pandas to split dataframe
y = df['Go']
print(X)
print(y)

# now create decision tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(X.values, y) # note use x.values here as y does not have col labels => keep consistent
tree.plot_tree(decision_tree, feature_names=features)
plt.show()

"""
RESULTS EXPLAINED:
Remember this is using past data to come to a decision on whether go to show
Inside box are conditions fulfilled to go down to next yes node, else no node

Rank <= 6.5 means comedian with rank satisfying this goes down yes arrow

gini is quality of split , always between 0.0 and 0.5 => 0.0 means all samples same result, 0.5 means 50:50 split 
gini uses formula: Gini = 1 - (x/n)^2 - (y/n)^2 => x = # positive y targets (# YES here), n = # samples, y = # negative y targets (# NO here)
is essentially showing the inequality among the frequency distr of our y targets

samples shows how many samples left at this node to classify

value = [6,7] means at this node 6 have a NO and 7 have a YES (GO) as classifier from known data

Goes for those that have higher rank, we are ofc more likely to go so these go down to next node which is next feature
then this feature has a constraint, Nationality <= 0.5 then same thing again until go through all features
note ths constraint on nationality is good as keeps us in UK 

not that those which are disregarded have gini = 0.0 means all samples same result (NO) and tree end here
BUT gini 0.0 at lowest depth with values = [0, 1] show after going thru tree, only one show worth going to 
and other discarded value = [1,0]
note experience node above this is gini 0.5 as sample = 2 as only 2 left
"""

# can also predict values using decision tree for set of input features e.g.
# use 2D array as it technically list in list
# e.g. here is all input same except comedy to show that it would not be classified as GO given different inputs
# const features => age = 40, experience = 10, nationality = 1 (USA)
# vary ranking feature to see at which point we go
# Decision Tree does not give us a 100% certain answer. It is based on the probability of an outcome, and the answer will vary
print("Prediction e.g. if comedy rank 6 then go or no? => ", decision_tree.predict([[40, 10, 6, 1]]))
print("Prediction e.g. if comedy rank 7 then go or no? => ", decision_tree.predict([[40, 10, 7, 1]]))