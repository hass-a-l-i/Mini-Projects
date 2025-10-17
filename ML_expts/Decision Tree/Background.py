# Decision tree basics

# have root node that then branches out using conditions into leaf nodes
# terminal nodes are once data has been classified = at the end of a certain branch
# tuned params at each branching of nodes used to classify
# like a bunch of nested if else statements
# used when working with non-linear data
# each node = a feature
# layers of a tree = depth

# function of decision tree is to divide data set into small groups until reach sets small enough to have a label
# entropy decreases as progress down decision tree
# number of branches increase mean more computationally expensive
# larger number of splits = can over-fit
# avoid over-fitting with pruning
# can do pruning while tree is growing or after all branches have been made

# regression and classification on labelled data = decision tree
# supervised learning
# decision tree because at each node a decision made based of some feature

# entropy = measure of uncertainty / randomness in data
# handles how the data is split => sum(prob(value_i) * log_2(prob(value_i)))
# info gain = decrease in entropy after data split => info gain (X,Y) = entropy(X) - entropy(Y|X)
# ofc ordering data as go down tree using constraints at each node so entropy decrease


