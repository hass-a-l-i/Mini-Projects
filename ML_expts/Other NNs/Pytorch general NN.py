# here we load csv dataset and prepare for pytorch define MLP model in pytorch and train + evaluate model

# load libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# first load data - is a set for indians and likelihood of getting diabetes
# binary classification => will or won't get diabetes is bitwise
# eight feature vars (# times pregnant, plasma glucose conc, blood pressure, skin thickness, insulin, BMI, diabetes pedigree, age
# output is 9th var which is our binary classification of diabetes => yes or no
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')  # remember this gives us 2D array, with rows as one list within list of all columns
x = dataset[:, 0:8]  # select features here using splice on secondary list (rows, i.e. chose cols 0 - 7 here within each row)
y = dataset[:, 8]  # select final (binary output ) vars with this splice

# now have to convert to pytorch tensors
X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # reshapes it into vector form

# now we define the model
# use sequential again for sequence of layers => need to ensure first layer has correct no input features, we need 8 as 8 features are inputs
# general rule is need network large enough to capture problem but small enough to run quickly
# we use RELU between layers and sigmoid for output layer activation function =? Relu between linear layers avoid vanishing gradient then classification of binary output with thresh 0.5 for output probabilities of sigmoid
"""
model = nn.Sequential(
    nn.Linear(8, 12),  # input features 8, we output 12 using 12 neurons
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),  # final putput is 1 i.e. binary
    nn.Sigmoid()
)
"""
# can then check model architecture by printing it
# print(model)

# it is better to use the inherited class method though
class DiabetesClassifier(nn.Module):
    def __init__(self):  # need to put all layers here in constructor
        super().__init__()  # used to allow us to call methods from inside parent lass (nn.module)
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

model = DiabetesClassifier()
print(model)

# now prepare training - decide loss function and optimiser
loss_func = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)  # pass on all params from model generated and also define learning rate for chosen adam optimizer

# now we train using set epochs and batches (remember batches of data passed until all data passed through model = one epoch then repeat) => need to optimise computational power here
# remember this is trial and error like choosing neuron number - need to find sweet spot on own for when design own NN problem
# easiest way to train is nested loop - one for batches other for epoch
n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    loss = 0
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_func(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch = {epoch}, loss = {loss}')


# now can evaluate the model performance - ideally separate data into train for training and test for evaluating but this skipped here for ease
# use predictions instead of validation to do this as our training set was all data here
# we can do this by converting output of prediction from sigmoid into a binary number (rounded) and seeing if classification is correct compared to label we know
# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)

accuracy = (y_pred.round() == y).float().mean()  # we find the accuracy as a float and find the mean number of 1s (which we know from the labels we have)
print(f"Accuracy {accuracy}")
# remember NN stochastic process so will get slightly diff values each time - best way to do it is run it multiple times the average our accuracy over N runs

# can now make new predictions => same process as above, probability output rounded to give binary
predictions = model(X)
rounded = predictions.round()
# could instead make direct class predictions
class_predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), class_predictions[i], y[i]))
# expect the output to reflect accuracy of model
