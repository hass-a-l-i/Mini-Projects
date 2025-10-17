# NN deep learning version using pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Define your PyTorch model class - as we do with function where we control architecture
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # fc is fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # args for linear layers = input features, output features

    def forward(self, x):  # decide how model is run with forward function
        x = torch.sigmoid_(self.fc1(x))
        x = self.fc2(x)
        return x


# Load and preprocess your data, split into labeled and unlabeled sets
np.random.seed(10)
X = np.random.rand(100, 2)  # 100 data points i.e. coords of each data point so each feature is x, y here
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # binary classification target
dataset = list(zip(X, y))  # total dataset to assign label for each data point
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.8, stratify=y)
# decide on size of input, output and hidden layers of NN
input_size = 2
output_size = 1
hidden_size = int((2/3 * input_size) + output_size)
print (hidden_size)
# Convert data to PyTorch tensors
X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)
X_labeled_tensor = torch.tensor(X_labeled, dtype=torch.float32)
y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.long)
y_unlabeled_tensor = torch.tensor(y_unlabeled, dtype=torch.long)
labeled_dataset = TensorDataset(X_labeled_tensor, y_labeled_tensor)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
X_validation_tensor = X_unlabeled_tensor
y_validation_tensor = y_unlabeled_tensor

# Initialize model, loss function, and optimizer
model = SimpleClassifier(input_size, hidden_size, output_size)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Active Learning Loop
num_iterations = 50
batch_size = 10

for iteration in range(num_iterations):
    # Implement a query strategy (e.g., uncertainty sampling)
    model.eval()
    with torch.no_grad():
        uncertainty = model(X_unlabeled_tensor)
        uncertainty_scores = torch.max(uncertainty, dim=1)[0]
    query_indices = uncertainty_scores.argsort()[-batch_size:]

    # Label the selected instances
    labeled_instances = X_unlabeled_tensor[query_indices]
    labeled_labels = y_unlabeled_tensor[query_indices]

    # Update the labeled and unlabeled datasets
    X_labeled_tensor = torch.cat((X_labeled_tensor, labeled_instances), dim=0)
    y_labeled_tensor = torch.cat((y_labeled_tensor, labeled_labels), dim=0)
    X_unlabeled_tensor = torch.cat((X_unlabeled_tensor[:query_indices[0]], X_unlabeled_tensor[query_indices[0] + 1:]), dim=0)
    y_validation_tensor = torch.cat((y_validation_tensor[:query_indices[0]], y_validation_tensor[query_indices[0] + 1:]), dim=0)

    # Retrain the model on the updated labeled dataset
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in labeled_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            batch_y = batch_y.unsqueeze(1)
            batch_y = batch_y.float()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate the model on a validation set
    y_validation = y_validation_tensor.tolist()
    prediction_tensor = model(X_unlabeled_tensor)
    prediction_array = prediction_tensor.cpu().detach().numpy()
    prediction = [item for items in prediction_array for item in items]
    av = sum(prediction) / len(prediction)  # made up classifier
    pred = []
    for i in prediction:
        if i >= av:
            pred.append(1)
        elif i < av:
            pred.append(0)

    validation_accuracy = accuracy_score(y_validation, pred)
    print(f"Iteration {iteration + 1}, Validation Accuracy: {validation_accuracy:.4f}")
