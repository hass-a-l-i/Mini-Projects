# pytorch is similar to numpy
# tensors are like n-D numpy arrays so use numpy commands on them - easy
# note don't lose artificial NNs for images as they lose spatial recognition of image
# CNNs extract features like edges and shapes using filters then reduce data complexity (keep information) with pooling layers
# remember NN part uses backpropagation to optimize weights and biases (by comparing to true values) so that it can accurately classify the data used to train
# saved W's and b's then applied to new test data to make predictions
# e.g. for face we break down NN into each perceptron being able to focus on feature of face like eyes etc...
# then these will fire if true (=1) features seen (we use MLPs (binary output) but artificial neurons can be used too where output between 0 and 1 (note if use sigmoid activation then get extremes of 0 and 1 only and basically behaves like an MLP)

# using CIFAR10 again
# first load library
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # to bypass certificate verification for download

# define relevant variables
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

# now can use device to see if better to train on cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data set  with built-in datasets in torchvision
# first we reformat the images from the dataset for ML compatibility
# resize images, make into tensor form and normalise with a mean and std => the values are chosen as images used from imagenet (3 values as 3 channels (color RGB, height, width))
all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])

# now directly make training dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             transform=all_transforms,  # this is where transforms come in
                                             download=True)

# and testing dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            transform=all_transforms,
                                            download=True)

# use a data loader, allows for data to be loaded without putting all in RAM at once - does it in batches (shuffle = true makes sure that at least one image of each class included in a batch)
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)


# now create the NN - have to make a class that includes nn.module from pytorch
# define layers with __init__ to create object to call - name layers then assign to actual layer from module e.g. pooling, conv etc...
# also create a forward method in the class to feed the info forward
class ConvNeuralNet(nn.Module):
    def __init__(self, no_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32,
                                     kernel_size=3)  # define no channels in and out of layer, kernel size is nxn of the filter matrix
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2,
                                      stride=2)  # pooling matrix nxn is first arg, second arg is # units the window moves before x by pooling matrix (i.e. dimension of output)

        self.fc1 = nn.Linear(1600,
                             128)  # this is fully connected layer i.e. layer with neurons, first arg in features and second is out features
        self.relu1 = nn.ReLU()  # activation function
        self.fc2 = nn.Linear(128, no_classes)  # another fully connected layer

    def forward(self, x):  # moves info through layers => via variable "out"
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)  # transposes our final vector so can feed to NN

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# now can call class to create model and set hyperparameters
model = ConvNeuralNet(num_classes)
# set loss func
loss_func = nn.CrossEntropyLoss()
# set optimizer (stochastic gradient descent is SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9, nesterov=True)  # weight decay ensures weights/biases do not get too large // momentum  = usually 0.9 => allows for quicker convergence of gradient descent (as otherwise fluctuates instead of going to minima so Nesterov momentum which smooths out the oscillations)
total_step = len(train_loader)  # to help us go through each batch of training (= one step)

# now we use epochs defined earlier to train model
for epoch in range(num_epochs):
    # load in the data in batches with train_loader object which has int i, and (images, label) tuples
    for i, (images, labels) in enumerate(train_loader):
        # convert our tensors to the configured device (it decides whether to loan data to cpu or gpu)
        images = images.to(device)
        labels = labels.to(device)

        # feed data forward through model by using images and predicting labels in outputs, then finding the loss between outputs and true labels
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # Backpropagation, invoke optimizer to update weights and biases, first zero gradient as each time calc gradient will be different, loss.backwards computes the gradient and then step updates b's and W's based on this
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# once run above we see loss minimises but fluctuates before end to be higher - could mean over-fitting or batch size too small

# now test model to see how right it was at predicting
with torch.no_grad():  # all below wrapped under this to ensure no gradients calculated again
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # get indices of max values from predicted
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # sums all predicted labels equal to originals

    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))





