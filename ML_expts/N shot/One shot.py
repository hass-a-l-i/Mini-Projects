# model now learns from one example as input
# similar to human learning

# we use identical siamese NN here with keras and MNIST
# SNNs are two identical NNs running on same data side by side - purpose is to produce similarity function
# the similarity func once trained then is applied to any inputs to test if two inputs similar (higher score = more similar) = one shot
# e.g. face recognition

# import libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import pickle
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
import random
# import tensorflow_addons as tfa
from matplotlib.colors import ListedColormap

# first load data as test and training sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# normalise the training and testing subsets
print("Before reshape:")
print("The shape of X_train is {} and Y_train is {} ".format(X_train.shape, Y_train.shape))
print("The shape of X_test is {} and Y_test is {} ".format(X_test.shape, Y_test.shape))
print("\n")
X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))  # do prods to combine the width and height of pics into one var, then reshape into a 2D array instead of 3D
X_test = X_test.astype('float32')
X_test /= 255
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))  # y does not need reshape is are just labels
# check shape of data
print("After reshape:")
print("The shape of X_train is {} and Y_train is {} ".format(X_train.shape, Y_train.shape))
print("The shape of X_test is {} and Y_test is {} ".format(X_test.shape, Y_test.shape))

# now arrange input into image pairs as SNN has two input channels
# define positive pairs as images belonging to same class and negatives as those belonging to different classes
# create object for this - same as classes in c++ => define objects with attributes and inheritance
class Pairs:
    def makePairs(self, x, y):
        num_classes = 10
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]  # returns indices of identical classes in y

        pairs = list()
        labels = list()

        for idx1 in range(len(x)):
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])  # random pair chosen with same label for positive pairs
            x2 = x[idx2]

            labels += list([1])  # add to label and pairs listing
            pairs += [[x1, x2]]

            label2 = random.randint(0, num_classes-1)   # for creating negative pairs
            while label2 == label1:
                label2 = random.randint(0, num_classes-1)

            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]

            labels += list([0])  # add negative pair to lists
            pairs += [[x1, x2]]

        return np.array(pairs), np.array(labels)

# construct pairs for test and train, making sure float for labels
p = Pairs()
pairs_train, labels_train = p.makePairs(X_train, Y_train)
pairs_test, labels_test = p.makePairs(X_test, Y_test)
labels_train = labels_train.astype('float32')
labels_test = labels_test.astype('float32')

# define our distance func (euclidean here) note K is keras backend functions which work direct on arrays
def euclideanDistance(v):
    x, y = v
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# finds shape of our array of euclidean dists
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# as using euclidean distance we define loss func as contrastive loss (if dist is below certain thresh (margin) then loss is 0, like step func)
def contrastive_loss(y_original, y_pred):
    sqaure_pred = K.square(y_pred)
    margin = 1
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_original * sqaure_pred + (1 - y_original) * margin_square)  # definition of loss here (around y_original scaled by square of predicted add 1-y original times margin square

# define own accuracy function and accuracy for compiling model
def compute_accuracy(y_original, y_pred):
    pred = y_pred.ravel() < 0.5  # ravel makes array 1D, choose ones less than 0.5 as normalised to 0.5
    return np.mean(pred == y_original)

def accuracy(y_original, y_pred):
    return K.mean(K.equal(y_original, K.cast(y_pred < 0.5, y_original.dtype)))

# construct NN
# (784,) input layer which is 28 x 28 pixel matrix
# use three fully connected RELU then two L2 normalisation layers
input = Input(shape=(784,))
x = Flatten()(input)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)  # expanding to get more features in each RELU layer
x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
dense = Model(input, x)  # is our NN

input1 = Input(shape=(784,))
input2 = Input(shape=(784,))
dense1 = dense(input1)
dense2 = dense(input2)

distance = Lambda(euclideanDistance, output_shape=eucl_dist_output_shape)([dense1, dense2])  # find our distance metric between output of the SNNs (dense 1 and dense 2)
model = Model([input1, input2], distance)  # define our model inputs so we can compile

# compile model to see architecture
model.compile(loss=contrastive_loss, optimizer="adam", metrics=[accuracy])
model.summary()

# train model for 10 epochs, capturing changes in training and test loss over training period
# pairs_train[:, 1] - slices keras tensor taking the values in first column only
# pairs_train[1:, :] - takes all values from 1 (second) row onwards
# labels_train[:] - creates a new list without affecting old one, creates a copy
history = model.fit([pairs_train[:, 0], pairs_train[:, 1]], labels_train[:], batch_size=128, epochs=10,  # pairs 0 and 1 are the two siamese NN inputs
                    validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test))

print(labels_train.shape, pairs_train.shape)
# now predict using trained model
y_pred_te = model.predict([pairs_test[:, 0], pairs_test[:, 1]])  # predict using the SNN inputs
te_acc = compute_accuracy(labels_test, y_pred_te)
print("The accuracy obtained on testing subset: {}".format(te_acc * 100))

# Plotting training and testing loss
def plot_losses():
    plt.figure(figsize=(20, 8))
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, (len(history.history['val_accuracy']) + 1))
    plt.plot(epochs, loss_values, 'y', label='Training loss')
    plt.plot(epochs, val_loss_values, 'g', label='Testing loss')
    plt.title('Model loss for one-shot training and testing subset of MNIST dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#plot_losses()

# we find test loss higher than training which is expected - overfitting unlikely


