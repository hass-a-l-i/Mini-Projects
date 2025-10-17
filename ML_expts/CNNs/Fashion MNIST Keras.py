# remember these are all MLP CNNs (i.e. feed forward)
# NB - feed forward means data goes in one direction and never loops back on self / backpropagation does forward and also feeds data from output back to input to adjust weights and biases and min error between actual and predicted
# feed forward only best for classification for e.g., but for optimization / regression we choose backpropagation along with it
# note fully connected layer at end of this NN is where we use the activation func to classify features

# here we use fashion MNIST - 28x28 grayscale images of fashion products, 70k images with 10 categories, train set 60k images and test is 10k
# first load data (and bypass certificates)
import ssl

import keras.models

ssl._create_default_https_context = ssl._create_unverified_context  # to bypass certificate verification for download
from keras.datasets import fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()  # remember x is our input data images and y is labels for them

# now can analyze data
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
print('Training data shape : Images:', train_X.shape, '// Labels:', train_Y.shape)  # output is (60k, 28, 28) i.e. 60k images of 28x28 size, similarly 60k size of labels with no other dimensions
print('Testing data shape : Images:', test_X.shape, '// Labels:', test_Y.shape)
classes = np.unique(train_Y)  # now check out classes to see if complete
no_classes = len(classes)
print('Total number of classes : ', no_classes)
print('List of classes : ', classes)

# now check data by viewing it here check first 3 images in test and training sets
"""
plt.figure(figsize=(5, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(train_X[i, :, :], cmap='gray')
    plt.title("TRAIN - Class : {}".format(train_Y[i]))
plt.show()
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(test_X[i, :, :], cmap='gray')
    plt.title("TEST - Class : {}".format(test_Y[i]))
plt.show()
"""

# now process data to make it in form we can use for NN
# each 28x28 image needs to be matrix 28x28x1 to feed to NN
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
print('Reshaped data : Training:', train_X.shape, '// Testing:', test_X.shape)
# now convert from int8 type to float32 and rescale (normalise) pixel values to be in range 0 - 1
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.0
test_X = test_X / 255.0

# now convert class labels (Y's) into one hot encoding (machine learning feedable) vector form
# only non-zero value in 1 x 10 vector for each image will be the class it belongs to
# e.g. of first image is printed
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
print('Original label:', train_Y[0])
print('Label after one-hot coding', train_Y_one_hot[0])

# now split data
from sklearn.model_selection import train_test_split
train_X, validation_X, train_label, validation_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13) # chose an 80:20 train:test split
# check shape of our data
print("Shape of train X = ", train_X.shape)
print("Shape of train label = ", train_label.shape)
print("Shape of validation X = ", validation_X.shape)
print("Shape of validation label = ", validation_label.shape)

# now map out architecture
# 3 convolutional layers with 3x3 filters for each layer, 32 of these for first layer, 64 for second and 128 for third
# 3 2x2 max pooling layers as well between each convolutional
# finally flatten layer which feeds into a dense layer of 128 neurons, output layer is then 10 units
# note only new thing below is dropout = applies dropout regularization to prevent over-fitting AND batch normalization normalizes the output of batches
# note activation = leaky relu as they fix problem of dying rectified linear units (relu's) as in deep NNs, during training relu's can die where weights don't cause activations to fire anymore so gradient past this point = 0
# leaky relu does not allow this to go to 0, instead gives it a small negative slope
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU

# define batch size, epochs and no classes
batch_size = 64
epochs = 20
num_classes = 10

# create our NN architecture
# conv layer -> leakyrelu activation (here to stop dying relu's) -> pooling layer => repeated 3 times, then flatten and pass through neurons (only 10)
"""
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dense(num_classes, activation='softmax'))  # using softmax as activation aka use probabilities for this one to classify images
"""

# now compile the model
# note we use adam again, and also can choose loss func from binary cross entropy and categorical, we choose latter here as multi-class problem is here and not binary
"""fashion_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])"""
# now visualise our CNN
"""fashion_model.summary()"""

# now we train our model
"""fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(validation_X, validation_label))"""
# remember we want high accuracy low loss which we get here
# note model may be over-fitting here as validation accuracy high but loss also high meaning model analysed training data well but not guaranteed to work on new input data

# evaluating the model now
"""
test_evaluation = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Validation (test) loss:', test_evaluation[0])
print('Validation (test) accuracy:', test_evaluation[1])
"""
# we notice high test loss so over fitting likely present
# this is also why training accuracy (seen in last epoch) and validation accuracy different

# to see how loss behaving we plot accuracy and loss plots between training and validation
# use history to access the progressive data for training accuracy, validation accuracy, training loss and validation loss
"""
accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
# below we plot, b is blue line, bo means blue circles
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
"""

# plots show validation accuracy stays within tight range after 4-5 epochs, before this validation accuracy increasing linearly with loss
# validation loss linearly decreasing with training loss until 4-5 epochs where it started to increase
# now we add dropout layers to reduce over fitting (dropout also forces the NNs still on to learn new features
"""
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))  # dropout layers added => runs the same CNN but randomly drops out 25% of units from layer, should be 0.2 - 0.5 to prevent over fitting. reduces number of params output
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))
"""
# print summary of CNN, and compile + train
"""
fashion_model.summary()
fashion_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(validation_X, validation_label))
fashion_model.save("fashion_MNIST_model_final.keras")  # .h5py indicates it's in the hierarchical data format used for storing large numerical datasets, we save as .keras here which includes this, a json config and a json metadata file which includes info of keras when model was run
"""

# now evaluate
"""
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Validation (test) loss:', test_eval[0])
print('Validation (test) accuracy:', test_eval[1])
"""
# note to save model and not have to keep running you save as above but also use csv logger to save history
"""
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('training.log', separator=',', append=False)
fashion_train_dropout = fashion_model.fit(train_X, train_label, callbacks=[csv_logger])
"""

# if now plot we will find validation loss and training loss are in sync (linear decrease) and validation accuracy and training accuracy also in sync (linear increase)
fashion_model_loaded = keras.models.load_model("fashion_MNIST_model_final.keras")
test_eval = fashion_model_loaded.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Validation (test) loss:', test_eval[0])
print('Validation (test) accuracy:', test_eval[1])


# predicting labels can now happen
predicted_classes = fashion_model_loaded.predict(test_X)
# predictions are floats here so need to round to get to class
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
# now lets see some examples where we matched
correct = np.where(predicted_classes == test_Y)[0]  # np.where helps us do this, makes an array of matched indices in correct
print("Found %d correct labels" % len(correct))
"""
for i, correct in enumerate(correct[:9]):  # lets display first 9 i.e. :9
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()
plt.show()
"""

# similarly can use np.where to find labels that were incorrect i.e. test labels and predicted labels not equal
incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
# if also plot above can see possible reasons for miss-classification => e.g. similar patterns may affect classifiers performance e.g. a similarly coloured jacket and shirt

# classification report helps identify misclassified => shows which classes were most high in error
# reflected using metrics such as precision and recall
# can see which classes the classifier lacks these metrics in any combination
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))


