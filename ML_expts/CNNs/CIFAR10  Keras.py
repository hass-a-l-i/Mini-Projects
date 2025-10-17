import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # to bypass certificate verification for download

# keras has functional or sequential API (application programming interface)
# functional for more complex CNNs and sequential for simple models with linear stack of layers
# first download CIFAR10 dataset = 60k color images in 10 classes, 6k images each class
# divide dataset into 50k training and 10k testing images
# labels are y's and images are x's here
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# next as images we normalise the pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# now verify data by displaying first 25 images from train set (using class names which are known and encoded)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
"""
plt.figure(figsize=(10, 10))                    # set size for each image
for i in range(25):                             # first 25
    plt.subplot(5, 5, i+1)                # no rows and cols, index incrementor (cifar labels are array so + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])                  # show ith image
    plt.xlabel(class_names[train_labels[i][0]])  # with label
plt.show()
"""

# now create convolutional base of network
# we do this by stacking Conv2D (2d convolution layer) and MaxPooling2D (takes conv and reduces data with dimensional reduction of matrix) layers
# CNNs here take tensors shape (image_height, image_width, color_channels) where batch size ignored (batch size = if you want to train network incrementally with batches of set datapoints form train set
# note batch size can mean faster training and use less memory but tradeoff is smaller batches mean less accurate predictions
# color_channels are (R,G,B)
# for this e.g. we process images inputs in our CNN of (32, 32, 3) which is format of cifar images => need to use input_shape
# can now build CNN structure, below is convolutional base (purpise of conv and pooling is to extract features from images and consolidate info in pooling layer, minimizing # params for network to compute)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # note we use relu activation function
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu'))  # first number is dimensionality of output space, larger width here means more params optimized so more accurate representation / second number is dimensions of convolution matrix
model.add(layers.MaxPooling2D((2, 2)))  # arg = dimension of pooling matrix
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# then build dense layers
# these layers feed outputs from prev layers to the neurons then the output of the neurons to the next layer
model.add(
    layers.Flatten())  # changes the 3D tensor to 1D tensor (a vector) which is needed to be fed to neurons in dense layer (is our input feature vector)
model.add(layers.Dense(64,
                       activation='relu'))  # dense layer with 64 neurons => is fully connected so each feature from flatten is fed 64 times to each neuron
model.add(layers.Dense(10))  # another dense layer of 10 neurons

# display architecture of CNN:
model.summary()
# notice you see output shape is a  3D tensor each layer with dimensions (image height,  image width, color channels)
# notice as go through layers, height and width decrease, means we have computational space to increase number of outputs from 32 to 64 which we did above
# none means batch size was not specified
# for flatten => can see (4, 4, 64) output tensors flattened to vector of shape (1024) which fed to dense layers

# now compile and train model
# adam is stochastic gradient descent
# loss function defined as cross entropy, logits true just means tensor inputs expected, but if false then probability inputs expected which likely would need softmax activation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# fit our model training sets over set number of epochs (remember memory of last epoch is starting point of next) and specify input images and labels
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# this is evaluation of model on our validation (test) set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)  # gives us loss and accuracy


# now evaluate model by plotting accuracy vs epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')  # this is validation set accuracy = i.e our test set
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
