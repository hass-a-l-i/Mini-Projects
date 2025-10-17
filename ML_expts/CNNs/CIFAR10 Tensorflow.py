# note CNNs not like decision trees or SVM as it does automatic feature extraction whereas former does manual
# tensorflow framework used here
# uses tensors of high ranks to store large volumes of complex information
# e.g. of 0D, 1D and 2D tensors in tensorflow
import tensorflow as tf

zero_D_tensor = tf.constant(20.3)
print(zero_D_tensor)
one_D_tensor = tf.constant([1, 3, 44, 22, 13])
print(one_D_tensor)
two_D_tensor = tf.constant([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
print(two_D_tensor)
# notice the outputs of above have 3 params, first is tensor itself, then its shape, and its data type
# note above we said constant, this is one type of tensor which does not change during execution, store params that remain const through training
# variable tensors (execute with tf.Variable()) can be changed during execution and usually contain weights and biases so that updates can occur
# placeholder tensors (execute with tf.function()) used to reverse a place for data to be used later => is an empty container tensor

# now we apply to CIFAR10 dataset (see info on CIFAR10 in Keras script)
# structure is as follows
# input is 32x32x3 tensor (width, height, color channel)
# then conv layer, 32 3x3 filters, then relu activation, then max pooling 2x2 filter
# then conv layer with 64 3x3 filters, and relu activation again with max pooling 2x2 after
# then flatten layer and dense layer with 128 neurons, then output layer is 10 units (as 10 classes)
# note relu used here as its introduces non linearities so network can learn more => keeps positives sames but makes negatives 0 so only keeps features with strong mapping (i.e. pattern detected)

# first load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# now analyse the data to see what we have - make it easier by encapsulating this in a show images function
import matplotlib.pyplot as plt


def show_images(input_images, classes, input_labels, no_samples, no_rows):
    plt.figure(figsize=(12, 12))
    for i in range(no_samples):
        plt.subplot(no_rows, no_rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(input_images[i])
        plt.xlabel(classes[input_labels[i][0]])
    plt.show()


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# show_images(train_images, class_names, train_labels, 12, 4)

# now preprocess data => normalise pixel values and make labels into numerical format with to_categorical
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, len(class_names))
test_labels = tf.keras.utils.to_categorical(test_labels, len(class_names))

# now build model with sequential so that we can stack layers as we need
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# define variables first
INPUT_SHAPE = (32, 32, 3)
FILTER1_SIZE = 32
FILTER2_SIZE = 64
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
FULLY_CONNECT_NUM = 128
NUM_CLASSES = len(class_names)

# model architecture can now be made
model = Sequential()
model.add(Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(POOL_SHAPE))
model.add(Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
model.add(MaxPooling2D(POOL_SHAPE))
model.add(Flatten())
model.add(Dense(FULLY_CONNECT_NUM, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# summary of network
model.summary()

# now we train, optimiser updates weights and biases, loss func for measuring misclassification errors, and metrics to check performance are accuracy, recall and precision
# note we also put metrics in list to put into the compiler
BATCH_SIZE = 32
EPOCHS = 30
METRICS = metrics = ['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=METRICS)

training_history = model.fit(train_images, train_labels,
                             epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(test_images, test_labels))

# now evaluate the model via a graph of epoochs vs each metric
import numpy as np
def show_performance_curve(training_result, metric, metric_label):
    train_perf = training_result.history[str(metric)]
    validation_perf = training_result.history['val_' + str(metric)]
    intersection_idx = np.argwhere(np.isclose(train_perf,
                                              validation_perf, atol=1e-2)).flatten()[0]
    intersection_value = train_perf[intersection_idx]

    plt.plot(train_perf, label=metric_label)
    plt.plot(validation_perf, label='val_' + str(metric))
    plt.axvline(x=intersection_idx, color='r', linestyle='--', label='Intersection')

    plt.annotate(f'Optimal Value: {intersection_value:.4f}',
                 xy=(intersection_idx, intersection_value),
                 xycoords='data',
                 fontsize=10,
                 color='green')

    plt.xlabel('Epoch')
    plt.ylabel(metric_label)
    plt.legend(loc='lower right')

show_performance_curve(training_history, 'accuracy', 'accuracy')
show_performance_curve(training_history, 'precision', 'precision')
# with the above we see accuracy c. 60% so 60/100 samples are correctly classified
# also see precision of c. 75% meaning out of 100 positive predictions, 75 ish of them are true positives and remaining are false positives (incorrectly predicts a positive for a class)

# we can use confusion matrix to see which classes model is best and worst at predicting => is like a heatmap with test and predicted classes on axis and then heat scale for how close test was to predicted
# can see from below that highest values in diagonal are best predicted classes
# can also see off diagonal classes are those which are confusing to model like planes and birds being misclassified
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
test_predictions = model.predict(test_images)  # predict out test image classes
test_predicted_labels = np.argmax(test_predictions, axis=1)  # gives us the label we predicted
test_true_labels = np.argmax(test_labels, axis=1)  # gives true label of image
cm = confusion_matrix(test_true_labels, test_predicted_labels)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.show()

# can improve model with applying different regularisation like L1 and L2 (to reduce noise) or use dropout layers
