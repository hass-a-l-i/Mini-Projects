# use a single layer here of LSTM and output layer
# here we set units in LSTM layer (neurons) as 125 with tanh activation

import Stock_price_predictor_base as SPPB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

# The LSTM architecture
model_lstm = Sequential()
model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(SPPB.n_steps, SPPB.features)))
model_lstm.add(Dense(units=1))
# Compiling the model with RMSprop optimizer and mse loss func
# note RMSprop is like Adam and is a 1st deriv (1st order) optimization algo too - main difference is learning rate is adaptive and not hyperparameter => will auto change over time
model_lstm.compile(optimizer="RMSprop", loss="mse")
model_lstm.summary()

# now train model over our batch size and epochs below
epoch = 50
batch = 32
model_lstm.fit(SPPB.X_train, SPPB.y_train, epochs=epoch, batch_size=batch)

# prediction
predicted_stock_price = model_lstm.predict(SPPB.X_test)
# inverse transform the values from normalised values to actual values
predicted_stock_price = SPPB.sc.inverse_transform(predicted_stock_price)

# now we have predicted prices from training data and real values we can plot graph to compare
def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("MasterCard Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("MasterCard Stock Price")
    plt.legend()
    plt.show()

plot_predictions(SPPB.test_set, predicted_stock_price)

# can also return root mean sq error (average deviation between test and predicted)
rmse = np.sqrt(mean_squared_error(SPPB.test_set, predicted_stock_price))
print("The root mean squared error is {:.2f}.".format(rmse))

# find accuracy
def accuracy_finder(test, predicted):
    acc = []
    for i in range(len(test)):
        accuracy = predicted[i]/test[i]
        acc.append(accuracy)
    mean_acc = sum(acc)/len(acc)
    print("Accuracy is ", mean_acc[0] * 100, "%")

accuracy_finder(SPPB.test_set, predicted_stock_price)