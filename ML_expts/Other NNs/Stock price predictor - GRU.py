# same as LSTM model we had but now put GRU layer there instead - same steps, see LSTM file for more detail

import Stock_price_predictor_base as SPPB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

# The GRU architecture
model_GRU = Sequential()
model_GRU.add(GRU(units=125, activation="tanh", input_shape=(SPPB.n_steps, SPPB.features)))
model_GRU.add(Dense(units=1))
# Compiling the model
model_GRU.compile(optimizer="RMSprop", loss="mse")
model_GRU.summary()

# train model over our batch size and epochs
epoch = 50
batch = 32
model_GRU.fit(SPPB.X_train, SPPB.y_train, epochs=epoch, batch_size=batch)

# prediction
predicted_stock_price = model_GRU.predict(SPPB.X_test)
# inverse transform the values from normalised values to actual values
predicted_stock_price = SPPB.sc.inverse_transform(predicted_stock_price)

# plot graph to compare
def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("MasterCard Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("MasterCard Stock Price")
    plt.legend()
    plt.show()

plot_predictions(SPPB.test_set, predicted_stock_price)

# root mean sq error
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

# note GRU had lower MSE and higher accuracy so was better performing here than LSTM

