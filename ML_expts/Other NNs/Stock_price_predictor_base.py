# here we use stock price dataset and train using LSTM and GRU models to forecast price

# import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# set our random seed for both tensorflow and numpy to be the same
tf.keras.utils.set_random_seed(455)
np.random.seed(455)

# first import dataset + clean/analyse - also change formatting of date to datetime in python
# given open price, high price, low price, close price and vol traded each day
dataset = (pd.read_csv(
    "Mastercard_stock_history.csv", index_col="Date", parse_dates=["Date"])
           .drop(["Dividends", "Stock Splits"], axis=1))
print(dataset.head())

# we use high to train price prediction model here - can use pandas describe func to gain insight into dataset
print(dataset.describe())
# can see for high tht the std dev is higher than mean so high variance
# use .isna().sum() to find if any missing values and data type
print(dataset.isna().sum())
# can create func to plot line graph to see trend of high price data - we go one step further and also split into train and test data here and reflect this on graph with change in line color => df.loc helps us do this to date col
def train_test_plot(dataset, t_start, t_end):
    dataset.loc[f"{t_start}":f"{t_end}", "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[f"{t_end + 1}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train set (Before {t_end + 1})", f"Test set ({t_end + 1} and beyond)"])
    plt.title("MasterCard stock price from %d - %d".format(t_start, t_end))
    plt.show()
# let's take data between 2016 and 2020
"""train_test_plot(dataset, 2016, 2020)"""


# now preprocess data for training with train test split => we can make our own with loc in pandas as did above for graph instead of using sklearn
def train_test_split(dataset, t_start, t_end):
    train = dataset.loc[f"{t_start}":f"{t_end}", "High"].values
    test = dataset.loc[f"{t_end+1}":, "High"].values
    return train, test
training_set, test_set = train_test_split(dataset, 2016, 2020)
# now we need to standardize training set to avoid anomalies, we use min max scaler (makes features in dataset between 0 and 1)
sc = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1)  # into vector form for NN feeding
training_set_scaled = sc.fit_transform(training_set)
# now we split training set into inputs (x train) and outputs (y train) for training => use a split sequence function
# no_steps here converts data list into input features until reach step value which it takes as an output feature, the goes through dataset taking every nth feature as output with group before this being that outputs input features
def split_sequence(sequence, no_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + no_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
# we split this for 60 no_steps below => this way our output is not just one column like classification, it is all data set which we can predict (is a hyperparameter you have to optimize for problem at hand)
n_steps = 60
X_train, y_train = split_sequence(training_set_scaled, n_steps)
# univariate data here so we put features a s1 and reshape training data for fitting LSTM model
features = 1
print("Shape of X_train is ", X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)  # args here are no. samples, no. time steps and no. features

# must preprocess and normalize test set too as we did with train set, so we can evaluate model
dataset_total = dataset.loc[:, "High"]
inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps :].values
inputs = inputs.reshape(-1, 1)
#scaling
inputs = sc.transform(inputs)

# Split into samples
X_test, y_test = split_sequence(inputs, n_steps)
# reshape as before
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)


# now we use this to do LSTM and GRU models respectively
