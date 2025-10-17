import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_val = []
y_val = []

label_x = []

# each data point here represents companies ad spend and corresponding sales
# we want to find for an arbitrary company, using regression, what sales expected for certain ad spend

with open("Advertising.csv", "r") as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        label_x.append(row[2])

with open("Advertising.csv", "r") as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(csvfile)
    for row in plots:
        x_val.append(row[2])
        y_val.append(row[4])

for i in range(len(x_val)):
    x_val[i] = float(x_val[i])

for i in range(len(y_val)):
    y_val[i] = float(y_val[i])

x_val = np.array(x_val)
y_val = np.array(y_val)

# split data set into training and testing sections
# training set finds optimum w and b, then testing set is what they are applied to to draw line of best fit
# reduces bias

X_train, x_test, Y_train, y_test = train_test_split(x_val, y_val, test_size=0.25, random_state=23)

X_train = np.array(X_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)

model = LinearRegression()

model.fit(X_train, Y_train)

# now can return gradient w and intercept b for model
b = model.intercept_
w_array = model.coef_
w = w_array[0]
print("w_opt = ", w)
print("b_opt = ", b)

# find our graphs now
x_in = 61.0
y_out = w * x_in + b

y_new = []


def regression_line(x, w, b):
    for index in range(len(x)):
        y = w * x[index] + b
        y_new.append(y)


regression_line(x_val, w, b)

# our plotting code to give OPTIMAL y from given x:
plt.scatter(x_in, y_out, color="blue")
plt.scatter(x_val, y_val, color="black")
plt.plot(x_val, y_new, color="red")
plt.xlabel(label_x[0] + " spending")
plt.ylabel("Units sold")
plt.show()
