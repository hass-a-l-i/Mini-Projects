import matplotlib.pyplot as plt
import csv

x_val = []
y_val = []

label_x = []

print("Input display intervals for epochs: ")
display_intervals = input()
print("Input number of epochs: ")
# deciding appropriate learning rate will determine divergence => need low enough to prevent divergence but high
# enough for efficient learning
e = input()
e = int(e)
print("Input alpha (learning rate, usually 0.01 - 0.001): ")
alph = input()
alph = float(alph)

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


# function to update w and b for ONE epoch as per partials minimizing mean square error (loss func)
def update_w_and_b(x, y, w, b, alpha):
    # initial assumption of dl/dw and dl/db approx 0
    dl_dw = 0.0
    dl_db = 0.0
    n = len(x)

    # our sums as seen in partial derivative equations (to minimize MSE)
    for index in range(n):
        dl_dw = dl_dw - (2 * x[index] * (y[index] - (w * x[index] + b)))
        dl_db = dl_db - (2 * (y[index] - (w * x[index] + b)))

    # update w and b now, takeaway as we are moving param in opposite direction
    # if derivative +ve, means func increasing, so we need to move param in opposite directio to minimize MSE
    w_new = w - ((1 / float(n)) * dl_dw * alpha)
    b_new = b - ((1 / float(n)) * dl_db * alpha)

    return w_new, b_new


# define loss function
def avg_loss(x, y, w, b):
    n = len(x)
    tot_error = 0.0
    # do sum of MSE as in loss eq
    for index in range(n):
        tot_error = tot_error + (y[index] - ((w * x[index]) + b)) ** 2
    # divided by N as in eq to find av
    return tot_error / n


# train the line over multiple epochs
def train(x, y, w, b, alpha, epochs):
    for epoch in range(epochs):
        # recursive here
        w, b = update_w_and_b(x, y, w, b, alpha)
        loss = avg_loss(x, y, w, b)

        # log the progress
        if epoch % int(display_intervals) == 0:
            print("epoch:", epoch, " loss:", loss)
            print("w : ", w, " b : ", b)

    return w, b


y_new = []


# define regression func
def regression_line(x, w, b):
    for index in range(len(x)):
        y = w * x[index] + b
        y_new.append(y)


# predict a y new given an x using our regression line
def predict(x, w, b):
    return w * x + b

# execute ML code
w_opt, b_opt = train(x_val, y_val, 0.0, 0.0, alph, e)
regression_line(x_val, w_opt, b_opt)

print("w_opt = ", w_opt)
print("b_opt = ", b_opt)

x_in = 61.0
y_out = predict(x_in,  w_opt, b_opt)

# our plotting code to give OPTIMAL y from given x:
plt.scatter(x_in, y_out, color="blue")
plt.scatter(x_val, y_val, color="black")
plt.title("Epoch : %i" % e)
plt.plot(x_val, y_new, color="red")
plt.xlabel(label_x[0] + " spending")
plt.ylabel("Units sold")
plt.show()
