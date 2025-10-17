from matplotlib import pyplot as plt
import numpy as np

y = [3, 4, 7, 9]
feature_1 = [6.00, 9.00, 16.00, 11.00]
feature_2 = [7.00, 12.00, 13.00, 12.00]
feature_3 = [6.00, 10.00, 14.00, 10.00]
feature_4 = [9.00, 8.00, 10.00, 15.00]

features = {'f1': feature_1, 'f2': feature_2, 'f3': feature_3, 'f4': feature_4}

# this is also why we add the ones here, for intercept b
X = np.vstack([feature_1, feature_2, feature_3, feature_4, np.ones(len(y))]).T
y = np.array(y)

# now we also include intercept
weights = np.linalg.lstsq(X, y, rcond=None)[0]
w_1, w_2, w_3, w_4, intercept = weights

# now find fit as sum of weights x respective features
feature_1_fit = np.array([w_1 * i for i in feature_1])
feature_2_fit = np.array([w_2 * i for i in feature_2])
feature_3_fit = np.array([w_3 * i for i in feature_3])
feature_4_fit = np.array([w_4 * i for i in feature_4])
features_fit = {'f1_fit': feature_1_fit, 'f2_fit': feature_2_fit, 'f3_fit': feature_3_fit, 'f4_fit': feature_4_fit}

# sum together as this is now our fit for y labels
# add intercept - now we get exact y label values
y_fit = feature_1_fit + feature_2_fit + feature_3_fit + feature_4_fit + intercept
y_fit = np.array(y_fit)

print("Labels :" , y)
print("Features:")
for name,vec in features.items():
    print(name, ' = ', vec)
print("Design Matrix X:")
print(X)
print("Weights = ", weights)
print("Fitted Features:")
for name,vec in features_fit.items():
    print(name, ' = ', vec)
print("Intercept b = ", intercept)
print("Sum of fitted features = ",  y_fit)
print("Actual y values = ", y)
print("Difference (residuals) = ", np.subtract(y_fit, y))
plt.ylabel("True y")
plt.xlabel("Fitted y (with intercept)")
plt.scatter(y_fit, y, c="black")
plt.plot(y_fit, y, linestyle='dashed')
plt.show()

# NOT EXACT FIT EVER, RESIDUAlS HERE ARE SMALL

