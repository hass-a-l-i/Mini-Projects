from matplotlib import pyplot as plt
import numpy as np

# labels and features for each
y = [3, 4, 7, 9]
feature_1 = [6.00, 9.00, 16.00, 11.00]
feature_2 = [7.00, 12.00, 13.00, 12.00]
feature_3 = [6.00, 10.00, 14.00, 10.00]
feature_4 = [9.00, 8.00, 10.00, 15.00]

features = {'f1': feature_1, 'f2': feature_2, 'f3': feature_3, 'f4': feature_4}

# create matrix of features
X = np.vstack([feature_1, feature_2, feature_3, feature_4, np.ones(len(y))]).T
y = np.array(y)

# case where b = 0, we fit weights w as follows
# [0] returns only weights but without this we get weights and intercept output
# gives rank of corresponding matrix Xw, residuals, singular values and weight vec for each feature vec
# we just want weights so [0]
weights = np.linalg.lstsq(X, y, rcond=None)[0][0:4]
w_1, w_2, w_3, w_4 = weights

# now find fit as sum of weights x respective features
feature_1_fit = np.array([w_1 * i for i in feature_1])
feature_2_fit = np.array([w_2 * i for i in feature_2])
feature_3_fit = np.array([w_3 * i for i in feature_3])
feature_4_fit = np.array([w_4 * i for i in feature_4])
features_fit = {'f1_fit': feature_1_fit, 'f2_fit': feature_2_fit, 'f3_fit': feature_3_fit, 'f4_fit': feature_4_fit}

# sum together as this is now our fit for y labels
# there is a difference here as we did not add intercept
y_fit = feature_1_fit + feature_2_fit + feature_3_fit + feature_4_fit
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
print("Sum of fitted features = ",  y_fit)
print("Actual y values = ", y)
print("Difference (residuals) = ", np.subtract(y_fit, y))
plt.scatter(y_fit, y, c="black")
plt.plot(y_fit, y, linestyle='dashed')
plt.ylabel("True y")
plt.xlabel("Fitted y (no intercept)")
plt.show()

"""
plt.scatter(feature_1, y, c="blue")
plt.scatter(feature_2, y, c="red")
plt.scatter(feature_3, y, c="orange")
plt.scatter(feature_4, y, c="black")
"""

