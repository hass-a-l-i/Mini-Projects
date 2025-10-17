import matplotlib.pyplot as plt
import numpy as np
import math

x = []
for i in range(0, 1000):
    x.append(np.random.randn())

y = []
for i in range(len(x)):
    y_point = math.exp(-(x[i]**2)/2)
    y.append(y_point)

print("Variance = ", np.var(x))
print("Mean = ", np.average(x))

plt.scatter(x, y, c='r', marker=".", s=10)
plt.show()