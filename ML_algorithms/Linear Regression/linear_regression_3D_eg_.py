import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# design matrix X is NxK = 15x2
# labels always Nx1 = 15x1
X = np.array([[0.8, 1.4], [1.9, 2.5], [3.1, 5.5]])
yy = np.array([1.1, 2.3, 2.9])
points = np.c_[X, yy]
c1, c2, c3 = points

# note this will mean plane needed to visualise as each feature own axis, 2 features here with final axis labels
##Prepare the Dataset
df2=pd.DataFrame(X,columns=['x1','x2'])
df2['y']=pd.Series(yy)

##Fit the algorithm
Regressor = LinearRegression()
Regressor.fit(X,yy)

## Prepare the data for Visualization
x_surf, y_surf = np.meshgrid(np.linspace(df2.x1.min(), df2.x1.max(), 100),np.linspace(df2.x2.min(), df2.x2.max(), 100))
onlyX = pd.DataFrame({'x1': x_surf.ravel(), 'x2': y_surf.ravel()})
fittedY=Regressor.predict(onlyX)

## convert the predicted result in an array
fittedY=np.array(fittedY)

# Visualize the Data for Multiple Linear Regression
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(20,10))

## find plane equation
def plane_eq(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    ax.set_title("%.2f(x1) + %.2f(x2) + %.2f(y) = %.2f"%(a, b, c, d))

### plot
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['x1'],df2['x2'],df2['y'],c='red', marker='o', alpha=0.5, label="(x1, x2, y)")
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3, label="see title")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.legend()
plane_eq(c1, c2, c3)
plt.show()

