import numpy as np
import pandas as pd
from common_funcs import polynomial_regression

# import data and split
df1 = pd.read_csv('Datasets/polynomial_eg.csv')

# quadratic noisy dataset
xx_quadratic = np.array(df1['Temperature'].values.tolist())
yy_quadratic = np.array(df1['Ice Cream Sales'].values.tolist())

# cubic (no noise) dataset
x_cubic = np.arange(-5, 5, .5)
y_cubic = [(i-2)*(i+3)*(i-1) for i in x_cubic]

polynomial_regression(3, x_cubic, y_cubic, 1, 0, 0, True, True)

