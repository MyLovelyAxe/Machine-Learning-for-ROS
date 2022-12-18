# Load the data set
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import future
import warnings
warnings.filterwarnings('ignore')


MSE = []
iteration = []
# type of data: <class 'pandas.core.frame.DataFrame'>
data = pd.read_csv(
    '/home/user/catkin_ws/src/machine_learning_course/dataset/test_brakes.csv')
print("type of data: ", type(data))

data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.drop(['payload'], axis=1, inplace=True)

# type of X: <type 'numpy.ndarray'>
# type of Y: <type 'numpy.ndarray'>
X = (data.iloc[:, 1].values).flatten()
Y = (data.iloc[:, :-1].values).flatten()


def compute_error_for_line_given_points(b, m, X, Y):
    totalError = 0  # sum of square error formula
    for i in range(0, len(X)):
        x = X
        y = Y
        totalError += (y[i]-(m*x[i] + b)) ** 2
        mse = (totalError)/totalError.size
        return mse
        # MSE.append(mse)
        # iteration.append(i)
        #print (MSE)
        # return totalError/ float(len(X))


# Initial values
m = 0
b = 0

# Hyperparameters
alpha = 0.0001  # The learning Rate
epochs = 15000  # The number of iterations to perform gradient descent

#n = float(len(X))
n = len(X)


##########################################################################
# OUR OPTIMIZATOR - we use MSE to look for the best paramaters m and b
# We use Gradient Descent
#########################################################################

for i in range(epochs):

    Y_pred = m*X + b
    # IMPLEMENTATION OF DISCUSSED ABOVE EQATIONS FOR GRADIENT DESCENT
    gradient_m = (-2/n) * sum(X * (Y - Y_pred))
    gradient_b = (-2/n) * sum(Y - Y_pred)

    # UPDATE OF m AND b
    m = m - alpha * gradient_m
    b = b - alpha * gradient_b
    mse = compute_error_for_line_given_points(b, m, X, Y)
    MSE.append(mse)
    iteration.append(i)

# performing the regression baseon od gradient descent
Y_pred = m*X + b

plt.figure(figsize=(10, 8))
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)],
         color='red')  # predicted

plt.xlabel("% of max speed of axis 1", fontsize=16)
plt.ylabel("stop distance [deg]", fontsize=16)
plt.title('Gradient descent', fontsize=18)
plt.show()

plt.figure(figsize=(10, 8))
i = np.arange(0, len(MSE), 1)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("error", fontsize=16)
plt.title('Gradient descent ERROR (MSE)', fontsize=18)
plt.plot(i, MSE)
plt.show()
