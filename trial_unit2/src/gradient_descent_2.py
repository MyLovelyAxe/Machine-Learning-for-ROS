# Load the data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import future


MSE = []
iteration = []

data = pd.read_csv('/home/user/catkin_ws/src/results/test.csv')


Y = (data.iloc[:, 1].values).flatten()
X = (data.iloc[:, :-1].values).flatten()


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
epochs = 100000  # The number of iterations to perform gradient descent

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
X_future = 1.4
Y_future = m*X_future + b

plt.figure(figsize=(10, 10))
plt.scatter(X, Y, label='ground truth')
plt.scatter(X_future, Y_future, label='prediction')
plt.plot(X, Y_pred, color='red', label='prediction')  # predicted

plt.xlabel("X", fontsize=16)
plt.ylabel("Y", fontsize=16)
plt.title('Gradient descent', fontsize=18)
plt.legend()
plt.show()
