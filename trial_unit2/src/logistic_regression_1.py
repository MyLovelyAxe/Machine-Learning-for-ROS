import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import datasets
import pandas as pd

################################
###### Definition of class #####
################################


class LogisticRegression:
    def __init__(self, lr=0.0001, num_iter=100000):
        self.lr = lr
        self.num_iter = num_iter
        self.logistic_loss = []

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):

        # shape of self.theta: (2,)
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if(i % 1000 == 0):
                self.logistic_loss.append(loss)
            if(i % 5000 == 0):
                print("i:", i, "loss: ", loss)

    def predict_prob(self, X):
        # if self.fit_intercept:
        #   X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()

################################
###### conder the dataset ######
################################


# pd.read_csv() return a result containing:
#   index and data
data = pd.read_csv(
    '/home/user/catkin_ws/src/machine_learning_course/dataset/test_logistic_data.csv')

# print(data.iloc[:, -1])
# data.iloc[:, :-1] return index and the data with every row and every but last column
# but np.asarray(data.iloc[:, :-1]) return only data without index
# shape of X: (58,2)
X = np.asarray(data.iloc[:, :-1])
# data.iloc[:, -1] return index and the data with every row and last column
# but np.asarray(data.iloc[:, -1]) return only data without index
# shape of y: (58,)
y = np.asarray(data.iloc[:, -1])

# give the labels (supervised machine learning)
# data.loc[y == 1]:
#   return a result containing only the part of 'data'
#   which has value '1' in corresponding position defined by 'y'
#   i.e. which has value '1' in position [:,-1]
#   a.k.a this step is to split the data by extracting data point through 'label', i.e. position [:,-1]
deviation_ok = data.loc[y == 1]
deviation_not = data.loc[y == 0]

plt.figure(figsize=(10, 8))

# although deviation_ok.iloc[:,0] contains index and value on position [:,0]
# it is still capable to be x-input of plt.scatter() as values of x-axis
plt.scatter(deviation_ok.iloc[:, 0],
            deviation_ok.iloc[:, 1], s=10, label='OK')
plt.scatter(deviation_not.iloc[:, 0],
            deviation_not.iloc[:, 1], s=10, label='Fail')

plt.xlabel("deviation in X direction", fontsize=16)
plt.ylabel("deviation in Y direction", fontsize=16)
plt.legend()
plt.show()

#################################
###### logistic regression ######
#################################

# creating a object to Logistic Regression
model = LogisticRegression(lr=0.1, num_iter=100000)
# Now we will learn our model
model.fit(X, y)

# print the error while training
plt.figure(figsize=(10, 8))
i = np.arange(0, len(model.logistic_loss), 1)
plt.plot(i, model.logistic_loss)
plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.show()


# print logistic regression applied to data set
plt.figure(figsize=(10, 8))
# check the definition of X and y, it represents that we can index X with y
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],  label='1')
plt.legend()
x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
# np.linspace(): num = 50 defaultly
# here is to create a scatter-plane with scale of 50x50
# xx1 contiains the x-coordinate of all 50x50 points
# xx2 contiains the y-coordinate of all 50x50 points
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                       np.linspace(x2_min, x2_max))
print("xx1 shape", xx1.shape)
print("xx2 shape", xx2.shape)
# numpy.ravel(): return a contiguous flattened array
# numpy._c(): translates slice objects to concatenation along the second axis
grid = np.c_[xx1.ravel(), xx2.ravel()]
print("grid shape: ", grid.shape)
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.xlabel("Deviation - x direction", fontsize=16)
plt.ylabel("Deviation - y direction", fontsize=16)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='red')
plt.show()
