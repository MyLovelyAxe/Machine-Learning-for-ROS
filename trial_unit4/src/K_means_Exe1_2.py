import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class Kmeans_dev:
    """
    class definition
    """

    def __init__(self, X, K):
        """
        class constructor
        """
        self.X = X
        self.output = {}
        # self.centroids is a vacant numpy.array with shape of (self.X.shape[1], 0)
        self.centroids = np.array([]).reshape(self.X.shape[1], 0)
        self.K = K
        self.m = self.X.shape[0]

    def start_centroid_pos(self, X, K):
        """
        Random initialization of K centroids.
        """
        # m: number of points, here is 23
        # n: dimensionality of points, which is 2, a.k.a coordinates of (x,y), here is 2
        m, n = X.shape[0], X.shape[1]
        centroids = np.zeros((K, n))

        # initialization of centroids with random k points from dataset X
        for i in range(K):
            # for i in range(1,K+1,1):
            centroids[i] = X[np.random.randint(0, m), :]

        return centroids

    def fit(self, n_iter):
        """
        Method to train the data set (position of centroids()
        """
        # randomly Initialize the centroids (callstart_centroid_pos() )
        self.centroids = self.start_centroid_pos(self.X, self.K)

        # compute Euclidian distances and assign clusters
        for n in range(n_iter):
            # EuclidianDistance: (23,0)
            EuclidianDistance = np.array([]).reshape(self.m, 0)
            for k in range(self.K):
                # the distance between all points and specific centroid
                # self.X: (23,2)
                # self.centroids[k]: (1,2)
                # tempDist: (23,)
                tempDist = np.sum((self.X-self.centroids[k, :])**2, axis=1)
                print("tempDist shape is: ", tempDist.shape)
                # EuclidianDistance: along with row: distances between same point to different centroids
                #                    along with colomn: distances between different points to same centroid
                EuclidianDistance = np.c_[EuclidianDistance, tempDist]
                print("EuclidianDistance shape is: ", EuclidianDistance.shape)
            C = np.argmin(EuclidianDistance, axis=1)+1
            print("centroid: ", C)

            # adjust the centroids
            Y = {}
            for k in range(self.K):
                # define keys for each centroid, keys are k+1
                Y[k+1] = np.array([]).reshape(2, 0)
            for i in range(self.m):
                # stack the points according to "C" to keys in Y, either key=1 or key=2
                # e.g. Y{1:array_1, 2:array_2}
                # array1 has shape of (2,point_num_belong_to_cluster_1)
                Y[C[i]] = np.c_[Y[C[i]], self.X[i]]
                print("the {}th point".format(i+1))
                print("key=1 shape: ", Y[1].shape)
                print("key=2 shape: ", Y[2].shape)
                print("key=3 shape: ", Y[3].shape)
                print("key=4 shape: ", Y[4].shape)
                print()

            for k in range(self.K):
                Y[k+1] = Y[k+1].T
            for k in range(self.K):
                # take the mean value of all points belonging to specific cluster as updated centriod for next iteration
                self.centroids[k, :] = np.mean(Y[k+1], axis=0)

            self.output = Y

    def predict(self):
        """
        Return of data set adherence to certain cluster
        """

        return self.output, self.centroids.T


# data set
X = np.array([[0.3, 8.3], [3, 8], [2, 9], [0.3, 8.9], [1.7, 9.7],
              [0.9, 10.5], [10.3, 2.1], [10, 2], [7, 7], [6.9, 6.5],
              [6, 6], [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6],
              [9, 11], [12, 5], [4.5, 4.8], [4.5, 3], [2, 8], [9, 3], [9, 7]])

"""
Plot the data set
"""

plt.figure(figsize=(8, 8))

plt.scatter(X[:, 0], X[:, 1])
plt.title('Data set')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()

"""
Run k-means algorithm on given data set. Printing the output (position of centrod) and adherence of point to
certain cluster
"""

K = 4  # number of cluster you would like to create

# creation of class object and training (fit)
kmeans = Kmeans_dev(X, K)
kmeans.fit(50)

color = ['blue', 'green', 'red', 'yellow']
labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4']

fig, axs = plt.subplots(5, 2, figsize=(14, 28), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.001)

axs = axs.ravel()

# you print the test results iteration by iteration (in order to show how the centroids moves)

for i in range(10):
    kmeans = Kmeans_dev(X, K)

    kmeans.fit(1)

    Output, Centroids = kmeans.predict()
    print("the shape of centroids: ", Centroids.shape)

    # draw points
    for k in range(K):
        axs[i].scatter(Output[k+1][:, 0], Output[k+1]
                       [:, 1], c=color[k], label=labels[k])

    # draw centroids
    print("iteration{}: ".format(i+1))
    print("centroids: ", Centroids)
    axs[i].scatter(Centroids[0, :], Centroids[1, :], s=300,
                   c='red', label='Centroids', marker='*')

    axs[i].set_title("Centroids movement. Iteration " + str(1+i))

plt.show()
