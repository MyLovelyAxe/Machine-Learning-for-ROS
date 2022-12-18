"""
We get data set taken by laser and plot
"""

from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# X = pd.read_csv(
#     '/home/user/catkin_ws/src/machine_learning_course/dataset/example_laser_data_points.csv')
X = pd.read_csv(
    '/home/user/catkin_ws/src/results/laser_data_points.csv')
X = np.asarray(X)

plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1],
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.title('Data set')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()

"""
Nnow you run the estimation of clusters (Elbow method) you shoud apply for our algorithm
You plot the the analysis 
"""

distortions = []

for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    estimator = km.fit(X)
    print(estimator)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker="o")
plt.xlabel("number of cluster centroids", fontsize=16)
plt.ylabel("sum of squared distances of points to their centroids", fontsize=16)
plt.title("show distortion of laser data0", fontsize=18)
plt.legend()
plt.show()
