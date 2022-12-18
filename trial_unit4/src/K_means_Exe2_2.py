"""
For more complex data set you are going to deploy an algorithm from sklearn library
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

# definition of object and setup
km = KMeans(n_clusters=5,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
# we learn our model (object) with provided data set
# the return of km.fit_predict():
# labels: ndarray of shape (n_samples,)
# Index of the cluster each sample belongs to.
y_km = km.fit_predict(X)

print(y_km)
print("y_km type: ", type(y_km))
print("y_km shape: ", y_km.shape)

"""
Now you plot our results - cluseters and centroids
"""

plt.figure(figsize=(7, 7))
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen',
            marker='s', edgecolor='black', label='Cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange',
            marker='o', edgecolor='black', label='Cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue',
            marker='v', edgecolor='black', label='Cluster 3')
plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1], s=50, c='orange',
            marker='>', edgecolor='black', label='Cluster 4')
plt.scatter(X[y_km == 4, 0], X[y_km == 4, 1], s=50, c='lightblue',
            marker='<', edgecolor='black', label='Cluster 5')

# cluster_centers_: ndarray of shape (n_clusters, n_features)
#                   Coordinates of cluster centers.
print(km.cluster_centers_)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
            :, 1], s=250, marker='*', c='red', edgecolor='black', label='Centroid')


for i in range(len(km.cluster_centers_)):
    print("Position of centroid (obstacle). X ::", km.cluster_centers_[
          i, 0], "Y ::", km.cluster_centers_[i, 1])

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()
