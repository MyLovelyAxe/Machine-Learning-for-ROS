import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_data():

    X = np.array([[0.3, 8.3], [3, 8], [2, 9], [0.3, 8.9], [1.7, 9.7],
                  [0.9, 10.5], [10.3, 2.1], [10, 2], [7, 7], [6.9, 6.5],
                  [6, 6], [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6],
                  [9, 11], [12, 5], [4.5, 4.8], [4.5, 3], [2, 8], [9, 3], [9, 7]])
    return X


def k_means(X, centroid_num):

    distortion = []

    for i in range(1, centroid_num):

        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X)
        # inertia_: float
        # Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
        distortion.append(km.inertia_)

    return distortion


def main():

    X = get_data()
    distortion = k_means(X, 11)

    plt.plot(range(1, 11), distortion, marker="o")
    plt.xlabel("number of clusters' centroids")
    plt.ylabel("sum of euclidean distance of dataset to centroids")
    plt.title("k means algorithm")
    plt.show()


if __name__ == "__main__":

    main()
