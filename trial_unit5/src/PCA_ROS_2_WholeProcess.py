#! /usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

############################
###### get radar data ######
############################

data_ori = pd.read_csv('/home/user/catkin_ws/src/results/radar_1_2.csv')
data_ori = np.asarray(data_ori)
# shape of original data: (670,4)
print("shape of original data: ", data_ori.shape)

##########################
###### standarising ######
##########################

data_std = StandardScaler().fit_transform(data_ori)
# shape of standarized data: (670,4)
print("shape of standarized data: ", data_std.shape)

###############################
###### covariance matrix ######
###############################

mean_std = np.mean(data_std, axis=0)
# shape of mean of standarized data: (4,)
print("shape of mean of standarized data: ", mean_std.shape)

cov_std = (np.dot((data_std - mean_std).T, (data_std - mean_std))) / \
    (data_std.shape[0]-1)
# shape of covariance matrix of standarized data: (4,4)
print("shape of covariance matrix of standarized data: ", cov_std.shape)
print("covariance matrix: ")
print(cov_std)

########################################
###### eigenvalue and eigenvector ######
########################################

# each colomn of eigenvectors are eigen vector of each eigen value
eigenvalues, eigenvectors = np.linalg.eig(cov_std)
print("shape of eigenvalues of standarized data: ", eigenvalues.shape)
print("eigenvalues: ")
print(eigenvalues)
print("shape of eigenvectors of standarized data: ", eigenvectors.shape)
print("eigenvectors: ")
print(eigenvectors)

# sort eigenvalues
# np.argsort() return indices of ascending order
# np.argsort().[::-1] return indices of descending order
des_idx_eigval = np.argsort(eigenvalues)[::-1]
print("eigen valus in descending order: ", eigenvalues[des_idx_eigval])

# compute importance of each component
importance_eigval = (eigenvalues / np.sum(eigenvalues)) * 100
print("importance of each eigenvectors: ", importance_eigval)

#################################
###### projection matrix W ######
#################################

# a = eigenvectors[:, des_idx_eigval[0]]
# print(a.shape)
# print(np.hstack((a, a)).shape)
# print(np.vstack((a, a)).shape)

# drop the eigenvectors with too low importances
matrix_w = np.hstack(
    (eigenvectors[:, des_idx_eigval[0]].reshape(4, 1), eigenvectors[:, des_idx_eigval[1]].reshape(4, 1)))
print("projection matrix: ")
print(matrix_w)

########################################
###### projection to new subspace ######
########################################

# data_std: (670,4) with 670 points in 4D space
# matrix_w: (4,2)
# np.dot(data_std, matrix_w): (670,2) with 670 points in 2D space
data_std = np.dot(data_std, matrix_w)
print("shape of standarized data in 2D subspace: ", data_std.shape)

plt.figure(figsize=(12, 8))
plt.scatter(data_std[:, 0], data_std[:, 1], marker='>',
            color='red', label='radar data in new subspace')
plt.xlabel('x-coordinate of obstacles')
plt.ylabel('y-coordinate of obstacles')
plt.title('radar data in new subspace')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
