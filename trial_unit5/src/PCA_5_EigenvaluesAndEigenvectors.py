import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(
    '/home/user/catkin_ws/src/machine_learning_course/dataset/displacement_xyz.csv')

data.drop(['Unnamed: 0'], axis=1, inplace=True)

# shape of data_std: (45,3)
# i.e. 45 points with 3-dimensional (x,y,z)
# we want to project them into a new subspace which is only 2-dimensional
data_std = StandardScaler().fit_transform(data)

print(data_std.shape)
print('\n')

###########################################
######  Eigenvalues and Eigenvectors ######
###########################################

"""
For our data set, you are going to compute the eigenvectors and eigenvalues.
In order to keep the course simple, you do not calculate these values from scratch.
You use the numpy.
"""

cov_mat = np.cov(data_std.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' % eigenvectors)
print('\nEigenvalues \n%s' % eigenvalues)
print('\n')

#################################
###### projection matrix W ######
#################################

"""
Now you sort our eignevalues in descending order in order to evaluate potential feature removal
"""

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i])
             for i in range(len(eigenvalues))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
print('\n')

"""
You commpute how each feature influences the characteristic of the data set - you compute explained variance
"""

# compute percentage of importance for each eigenvalues
eig_val_sum = sum(eigenvalues)
sorted_eig_val = [(j/eig_val_sum)*100 for j in eigenvalues]
print('the percentage of importance for each eigenvalues: ')
print(sorted_eig_val)
print('\n')

plt.figure(figsize=(8, 10))
plt.bar(range(3), sorted_eig_val, alpha=1.0, align='center',
        color='green', label='individual explained variance')
plt.xlabel('principle components')
plt.ylabel('explained variance ratio')
plt.title('principle components', fontsize=14)
plt.show()

"""
constructing projection matrix
"""
# because of the result above
# we know that the least eigenvalue from component 'Z' is almost negligable
# so we drop this component, and keep 'X' and 'Y'

matrix_w = np.array([]).reshape(0, 3)
for k in eig_pairs[:2]:
    # np.vstack take only 1 argument
    # so contain all elements which you want to stack vertically in one ()
    matrix_w = np.vstack((matrix_w, k[1]))
matrix_w = matrix_w.T
print('projection matrix W:')
print(matrix_w)
print('\n')

#############################################
###### Projection to new feature space ######
#############################################

# data_std has shape of (45,3), which contains 45 points with 3-dimensional (x,y,z)
# we drop the 'Z-dimension', and project them into 2-dimensional subspace (x,y)
# matrix_w has shape of (3,2)
# then data_std * matrix_w = (45,3) * (3,2) = (45,2)
# i.e. 45 points with 2-dimension (x,y)
projected_data = np.dot(data_std, matrix_w)

plt.figure(figsize=(12, 8))
plt.scatter(projected_data[:, 0], projected_data[:, 1],
            color='blue', label='projected data')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('the projected data in 2D subspace')
plt.show()
