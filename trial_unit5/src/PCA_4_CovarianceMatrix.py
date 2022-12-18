import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(
    '/home/user/catkin_ws/src/machine_learning_course/dataset/displacement_xyz.csv')

data.drop(['Unnamed: 0'], axis=1, inplace=True)

data_std = StandardScaler().fit_transform(data)

"""
For our example data set you compute covariance matrix
"""

mean_vec = np.mean(data_std, axis=0)
cov_mat = (data_std - mean_vec).T.dot((data_std - mean_vec)) / \
    (data_std.shape[0]-1)
print('Covariance matrix \n%s' % cov_mat)
