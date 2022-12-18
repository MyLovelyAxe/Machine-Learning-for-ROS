import numpy as np
import pandas as pd
import statsmodels as sm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d

data = pd.read_csv(
    '/home/user/catkin_ws/src/machine_learning_course/dataset/displacement_xyz.csv')


data.drop(['Unnamed: 0'], axis=1, inplace=True)

"""
You are going to standarized our data set (each feature independently).
You are going to depoly StandardScaler() from scikit.learn Python library
"""

data_std = StandardScaler().fit_transform(data)


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

xt = data_std[:, 0]
yt = data_std[:, 1]
zt = data_std[:, 2]

ax.scatter3D(xt, yt, zt, c='crimson', s=60)

ax.set_xlabel('x [mm]', fontsize=14)
ax.set_ylabel('y [mm]', fontsize=14)
ax.set_zlabel('z [mm]', fontsize=14)
plt.title("Standarized Data Set. Displacement",  fontsize=14)
plt.show()
