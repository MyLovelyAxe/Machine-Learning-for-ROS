"""
First you are going to load the data set and display
"""

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


fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection="3d")

x = data[['x']]
y = data[['y']]
z = data[['z']]


ax.scatter3D(x, y, z, c='blue', s=60)


ax.set_xlabel('x [mm]', fontsize=14)
ax.set_ylabel('y [mm]', fontsize=14)
ax.set_zlabel('z [mm]', fontsize=14)
plt.title("Data Set. Displacement",  fontsize=14)
plt.show()
