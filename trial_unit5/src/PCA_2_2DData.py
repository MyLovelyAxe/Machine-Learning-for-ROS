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


"""
You are going to search for correlation (dependency) between directions (features) 
"""

plt.figure(figsize=(14, 10))


x = data[['x']]
y = data[['y']]
z = data[['z']]

plt.subplot(131)
plt.scatter(x, y, color='deeppink')
plt.xlabel("x [mm]", fontsize=14)
plt.ylabel('y [mm]', fontsize=14)
plt.title("Displacement XY",  fontsize=14)

plt.subplot(132)
plt.scatter(x, z, color='navy')
plt.xlabel("x [mm]", fontsize=14)
plt.ylabel('z [mm]', fontsize=14)
plt.title("Displacement XZ",  fontsize=14)

plt.subplot(133)
plt.scatter(y, z)
plt.xlabel("y [mm]", fontsize=14)
plt.ylabel('z [mm]', fontsize=14)
plt.title("Displacement YZ",  fontsize=14)

plt.show()
