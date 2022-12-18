#! /usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

radar_data = pd.read_csv('/home/user/catkin_ws/src/results/radar_1_2.csv')
radar_data = np.asarray(radar_data)

print(radar_data.shape)

x1 = radar_data[:, 0]
y1 = radar_data[:, 1]
x2 = radar_data[:, 2]
y2 = radar_data[:, 3]

plt.figure(figsize=(12, 8))
plt.scatter(x1, y1, edgecolor='blue', marker='o', label='radar_1')
plt.scatter(x2, y2, edgecolor='red', marker='o', label='radar_2')
plt.xlabel('x-coordinate of obstacles')
plt.xlabel('y-coordinate of obstacles')
plt.title('radar data of obstacles')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
