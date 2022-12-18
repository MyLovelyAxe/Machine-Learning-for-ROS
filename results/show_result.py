import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('/home/user/catkin_ws/src/results/test.csv')

X = np.asarray(data.iloc[:, :-1])
y = np.asarray(data.iloc[:, -1])

plt.figure(figsize=(10, 10))
plt.scatter(X, y)
plt.xlabel("X", fontsize=16)
plt.ylabel("Y", fontsize=16)
plt.title("Robot position", fontsize=18)

plt.show()
