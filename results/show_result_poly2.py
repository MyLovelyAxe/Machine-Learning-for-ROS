import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('/home/user/catkin_ws/src/results/test.csv')

X = np.asarray(data.iloc[:, :-1]).flatten()
y = np.asarray(data.iloc[:, -1]).flatten()

order1 = np.polyfit(X, y, 1)
order2 = np.polyfit(X, y, 2)
x = np.linspace(0, 2.5, 20)
y_pred1 = np.polyval(order1, x)
y_pred2 = np.polyval(order2, x)

plt.figure(figsize=(10, 10))
plt.plot(x, y_pred1, '-r', label="order 1")
plt.plot(x, y_pred2, '--g', label="order 2")
plt.scatter(X, y, label="original")
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)
plt.legend()
plt.title("Robot position", fontsize=18)

plt.show()
