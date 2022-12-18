from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# data set
day = np.linspace(1, 10, 10)
future_day = np.linspace(1, 12, 12)
temp = [22, 18, 19, 26, 28, 18, 21, 27, 28, 26]

# simple regression line without optimizing the slope

m = (temp[-1] - temp[0])/(10-1)
#y = m*x + b
b = temp[-1] - m*10

# plot the data and line
plt.figure(figsize=(10, 8))
plt.axes(xlim=(0, 15), ylim=(15, 30), autoscale_on=False)
plt.scatter(day, temp)
plt.plot(future_day, (m*future_day+b))
plt.xlabel("day", fontsize=14)
plt.ylabel("temperature [C]", fontsize=14)
plt.title("Registrered temperature [C]", fontsize=16)

plt.show()
